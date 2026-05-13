# 生成到 Judge Prompt 打包流程说明

这个目录把 `generate.py` 到 `judge prompt` 打包之间的核心代码整理成一个可读快照，方便检查和使用。

重要约定：

- `source_snapshot/` 是三端统一后的代码快照，便于阅读和对账。
- 实际运行命令建议仍在 benchmark 仓库根目录执行，使用原始路径，例如 `python generate.py ...`、`python tools/prompts/pack_prompt.py ...`。
- 三端基准为 159 当前实验代码 `b6f71e1`，并额外统一修改了 `benchmark_core/prompt.py` 中的 `Claim_Segment_Prompt`：不限制 claim 数量，不限制 claim 文本长度。
- 本目录不包含模型权重、输入数据、输出产物、日志和历史备份文件。

## 目录内容

- `README.md`：当前说明文档。
- `COMPATIBILITY_BRANCHES.md`：兼容分支目的、位置和人工审核重点说明。
- `MANIFEST.json`：结构化文件清单、hash、角色说明。
- `FILE_MANIFEST.tsv`：便于肉眼检查的清单。
- `CHECKSUMS.sha256`：代码快照校验用 sha256。
- `BUNDLE_CHECKSUMS.sha256`：整个整理目录校验用 sha256。
- `source_snapshot/`：整理后的代码快照。
- `manual_audit_slices/full_flow_slice_20260507/`：全流程输入输出抽样切片，覆盖 clean input、generation prompt、模型输出和 judge prompt cache。

主要入口文件：

- `generate.py`：直接本地模型生成入口。
- `tools/prompts/pack_prompt.py`：把 `gen_only.jsonl` 打包成 judge prompts。
- `tools/prompts/build_generate_prompt_pack.py`：把输入数据打成本地 vLLM generation prompt pack。
- `tools/prompts/run_generate_prompt_pack.py`：执行 generation prompt pack，只走本地 vLLM 后端。
- `scripts/run_completed_annotations_generate_and_pack.py`：单模型端到端 wrapper。
- `scripts/run_manifest_models_completed_annotations.py`：多模型自动化实验 runner。
- `scripts/probe_manifest_model_continuation.py`：本地 tokenizer assistant 续写能力预检。
- `scripts/monitor_completed_annotation_runs.py`：实验监控和异常检测。

## 数据输入格式

核心输入通常是：

```text
workflow_data/annotation_exports/completed_annotation_records/purified_cases.jsonl
workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl
```

每行是一条 case/sample 级记录。生成和打包流程主要依赖这些字段：

- `id` / `annotation_uid`：样本级唯一 ID。避免同一个 `case_id` 多条 sample 时互相覆盖。
- `case_id` / `original_case_id`：原题号或 case 号，用于追踪。
- `question` / `problem`：题面。
- `answer` / `standard_solution`：参考答案或标准解。
- `reference_steps` / `steps`：参考步骤列表。
- `reference_claims_by_step` / `claims_by_step`：按 step 组织的 reference claims。
- `reference_step_dependencies` / `step_dependencies`：step dependency 信息。

`generate.py` 会读取这些字段，构造逐步续写 prompt，并生成每个 step prefix 对应的模型输出。

## 流程 A：直接跑本地模型生成，再打包 judge prompts

适合本地 HF/vLLM 模型。

### 1. 小规模 smoke 生成

```bash
cd /home/zhaipengxiang/benchmark
python generate.py   --input_path workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl   --out_root artifacts/model_outputs/completed_annotations_smoke   --tag smoke_manual   --max_cases 10
```

输出目录形如：

```text
artifacts/model_outputs/completed_annotations_smoke/<model_name>_smoke_manual/
```

关键输出：

- `gen_only.jsonl`：下一步 judge prompt 打包输入。
- `gen_only_pretty.json`：便于人工阅读。
- `run_info.json`：生成 manifest。

### 2. 检查 smoke 输出

重点检查：

- `gen_only.jsonl` 行数是否等于输入 case 数。
- `gen_output` 是否大量为空。
- 是否出现 replacement character / 乱码 / chat template 泄漏。
- `gen_prefix` 是否有内容。
- 每条记录是否保留 `steps`、`claims_by_step`、`step_dependencies` 等 judge 打包字段。

### 3. 打包 judge prompts

```bash
python tools/prompts/pack_prompt.py   --gen_file artifacts/model_outputs/completed_annotations_smoke/<model_name>_smoke_manual/gen_only.jsonl   --out_dir artifacts/model_outputs/completed_annotations_smoke/<model_name>_smoke_manual/packed_prompts   --write_all
```

输出：

```text
packed_prompts/cache_prompts/case_<id>_cache.jsonl
packed_prompts/cache_prompts/ALL_cache.jsonl
```

`pack_prompt.py` 会为每个 scored prefix 生成四类 judge 请求：

- `pairwise`：当前生成 step prefix vs dependency claim。
- `holistic`：当前生成 step prefix vs 已有参考前缀。
- `selfjudge_without_reference`：只检查生成片段内部自洽。
- `selfjudge_with_reference`：当前生成片段 vs 当前 step 的 reference claim。

默认沿用旧 scoring window：不打包最后两个 generated prefix 位置。

## 流程 B：单模型端到端 wrapper

推荐用于正式单模型跑法，因为它会等待 GPU、设置 Config、调用 `generate.py`、再调用 `pack_prompt.py`，并持续写 status JSON。

```bash
cd /home/zhaipengxiang/benchmark
/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12   scripts/run_completed_annotations_generate_and_pack.py   --input-path workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl   --out-root artifacts/model_outputs/completed_annotations_manual   --tag smoke_<model_slug>   --model-path /data/pretrain/<model_dir>   --gpus 0,1   --tensor-parallel-size 2   --dtype bfloat16   --max-model-len 12288   --max-num-seqs 64   --max-num-batched-tokens 16384   --gpu-memory-utilization 0.82   --min-tokens 16   --max-cases 10   --wait-gpu-free-mib 50000   --wait-gpu-max-util 10   --wait-poll-seconds 120   --status-path logs/manual_smoke_status.json   --write-all-prompts
```

通过 smoke 后，把 input 换成全量：

```bash
--input-path workflow_data/annotation_exports/completed_annotation_records/purified_cases.jsonl
--tag full_<model_slug>
--max-cases 100000
--status-path logs/manual_full_status.json
```

重要参数：

- `--gpus`：选择 GPU，例如 `0,1`。
- `--tensor-parallel-size`：通常与 GPU 数一致。
- `--wait-gpu-free-mib` / `--wait-gpu-max-util`：避免抢占已有任务。
- `--chat-template-no-system-role`：给不支持 system role 的模型使用。
- `--chat-template-system-suffix`：给特殊模板补充 system 内容。
- `--chat-template-first-user-prefix`：给特殊模板补充首个 user 前缀。
- `--sampling-bad-words-json`：屏蔽模型容易泄漏的特殊 token。
- `--sampling-stop-token-ids-json`：设置 stop token ids。
- `--sampling-ignore-eos`：模型过早 EOS 时可尝试，但要先 smoke 验证。
- `--min-tokens`：降低空续写概率，但不能保证语义正确。

## 流程 C：Generation Prompt Pack / vLLM 路径

适合希望先打包 generation prompt，再用本地 vLLM 批量生成的场景。当前生成阶段不走 API；Kimi/Moonshot、DeepSeek API 的 partial/prefill 逻辑不在这条流程里。

### 1. 构建 generation prompt pack

默认格式是 `vllm-messages`：最后一条 assistant message 带 `prefix: true`，具体如何渲染为模型可续写的 prompt 由 `VLLMRunner` 统一处理。

```bash
python tools/prompts/build_generate_prompt_pack.py   --input-path workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl   --output tools/prompts/output_vllm_generation_prompt_packs/test/small_test_generate_prompts.jsonl   --model /data/pretrain/<model_dir>   --prompt-format auto   --max-cases 10   --manifest tools/prompts/output_vllm_generation_prompt_packs/test/small_test_generate_prompts.manifest.json
```

可选 `--prompt-format`：

- `auto`：等价于 `vllm-messages`。
- `vllm-messages`：保留 messages，交给 `VLLMRunner` 按模型选择续写渲染方式。
- `chat-template`：提前用本地 tokenizer 渲染 `continue_final_message=True`，主要用于本地调试，不作为默认正式路径。

### 2. dry-run 检查 prompt pack

```bash
python tools/prompts/run_generate_prompt_pack.py   --prompt-pack tools/prompts/output_vllm_generation_prompt_packs/test/small_test_generate_prompts.jsonl   --out-dir /tmp/vllm_small_dryrun   --backend vllm   --model /data/pretrain/<model_dir>   --dry-run
```

如果这里报 `has no prompt/messages field`，基本说明传错文件，例如传了 manifest、cache、batch wrapper，而不是 generation prompt pack。

### 3. 实际调用模型

```bash
python tools/prompts/run_generate_prompt_pack.py   --prompt-pack tools/prompts/output_vllm_generation_prompt_packs/test/small_test_generate_prompts.jsonl   --out-dir gen_output/vllm_small_test_prompt_pack_run   --backend vllm   --model /data/pretrain/<model_dir>   --gpus 0,1   --batch-size 32   --max-empty-retries 2
```

输出：

- `prompt_outputs.jsonl`：每条 prompt 的原始输出。
- `gen_only_from_prompt_pack.jsonl`：聚合成和 `generate.py` 类似的 `gen_only` 格式。
- `run_info.json`：prompt 数、空输出数、耗时等。

然后继续：

```bash
python tools/prompts/pack_prompt.py   --gen_file gen_output/vllm_small_test_prompt_pack_run/gen_only_from_prompt_pack.jsonl   --out_dir gen_output/vllm_small_test_prompt_pack_run/packed_prompts   --write_all
```

## 流程 D：多模型 manifest 自动化

159 服务器正式多模型实验主要用：

```bash
cd /home/zhaipengxiang/benchmark
COMPLETED_MANIFEST_PY=/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12 COMPLETED_MANIFEST_OUT_ROOT=/home/zhaipengxiang/benchmark/artifacts/model_outputs/completed_annotations_manifest_models COMPLETED_MANIFEST_LOG_ROOT=/home/zhaipengxiang/benchmark/logs/completed_annotations_manifest_models python scripts/run_manifest_models_completed_annotations.py
```

只跑指定模型：

```bash
COMPLETED_MANIFEST_MODEL_ALLOWLIST="<model_name_1>,<model_name_2>" python scripts/run_manifest_models_completed_annotations.py
```

该脚本会：

1. 读取 `model/download_manifest.tsv`。
2. 解析本地模型路径，默认在 `/data/pretrain` 下找。
3. 对每个模型先跑 tokenizer continuation probe。
4. 跑 10-case smoke。
5. 检查 smoke 的 `gen_only` 和 packed prompts。
6. smoke 通过后跑 full。
7. 对 full 输出和 judge prompt cache 做完整性检查。
8. 写入 `logs/completed_annotations_manifest_models/model_results.jsonl`、`summary.json`、`continuation_issues.md`、`continuation_compatibility_report.md`。

## Assistant 续写逻辑

本实验不是普通“让模型完整回答题目”，而是 continuation：给定参考解前缀，让模型从 assistant 的最后一段继续写。

本地 HF/vLLM 路径：

- 优先使用 tokenizer 的 `apply_chat_template(..., continue_final_message=True)`。
- 如果模型不支持该参数或 chat template 不能渲染 assistant-prefix，预检会记录失败。
- 不把任务改写成语义 cue/fallback prompt，避免改变实验任务定义。

特殊模型适配：

- `--chat-template-no-system-role`：把 system 合并进第一个 user。
- `--chat-template-system-suffix`：给 system 内容补模型特定说明。
- `--chat-template-first-user-prefix`：给首个 user 内容加前缀。
- `--sampling-bad-words-json` 和 `--sampling-stop-token-ids-json`：处理特殊 token 泄漏、工具 token 泄漏、过早/过晚停止。

## 容错和停止规则

推荐顺序：先 preflight，再 10-case smoke，再 full。

preflight 阶段：

- 如果 tokenizer 没有 chat template，或不支持 assistant continuation，记录并跳过该模型。
- 如果 chat template 渲染报错，先记录原因，不直接上 GPU 全量跑。

smoke 阶段：

- 如果出现大量空输出，停止该模型，不进入 full。
- 如果出现 replacement character / 乱码 / chat template 泄漏，停止该模型。
- 如果 runtime traceback、OOM、KV-cache 错误重复出现，停止该模型。
- 如果 prompt/cache mismatch，停止该模型。

full 阶段：

- 如果评分窗口内空输出超过阈值，停止该模型并保留日志。
- 如果 GPU 忙但 30 分钟没有进展，检查日志；确认非偶发问题后停止该模型。
- 不杀无关用户进程。
- 不因为一次 SSH/网络瞬断停止实验。

API prompt pack 路径：

- `run_generate_prompt_pack.py` 支持 `--max-empty-retries`，只对空输出做原 prompt 重试。
- 重试仍为空会记录 `empty_generation_indices` 和 `empty_generation_count`，供后续验证决定是否停止。

## Judge Prompt 打包输出结构

`pack_prompt.py` 输出目录：

```text
<out_dir>/cache_prompts/
  case_<case_id>_cache.jsonl
  ALL_cache.jsonl   # 仅 --write_all 时生成
```

每行包含：

- `request_id`：请求 ID。
- `route`：`pairwise` / `holistic` / `selfjudge_without_reference` / `selfjudge_with_reference`。
- `idx`：生成 step prefix 位置。
- `prompt`：judge messages。
- `schema`：期望 JSON schema。
- `meta`：step id、claim id、依赖 claim、prefix 长度等定位信息。

## 可选：执行 judge prompt

如果要继续执行 judge prompt cache：

```bash
python tools/prompts/stage2_judge_from_cache_prompts.py   --prompt_dir <packed_prompts/cache_prompts>   --gen_file <gen_only.jsonl>   --run_dir <judge_run_dir>   --max_workers 32   --batch_size 256   --max_tokens 512
```

本文件依赖 API runner，需要本地环境有 `openai` 等依赖和 API key。

## 当前 no-limit claim 切分要求

`benchmark_core/prompt.py` 中 `Claim_Segment_Prompt` 已统一为：

- 不写 `≤10 items`。
- 不写 `≤80 chars`。
- schema 中无 `maxLength`。
- schema 中无 `maxItems`。
- 明确提示：`Do not impose or mention any fixed number of propositions.`

这点对三端都已校验。

## 三端一致性检查命令

在三端 repo 根目录分别执行：

```bash
python -m py_compile   generate.py runner.py   benchmark_core/*.py   tools/prompts/build_generate_prompt_pack.py   tools/prompts/run_generate_prompt_pack.py   tools/prompts/pack_prompt.py   tools/prompts/stage2_judge_from_cache_prompts.py   scripts/run_completed_annotations_generate_and_pack.py   scripts/run_manifest_models_completed_annotations.py   scripts/probe_manifest_model_continuation.py   scripts/monitor_completed_annotation_runs.py
```

校验整理目录快照：

```bash
cd generation_to_judge_pipeline
sha256sum -c CHECKSUMS.sha256
```

macOS 没有 `sha256sum` 时可用：

```bash
python - <<'PY'
import hashlib
from pathlib import Path
ok = True
for line in Path('CHECKSUMS.sha256').read_text().splitlines():
    expected, path = line.split('  ', 1)
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        print('BAD', path, expected, actual)
        ok = False
print('OK' if ok else 'FAILED')
raise SystemExit(0 if ok else 1)
PY
```
