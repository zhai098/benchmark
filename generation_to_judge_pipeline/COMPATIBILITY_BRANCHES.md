# 兼容分支说明：生成到 Judge Prompt 打包流程

本文说明当前生成实验代码中新增的条件分支、适配逻辑和容错逻辑。核心原则是：**不改变实验任务语义**，仍然执行“assistant-prefix 续写生成 -> `gen_only.jsonl` -> judge prompt 打包”。当前生成阶段只保留本地 vLLM 路径；兼容分支只用于不同本地模型的原生续写渲染、处理历史数据格式、提前发现会浪费算力的异常。

## 总体原则

- 本地 HF/vLLM 模型：优先使用 tokenizer 原生 `continue_final_message=True`。
- DeepSeek-V4 类特殊模型：如果模型目录提供原生 encoding 脚本，使用其原生消息编码。
- prompt pack 默认输出 `vllm-messages`，最后一条 assistant message 只带 `prefix: true`，不再带 API 用的 `partial: true`。
- 如果某模型没有可用的本地 assistant-prefix continuation 机制，不改写成语义 cue/fallback prompt，而是在 preflight/smoke 阶段记录并停止。
- 空输出重试只重试原 prompt，不改变 prompt 内容。
- 兼容历史字段名和旧 cache 命名，是为了让已有数据能进入同一条 pipeline，不改变评分口径。

## 1. 本地 HF/vLLM 续写适配

位置：`source_snapshot/runner.py:233`

目的：把 chat messages 转成本地 vLLM 可用的 prompt 文本，同时保留“最后一条 assistant 是待续写前缀”的语义。

关键位置：

- `runner.py:243`：识别最后一条 assistant message 是否带 `prefix`。
- `runner.py:257`：如果 tokenizer 有 `apply_chat_template`，优先使用官方 chat template。
- `runner.py:260`：检查 tokenizer 是否支持 `continue_final_message`。
- `runner.py:261`：不支持原生 assistant continuation 时直接报错，避免悄悄改成普通问答 prompt。
- `runner.py:268`：根据是否 continuation 控制 `add_generation_prompt`。
- `runner.py:270`：支持时传 `continue_final_message=True`。
- `runner.py:272`：部分 reasoning template 需要 `reasoning_effort`，按模板内容补默认值。
- `runner.py:277`：去掉 assistant prefill 后模板错误追加的 EOS，避免模型直接停止或泄漏模板 token。

为什么需要：不同 HF tokenizer 的 chat template 行为不同。有些会支持 continuation，有些会在 assistant 前缀后追加 EOS，有些会需要额外模板参数。这里的分支是为了使用模型原生模板，同时保持 continuation 任务定义。

## 2. 特殊 tokenizer / 模型模板适配

位置：`source_snapshot/runner.py:130`

目的：支持不接受 system role、需要 system suffix、或需要首个 user 前缀的模型模板。

关键位置：

- `runner.py:130`：读取 `generation_chat_template_first_user_prefix`。
- `runner.py:131`：给 system message 追加模型需要的 suffix。
- `runner.py:138`：给第一个 user message 加 prefix。
- `runner.py:148`：DeepSeek-V4 如果有 `encoding/encoding_dsv4.py`，走原生编码器。
- `runner.py:163`：DeepSeek-V4 continuation 时对最后 assistant 设置 `wo_eos=True`。

为什么需要：部分模型模板不能直接处理标准 `system/user/assistant` 三段结构；如果硬套模板，会造成渲染失败、空输出或模板 token 泄漏。这里的适配只调整模板渲染方式，不改变续写任务内容。

## 3. 已移出生成阶段的 API 适配

历史版本曾在 generation prompt pack 中支持 Kimi/Moonshot `partial: true` 和 DeepSeek API backend。当前生成阶段明确只跑本地 vLLM，因此：

- `source_snapshot/benchmark_core/prompt.py` 不再生成 `partial: true`。
- `source_snapshot/tools/prompts/build_generate_prompt_pack.py` 不再提供 `moonshot-partial` / `messages` 格式。
- `source_snapshot/tools/prompts/run_generate_prompt_pack.py` 的 `--backend` 只允许 `vllm`。
- `source_snapshot/runner.py` 中仍保留 API runner 类，供仓库其他历史流程使用；它们不再是本 generation prompt pack 路径的一部分。

## 4. Generation Prompt Pack 多格式适配

位置：`source_snapshot/tools/prompts/build_generate_prompt_pack.py:121`

目的：同一份输入数据可以打成本地 vLLM messages prompt pack；可选提前渲染 tokenizer chat template 仅用于本地调试。

关键位置：

- `build_generate_prompt_pack.py:121`：构造标准 system/user/assistant 三段 messages。
- `build_generate_prompt_pack.py:126`：assistant 前缀只写 `prefix: true`。
- `build_generate_prompt_pack.py:136`：`chat-template` 可选预渲染。
- `build_generate_prompt_pack.py:163`：`auto` 固定解析为 `vllm-messages`。
- `build_generate_prompt_pack.py:220`：记录 `continuation_mode`，方便追踪使用了哪种续写机制。

为什么需要：正式路径统一交给 `VLLMRunner` 做模型分支，prompt pack 只保存可追踪、可复核的 assistant-prefix messages，避免 API partial 和 tokenizer 预渲染混在一起。

## 5. Prompt Pack 执行层兼容

位置：`source_snapshot/tools/prompts/run_generate_prompt_pack.py:146`

目的：执行 generation prompt pack 时只调用本地 vLLM，同时保持输入格式严格。

关键位置：

- `run_generate_prompt_pack.py:146`：统一调用 `model.generate(prompts, None)`。
- `run_generate_prompt_pack.py:169`：空输出重试，只重试原 prompt，不改变 prompt 语义。
- `run_generate_prompt_pack.py:189`：可选检测“重启答题/元分析”输出并用 continuation guard 重试。
- `run_generate_prompt_pack.py:248`：严格只接受顶层 `prompt` 或 `messages`。
- `run_generate_prompt_pack.py:256`：构造 `VLLMRunner`，非 vLLM backend 直接拒绝。
- `run_generate_prompt_pack.py:341`：命令行 `--backend` 只允许 `vllm`。

为什么需要：删除 API runner 和签名探测后，调用链更短、更可读；输入格式保持严格，是为了避免误传 manifest/cache 文件。

## 6. `generate.py` 对历史数据和 runner 返回格式的兼容

位置：`source_snapshot/generate.py:35`

目的：保证清洗后的标注数据、旧格式 step 和本地 vLLM 返回值都能进入同一个 `gen_only.jsonl` 结构。

关键位置：

- `generate.py:20`：只构造 `VLLMRunner`，不再有 API runner fallback。
- `generate.py:41`：兼容 runner 返回 `(reasonings, generations)`。
- `generate.py:48`：兼容只返回 generations 的 runner。
- `generate.py:52`：统一调用 `reasoning_model.generate(prompts, None)`。
- `generate.py:84`：兼容旧数据里 step 被存成字符串化 dict 的情况。
- `generate.py:105`：兼容 `reference_steps / steps / segments` 三种字段名。
- `generate.py:146`：清理 `<think>`、tool call 特殊 token，避免污染 judge prompt。
- `generate.py:174`：优先使用 `annotation_uid/id/sample_idx` 构造唯一 ID，避免同题多 sample 覆盖。

为什么需要：输入数据经历过多轮清洗和历史版本，字段名不完全一致。这里的兼容保证输出统一为后续 `pack_prompt.py` 能消费的 `gen_only.jsonl`，但生成模型调用只走本地 vLLM。

## 7. 单模型 wrapper 的运行参数兼容

位置：`source_snapshot/scripts/run_completed_annotations_generate_and_pack.py:99`

目的：不修改 `benchmark_core/config.py` 文件本身，而是在运行时临时注入模型、GPU、tokenizer、采样参数，方便同一脚本跑不同模型。

关键位置：

- `run_completed_annotations_generate_and_pack.py:80`：等待 GPU 空闲，避免抢占已有任务。
- `run_completed_annotations_generate_and_pack.py:147`：有 `--config-format` 才注入 vLLM config。
- `run_completed_annotations_generate_and_pack.py:149`：有 `--load-format` 才注入。
- `run_completed_annotations_generate_and_pack.py:151`：有 `--tokenizer-mode` 才注入。
- `run_completed_annotations_generate_and_pack.py:160`：解析 chat template kwargs。
- `run_completed_annotations_generate_and_pack.py:175`：可选 bad words。
- `run_completed_annotations_generate_and_pack.py:179`：可选 stop token ids。
- `run_completed_annotations_generate_and_pack.py:183`：可选 ignore eos。
- `run_completed_annotations_generate_and_pack.py:221`：生成成功后才进入 judge prompt 打包。

为什么需要：不同模型需要不同 vLLM 加载参数和采样参数；写在命令行里比反复修改全局 Config 更可追踪，也能在 status JSON 中记录实际参数。

## 8. 多模型自动化中的失败归因和模型定制

位置：`source_snapshot/scripts/run_manifest_models_completed_annotations.py:138`

目的：多模型自动实验时，先预检、再 smoke、再 full；对不可用模型记录原因，避免浪费 GPU。

关键位置：

- `run_manifest_models_completed_annotations.py:138`：把失败归因成 tokenizer 不支持、空续写、runtime error 等。
- `run_manifest_models_completed_annotations.py:400`：按模型名加 sampling overrides。
- `run_manifest_models_completed_annotations.py:403`：Nemotron 屏蔽 replacement character。
- `run_manifest_models_completed_annotations.py:407`：特定 Nemotron 忽略 EOS，防止 assistant prefill 下直接空输出。
- `run_manifest_models_completed_annotations.py:905`：每个模型先跑 continuation probe。
- `run_manifest_models_completed_annotations.py:915`：预检不过就停止，不进入 GPU full。
- `run_manifest_models_completed_annotations.py:931`：先跑 10-case smoke。
- `run_manifest_models_completed_annotations.py:951`：smoke 不通过就停止，不进入 full。

为什么需要：不同模型失败模式不同。预检和 smoke 可以节省大量 GPU 时间，避免模型已经明显不适配时继续跑全量。

## 9. Tokenizer continuation 预检

位置：`source_snapshot/scripts/probe_manifest_model_continuation.py:120`

目的：在真正加载模型、占用 GPU 前，只用 tokenizer 检查是否能渲染 assistant-prefix continuation。

关键位置：

- `probe_manifest_model_continuation.py:131`：加载 tokenizer。
- `probe_manifest_model_continuation.py:134`：没有 `apply_chat_template` 时直接失败。
- `probe_manifest_model_continuation.py:144`：检查签名。
- `probe_manifest_model_continuation.py:146`：判断是否支持 `continue_final_message`。
- `probe_manifest_model_continuation.py:176`：DeepSeek-V4 原生 encoding 分支。
- `probe_manifest_model_continuation.py:195`：构造 tokenizer continuation 渲染参数。
- `probe_manifest_model_continuation.py:204`：实际尝试渲染。
- `probe_manifest_model_continuation.py:207`：去掉 trailing EOS 后检查渲染是否有效。

为什么需要：很多失败可以在 tokenizer 层发现，不需要启动 vLLM 和占 GPU。

## 10. Judge prompt 文件名和历史 cache 兼容

位置：`source_snapshot/tools/prompts/pack_prompt.py:31`

目的：避免同题多 sample 打包时互相覆盖，并兼容旧 cache 文件名。

关键位置：

- `pack_prompt.py:31`：优先使用 `annotation_uid/id/uid/qid/uuid/case_id` 生成安全文件名。
- `pack_prompt.py:49`：优先使用已存的 `gen_prefix`；没有则从 `gen_output` 重新切 prefix。
- `stage2_judge_from_cache_prompts.py:145`：兼容 `case_<id>.cache.jsonl` 和 `case_<id>_cache.jsonl` 两种命名。

为什么需要：同一题可能有多条 sample。如果只用 `case_id`，下游 cache 文件可能覆盖。旧文件名兼容是为了能继续读取历史打包产物。

## 11. 哪些不是“改变任务”的分支

下面这些分支只是工程适配或保护，不改变生成/评分任务：

- tokenizer 支持时使用 `continue_final_message`。
- DeepSeek-V4 使用目录内原生 encoding。
- 清理 `<think>`、tool call 特殊 token。
- 根据 `annotation_uid` 避免文件覆盖。
- 空输出重试原 prompt。
- preflight/smoke 失败后停止。

## 12. 需要人工审核重点关注的分支

如果人工审核代码，建议重点看这些点：

1. `runner.py:261` 是否仍保持“不支持原生 continuation 就失败”，而不是自动改写 prompt。
2. `run_generate_prompt_pack.py:248` 是否仍只接受顶层 `prompt/messages`，避免误用非 prompt 文件。
3. `run_manifest_models_completed_annotations.py:915` 和 `:951` 是否仍会在 preflight/smoke 失败时停止。
4. `benchmark_core/prompt.py` 中 `Claim_Segment_Prompt` 是否无 `≤10`、无 `≤80`、无 `maxLength/maxItems`。
5. `pack_prompt.py:31` 是否优先使用样本级唯一 ID，避免同题多 sample 覆盖。

## 13. 一句话总结

这些兼容分支的目的，是在不改变 continuation 实验定义的前提下，让不同模型和不同历史数据格式能进入同一套稳定流程，并在全量实验前尽早发现会造成空输出、乱码、模板泄漏或 GPU 浪费的问题。
