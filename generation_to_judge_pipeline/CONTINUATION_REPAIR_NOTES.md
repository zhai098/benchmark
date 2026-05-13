# 本地 vLLM 续写链路说明

当前生成阶段不走 API。`generate.py` 和 `tools/prompts/run_generate_prompt_pack.py` 都只调用本地 `VLLMRunner`，prompt 中最后一条 assistant message 用 `prefix: true` 表示“这是已写出的解答前缀，请从这里继续”。

## 关键约定

- prompt 构建层只产出 messages，不再写 `partial: true`。
- `VLLMRunner` 统一负责把 messages 渲染为模型实际接收的 prompt。
- 标准 HF/vLLM chat template 使用 `continue_final_message=True`。
- DeepSeek-V4 类模型如果自带 `encoding/encoding_dsv4.py`，使用原生编码器并对最后 assistant 设置 `wo_eos=True`。
- 如果本地 tokenizer/chat template 没有可用的 assistant continuation 机制，直接失败；不自动改写成普通“请继续”语义 cue。

## 当前代码位置

- `benchmark_core/prompt.py`：`Generate_Prompt` 只产出 `prefix: true` messages。
- `tools/prompts/build_generate_prompt_pack.py`：`auto` 固定为 `vllm-messages`；`chat-template` 只用于本地调试预渲染。
- `tools/prompts/run_generate_prompt_pack.py`：`--backend` 只允许 `vllm`，统一调用 `model.generate(prompts, None)`。
- `runner.py`：`VLLMRunner._chat_messages_to_prompt_text()` 按模型选择 DeepSeek-V4 原生编码、HF `continue_final_message`，或普通无 chat template 的文本拼接。

## 异常处理

- 空输出或 `<Error: ...>` 输出会按原 prompt 重试。
- 可选 `--enforce-continuation-guard` 会在 system message 里加入严格续写约束，用于修复模型“重新开始答题”的行为。
- 检测到明显 `The user wants me...` / `Let me parse...` 等重启式输出时，可按 `--max-restart-retries` 重试。
- 这些重试不改变任务语义，不把 continuation 改成新的 user 指令。

## 人工检查重点

1. prompt JSONL 每条应有顶层 `prompt` 字段。
2. `prompt` 为 messages 时，最后一条 assistant 应有 `prefix: true`，不应有 `partial: true`。
3. `run_info.json` 中 `empty_count`、`error_count`、`restart_like_count` 应低；如果 smoke 阶段系统性异常，应停止全量。
4. `gen_only_from_prompt_pack.jsonl` 的 `gen_output` 应是接续前缀的数学文本，而不是重新复述题目。
