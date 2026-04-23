# Pretrain Download List

这份目录用于整理本轮 benchmark 的待下载模型清单。

文件说明：

- `download_manifest.tsv`: 机器可读清单，包含 Hugging Face 来源、可选子目录、目标下载目录、是否 gated、以及 continuation 相关备注。

约定：

- 所有目标目录都放在 `data/pretrain` 下面。
- `tier=main` 表示优先进入 benchmark 主表。
- `tier=shadow` 表示值得纳入扩展 benchmark，但建议先做一次 smoke test。
- `include_pattern` 只有在模型位于仓库子目录时才需要填写。

几个特殊项：

- `tencent/Tencent-Hunyuan-Large` 不是单模型仓库，`Hunyuan-A52B-Instruct` 需要按子目录下载。
- `meta-llama/*` 与 `google/gemma-*` 为 gated 模型，下载前通常需要先在 Hugging Face 页面接受许可。
- `mistralai/Mistral-*` 在 vLLM 里通常建议走 `tokenizer_mode=mistral` / `config_format=mistral` 路线。
