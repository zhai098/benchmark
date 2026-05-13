# 全流程输入输出抽样切片索引

生成时间：2026-05-07

## 切片范围

- 来源：`/Users/zhaipengxiang/Desktop/completed_annotation_records_full.zip` 中的 `purified_cases.jsonl`。
- 样本数：7。
- 覆盖：WangZX、Xiechenzhe、chenmingrui、liyanheng、qutianyi。
- 远端真实模型输出来源：159 服务器 `granite-4.1-8b_completed_annotations_full_granite_4_1_8b_4shard_combined`。

## 文件说明

- `stage_01_clean_input_slice.jsonl`：清洗后的 pipeline 输入，每行一个 completed 正确 sample。
- `stage_02_generation_prompts_from_generate_py_slice.jsonl`：从真实 `gen_only.jsonl` 的 `prompts` 字段抽出的 generate.py 实际送模 prompt。
- `stage_02b_moonshot_partial_generation_prompt_pack_slice.jsonl`：用当前 `build_generate_prompt_pack.py` 重新生成的 Kimi/Moonshot messages prompt。
- `stage_03_model_gen_only_granite4_1_8b_slice.jsonl`：真实模型生成输出切片。
- `stage_04_judge_cache_granite4_1_8b_slice.jsonl`：真实 judge prompt cache rows，保留全部 filtered rows。
- `per_sample_json/`：每个 sample 的输入、prompt、输出、judge rows 合并 JSON，适合机器核验。
- `per_sample_markdown/`：每个 sample 的中文可读卡片，适合人工逐条核验。
- `trace_table.csv`：每条 sample 的行数、step 数、claim 数、prompt 数、judge route 数。

## 样本清单

|序号|题号|标注者|sample_idx|annotation_uid|steps|claims|
|---:|---|---|---:|---|---:|---:|
|1|q-1|WangZX|2|`q-1__WangZX__dev-1776827054895-o7cnww__sample_2`|5|21|
|2|q-235|Xiechenzhe|0|`q-235__Xiechenzhe__dev-1777129480081-jcqn72__sample_0`|6|47|
|3|q-122|chenmingrui|1|`q-122__chenmingrui__dev-1777734001062-awk0xz__sample_1`|7|10|
|4|q-565|liyanheng|5|`q-565__liyanheng__dev-1776655631702-gk38nd__sample_5`|6|29|
|5|q-341|qutianyi|0|`q-341__qutianyi__dev-1776655218740-88nccp__sample_0`|7|64|
|6|q-441|qutianyi|1|`q-441__qutianyi__dev-1776932087401-vma9h3__sample_1`|4|22|
|7|q-448|qutianyi|0|`q-448__qutianyi__dev-1776932087401-vma9h3__sample_0`|6|68|

## 生成校验

- stage_01 clean input rows: 7
- stage_02 generate.py actual prompt rows: 41
- stage_02b moonshot partial prompt rows: 34
- stage_03 gen_only rows: 7
- stage_04 remote judge cache rows: 245
- stage_04b local repack judge cache rows: 245
- remote/local repack route-count match: True

## 人工审核建议顺序

1. 先看 `trace_table.csv` 确认每条样本的 step/claim/prompt/judge 数量是否合理。
2. 再打开 `per_sample_markdown/*.md`，从题目、标准解、保留 sample、标注 step/claim 开始看。
3. 若要核验送模内容，打开 `stage_02_generation_prompts_from_generate_py_slice.jsonl` 或 `stage_02b_moonshot_partial_generation_prompt_pack_slice.jsonl`。
4. 若要核验 judge 打包，打开 `stage_04_judge_cache_granite4_1_8b_slice.jsonl`，按 `case_id=annotation_uid` 过滤。
