# 标注网页（客户端 + 服务器端审核）

## 你关心的关键能力
- 客户端（`/`）：
  - 五步流程 + 提交前总览：多采样验证、step 切分、claim 整理、claim 校验、依赖关系标注、最终总览提交。
  - 任务列表可收起，主工作区更大。
  - Step 切分支持在完整 solution 上“光标打点”切分，可随时删除切分点回退修改。
  - 多采样验证支持翻译按钮（调用后端翻译 API）。
  - Claim 阶段改为“整理预切分 claim”：每个 solution 的 claim 由上游预切分提供，标注者仅需将 claim 归入对应 step，再进行 claim 正确性检查与依赖关系标注。
  - 手动保存 + 完成提交自动保存（不做实时自动保存）。
- 审核端（`/review`）：
  - 汇总全部标注结果，表格检查 + 详情查看。

## 启动
```bash
pip install flask
python annotation_app/app.py
```

打开：
- http://127.0.0.1:5000/
- http://127.0.0.1:5000/review

## JSONL 输入格式（示例）
```json
{
  "id": "q-1",
  "question": "...",
  "reference_answer": "...",
  "known_solutions": ["方法A", "方法B"],
  "samples": [
    {"solution": "模型解1"},
    {"solution": "模型解2"}
  ]
}
```

## 预切分 Claim 输入建议
`samples[i]` 建议包含预切分 claim 字段之一（前端已做兼容）：

- `claims`: `["claim1", "claim2"]`
- `claims`: `[{ "text": "...", "step_id": "s1" }]`
- `claims_by_step`: `[{ "step_id": "s1", "claims": ["...", "..."] }]`

Step 3 中标注者将这些 claim 归并到 Step 2 切分出的 step 后，进入 Step 4 做 claim 正确性检查与修正。
