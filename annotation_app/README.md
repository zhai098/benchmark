# 标注网页（客户端 + 服务器端审核）

## 你关心的关键能力
- 客户端（`/`）：
  - 五步流程 + 提交前总览：多采样验证、step 切分、claim 生成、claim 校验、依赖关系标注、最终总览提交。
  - 任务列表可收起，主工作区更大。
  - Step 切分支持在完整 solution 上“光标打点”切分，可随时删除切分点回退修改。
  - 多采样验证支持翻译按钮（调用后端翻译 API）。
  - Claim 生成通过服务端 OpenAI 官方 SDK 调用模型（支持 model/temperature/max_tokens 参数），失败时可 fallback 到本地分句。
  - 手动保存 + 完成提交自动保存（不做实时自动保存）。
- 审核端（`/review`）：
  - 汇总全部标注结果，表格检查 + 详情查看。

## 启动
```bash
pip install flask openai
# 必填：OpenAI API Key
export OPENAI_API_KEY="你的key"
# 可选：兼容网关
# export OPENAI_BASE_URL="https://xxx/v1"
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

## Claim 生成 API（OpenAI SDK）
前端会把 Step 2 生成的 `steps` 发送到后端 `/api/generate_claims`：

请求体示例：
```json
{
  "steps": [
    {"id": "s1", "text": "第一步..."},
    {"id": "s2", "text": "第二步..."}
  ],
  "model": "gpt-4o-mini",
  "temperature": 0,
  "max_tokens": 1500,
  "allow_fallback": true
}
```

返回体示例：
```json
{
  "claims_by_step": [
    {"step_id": "s1", "claims": ["...", "..."]},
    {"step_id": "s2", "claims": ["..."]}
  ],
  "source": "openai_sdk",
  "model": "gpt-4o-mini"
}
```

说明：
- 当 OpenAI 调用失败且 `allow_fallback=true` 时，会退化到本地分句并返回 `source: "fallback"`。
- 若要严格只用模型结果，请把 `allow_fallback` 设为 `false`，失败时接口会直接报错。
