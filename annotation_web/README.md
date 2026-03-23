# 标注辅助网页（客户端 + 服务器端）

## 启动
```bash
python annotation_web/server.py
```

- 客户端（标注者）: `http://127.0.0.1:8080/`
- 审核端（审核员）: `http://127.0.0.1:8080/review`

## 功能覆盖
- 多采样验证：判定正确/错误、方法分类、新分类创建与摘要。
- Step 切分：仅切分文本，可编辑/删除/新增。
- Claim 切分：固定走 OpenAI Python SDK 生成初稿，客户端人工修正。
- 依赖判定：仅允许依赖前序 claim。
- 保存策略：支持手动保存，提交问题时自动保存并标记 submitted。
- 审核端：汇总所有标注者进度，按标注者+任务查看完整结果。

## Claim 生成（OpenAI SDK）

### 1) 环境变量
在启动服务前设置 API Key（不要放到前端）：

```bash
export OPENAI_API_KEY="sk-..."
python annotation_web/server.py
```

### 2) 可选参数
`POST /api/task/generate_claims` 支持以下 SDK 控制参数（请求体）：

- `model`：模型名（默认 `gpt-4.1-mini`）
- `temperature`：采样温度（默认 `0.2`）
- `max_tokens`：最大输出 token（默认 `1200`）
- `base_url`：可选，自定义 OpenAI 兼容服务地址
- `allow_fallback`：`true/false`；仅在 SDK 调用失败时，显式允许回退到 `fallback_segment_claims`

### 3) 请求示例

```json
{
  "annotator": "alice",
  "task_id": 1,
  "step_segments": ["Step 1 ...", "Step 2 ..."],
  "model": "gpt-4.1-mini",
  "temperature": 0.2,
  "max_tokens": 1200,
  "base_url": "",
  "allow_fallback": false
}
```

### 4) 返回与错误码行为

- 成功：`200`
  - `claims`：前端当前使用的扁平 claim 列表
  - `claims_by_step`：结构化返回（`[{ "step_id": "...", "claims": ["..."] }]`）
  - `source`：`openai_sdk` 或 `fallback_segment_claims`
- SDK 失败且 `allow_fallback=false`：`502`
  - 返回 `{ "error": "claim generation failed", "details": "...", "source": "openai_sdk" }`
- 参数格式错误 / JSON 结构不合法：`502`（由服务端验证与 SDK 调用阶段抛错）

## 数据存储
- 标注存储：`annotation_web/data/annotations/<annotator>.json`
- 数据集缓存：`annotation_web/data/dataset_cache.json`
- 指南文档：`annotation_web/guideline.md`
