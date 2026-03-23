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
- Claim 切分：服务端 API 自动生成初稿，客户端人工修正。
- 依赖判定：仅允许依赖前序 claim。
- 保存策略：支持手动保存，提交问题时自动保存并标记 submitted。
- 审核端：汇总所有标注者进度，按标注者+任务查看完整结果。

## 数据存储
- 标注存储：`annotation_web/data/annotations/<annotator>.json`
- 数据集缓存：`annotation_web/data/dataset_cache.json`
- 指南文档：`annotation_web/guideline.md`
