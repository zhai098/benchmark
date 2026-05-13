# 标注应用（annotation_app）

## 启动
单应用启动：
```bash
pip install flask
python annotation_app/app.py
```

打开：
- http://127.0.0.1:5000/ （首页：标注入口 + 指南）
- http://127.0.0.1:5000/annotator （标注工作台）
- http://127.0.0.1:5000/review （Reviewer 面板，需要 reviewer key）

正式部署请不要直接使用 Flask 开发服务器。生产运行方式见：

- `annotation_app/DEPLOYMENT.md`
- `annotation_app/wsgi.py`
- `annotation_app/gunicorn.conf.py`

## 当前核心能力
- LaTeX 公式渲染（KaTeX，本地静态资源托管，支持行内与块级，异常表达式自动降级为原文展示）。
- 一键复制 solution 原始文本（复制的是原始字符串而不是渲染后的 DOM）。
- 自动保存（防抖）+ 手动保存 + 提交时最终保存 + 刷新前 `sendBeacon` 保存。
- 登录后自动按 `annotator_id + device_id + case_id` 恢复完整状态。
- 每个 sample 独立工作流：
  - 错误样本立即从当前工作管线剔除。
  - 正确样本完成完整流程后，才进入 `correct_solutions` 供后续样本参考。

## 存储布局
进度文件路径：

`annotation_app/data/annotations/{annotator_id}/{device_id}/{case_id}.json`

每个文件至少包含：
- annotator_id
- device_id
- case_id
- current_workflow_state
- current_annotations
- sample_decisions
- correct_solutions
- created_at_utc / updated_at_utc

运行日志路径：

- `annotation_app/data/logs/access.log`
- `annotation_app/data/logs/app.log`

这些日志仅用于部署排障与访问留痕，不改变现有记录格式。

## 兼容性/迁移
- 新逻辑写入 `data/annotations/...`。
- 旧版 `data/records/*.json` 不会被覆盖；审核页面会继续读取并标记为 `legacy/...`。


## 角色与权限
- 默认角色为 annotator。
- annotator 仅可访问标注工作流，无法访问 reviewer 记录接口。
- reviewer 通过首页登录后可访问 `/review` 和 `/api/review_records`，并可编辑 `guideline.md`。
- reviewer key 默认值为 `reviewer`，可通过 `annotation_app/data/.review_key` 覆盖。
