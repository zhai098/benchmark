# 标注应用（annotation_app）

## 启动
```bash
pip install flask
python annotation_app/app.py
```

打开：
- http://127.0.0.1:5000/
- http://127.0.0.1:5000/review

## 当前核心能力
- LaTeX 公式渲染（KaTeX，支持行内与块级，异常表达式自动降级为原文展示）。
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

## 兼容性/迁移
- 新逻辑写入 `data/annotations/...`。
- 旧版 `data/records/*.json` 不会被覆盖；审核页面会继续读取并标记为 `legacy/...`。
