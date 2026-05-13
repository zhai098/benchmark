# annotation_app 部署说明

本部署方案只调整运行方式与日志，不修改现有记录格式、目录结构、接口返回或页面流程。

## 1. 推荐架构

```text
标注者浏览器 -> Nginx(:80/:443) -> frontend(:3001) -> annotation_app(:5050)
```

- 外部用户访问服务器 IP 或域名
- `frontend` 负责包装与转发
- `annotation_app` 负责实际页面、静态资源和保存接口

## 2. 本地开发运行

```bash
python annotation_app/app.py
```

可选环境变量：

```bash
ANNOTATION_APP_HOST=0.0.0.0
ANNOTATION_APP_PORT=5000
ANNOTATION_APP_DEBUG=1
```

## 3. 生产运行（推荐）

使用 Gunicorn 托管现有 Flask app：

```bash
gunicorn -c annotation_app/gunicorn.conf.py annotation_app.wsgi:application
```

也可以直接使用仓库里准备好的脚本：

```bash
./scripts/start_annotation_backend.sh
```

默认只监听本机 `127.0.0.1:5050`，适合由 Nginx 反向代理到：

- `/annotator`
- `/review`
- `/api/*`
- `/static/*`

如果必须直接通过服务器 IP 暴露后端，可临时把 `annotation_app/gunicorn.conf.py` 中的 `bind` 改成：

```python
bind = "0.0.0.0:5050"
```

frontend 生产运行方式：

```bash
cd frontend
BACKEND_URL=http://127.0.0.1:5050 PORT=3001 npm run build
BACKEND_URL=http://127.0.0.1:5050 PORT=3001 npm run start:prod
```

也可以直接使用仓库里的脚本：

```bash
BACKEND_URL=http://127.0.0.1:5050 ./scripts/start_annotation_frontend.sh
```

默认 `frontend` 监听本机 `127.0.0.1:3001`，这样：

- 本地 `curl` / `cloudflared` 的源站地址固定一致
- 避免 macOS 上 `127.0.0.1` 与 `::1` 混用时出现资源可达性不一致

如果你需要让同一局域网内其他设备直接访问 `frontend`，可以临时改为：

```bash
BACKEND_URL=http://127.0.0.1:5050 HOSTNAME=0.0.0.0 PORT=3001 ./scripts/start_annotation_frontend.sh
```

如果要用 cloudflared 把本机 `frontend(:3001)` 暴露给外网，推荐使用仓库里的脚本：

```bash
./scripts/start_annotation_tunnel.sh
```

默认它会连到：

```text
http://127.0.0.1:3001
```

默认还会使用：

```text
protocol=http2
```

这是为了避开校园网/VPN 环境下常见的 QUIC/UDP 7844 不稳定问题；如果你在 tunnel 日志里看到：

- `sendmsg: network is unreachable`
- `failed to dial to edge with quic`
- `control stream encountered a failure while serving`

优先保持 `http2`，不要切回 `quic`。

如果你确认当前网络对 QUIC 稳定，才手动改为：

```bash
TUNNEL_PROTOCOL=quic ./scripts/start_annotation_tunnel.sh
```

## 4. 本地数据与日志

现有业务记录格式保持不变：

- `annotation_app/data/annotations/{annotator_id}/{device_id}/{case_id}.json`
- `annotation_app/data/records/*.json`

新增运行日志：

- `annotation_app/data/logs/access.log`
- `annotation_app/data/logs/app.log`

其中：

- `access.log` 记录访问路径、状态码、耗时、客户端 IP
- `app.log` 记录应用启动、保存动作、异常

页面数学渲染也已改为本地静态资源提供：

- `annotation_app/static/vendor/katex/katex.min.css`
- `annotation_app/static/vendor/katex/katex.min.js`
- `annotation_app/static/vendor/katex/auto-render.min.js`

这样外网用户访问时不再依赖 `jsdelivr` 等外部 CDN，避免页面只剩裸 HTML 或公式不渲染。

## 5. Nginx 反向代理

仓库已提供模板：

- `deploy/nginx.annotation_frontend.conf`

它会把公网请求转发到本机 `frontend(:3001)`。

## 6. 对外可访问检查

至少验证以下几点：

1. 服务器监听正常：

```bash
lsof -iTCP:5050 -sTCP:LISTEN
```

2. 本机探活：

```bash
curl -I http://127.0.0.1:5050/annotator
```

补充静态资源检查：

```bash
curl -I http://127.0.0.1:3001/static/styles.css
curl -I http://127.0.0.1:3001/static/app.js
curl -I http://127.0.0.1:3001/static/vendor/katex/katex.min.css
```

3. 远端机器通过 IP 访问：

```text
http://<服务器IP>/annotator
```

4. 保存后检查数据与日志：

- `annotation_app/data/annotations/...` 中有更新
- `annotation_app/data/logs/access.log` 有请求记录
- `annotation_app/data/logs/app.log` 有保存事件或异常记录

也可直接运行：

```bash
./scripts/check_annotation_deploy.sh
```
