# Frontend dev workflow

For full-stack development, run two terminals:

## Terminal A (Flask backend)
```bash
python annotation_app/app.py
```

## Terminal B (Next.js frontend proxy)
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.
The repo scripts bind Next explicitly to `127.0.0.1` by default so Linux hosts with `localhost -> 127.0.1.1` do not break local tunnels or reverse proxies.

## Routing behavior
The Next.js app is intentionally a thin shell that forwards product routes to the Flask annotation app:
- `/` -> Flask `/annotator` (annotation workspace is the main entry)
- `/annotator` -> Flask `/annotator`
- `/review` -> Flask `/review`
- `/api/*` and `/static/*` -> Flask backend

You can override the backend target:

```bash
BACKEND_URL=http://127.0.0.1:5050 npm run dev
```

For production-style local proxy runs, both `npm run start` and the repo helper bind `next start` to `127.0.0.1:3001` by default so that `cloudflared` and local smoke checks use the same loopback origin.
