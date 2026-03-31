# Frontend dev workflow

For full-stack development, run **two terminals**:

## Terminal A (Flask backend)
```bash
python annotation_app/app.py
```

## Terminal B (Next.js frontend)
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

- `/` uses the Next landing page.
- `/annotator`, `/review`, and `/api/*` are rewritten to Flask (`http://127.0.0.1:5000`) in dev.
- You can override backend target with:

```bash
BACKEND_URL=http://127.0.0.1:5000 npm run dev
```
