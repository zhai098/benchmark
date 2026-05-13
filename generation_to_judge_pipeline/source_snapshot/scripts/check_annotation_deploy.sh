#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:5050}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:3001}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/7] Checking backend HTML..."
curl -fsS -o /dev/null -I "${BACKEND_URL}/annotator"
echo "OK: ${BACKEND_URL}/annotator"

echo "[2/7] Checking frontend HTML..."
curl -fsS -o /dev/null -I "${FRONTEND_URL}/annotator"
echo "OK: ${FRONTEND_URL}/annotator"

echo "[3/7] Checking frontend CSS..."
curl -fsS -o /dev/null -I "${FRONTEND_URL}/static/styles.css?v=20260402b"
echo "OK: ${FRONTEND_URL}/static/styles.css"

echo "[4/7] Checking frontend app.js..."
curl -fsS -o /dev/null -I "${FRONTEND_URL}/static/app.js?v=20260402b"
echo "OK: ${FRONTEND_URL}/static/app.js"

echo "[5/7] Checking local KaTeX assets..."
curl -fsS -o /dev/null -I "${FRONTEND_URL}/static/vendor/katex/katex.min.css?v=20260415a"
curl -fsS -o /dev/null -I "${FRONTEND_URL}/static/vendor/katex/katex.min.js?v=20260415a"
curl -fsS -o /dev/null -I "${FRONTEND_URL}/static/vendor/katex/auto-render.min.js?v=20260415a"
echo "OK: ${FRONTEND_URL}/static/vendor/katex/*"

echo "[6/7] Checking log directory..."
test -d "${ROOT_DIR}/annotation_app/data/logs"
echo "OK: annotation_app/data/logs exists"

echo "[7/7] Checking business data directory..."
test -d "${ROOT_DIR}/annotation_app/data/annotations"
echo "OK: annotation_app/data/annotations exists"

echo "Deployment checks passed."
