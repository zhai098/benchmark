#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ROOT_DIR}/.conda/benchmark-web"
export ANNOTATION_APP_BIND="${ANNOTATION_APP_BIND:-127.0.0.1:5050}"
ENV_NAME="${ANNOTATION_APP_ENV_NAME:-benchmark-web}"

cd "${ROOT_DIR}"
echo "Starting backend on ${ANNOTATION_APP_BIND}"

if [ -x "${ENV_PREFIX}/bin/gunicorn" ]; then
  exec "${ENV_PREFIX}/bin/gunicorn" -c annotation_app/gunicorn.conf.py annotation_app.wsgi:application
fi

if command -v gunicorn >/dev/null 2>&1; then
  exec gunicorn -c annotation_app/gunicorn.conf.py annotation_app.wsgi:application
fi

if command -v conda >/dev/null 2>&1; then
  if conda run -n "${ENV_NAME}" python -c "import gunicorn" >/dev/null 2>&1; then
    exec conda run -n "${ENV_NAME}" gunicorn -c annotation_app/gunicorn.conf.py annotation_app.wsgi:application
  fi
fi

echo "Gunicorn not found."
echo "Supported setups:"
echo "  1) Prefix env: conda env update -p ${ENV_PREFIX} -f ${ROOT_DIR}/environment.yml"
echo "  2) Named env:  conda env create -f ${ROOT_DIR}/environment.yml"
echo "Then rerun this script, or set ANNOTATION_APP_ENV_NAME to your conda env name."
exit 1
