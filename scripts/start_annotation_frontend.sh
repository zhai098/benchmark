#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
BUILD_STAMP_FILE="${FRONTEND_DIR}/.next/.backend_url"

export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:5050}"
export HOSTNAME="${HOSTNAME:-127.0.0.1}"
export PORT="${PORT:-3001}"

cd "${FRONTEND_DIR}"

if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies with npm ci..."
  npm ci
fi

NEEDS_BUILD=0
if [ ! -d .next ]; then
  NEEDS_BUILD=1
elif [ ! -f "${BUILD_STAMP_FILE}" ]; then
  NEEDS_BUILD=1
elif [ "$(cat "${BUILD_STAMP_FILE}")" != "${BACKEND_URL}" ]; then
  NEEDS_BUILD=1
fi

if [ "${NEEDS_BUILD}" -eq 1 ]; then
  echo "Building frontend..."
  rm -rf .next
  npm run build
  printf '%s' "${BACKEND_URL}" > "${BUILD_STAMP_FILE}"
fi

echo "Starting frontend with BACKEND_URL=${BACKEND_URL} on ${HOSTNAME}:${PORT}"
exec npx next start -H "${HOSTNAME}" -p "${PORT}"
