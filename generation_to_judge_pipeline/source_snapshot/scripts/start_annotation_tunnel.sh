#!/usr/bin/env bash
set -euo pipefail

FRONTEND_TUNNEL_URL="${FRONTEND_TUNNEL_URL:-http://127.0.0.1:3001}"
TUNNEL_PROTOCOL="${TUNNEL_PROTOCOL:-http2}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed."
  echo "Install it first, for example on macOS: brew install cloudflared"
  exit 1
fi

echo "Starting cloudflared tunnel to ${FRONTEND_TUNNEL_URL} with protocol=${TUNNEL_PROTOCOL}"
exec cloudflared tunnel --url "${FRONTEND_TUNNEL_URL}" --protocol "${TUNNEL_PROTOCOL}"
