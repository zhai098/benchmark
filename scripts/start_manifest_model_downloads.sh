#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zhaipengxiang/benchmark
PY=/home/zhaipengxiang/miniconda3/bin/python
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO}/logs/manifest_model_downloads_${STAMP}"
mkdir -p "${LOG_DIR}"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600

cd "${REPO}"
echo "${LOG_DIR}" > logs/manifest_model_downloads_latest.txt
exec > >(tee -a "${LOG_DIR}/download_driver.log") 2>&1

echo "[download] started_at=$(date -Is)"
echo "[download] log_dir=${LOG_DIR}"
echo "[download] python=${PY}"
df -h /data /home || true

"${PY}" scripts/download_manifest_models.py \
  --manifest model/download_manifest.tsv \
  --pretrain-root /data/pretrain \
  --log-dir "${LOG_DIR}" \
  --parallel-models "${PARALLEL_MODELS:-3}" \
  --per-model-workers "${PER_MODEL_WORKERS:-8}"

echo "[download] finished_at=$(date -Is)"
