#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaipengxiang/benchmark

unset LD_LIBRARY_PATH
unset PYTHONSTARTUP
unset PYTHON_BASIC_REPL

PY=/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12
MODEL=/data/pretrain/Granite/granite-4.1-8b
BASE_OUT=/home/zhaipengxiang/benchmark/artifacts/model_outputs/completed_annotations

mkdir -p "$BASE_OUT" logs

"$PY" scripts/run_completed_annotations_generate_and_pack.py \
  --input-path workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl \
  --out-root "$BASE_OUT" \
  --tag completed_annotations_test_granite_4_1_8b \
  --model-path "$MODEL" \
  --gpus 0 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.35 \
  --max-cases 10 \
  --wait-gpu-free-mib 30000 \
  --wait-gpu-max-util 100 \
  --wait-poll-seconds 60 \
  --status-path logs/completed_annotations_test_status.json \
  --write-all-prompts

"$PY" scripts/run_completed_annotations_generate_and_pack.py \
  --input-path workflow_data/annotation_exports/completed_annotation_records/purified_cases.jsonl \
  --out-root "$BASE_OUT" \
  --tag completed_annotations_full_granite_4_1_8b \
  --model-path "$MODEL" \
  --gpus 0 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.35 \
  --max-cases 100000 \
  --wait-gpu-free-mib 30000 \
  --wait-gpu-max-util 100 \
  --wait-poll-seconds 60 \
  --status-path logs/completed_annotations_full_status.json \
  --write-all-prompts
