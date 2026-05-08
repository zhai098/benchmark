#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaipengxiang/benchmark
PY=/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12
MODEL=/data/pretrain/Mistral/Mistral-Small-3.2-24B-Instruct-2506
BASE_OUT=/home/zhaipengxiang/benchmark/artifacts/model_outputs/completed_annotations
mkdir -p "$BASE_OUT" logs

$PY scripts/run_completed_annotations_generate_and_pack.py \
  --input-path workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl \
  --out-root "$BASE_OUT" \
  --tag completed_annotations_test_mistral_small_3_2 \
  --model-path "$MODEL" \
  --gpus 0,1 \
  --tensor-parallel-size 2 \
  --config-format mistral \
  --load-format mistral \
  --max-cases 10 \
  --wait-gpu-free-mib 50000 \
  --wait-gpu-max-util 10 \
  --wait-poll-seconds 120 \
  --status-path logs/completed_annotations_test_status.json \
  --write-all-prompts

$PY scripts/run_completed_annotations_generate_and_pack.py \
  --input-path workflow_data/annotation_exports/completed_annotation_records/purified_cases.jsonl \
  --out-root "$BASE_OUT" \
  --tag completed_annotations_full_mistral_small_3_2 \
  --model-path "$MODEL" \
  --gpus 0,1 \
  --tensor-parallel-size 2 \
  --config-format mistral \
  --load-format mistral \
  --max-cases 100000 \
  --wait-gpu-free-mib 50000 \
  --wait-gpu-max-util 10 \
  --wait-poll-seconds 120 \
  --status-path logs/completed_annotations_full_status.json \
  --write-all-prompts
