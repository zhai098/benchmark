#!/usr/bin/env bash
set -e

# 使用你配置的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME="openai/gpt-oss-20b"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 12288 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --api-key EMPTY
  # 注意：enable_prefix_caching=False → 不需要加 --enable-prefix-caching
  # 如果以后想开前缀缓存，只要加上：
  #   --enable-prefix-caching
