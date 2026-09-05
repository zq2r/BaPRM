#!/usr/bin/env bash
set -euo pipefail
set -x

cd /inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Judge model. Change this path if your Qwen model is elsewhere.
JUDGE_MODEL=${JUDGE_MODEL:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model/Qwen2.5-32B-Instruct"}
JUDGE_SERVED_NAME=${JUDGE_SERVED_NAME:-"Qwen2.5-32B-Instruct"}

HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8888}
TP_SIZE=${TP_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# Use 2 GPUs for Qwen2.5-32B judge by default.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4"}

python data_pipeline/start_vllm_server.py \
  --model "${JUDGE_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${JUDGE_SERVED_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.80