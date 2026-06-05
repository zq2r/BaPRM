#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ===== user config =====
model_name=${model_name:-"InternVL3-2B"}

# 只用这一个变量；写几个 benchmark 就顺序跑几个
#   MathVista
#   MathVision
#   MathVerse
#   OlympiadBench
benchs=${benchs:-"MathVista MathVision OlympiadBench MathVerse"}

MODEL_ROOT="/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model"
BASE_MODEL="${MODEL_ROOT}/${model_name}"

GPUS=${GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-63711}

# FORCE_REBUILD=1 时会删掉旧的 base-prm-token-init 重新生成
FORCE_REBUILD=${FORCE_REBUILD:-0}

CKPT_ROOT="${REPO_ROOT}/log/base-${model_name}"
CKPT="${CKPT_ROOT}/base-prm-token-init"

mkdir -p "${CKPT_ROOT}"

if [ ! -d "${BASE_MODEL}" ]; then
  echo "ERROR: base model not found: ${BASE_MODEL}"
  exit 1
fi

if [ "${FORCE_REBUILD}" = "1" ]; then
  rm -rf "${CKPT}"
fi

# ===== prepare base checkpoint with <prm> token, no training =====
if [ ! -f "${CKPT}/config.json" ]; then
  mkdir -p "${CKPT}"

  BASE_MODEL="${BASE_MODEL}" CKPT="${CKPT}" python - <<'PY'
import os
import torch
from transformers import AutoTokenizer

from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.internvl_chat.modeling_internvl_chat_beta_binom import InternVLChatModel
from internvl.train.constants import (
    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
    QUAD_START_TOKEN, QUAD_END_TOKEN,
    REF_START_TOKEN, REF_END_TOKEN,
    BOX_START_TOKEN, BOX_END_TOKEN,
    PRM_TOKEN,
)

base_model = os.environ["BASE_MODEL"]
ckpt = os.environ["CKPT"]

print(f"[prepare] base_model = {base_model}")
print(f"[prepare] ckpt = {ckpt}")

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    add_eos_token=False,
    trust_remote_code=True,
    use_fast=False,
)
tokenizer.model_max_length = 8192

token_list = [
    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
    QUAD_START_TOKEN, QUAD_END_TOKEN,
    REF_START_TOKEN, REF_END_TOKEN,
    BOX_START_TOKEN, BOX_END_TOKEN,
    PRM_TOKEN,
]
num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
print(f"[prepare] num_new_tokens = {num_new_tokens}")

config = InternVLChatConfig.from_pretrained(base_model)

model = InternVLChatModel.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    config=config,
)

# 避免 kappa_head 出现 meta tensor
if hasattr(model, "kappa_head"):
    for m in model.kappa_head.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

if num_new_tokens > 0:
    model.language_model.resize_token_embeddings(len(tokenizer))

    output_embeddings = model.language_model.get_output_embeddings().weight.data
    output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings[-num_new_tokens:] = output_avg

    model.config.llm_config.vocab_size = len(tokenizer)
    model.language_model.config.vocab_size = len(tokenizer)

model.prm_token_id = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

tokenizer.save_pretrained(ckpt)
model.save_pretrained(ckpt, safe_serialization=False)

print(f"[prepare] saved base PRM-init checkpoint to {ckpt}")
PY
fi

run_one_benchmark() {
  local BENCH="$1"
  local MASTER_PORT="$2"

  case "${BENCH}" in
    MathVista)
      EVAL_SCRIPT="eval/prm/evaluate_mathvista_prm_normal.py"
      DATASET_NAME="mathvista_prm"
      ROOT="datasets/MathVista/extracted_images"
      ANNOTATION="datasets/MathVista/MathVista_rollout_annotation_InternVL8B_oversample.json"
      ;;
    MathVision)
      EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_normal.py"
      DATASET_NAME="mathvision_prm"
      ROOT="datasets/MathVision/extracted_images"
      ANNOTATION="datasets/MathVision/MathVision_rollout_annotation_InternVL8B_oversample.json"
      ;;
    MathVerse)
      EVAL_SCRIPT="eval/prm/evaluate_mathverse_prm_normal.py"
      DATASET_NAME="mathverse_prm"
      ROOT="datasets/MathVerse/extracted_images"
      ANNOTATION="datasets/MathVerse/MathVerse_rollout_annotation_InternVL8B_oversample.json"
      ;;
    OlympiadBench)
      EVAL_SCRIPT="eval/prm/evaluate_olympiadbench_prm_normal.py"
      DATASET_NAME="olympiadbench_prm"
      ROOT="."
      ANNOTATION="datasets/OlympiadBench/OlympiadBench_rollout_annotation_InternVL8B_oversample.json"
      ;;
    *)
      echo "ERROR: Unknown benchmark: ${BENCH}"
      echo "Supported: MathVista, MathVision, MathVerse, OlympiadBench"
      exit 1
      ;;
  esac

  OUT_DIR="${CKPT_ROOT}/eval_base_${BENCH}/$(basename "${CKPT}")"
  mkdir -p "${OUT_DIR}"

  if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "ERROR: eval script not found: ${EVAL_SCRIPT}"
    exit 1
  fi

  if [ ! -f "${ANNOTATION}" ]; then
    echo "ERROR: annotation file not found: ${ANNOTATION}"
    exit 1
  fi

  echo "========== Base Eval Config =========="
  echo "model_name: ${model_name}"
  echo "BASE_MODEL: ${BASE_MODEL}"
  echo "BENCH: ${BENCH}"
  echo "EVAL_SCRIPT: ${EVAL_SCRIPT}"
  echo "DATASET_NAME: ${DATASET_NAME}"
  echo "CKPT: ${CKPT}"
  echo "ANNOTATION: ${ANNOTATION}"
  echo "OUT_DIR: ${OUT_DIR}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "GPUS: ${GPUS}"
  echo "MASTER_PORT: ${MASTER_PORT}"
  echo "======================================"

  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${EVAL_SCRIPT}" \
    --checkpoint "${CKPT}" \
    --datasets "${DATASET_NAME}" \
    --root "${ROOT}" \
    --annotation "${ANNOTATION}" \
    --out-dir "${OUT_DIR}"
}

idx=0
for BENCH in ${benchs}; do
  MASTER_PORT=$((MASTER_PORT_BASE + idx))
  run_one_benchmark "${BENCH}" "${MASTER_PORT}"
  idx=$((idx + 1))
done

echo "All requested benchmarks finished: ${benchs}"