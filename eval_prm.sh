#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

# =========================
# User configs
# =========================
model_name=${model_name:-"InternVL2_5-2B"}

# Choose PRM mode:
#   beta
#   normal
PRM_MODE=${PRM_MODE:-"beta"}

# Choose benchmark:
#   MathVista
#   MathVision
#   MathVerse
#   OlympiadBench
BENCH=${BENCH:-"MathVerse"}
MASTER_PORT=${MASTER_PORT:-63704}

case "${PRM_MODE}" in
  beta)
    CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/log/beta-${model_name}-visualprm400k"}
    SCRIPT_SUFFIX="beta_binomial"
    ;;
  normal)
    CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/log/normal-${model_name}-visualprm400k"}
    SCRIPT_SUFFIX="normal"
    ;;
  *)
    echo "ERROR: Unknown PRM_MODE=${PRM_MODE}"
    echo "Supported: beta, normal"
    exit 1
    ;;
esac

# If CKPT is not provided, use the latest checkpoint under CKPT_ROOT.
if [ -z "${CKPT:-}" ]; then
  CKPT=$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n 1)
fi

if [ -z "${CKPT:-}" ]; then
  echo "ERROR: No checkpoint found under ${CKPT_ROOT}"
  exit 1
fi

GPUS=${GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# Only used by beta-binomial PRM.
SCORE_MODE=${SCORE_MODE:-"mu_minus_lambda_sigma"}

# =========================
# Benchmark mapping
# =========================
case "${BENCH}" in
  MathVista)
    EVAL_SCRIPT="eval/prm/evaluate_mathvista_prm_${SCRIPT_SUFFIX}.py"
    DATASET_NAME="mathvista_prm"
    DEFAULT_ROOT="datasets/MathVista/extracted_images"
    DEFAULT_ANNOTATION="datasets/MathVista/MathVista_rollout_annotation_InternVL8B_oversample.json"
    ;;

  MathVision)
    EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_${SCRIPT_SUFFIX}.py"
    DATASET_NAME="mathvision_prm"
    DEFAULT_ROOT="datasets/MathVision/extracted_images"
    DEFAULT_ANNOTATION="datasets/MathVision/MathVision_rollout_annotation_InternVL8B_oversample.json"
    ;;

  MathVerse)
    EVAL_SCRIPT="eval/prm/evaluate_mathverse_prm_${SCRIPT_SUFFIX}.py"
    DATASET_NAME="mathverse_prm"
    DEFAULT_ROOT="datasets/MathVerse/extracted_images"
    DEFAULT_ANNOTATION="datasets/MathVerse/MathVerse_rollout_annotation_InternVL8B_oversample.json"
    ;;

  OlympiadBench)
    EVAL_SCRIPT="eval/prm/evaluate_olympiadbench_prm_${SCRIPT_SUFFIX}.py"
    DATASET_NAME="olympiadbench_prm"
    DEFAULT_ROOT="."
    DEFAULT_ANNOTATION="datasets/OlympiadBench/OlympiadBench_rollout_annotation_InternVL8B_oversample.json"
    ;;

  *)
    echo "ERROR: Unknown BENCH=${BENCH}"
    echo "Supported: MathVista, MathVision, MathVerse, OlympiadBench"
    exit 1
    ;;
esac

ROOT=${ROOT:-"${DEFAULT_ROOT}"}
ANNOTATION=${ANNOTATION:-"${DEFAULT_ANNOTATION}"}
OUT_DIR=${OUT_DIR:-"${CKPT_ROOT}/eval_${PRM_MODE}_${BENCH}/$(basename "${CKPT}")"}

mkdir -p "${OUT_DIR}"

echo "========== Eval Config =========="
echo "PRM_MODE: ${PRM_MODE}"
echo "BENCH: ${BENCH}"
echo "EVAL_SCRIPT: ${EVAL_SCRIPT}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "CKPT_ROOT: ${CKPT_ROOT}"
echo "CKPT: ${CKPT}"
echo "ROOT: ${ROOT}"
echo "ANNOTATION: ${ANNOTATION}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUS: ${GPUS}"
if [ "${PRM_MODE}" = "beta" ]; then
  echo "SCORE_MODE: ${SCORE_MODE}"
fi
echo "================================="

if [ ! -d "${CKPT}" ]; then
  echo "ERROR: checkpoint does not exist: ${CKPT}"
  exit 1
fi

if [ ! -f "${ANNOTATION}" ]; then
  echo "ERROR: annotation file does not exist: ${ANNOTATION}"
  exit 1
fi

if [ ! -f "${EVAL_SCRIPT}" ]; then
  echo "ERROR: eval script does not exist: ${EVAL_SCRIPT}"
  exit 1
fi

EXTRA_ARGS=()
if [ "${PRM_MODE}" = "beta" ]; then
  EXTRA_ARGS+=(--score-mode "${SCORE_MODE}")
fi

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
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"