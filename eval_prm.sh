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
model_name=${model_name:-"InternVL3-8B"}

# Choose PRM mode:
# beta
# normal
# ensemble
PRM_MODE=${PRM_MODE:-"beta"}

# 写一个就跑一个，写多个就顺序跑多个
# Supported: MathVista MathVision MathVerse OlympiadBench
benchs=${benchs:-"MathVision OlympiadBench MathVerse"}

# Default rollout annotation variant when ANNOTATION is not set explicitly.
# Supported examples: oversample, oversample_2
ANNOTATION_TAG=${ANNOTATION_TAG:-"oversample_2"}

MASTER_PORT=${MASTER_PORT:-63702}

case "${PRM_MODE}" in
  beta)
    CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/log/beta-${model_name}-visualprm400k"}
    SCRIPT_SUFFIX="beta_binomial"
    ;;
  normal)
    CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/log/normal-${model_name}-visualprm400k"}
    SCRIPT_SUFFIX="normal"
    ;;
  ensemble)
    CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/log/ensemble-${model_name}-visualprm400k"}
    SCRIPT_SUFFIX="ensemble"
    ;;
  *)
    echo "ERROR: Unknown PRM_MODE=${PRM_MODE}"
    echo "Supported: beta, normal, ensemble"
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

if [ ! -d "${CKPT}" ]; then
  echo "ERROR: checkpoint does not exist: ${CKPT}"
  exit 1
fi

run_one_benchmark() {
  local BENCH="$1"
  local PORT="$2"

  # =========================
  # Benchmark mapping
  # =========================
  case "${BENCH}" in
    MathVista)
      EVAL_SCRIPT="eval/prm/evaluate_mathvista_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathvista_prm"
      DEFAULT_ROOT="datasets/MathVista/extracted_images"
      DEFAULT_ANNOTATION="datasets/MathVista/MathVista_rollout_annotation_InternVL8B_${ANNOTATION_TAG}.json"
      ;;
    MathVision)
      EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathvision_prm"
      DEFAULT_ROOT="datasets/MathVision/extracted_images"
      DEFAULT_ANNOTATION="datasets/MathVision/MathVision_rollout_annotation_InternVL8B_${ANNOTATION_TAG}.json"
      ;;
    MathVerse)
      EVAL_SCRIPT="eval/prm/evaluate_mathverse_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathverse_prm"
      DEFAULT_ROOT="datasets/MathVerse/extracted_images"
      DEFAULT_ANNOTATION="datasets/MathVerse/MathVerse_rollout_annotation_InternVL8B_${ANNOTATION_TAG}.json"
      ;;
    OlympiadBench)
      EVAL_SCRIPT="eval/prm/evaluate_olympiadbench_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="olympiadbench_prm"
      DEFAULT_ROOT="."
      DEFAULT_ANNOTATION="datasets/OlympiadBench/OlympiadBench_rollout_annotation_InternVL8B_${ANNOTATION_TAG}.json"
      ;;
    *)
      echo "ERROR: Unknown benchmark: ${BENCH}"
      echo "Supported: MathVista, MathVision, MathVerse, OlympiadBench"
      exit 1
      ;;
  esac

ROOT_THIS="${ROOT:-${DEFAULT_ROOT}}"
ANNOTATION_THIS="${ANNOTATION:-${DEFAULT_ANNOTATION}}"

if [ -n "${ANNOTATION:-}" ]; then
  ANNOTATION_DIR_TAG="$(basename "${ANNOTATION_THIS}" .json)"
else
  ANNOTATION_DIR_TAG="${ANNOTATION_TAG}"
fi

# OlympiadBench 的 annotation 可能包含作者机器上的绝对路径：
# /storage1/jiaxinh/Active/jinyuan/MM-PRM/...
# 这里自动重写成当前 REPO_ROOT 下的路径。
if [ "${BENCH}" = "OlympiadBench" ]; then
  FIXED_ANNOTATION="${CKPT_ROOT}/fixed_annotations/OlympiadBench_${ANNOTATION_DIR_TAG}.json"
  mkdir -p "$(dirname "${FIXED_ANNOTATION}")"

  SRC_ANNOTATION="${ANNOTATION_THIS}" \
  DST_ANNOTATION="${FIXED_ANNOTATION}" \
  REPO_ROOT="${REPO_ROOT}" \
  python - <<'PY'
import json
import os

src = os.environ["SRC_ANNOTATION"]
dst = os.environ["DST_ANNOTATION"]
repo_root = os.environ["REPO_ROOT"]

old_prefixes = [
    "/storage1/jiaxinh/Active/jinyuan/MM-PRM",
]

def fix_obj(x):
    if isinstance(x, dict):
        return {k: fix_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [fix_obj(v) for v in x]
    if isinstance(x, str):
        y = x
        for old in old_prefixes:
            if old in y:
                y = y.replace(old, repo_root)
        return y
    return x

with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

data = fix_obj(data)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"[fix annotation] {src} -> {dst}")
PY

  ANNOTATION_THIS="${FIXED_ANNOTATION}"

fi

OUT_DIR="${CKPT_ROOT}/eval_${PRM_MODE}_${BENCH}/${ANNOTATION_DIR_TAG}/$(basename "${CKPT}")"

  mkdir -p "${OUT_DIR}"

  if [ ! -f "${ANNOTATION_THIS}" ]; then
    echo "ERROR: annotation file does not exist: ${ANNOTATION_THIS}"
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

  echo "========== Eval Config =========="
  echo "PRM_MODE: ${PRM_MODE}"
  echo "BENCH: ${BENCH}"
  echo "EVAL_SCRIPT: ${EVAL_SCRIPT}"
  echo "DATASET_NAME: ${DATASET_NAME}"
  echo "CKPT_ROOT: ${CKPT_ROOT}"
  echo "CKPT: ${CKPT}"
  echo "ROOT: ${ROOT_THIS}"
  echo "ANNOTATION: ${ANNOTATION_THIS}"
  echo "ANNOTATION_TAG: ${ANNOTATION_TAG}"
  echo "ANNOTATION_DIR_TAG: ${ANNOTATION_DIR_TAG}"
  echo "OUT_DIR: ${OUT_DIR}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "GPUS: ${GPUS}"
  echo "MASTER_PORT: ${PORT}"
  if [ "${PRM_MODE}" = "beta" ]; then
    echo "SCORE_MODE: ${SCORE_MODE}"
  fi
  echo "================================="

  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${EVAL_SCRIPT}" \
    --checkpoint "${CKPT}" \
    --datasets "${DATASET_NAME}" \
    --root "${ROOT_THIS}" \
    --annotation "${ANNOTATION_THIS}" \
    --out-dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
}

idx=0
for BENCH in ${benchs}; do
  PORT=$((MASTER_PORT + idx))
  run_one_benchmark "${BENCH}" "${PORT}"
  idx=$((idx + 1))
done

echo "All requested benchmarks finished: ${benchs}"
