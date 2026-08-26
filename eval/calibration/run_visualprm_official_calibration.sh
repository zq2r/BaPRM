#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"

REPO_ROOT="$(
  cd "${SCRIPT_DIR}/../.."
  pwd
)"

cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

# =========================
# Config
# =========================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

GPUS="${GPUS:-4}"

MASTER_PORT="${MASTER_PORT:-63741}"

MODEL="${MODEL:-${REPO_ROOT}/../../model/VisualPRM-8B-v1_1}"

ANNOTATION="${ANNOTATION:-${REPO_ROOT}/outputs/calibration/mathvision/mathvision_calibration_annotation_with_source_labels.json}"

IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/datasets/MathVision/extracted_images}"

MC_LABELS="${MC_LABELS:-${REPO_ROOT}/outputs/calibration/mathvision/mc_labels_full_internvl3_n16.jsonl}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/calibration/mathvision/visualprm_official}"

EXPECTED_PREFIXES="${EXPECTED_PREFIXES:-3513}"

MAX_NUM="${MAX_NUM:-12}"

EVAL_OUTPUT="${OUT_DIR}/mathvision_visualprm_official.json"

PRED_OUTPUT="${OUT_DIR}/predictions_visualprm_8b_v1_1.jsonl"

SUMMARY_OUTPUT="${OUT_DIR}/predictions_visualprm_8b_v1_1.summary.json"

METRICS_JSON="${OUT_DIR}/metrics_visualprm_8b_v1_1.json"

METRICS_CSV="${OUT_DIR}/metrics_visualprm_8b_v1_1.csv"


# =========================
# Checks
# =========================

if [[ ! -d "${MODEL}" ]]; then
    echo "[ERROR] Model does not exist:"
    echo "${MODEL}"
    exit 1
fi

if [[ ! -f "${ANNOTATION}" ]]; then
    echo "[ERROR] Annotation does not exist:"
    echo "${ANNOTATION}"
    exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
    echo "[ERROR] Image root does not exist:"
    echo "${IMAGE_ROOT}"
    exit 1
fi

if [[ ! -f "${MC_LABELS}" ]]; then
    echo "[ERROR] MC labels do not exist:"
    echo "${MC_LABELS}"
    exit 1
fi

mkdir -p "${OUT_DIR}"


echo "========================================"
echo "Official VisualPRM calibration"
echo "Model:        ${MODEL}"
echo "Annotation:   ${ANNOTATION}"
echo "Image root:   ${IMAGE_ROOT}"
echo "MC labels:    ${MC_LABELS}"
echo "Output dir:   ${OUT_DIR}"
echo "GPUs:         ${GPUS}"
echo "Prefixes:     ${EXPECTED_PREFIXES}"
echo "========================================"


# =========================
# 1. VisualPRM inference
# =========================

torchrun \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  eval/prm/evaluate_mathvision_visualprm_official.py \
  --checkpoint "${MODEL}" \
  --annotation "${ANNOTATION}" \
  --root "${IMAGE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --output-name "mathvision_visualprm_official.json" \
  --expected-prefixes "${EXPECTED_PREFIXES}" \
  --max-num "${MAX_NUM}" \
  --overwrite \
  2>&1 | tee "${OUT_DIR}/inference.log"


# =========================
# 2. Extract terminal scores
# =========================

python \
  eval/calibration/extract_visualprm_official_calibration_predictions.py \
  --evaluator-output "${EVAL_OUTPUT}" \
  --mc-labels "${MC_LABELS}" \
  --output "${PRED_OUTPUT}" \
  --summary-output "${SUMMARY_OUTPUT}" \
  --model-name "VisualPRM-8B-v1_1" \
  --checkpoint "${MODEL}" \
  --expected-prefixes "${EXPECTED_PREFIXES}" \
  --overwrite


# =========================
# 3. Calibration metrics
# =========================

python \
  eval/calibration/compute_calibration_metrics.py \
  --input "${PRED_OUTPUT}" \
  --output-json "${METRICS_JSON}" \
  --output-csv "${METRICS_CSV}" \
  --num-bins 10 \
  --expected-prefixes "${EXPECTED_PREFIXES}" \
  --overwrite


echo
echo "========================================"
echo "[PASS] VisualPRM calibration completed."
echo
echo "Predictions:"
echo "${PRED_OUTPUT}"
echo
echo "Metrics:"
echo "${METRICS_JSON}"
echo "========================================"