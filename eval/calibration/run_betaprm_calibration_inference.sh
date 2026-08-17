#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

GPUS="${GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-63731}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/log/beta-InternVL3-8B-visualprm400k/checkpoint-1103}"

ANNOTATION="${ANNOTATION:-${REPO_ROOT}/outputs/calibration/mathvision/mathvision_calibration_annotation_with_source_labels.json}"

IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/datasets/MathVision/extracted_images}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/calibration/mathvision/betaprm_checkpoint1103}"

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "${ANNOTATION}" ]]; then
    echo "[ERROR] Annotation does not exist: ${ANNOTATION}" >&2
    exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
    echo "[ERROR] Image root does not exist: ${IMAGE_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "========================================"
echo "BetaPRM calibration inference"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Annotation:      ${ANNOTATION}"
echo "Image root:      ${IMAGE_ROOT}"
echo "Output dir:      ${OUT_DIR}"
echo "Visible GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo "Processes:       ${GPUS}"
echo "Master port:     ${MASTER_PORT}"
echo "Mini batch size: ${MINI_BATCH_SIZE}"
echo "========================================"

torchrun \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    eval/prm/evaluate_mathvision_prm_beta_binomial.py \
    --checkpoint "${CHECKPOINT}" \
    --datasets mathvision_prm \
    --root "${IMAGE_ROOT}" \
    --annotation "${ANNOTATION}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    --score-mode mu \
    --skip-uncertainty-diagnose \
    2>&1 | tee "${OUT_DIR}/inference.log"
