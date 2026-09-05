#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

TTS_DATA_ROOT=${TTS_DATA_ROOT:-"${REPO_ROOT}/datasets/TTS_FINAL"}

GENERATOR_MODEL=${GENERATOR_MODEL:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model/InternVL3-8B"}
JUDGE_API_BASE=${JUDGE_API_BASE:-"http://127.0.0.1:8888/v1"}
JUDGE_MODEL=${JUDGE_MODEL:-"Qwen2.5-32B-Instruct"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2"}

NUM_ROLLOUTS=${NUM_ROLLOUTS:-16}
OVERSAMPLE=${OVERSAMPLE:-2.0}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-30}

run_one() {
  local BENCH="$1"
  local IMAGE_ROOT="$2"

  local INPUT="${TTS_DATA_ROOT}/${BENCH}/seed_dataset.json"
  local OUTPUT="${TTS_DATA_ROOT}/${BENCH}/rollout_annotation_internvl3_8b_n16.json"

  if [ ! -f "${INPUT}" ]; then
    echo "ERROR: seed dataset missing: ${INPUT}" >&2
    exit 1
  fi

  python -m eval.build_eval_rollouts_annotation \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --image_root "${IMAGE_ROOT}" \
    --generator_model "${GENERATOR_MODEL}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --judge_api_base "${JUDGE_API_BASE}" \
    --judge_model "${JUDGE_MODEL}" \
    --select_by_llm_quality \
    --oversample "${OVERSAMPLE}" \
    --flush_every 1
}

#run_one "MathVision" "datasets/MathVision"
#run_one "MathVerse"  "datasets/MathVerse"
run_one "MathVista"  "datasets/MathVista"
#run_one "MMStar"     "datasets/MMStar"

echo
echo "[PASS] All final TTS rollout pools are complete."
