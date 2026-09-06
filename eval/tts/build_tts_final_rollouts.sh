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

# =========================
# Basic configs
# =========================
TTS_DATA_ROOT=${TTS_DATA_ROOT:-"${REPO_ROOT}/datasets/TTS_FINAL"}

benchs=${benchs:-"MathVision MathVerse MathVista MMStar"}

# =========================
# Rollout policy
# =========================
# Supported:
#   internvl3_8b
#   qwen3vl_8b
#
# Keep the old behavior as default.
ROLLOUT_POLICY=${ROLLOUT_POLICY:-"qwen3vl_8b"}

# InternVL3-8B: existing local generation path.
GENERATOR_MODEL=${GENERATOR_MODEL:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model/InternVL3-8B"}

# Qwen3-VL-8B: new external generation server.
GENERATOR_SERVER=${GENERATOR_SERVER:-"http://127.0.0.1:18080"}
GENERATOR_TIMEOUT=${GENERATOR_TIMEOUT:-1200}

MAX_REFILL_ROUNDS=${MAX_REFILL_ROUNDS:-0}
ALLOW_INCOMPLETE=${ALLOW_INCOMPLETE:-1}

# =========================
# Judge
# =========================
JUDGE_API_BASE=${JUDGE_API_BASE:-"http://127.0.0.1:8888/v1"}
JUDGE_MODEL=${JUDGE_MODEL:-"Qwen2.5-32B-Instruct"}

# =========================
# Rollout configs
# =========================
NUM_ROLLOUTS=${NUM_ROLLOUTS:-16}
OVERSAMPLE=${OVERSAMPLE:-2.0}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-30}

TIMEOUT=${TIMEOUT:-60}
RETRIES=${RETRIES:-5}


# =========================
# Resolve rollout backend
# =========================
case "${ROLLOUT_POLICY}" in
  internvl3_8b)
    ROLLOUT_TAG="internvl3_8b"

    GENERATOR_ARGS=(
      --generator_model "${GENERATOR_MODEL}"
    )
    ;;

  qwen3vl_8b)
    ROLLOUT_TAG="qwen3vl_8b"

    curl -fsS "${GENERATOR_SERVER}/healthz" >/dev/null || {
      echo "ERROR: Qwen3-VL generator server unavailable: ${GENERATOR_SERVER}" >&2
      exit 1
    }

    GENERATOR_ARGS=(
      --generator_server "${GENERATOR_SERVER}"
      --generator_timeout "${GENERATOR_TIMEOUT}"
    )
    ;;

  *)
    echo "ERROR: Unsupported ROLLOUT_POLICY=${ROLLOUT_POLICY}" >&2
    echo "Supported: internvl3_8b, qwen3vl_8b" >&2
    exit 1
    ;;
esac


run_one() {
  local BENCH="$1"

  local INPUT="${TTS_DATA_ROOT}/${BENCH}/seed_dataset.json"
  local IMAGE_ROOT="datasets/${BENCH}"
  local OUTPUT="${TTS_DATA_ROOT}/${BENCH}/rollout_annotation_${ROLLOUT_TAG}_n${NUM_ROLLOUTS}.json"

  if [ ! -f "${INPUT}" ]; then
    echo "ERROR: Missing seed dataset: ${INPUT}" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Benchmark:      ${BENCH}"
  echo "Rollout policy: ${ROLLOUT_POLICY}"
  echo "Input:          ${INPUT}"
  echo "Output:         ${OUTPUT}"
  echo "============================================================"

  REFILL_ARGS=(
  --max_refill_rounds "${MAX_REFILL_ROUNDS}"
  )

  if [ "${ALLOW_INCOMPLETE}" = "1" ]; then
    REFILL_ARGS+=(--allow_incomplete)
  fi

  python -m eval.build_eval_rollouts_annotation \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --image_root "${IMAGE_ROOT}" \
    "${GENERATOR_ARGS[@]}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --judge_api_base "${JUDGE_API_BASE}" \
    --judge_model "${JUDGE_MODEL}" \
    --select_by_llm_quality \
    --oversample "${OVERSAMPLE}" \
    --timeout "${TIMEOUT}" \
    --retries "${RETRIES}" \
    --flush_every 1 \
    "${REFILL_ARGS[@]}"
}

# run_one "MathVision"
run_one "MathVerse"
run_one "MathVista"
run_one "MMStar"

for BENCH in ${benchs}; do
  run_one "${BENCH}"
done

echo
echo "[PASS] All requested TTS rollout pools are complete."