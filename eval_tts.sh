#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# ============================================================
# User configs
# ============================================================

model_name=${model_name:-"InternVL3-8B"}

# Can specify one or multiple:
# beta normal ensemble bayesian
PRM_MODES=${PRM_MODES:-"normal beta"}

# Can specify one or multiple:
# MathVision MathVerse MathVista MMStar
benchs=${benchs:-"MathVision MathVerse MathVista MMStar"}

GPUS=${GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}

MASTER_PORT=${MASTER_PORT:-63810}

N_GRID=${N_GRID:-"1,2,4,8,16"}

# Because the current rollout pool was quality-ranked before keeping 16,
# use random subsets rather than [:N].
TTS_SUBSET_MODE=${TTS_SUBSET_MODE:-"random"}
TTS_REPEATS=${TTS_REPEATS:-20}
TTS_SEED=${TTS_SEED:-42}

# IAS settings.
TTS_ENABLE_IAS=${TTS_ENABLE_IAS:-1}
TTS_IAS_CONFIDENCE=${TTS_IAS_CONFIDENCE:-0.99}
TTS_IAS_MAX_N=${TTS_IAS_MAX_N:-16}

# 1 = reuse OUT_DIR/prm_output.json when it exists.
# 0 = rerun PRM inference and overwrite fixed outputs.
REUSE_RAW=${REUSE_RAW:-0}

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

# BayesianPRM settings.
BAYESIAN_FROM_PRIOR_ENSEMBLE=${BAYESIAN_FROM_PRIOR_ENSEMBLE:-1}

BELIEF_USE_CONSERVATISM=${BELIEF_USE_CONSERVATISM:-auto}
BELIEF_CONSERVATISM_BETA=${BELIEF_CONSERVATISM_BETA:-}

# Optional explicit checkpoints:
BETA_CKPT=${BETA_CKPT:-}
NORMAL_CKPT=${NORMAL_CKPT:-}
ENSEMBLE_CKPT=${ENSEMBLE_CKPT:-}
BAYESIAN_CKPT=${BAYESIAN_CKPT:-}


is_true() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}


find_latest_checkpoint() {
  local ROOT="$1"

  find "${ROOT}" \
    -maxdepth 1 \
    -type d \
    -name "checkpoint-*" \
    2>/dev/null \
    | sort -V \
    | tail -n 1
}


resolve_mode() {
  local MODE="$1"

  case "${MODE}" in
    beta)
      SCRIPT_SUFFIX="beta_binomial"
      CKPT_ROOT="${REPO_ROOT}/log/beta-${model_name}-visualprm400k"
      CKPT="${BETA_CKPT}"
      ;;

    normal)
      SCRIPT_SUFFIX="normal"
      CKPT_ROOT="${REPO_ROOT}/log/normal-${model_name}-visualprm400k"
      CKPT="${NORMAL_CKPT}"
      ;;

    ensemble)
      SCRIPT_SUFFIX="ensemble"
      CKPT_ROOT="${REPO_ROOT}/log/ensemble-${model_name}-visualprm400k"
      CKPT="${ENSEMBLE_CKPT}"
      ;;

    bayesian)
      SCRIPT_SUFFIX="bayesian"

      if is_true "${BAYESIAN_FROM_PRIOR_ENSEMBLE}"; then
        CKPT_ROOT="${REPO_ROOT}/log/bayesian-prior-${model_name}-visualprm400k"
      else
        CKPT_ROOT="${REPO_ROOT}/log/bayesian-${model_name}-visualprm400k"
      fi

      CKPT="${BAYESIAN_CKPT}"
      ;;

    *)
      echo "ERROR: unknown PRM mode: ${MODE}"
      exit 1
      ;;
  esac

  if [ -z "${CKPT}" ]; then
    CKPT="$(find_latest_checkpoint "${CKPT_ROOT}")"
  fi

  if [ -z "${CKPT}" ] || [ ! -d "${CKPT}" ]; then
    echo "ERROR: checkpoint not found for ${MODE}"
    echo "CKPT_ROOT=${CKPT_ROOT}"
    echo "CKPT=${CKPT}"
    exit 1
  fi
}


resolve_benchmark() {
  local BENCH="$1"

  case "${BENCH}" in
    MathVision)
      EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathvision_prm"
      ROOT_THIS="datasets/MathVision/extracted_images"
      ANNOTATION_THIS="datasets/TTS/MathVision/rollout_annotation_internvl3_8b_n16.json"
      ;;

    MathVerse)
      EVAL_SCRIPT="eval/prm/evaluate_mathverse_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathverse_prm"
      ROOT_THIS="datasets/MathVerse/extracted_images"
      ANNOTATION_THIS="datasets/TTS/MathVerse/rollout_annotation_internvl3_8b_n16.json"
      ;;

    MathVista)
      EVAL_SCRIPT="eval/prm/evaluate_mathvista_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathvista_prm"
      ROOT_THIS="datasets/MathVista/extracted_images"
      ANNOTATION_THIS="datasets/TTS/MathVista/rollout_annotation_internvl3_8b_n16.json"
      ;;

    MMStar)
      # MMStar uses the generic single-image MathVision evaluator.
      # root/annotation are overridden below.
      EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_${SCRIPT_SUFFIX}.py"
      DATASET_NAME="mathvision_prm"
      ROOT_THIS="datasets/MMStar"
      ANNOTATION_THIS="datasets/TTS/MMStar/rollout_annotation_internvl3_8b_n16.json"
      ;;

    *)
      echo "ERROR: unknown benchmark: ${BENCH}"
      exit 1
      ;;
  esac

  if [ ! -f "${ANNOTATION_THIS}" ]; then
    echo "ERROR: annotation missing:"
    echo "${ANNOTATION_THIS}"
    exit 1
  fi

  if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "ERROR: evaluator missing:"
    echo "${EVAL_SCRIPT}"
    exit 1
  fi
}


validate_rollout_annotation() {
  local ANNOTATION_PATH="$1"

  ANNOTATION_PATH="${ANNOTATION_PATH}" python - <<'PY2'
import json
import os
import sys

path = os.environ["ANNOTATION_PATH"]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

bad = []
for i, item in enumerate(data):
    n_solution = len(item.get("solutions_splits", []))
    n_label = len(item.get("labels", []))
    if n_solution != 16 or n_label != 16:
        bad.append((i, n_solution, n_label))

if bad:
    print(f"ERROR: incomplete rollout pool: {path}")
    print(f"bad items: {len(bad)}")
    for i, n_solution, n_label in bad[:50]:
        print(
            f"  item={i}: solutions_splits={n_solution}, labels={n_label}"
        )
    sys.exit(1)

print(
    f"[rollout check] PASS: {len(data)} items, "
    f"all have exactly 16 candidates."
)
PY2
}


find_latest_raw_json() {
  local DIR="$1"
  local PREFIX="$2"

  find "${DIR}" \
    -maxdepth 1 \
    -type f \
    -name "${PREFIX}_*.json" \
    ! -name "*_score.json" \
    ! -name "*_uncertainty_diag.json" \
    -printf '%T@ %p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}


find_latest_score_json() {
  local DIR="$1"
  local PREFIX="$2"

  find "${DIR}" \
    -maxdepth 1 \
    -type f \
    -name "${PREFIX}_*_score.json" \
    -printf '%T@ %p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}


MANIFEST="$(mktemp)"
export MANIFEST


run_one() {
  local MODE="$1"
  local BENCH="$2"
  local PORT="$3"

  resolve_mode "${MODE}"
  resolve_benchmark "${BENCH}"
  validate_rollout_annotation "${ANNOTATION_THIS}"

  EXTRA_ARGS=()

  # Beta raw inference only.
  # TTS risk-budget sweep is performed by evaluate_prm_tts.py.
  if [ "${MODE}" = "beta" ]; then
    EXTRA_ARGS+=(
      --score-mode mu
      --skip-uncertainty-diagnose
    )
  fi

  BAYES_TAG=""

  if [ "${MODE}" = "bayesian" ]; then
    EXTRA_ARGS+=(
      --belief-use-conservatism "${BELIEF_USE_CONSERVATISM}"
    )

    if [ -n "${BELIEF_CONSERVATISM_BETA}" ]; then
      EXTRA_ARGS+=(
        --belief-conservatism-beta "${BELIEF_CONSERVATISM_BETA}"
      )
    fi

    BAYES_TAG="cons-${BELIEF_USE_CONSERVATISM}"

    if [ -n "${BELIEF_CONSERVATISM_BETA}" ]; then
      BAYES_TAG="${BAYES_TAG}_beta-${BELIEF_CONSERVATISM_BETA}"
    fi

    BAYES_TAG="$(echo "${BAYES_TAG}" | tr '/ ' '__')"
  fi

  if [ "${MODE}" = "bayesian" ]; then
    OUT_DIR="${REPO_ROOT}/outputs/tts/${MODE}/${BENCH}/${BAYES_TAG}/$(basename "${CKPT}")"
  else
    OUT_DIR="${REPO_ROOT}/outputs/tts/${MODE}/${BENCH}/$(basename "${CKPT}")"
  fi

  mkdir -p "${OUT_DIR}"

  FIXED_RAW_JSON="${OUT_DIR}/prm_output.json"
  FIXED_SCORE_JSON="${OUT_DIR}/prm_score.json"
  TTS_JSON="${OUT_DIR}/tts_results.json"

  echo "============================================"
  echo "MODE:             ${MODE}"
  echo "BENCH:            ${BENCH}"
  echo "CKPT:             ${CKPT}"
  echo "EVAL_SCRIPT:      ${EVAL_SCRIPT}"
  echo "ANNOTATION:       ${ANNOTATION_THIS}"
  echo "ROOT:             ${ROOT_THIS}"
  echo "OUT_DIR:          ${OUT_DIR}"
  echo "PRM_OUTPUT:       ${FIXED_RAW_JSON}"
  echo "TTS_OUTPUT:       ${TTS_JSON}"
  echo "SUBSET_MODE:      ${TTS_SUBSET_MODE}"
  echo "REPEATS:          ${TTS_REPEATS}"
  echo "============================================"

  if [ "${REUSE_RAW}" = "1" ] && [ -f "${FIXED_RAW_JSON}" ]; then
    RAW_JSON="${FIXED_RAW_JSON}"
    echo "[reuse] ${RAW_JSON}"
  else
    # Remove only old timestamped evaluator outputs for this target.
    # Fixed files prm_output.json / prm_score.json / tts_results.json are unaffected.
    rm -f "${OUT_DIR}/${DATASET_NAME}_"*.json 2>/dev/null || true

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
      --mini-batch-size "${MINI_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      "${EXTRA_ARGS[@]}"

    RAW_JSON="$(find_latest_raw_json "${OUT_DIR}" "${DATASET_NAME}")"

    if [ -z "${RAW_JSON}" ] || [ ! -f "${RAW_JSON}" ]; then
      echo "ERROR: raw evaluator output not found."
      exit 1
    fi

    SCORE_JSON="$(find_latest_score_json "${OUT_DIR}" "${DATASET_NAME}")"

    mv -f "${RAW_JSON}" "${FIXED_RAW_JSON}"
    RAW_JSON="${FIXED_RAW_JSON}"

    if [ -n "${SCORE_JSON}" ] && [ -f "${SCORE_JSON}" ]; then
      mv -f "${SCORE_JSON}" "${FIXED_SCORE_JSON}"
    fi

    # Clean any remaining timestamped JSON generated by evaluator.
    rm -f "${OUT_DIR}/${DATASET_NAME}_"*.json 2>/dev/null || true
  fi

  if [ ! -f "${RAW_JSON}" ]; then
    echo "ERROR: PRM output does not exist: ${RAW_JSON}"
    exit 1
  fi

  TTS_ARGS=(
    --input-json "${RAW_JSON}"
    --output-json "${TTS_JSON}"
    --prm-mode "${MODE}"
    --n-grid "${N_GRID}"
    --subset-mode "${TTS_SUBSET_MODE}"
    --repeats "${TTS_REPEATS}"
    --seed "${TTS_SEED}"
  )

  if is_true "${TTS_ENABLE_IAS}"; then
    case "${MODE}" in
      normal|beta|bayesian)
        TTS_ARGS+=(
          --ias
          --ias-confidence "${TTS_IAS_CONFIDENCE}"
          --ias-max-n "${TTS_IAS_MAX_N}"
        )
        ;;
    esac
  fi

  python eval/tts/evaluate_prm_tts.py \
    "${TTS_ARGS[@]}"

  printf "%s\t%s\t%s\n" \
    "${MODE}" \
    "${BENCH}" \
    "${TTS_JSON}" \
    >> "${MANIFEST}"
}


idx=0

for MODE in ${PRM_MODES}; do
  for BENCH in ${benchs}; do
    PORT=$((MASTER_PORT + idx))

    run_one \
      "${MODE}" \
      "${BENCH}" \
      "${PORT}"

    idx=$((idx + 1))
  done
done

# ============================================================
# Final summary
# ============================================================

# No timestamp. Different mode/benchmark combinations get different summary folders.
SUMMARY_TAG="$(printf '%s__%s' "${PRM_MODES}" "${benchs}" | tr '/ ,:' '_____')"
export SUMMARY_TAG

python - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest = os.environ["MANIFEST"]
summary_tag = os.environ["SUMMARY_TAG"]

rows = []

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        mode, bench, path = line.rstrip("\n").split("\t")

        with open(path, "r", encoding="utf-8") as g:
            data = json.load(g)

        norm_max_n = (
            data.get("ias", {}).get(
                "max_n",
                max(data["n_grid"]),
            )
        )

        for n, result in data["results"].items():
            n_int = int(n)

            rows.append({
                "mode": mode,
                "benchmark": bench,
                "strategy": "fixed",
                "N": str(n_int),
                "average_n": float(n_int),
                "budget_ratio": float(n_int) / norm_max_n,
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "oracle_pass_mean": result["oracle_pass_mean"],
                "oracle_pass_std": result["oracle_pass_std"],
            })

        if "ias" in data:
            result = data["ias"]

            rows.append({
                "mode": mode,
                "benchmark": bench,
                "strategy": "ias",
                "N": "IAS",
                "average_n": result["average_n"],
                "budget_ratio": result["budget_ratio"],
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "oracle_pass_mean": result["oracle_pass_mean"],
                "oracle_pass_std": result["oracle_pass_std"],
            })

rows.sort(
    key=lambda x: (
        x["mode"],
        x["benchmark"],
        0 if x["strategy"] == "fixed" else 1,
        x["average_n"],
    )
)

out_root = Path("outputs/tts/summaries") / summary_tag
out_root.mkdir(parents=True, exist_ok=True)

json_path = out_root / "tts_summary.json"
csv_path = out_root / "tts_summary.csv"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "benchmark",
            "strategy",
            "N",
            "average_n",
            "budget_ratio",
            "accuracy_mean",
            "accuracy_std",
            "oracle_pass_mean",
            "oracle_pass_std",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 120)
print(
    f"{'Mode':10s} "
    f"{'Benchmark':12s} "
    f"{'Method':7s} "
    f"{'N':>5s} "
    f"{'Avg.N':>8s} "
    f"{'Budget':>8s} "
    f"{'Accuracy':>18s} "
    f"{'Oracle pass':>18s}"
)
print("-" * 120)

for r in rows:
    acc = (
        f"{100*r['accuracy_mean']:.2f}"
        f" ± {100*r['accuracy_std']:.2f}"
    )

    oracle = (
        f"{100*r['oracle_pass_mean']:.2f}"
        f" ± {100*r['oracle_pass_std']:.2f}"
    )

    print(
        f"{r['mode']:10s} "
        f"{r['benchmark']:12s} "
        f"{r['strategy']:7s} "
        f"{r['N']:>5s} "
        f"{r['average_n']:8.2f} "
        f"{r['budget_ratio']:8.4f} "
        f"{acc:>18s} "
        f"{oracle:>18s}"
    )

print("=" * 120)
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
PY

rm -f "${MANIFEST}"

echo "All TTS evaluations finished."