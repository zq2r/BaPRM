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
# normal beta bayesian
PRM_MODES=${PRM_MODES:-"bayesian"}

GPUS=${GPUS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"7"}

MASTER_PORT=${MASTER_PORT:-63831}

EVAL_SCRIPT=${EVAL_SCRIPT:-"eval/prm/evaluate_visualprocessbench_prm.py"}

ANNOTATION=${ANNOTATION:-"${REPO_ROOT}/datasets/VisualProcessBench/test.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-"${REPO_ROOT}/datasets/VisualProcessBench"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs/visualprocessbench"}

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-0}

# VisualProcessBench official/public evaluator settings.
MAX_NUM=${MAX_NUM:-6}
GRID_MAX_COLS=${GRID_MAX_COLS:-3}

# 1: automatically search one global threshold.
# 0: use FIXED_THRESHOLD.
AUTO_THRESHOLD=${AUTO_THRESHOLD:-1}
FIXED_THRESHOLD=${FIXED_THRESHOLD:-0.5}

# BetaPRM VisualProcessBench score:
# score = mu - lambda * sigma
RISK_LAMBDA=${RISK_LAMBDA:-0.5}

# ============================================================
# BayesianPRM settings
# ============================================================

# Checkpoint selector.
ENSEMBLE_PRM_USE_PRIOR_NETWORK=${ENSEMBLE_PRM_USE_PRIOR_NETWORK:-True}
ENSEMBLE_PRM_NUM_HEADS=${ENSEMBLE_PRM_NUM_HEADS:-10}
ENSEMBLE_PRM_PRIOR_SCALE=${ENSEMBLE_PRM_PRIOR_SCALE:-10}
BELIEF_BETA_KL=${BELIEF_BETA_KL:-0.1}

# Same beta2 grid as eval_tts.sh.
BAYESIAN_BETA2_GRID=${BAYESIAN_BETA2_GRID:-"0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,1.0"}

# BayesianPRM conservatism is enabled for beta2 sweep.
BELIEF_USE_CONSERVATISM=${BELIEF_USE_CONSERVATISM:-true}

# Optional explicit checkpoints.
# Empty means automatically use the latest checkpoint.
NORMAL_CKPT=${NORMAL_CKPT:-}
BETA_CKPT=${BETA_CKPT:-}
BAYESIAN_CKPT=${BAYESIAN_CKPT:-}


# ============================================================
# Helpers
# ============================================================

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


find_latest_file() {
  local DIR="$1"
  local PATTERN="$2"

  find "${DIR}" \
    -maxdepth 1 \
    -type f \
    -name "${PATTERN}" \
    -printf '%T@ %p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}


resolve_mode() {
  local MODE="$1"

  case "${MODE}" in
    normal)
      CKPT_ROOT="${REPO_ROOT}/log/normal-${model_name}-visualprm400k"
      CKPT="${NORMAL_CKPT}"
      ;;

    beta)
      CKPT_ROOT="${REPO_ROOT}/log/beta-${model_name}-visualprm400k"
      CKPT="${BETA_CKPT}"
      ;;

    bayesian)
      if is_true "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"; then
        CKPT_ROOT="${REPO_ROOT}/log/bayesian-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-beta${BELIEF_BETA_KL}-${model_name}-visualprm400k"
      else
        CKPT_ROOT="${REPO_ROOT}/log/bayesian-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-beta${BELIEF_BETA_KL}-${model_name}-visualprm400k"
      fi

      CKPT="${BAYESIAN_CKPT}"
      ;;

    *)
      echo "ERROR: unknown PRM mode: ${MODE}"
      echo "Supported: normal beta bayesian"
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


# ============================================================
# Basic checks
# ============================================================

if [ ! -f "${EVAL_SCRIPT}" ]; then
  echo "ERROR: evaluator not found:"
  echo "${EVAL_SCRIPT}"
  exit 1
fi

if [ ! -f "${ANNOTATION}" ]; then
  echo "ERROR: VisualProcessBench annotation not found:"
  echo "${ANNOTATION}"
  exit 1
fi

if [ ! -d "${IMAGE_ROOT}" ]; then
  echo "ERROR: VisualProcessBench image root not found:"
  echo "${IMAGE_ROOT}"
  exit 1
fi


# ============================================================
# Run one PRM
# ============================================================

MANIFEST="$(mktemp)"
export MANIFEST

run_one() {
  local MODE="$1"
  local PORT="$2"

  resolve_mode "${MODE}"

  EXTRA_ARGS=(
    --prm-mode "${MODE}"
  )

  # ----------------------------------------------------------
  # Threshold
  # ----------------------------------------------------------

  if is_true "${AUTO_THRESHOLD}"; then
    EXTRA_ARGS+=(
      --auto-threshold
    )
  else
    EXTRA_ARGS+=(
      --threshold "${FIXED_THRESHOLD}"
    )
  fi

  # ----------------------------------------------------------
  # BetaPRM
  # ----------------------------------------------------------

  if [ "${MODE}" = "beta" ]; then
    EXTRA_ARGS+=(
      --risk-lambda "${RISK_LAMBDA}"
    )
  fi

  # ----------------------------------------------------------
  # BayesianPRM
  # ----------------------------------------------------------

  BAYES_TAG=""

  if [ "${MODE}" = "bayesian" ]; then
    EXTRA_ARGS+=(
      --belief-use-conservatism "${BELIEF_USE_CONSERVATISM}"
      --bayesian-beta2-grid "${BAYESIAN_BETA2_GRID}"
    )

    BAYES_TAG="beta2-grid"
  fi

  # ----------------------------------------------------------
  # Output directory
  # ----------------------------------------------------------

  if [ "${MODE}" = "bayesian" ]; then
    OUT_DIR="${OUTPUT_ROOT}/${MODE}/${BAYES_TAG}/$(basename "${CKPT}")"
  else
    OUT_DIR="${OUTPUT_ROOT}/${MODE}/$(basename "${CKPT}")"
  fi

  mkdir -p "${OUT_DIR}"

  echo
  echo "============================================================"
  echo "VisualProcessBench"
  echo "MODE:             ${MODE}"
  echo "CKPT:             ${CKPT}"
  echo "ANNOTATION:       ${ANNOTATION}"
  echo "IMAGE_ROOT:       ${IMAGE_ROOT}"
  echo "OUT_DIR:          ${OUT_DIR}"
  echo "GPUS:             ${GPUS}"
  echo "CUDA:             ${CUDA_VISIBLE_DEVICES}"
  echo "AUTO_THRESHOLD:   ${AUTO_THRESHOLD}"

  if [ "${MODE}" = "beta" ]; then
    echo "RISK_LAMBDA:      ${RISK_LAMBDA}"
  fi

  if [ "${MODE}" = "bayesian" ]; then
    echo "CONSERVATISM:     ${BELIEF_USE_CONSERVATISM}"
    echo "BETA2_GRID:       ${BAYESIAN_BETA2_GRID}"
  fi

  echo "============================================================"
  echo

  # Remove only old timestamped outputs in this exact target folder.
  # Fixed metrics.json / prm_output.json are kept until the new run
  # completes successfully.
  rm -f "${OUT_DIR}/visualprocessbench_"*.json 2>/dev/null || true
  rm -f "${OUT_DIR}/metrics_"*.json 2>/dev/null || true

  python -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${EVAL_SCRIPT}" \
    --checkpoint "${CKPT}" \
    --annotation "${ANNOTATION}" \
    --image-root "${IMAGE_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --dynamic \
    --max-num "${MAX_NUM}" \
    --grid-max-cols "${GRID_MAX_COLS}" \
    "${EXTRA_ARGS[@]}"

  RAW_JSON="$(find_latest_file "${OUT_DIR}" 'visualprocessbench_*.json')"
  METRICS_JSON="$(find_latest_file "${OUT_DIR}" 'metrics_*.json')"

  if [ -z "${RAW_JSON}" ] || [ ! -f "${RAW_JSON}" ]; then
    echo "ERROR: VisualProcessBench raw output not found."
    exit 1
  fi

  if [ -z "${METRICS_JSON}" ] || [ ! -f "${METRICS_JSON}" ]; then
    echo "ERROR: VisualProcessBench metrics output not found."
    exit 1
  fi

  FIXED_RAW_JSON="${OUT_DIR}/prm_output.json"
  FIXED_METRICS_JSON="${OUT_DIR}/metrics.json"

  mv -f "${RAW_JSON}" "${FIXED_RAW_JSON}"
  mv -f "${METRICS_JSON}" "${FIXED_METRICS_JSON}"

  printf "%s\t%s\t%s\t%s\n" \
    "${MODE}" \
    "${CKPT}" \
    "${FIXED_METRICS_JSON}" \
    "${OUT_DIR}" \
    >> "${MANIFEST}"

  echo
  echo "[DONE] ${MODE}"
  echo "PRM output: ${FIXED_RAW_JSON}"
  echo "Metrics:    ${FIXED_METRICS_JSON}"

  if [ "${MODE}" = "bayesian" ]; then
    echo
    python - "${FIXED_METRICS_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)

best_beta2 = m.get("best_beta2")

if best_beta2 is None:
    best_beta2 = m.get("overall", {}).get("best_beta2")

print("BayesianPRM best beta2:", best_beta2)

sweep = m.get("beta2_sweep", [])

if sweep:
    print()
    print(f"{'beta2':>10} {'threshold':>12} {'Overall':>12}")
    print("-" * 36)

    for x in sweep:
        beta2 = x.get("beta2")
        threshold = x.get("threshold")

        if threshold is None:
            threshold = x.get("threshold_used")

        overall = x.get("overall")

        if isinstance(overall, dict):
            overall = overall.get("micro_over_sources")

        if overall is None:
            metrics = x.get("metrics", {})
            overall = metrics.get(
                "overall", {}
            ).get("micro_over_sources")

        beta2_str = "-" if beta2 is None else f"{float(beta2):g}"
        threshold_str = "-" if threshold is None else f"{float(threshold):.4f}"
        overall_str = "-" if overall is None else f"{100.0 * float(overall):.2f}"

        print(
            f"{beta2_str:>10} "
            f"{threshold_str:>12} "
            f"{overall_str:>12}"
        )
PY
  fi

  echo
}


# ============================================================
# Run all requested modes
# ============================================================

idx=0

for MODE in ${PRM_MODES}; do
  PORT=$((MASTER_PORT + idx))

  run_one \
    "${MODE}" \
    "${PORT}"

  idx=$((idx + 1))
done


# ============================================================
# Final summary
# ============================================================

SUMMARY_TAG="$(printf '%s' "${PRM_MODES}" | tr '/ ,:' '_____')"
export SUMMARY_TAG

python - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest = os.environ["MANIFEST"]
summary_tag = os.environ["SUMMARY_TAG"]

SOURCE_ORDER = [
    "MathVision",
    "MathVerse",
    "MMMU",
    "DynaMath",
    "WeMath",
]


def canonical_source(name):
    raw = str(name)
    key = raw.lower().replace("_", "").replace("-", "")

    if key.startswith("mathvision"):
        return "MathVision"

    if key.startswith("mathverse"):
        return "MathVerse"

    if key.startswith("mmmu"):
        return "MMMU"

    if key.startswith("dynamath"):
        return "DynaMath"

    if key.startswith("wemath"):
        return "WeMath"

    return raw


def get_best_beta2(m):
    value = m.get("best_beta2")

    if value is not None:
        return float(value)

    value = m.get("overall", {}).get("best_beta2")

    if value is not None:
        return float(value)

    return None


rows = []

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        mode, ckpt, metrics_path, out_dir = (
            line.rstrip("\n").split("\t")
        )

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as g:
            m = json.load(g)

        per_source_raw = m.get(
            "per_source",
            {},
        )

        per_source = {
            canonical_source(k): v
            for k, v in per_source_raw.items()
        }

        overall = m.get(
            "overall",
            {},
        )

        row = {
            "mode": mode,
            "checkpoint": ckpt,
            "best_beta2": (
                get_best_beta2(m)
                if mode == "bayesian"
                else None
            ),
            "threshold": overall.get(
                "threshold_used"
            ),
            "overall": overall.get(
                "micro_over_sources"
            ),
            "MathVision": (
                per_source
                .get("MathVision", {})
                .get("macro_f1")
            ),
            "MathVerse": (
                per_source
                .get("MathVerse", {})
                .get("macro_f1")
            ),
            "MMMU": (
                per_source
                .get("MMMU", {})
                .get("macro_f1")
            ),
            "DynaMath": (
                per_source
                .get("DynaMath", {})
                .get("macro_f1")
            ),
            "WeMath": (
                per_source
                .get("WeMath", {})
                .get("macro_f1")
            ),
            "metrics_path": metrics_path,
            "out_dir": out_dir,
        }

        rows.append(row)


out_root = (
    Path("outputs/visualprocessbench/summaries")
    / summary_tag
)
out_root.mkdir(
    parents=True,
    exist_ok=True,
)

json_path = (
    out_root
    / "visualprocessbench_summary.json"
)

csv_path = (
    out_root
    / "visualprocessbench_summary.csv"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        rows,
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(
    csv_path,
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "checkpoint",
            "best_beta2",
            "threshold",
            "overall",
            *SOURCE_ORDER,
            "metrics_path",
            "out_dir",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


def fmt(v):
    if v is None:
        return "-"
    return f"{100.0 * float(v):.2f}"


def fmt_beta2(v):
    if v is None:
        return "-"
    return f"{float(v):g}"


print()
print("=" * 116)
print("VisualProcessBench Results (%)")
print("-" * 116)

print(
    f"{'Model':<12}"
    f"{'Beta2':>10}"
    f"{'Thr.':>10}"
    f"{'Overall':>12}"
    f"{'MathVision':>14}"
    f"{'MathVerse':>14}"
    f"{'MMMU':>10}"
    f"{'DynaMath':>12}"
    f"{'WeMath':>11}"
)

print("-" * 116)

for r in rows:
    threshold = r["threshold"]

    threshold_str = (
        "-"
        if threshold is None
        else f"{float(threshold):.4f}"
    )

    print(
        f"{r['mode']:<12}"
        f"{fmt_beta2(r['best_beta2']):>10}"
        f"{threshold_str:>10}"
        f"{fmt(r['overall']):>12}"
        f"{fmt(r['MathVision']):>14}"
        f"{fmt(r['MathVerse']):>14}"
        f"{fmt(r['MMMU']):>10}"
        f"{fmt(r['DynaMath']):>12}"
        f"{fmt(r['WeMath']):>11}"
    )

print("=" * 116)
print()
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
PY

rm -f "${MANIFEST}"

echo
echo "All VisualProcessBench evaluations finished."