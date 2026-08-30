#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# ============================================================
# Offline / runtime settings
# ============================================================
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# ============================================================
# User configs
# ============================================================
MODEL_NAME=${MODEL_NAME:-"InternVL3-8B"}

# Run one or multiple benchmarks sequentially.
# Supported: MathVision MathVerse MathVista MMStar
benchs=${benchs:-"MathVision MathVerse MathVista MMStar"}

# BayesianPRM experiment selector.
ENSEMBLE_PRM_USE_PRIOR_NETWORK=${ENSEMBLE_PRM_USE_PRIOR_NETWORK:-True}
ENSEMBLE_PRM_NUM_HEADS=${ENSEMBLE_PRM_NUM_HEADS:-10}
ENSEMBLE_PRM_PRIOR_SCALE=${ENSEMBLE_PRM_PRIOR_SCALE:-10}
BELIEF_BETA_KL=${BELIEF_BETA_KL:-0.05}

GPUS=${GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
MASTER_PORT=${MASTER_PORT:-64900}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_BINS=${NUM_BINS:-10}

if [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "True" ] || \
   [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "true" ]; then
  DEFAULT_CKPT_ROOT="${REPO_ROOT}/log/bayesian-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-beta${BELIEF_BETA_KL}-${MODEL_NAME}-visualprm400k"
else
  DEFAULT_CKPT_ROOT="${REPO_ROOT}/log/bayesian-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-beta${BELIEF_BETA_KL}-${MODEL_NAME}-visualprm400k"
fi

CKPT_ROOT=${CKPT_ROOT:-"${DEFAULT_CKPT_ROOT}"}
CHECKPOINT=${CHECKPOINT:-}

# Eval-time conservatism:
#   auto  -> use checkpoint config
#   true  -> force enable
#   false -> force disable
BELIEF_USE_CONSERVATISM=${BELIEF_USE_CONSERVATISM:-auto}

# Leave empty to use checkpoint config.
BELIEF_CONSERVATISM_BETA=${BELIEF_CONSERVATISM_BETA:-0.001}



# 1: reuse existing prm_output.json if present.
# 0: rerun BayesianPRM inference.
REUSE_INFERENCE=${REUSE_INFERENCE:-0}

# Optional benchmark-specific overrides.
MATHVISION_ANNOTATION=${MATHVISION_ANNOTATION:-}
MATHVISION_MC_LABELS=${MATHVISION_MC_LABELS:-}

MATHVERSE_ANNOTATION=${MATHVERSE_ANNOTATION:-}
MATHVERSE_MC_LABELS=${MATHVERSE_MC_LABELS:-}

MATHVISTA_ANNOTATION=${MATHVISTA_ANNOTATION:-}
MATHVISTA_MC_LABELS=${MATHVISTA_MC_LABELS:-}

MMSTAR_ANNOTATION=${MMSTAR_ANNOTATION:-}
MMSTAR_MC_LABELS=${MMSTAR_MC_LABELS:-}

# All four calibration annotations use the common schema consumed by
# evaluate_mathvision_prm_bayesian.py, so we reuse one evaluator and
# override --root / --annotation.
EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_bayesian.py"
EXTRACT_SCRIPT="eval/calibration/extract_bayesianprm_calibration_predictions.py"
METRIC_SCRIPT="eval/calibration/compute_calibration_metrics.py"

# ============================================================
# Helpers
# ============================================================
find_latest_checkpoint() {
  find "${CKPT_ROOT}" \
    -maxdepth 1 \
    -type d \
    -name 'checkpoint-*' \
    2>/dev/null \
    | sort -V \
    | tail -n 1
}

if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(find_latest_checkpoint)"
fi

if [ -z "${CHECKPOINT}" ] || [ ! -d "${CHECKPOINT}" ]; then
  echo "ERROR: BayesianPRM checkpoint not found." >&2
  echo "CKPT_ROOT=${CKPT_ROOT}" >&2
  echo "CHECKPOINT=${CHECKPOINT}" >&2
  exit 1
fi

for required in "${EVAL_SCRIPT}" "${EXTRACT_SCRIPT}" "${METRIC_SCRIPT}"; do
  if [ ! -f "${required}" ]; then
    echo "ERROR: required script does not exist: ${required}" >&2
    exit 1
  fi
done

VISIBLE_GPU_COUNT="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
if [ "${VISIBLE_GPU_COUNT}" -ne "${GPUS}" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPU_COUNT} GPUs, but GPUS=${GPUS}" >&2
  exit 1
fi

count_jsonl_records() {
  local PATH_THIS="$1"

  JSONL_PATH="${PATH_THIS}" python - <<'PY'
import os

path = os.environ["JSONL_PATH"]
count = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1

print(count)
PY
}

validate_annotation_against_mc_labels() {
  local ANNOTATION_PATH="$1"
  local MC_LABELS_PATH="$2"
  local EXPECTED_PREFIXES_THIS="$3"

  ANNOTATION_PATH="${ANNOTATION_PATH}" \
  MC_LABELS_PATH="${MC_LABELS_PATH}" \
  EXPECTED_PREFIXES="${EXPECTED_PREFIXES_THIS}" \
  python - <<'PY'
import json
import os

ann_path = os.environ["ANNOTATION_PATH"]
labels_path = os.environ["MC_LABELS_PATH"]
expected = int(os.environ["EXPECTED_PREFIXES"])

with open(ann_path, "r", encoding="utf-8") as f:
    annotation = json.load(f)

if not isinstance(annotation, list) or not annotation:
    raise ValueError(f"Invalid calibration annotation: {ann_path}")

flat_ids = []

for i, item in enumerate(annotation):
    prefix_ids = item.get("prefix_ids")
    solutions = item.get("solutions_splits")

    if not isinstance(prefix_ids, list) or not isinstance(solutions, list):
        raise ValueError(
            f"annotation item {i}: "
            "missing prefix_ids/solutions_splits lists"
        )

    if len(prefix_ids) != len(solutions):
        raise ValueError(
            f"annotation item {i}: "
            f"prefix_ids={len(prefix_ids)} "
            f"!= solutions_splits={len(solutions)}"
        )

    flat_ids.extend(prefix_ids)

if len(flat_ids) != expected:
    raise ValueError(
        f"annotation contains {len(flat_ids)} prefixes, "
        f"MC labels contain {expected}"
    )

if len(set(flat_ids)) != len(flat_ids):
    raise ValueError(
        "duplicate prefix_id in calibration annotation"
    )

label_ids = []

with open(labels_path, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        if not line.strip():
            continue

        obj = json.loads(line)
        prefix_id = obj.get("prefix_id")

        if not isinstance(prefix_id, str) or not prefix_id:
            raise ValueError(
                f"invalid prefix_id at MC labels line {lineno}"
            )

        label_ids.append(prefix_id)

if len(label_ids) != expected:
    raise ValueError(
        f"MC label file contains {len(label_ids)} prefixes, "
        f"expected {expected}"
    )

if len(set(label_ids)) != len(label_ids):
    raise ValueError(
        "duplicate prefix_id in MC labels"
    )

if set(flat_ids) != set(label_ids):
    missing = sorted(set(label_ids) - set(flat_ids))
    extra = sorted(set(flat_ids) - set(label_ids))

    raise ValueError(
        "annotation/MC-label prefix mismatch: "
        f"missing={len(missing)}, extra={len(extra)}, "
        f"missing examples={missing[:5]}, "
        f"extra examples={extra[:5]}"
    )

print(
    f"[calibration check] PASS: "
    f"questions={len(annotation)}, prefixes={expected}"
)
PY
}

resolve_benchmark() {
  local BENCH="$1"

  case "${BENCH}" in
    MathVision)
      SLUG="mathvision"
      IMAGE_ROOT="datasets/MathVision/extracted_images"
      CALIB_DIR="outputs/calibration/mathvision"

      # Keep exactly the same MathVision calibration protocol used by
      # the existing NormalPRM evaluation.
      ANNOTATION_THIS="${MATHVISION_ANNOTATION:-outputs/calibration/mathvision/mathvision_calibration_annotation_with_source_labels.json}"
      MC_LABELS_THIS="${MATHVISION_MC_LABELS:-outputs/calibration/mathvision/mc_labels_full_internvl3_n16.jsonl}"
      ;;

    MathVerse)
      SLUG="mathverse"
      IMAGE_ROOT="datasets/MathVerse/extracted_images"
      CALIB_DIR="outputs/calibration/mathverse"

      ANNOTATION_THIS="${MATHVERSE_ANNOTATION:-outputs/calibration/mathverse/prm_annotation_1000.json}"
      MC_LABELS_THIS="${MATHVERSE_MC_LABELS:-outputs/calibration/mathverse/mc_labels_internvl3_n16.jsonl}"
      ;;

    MathVista)
      SLUG="mathvista"
      IMAGE_ROOT="datasets/MathVista/extracted_images"
      CALIB_DIR="outputs/calibration/mathvista"

      ANNOTATION_THIS="${MATHVISTA_ANNOTATION:-outputs/calibration/mathvista/prm_annotation_1000.json}"
      MC_LABELS_THIS="${MATHVISTA_MC_LABELS:-outputs/calibration/mathvista/mc_labels_internvl3_n16.jsonl}"
      ;;

    MMStar)
      SLUG="mmstar"
      IMAGE_ROOT="datasets/MMStar"
      CALIB_DIR="outputs/calibration/mmstar"

      ANNOTATION_THIS="${MMSTAR_ANNOTATION:-outputs/calibration/mmstar/prm_annotation_1000.json}"
      MC_LABELS_THIS="${MMSTAR_MC_LABELS:-outputs/calibration/mmstar/mc_labels_internvl3_n16.jsonl}"
      ;;

    *)
      echo "ERROR: unsupported benchmark: ${BENCH}" >&2
      echo "Supported: MathVision MathVerse MathVista MMStar" >&2
      exit 1
      ;;
  esac

  if [ ! -d "${IMAGE_ROOT}" ]; then
    echo "ERROR: image root does not exist: ${IMAGE_ROOT}" >&2
    exit 1
  fi

  if [ ! -f "${ANNOTATION_THIS}" ]; then
    echo "ERROR: calibration annotation does not exist: ${ANNOTATION_THIS}" >&2
    exit 1
  fi

  if [ ! -f "${MC_LABELS_THIS}" ]; then
    echo "ERROR: MC labels do not exist: ${MC_LABELS_THIS}" >&2
    exit 1
  fi
}

find_latest_raw_json() {
  local DIR="$1"

  find "${DIR}" \
    -maxdepth 1 \
    -type f \
    -name 'mathvision_prm_*.json' \
    ! -name '*_score.json' \
    ! -name '*_uncertainty_diag.json' \
    -printf '%T@ %p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

find_latest_score_json() {
  local DIR="$1"

  find "${DIR}" \
    -maxdepth 1 \
    -type f \
    -name 'mathvision_prm_*_score.json' \
    -printf '%T@ %p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

# Resolve the actual beta_2 / conservatism setting used at evaluation.
# If no override is given, read them from checkpoint config.json.
mapfile -t BAYES_EVAL_SETTINGS < <(
  CHECKPOINT="${CHECKPOINT}" \
  CONS_OVERRIDE="${BELIEF_USE_CONSERVATISM}" \
  BETA_OVERRIDE="${BELIEF_CONSERVATISM_BETA}" \
  python - <<'PY'
import json
import os

checkpoint = os.environ["CHECKPOINT"]
cons_override = os.environ["CONS_OVERRIDE"].strip().lower()
beta_override = os.environ["BETA_OVERRIDE"].strip()

config_path = os.path.join(checkpoint, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

if cons_override == "auto":
    use_conservatism = bool(
        config.get("belief_use_conservatism", False)
    )
elif cons_override in {"true", "1", "yes", "y"}:
    use_conservatism = True
elif cons_override in {"false", "0", "no", "n"}:
    use_conservatism = False
else:
    raise ValueError(
        f"Invalid BELIEF_USE_CONSERVATISM="
        f"{cons_override!r}"
    )

if beta_override:
    beta = float(beta_override)
else:
    beta = float(
        config.get("belief_conservatism_beta", 0.1)
    )

if beta <= 0:
    raise ValueError(
        f"belief_conservatism_beta must be > 0, got {beta}"
    )

print("true" if use_conservatism else "false")
print(beta)
PY
)

RESOLVED_USE_CONSERVATISM="${BAYES_EVAL_SETTINGS[0]}"
RESOLVED_CONSERVATISM_BETA="${BAYES_EVAL_SETTINGS[1]}"

EVAL_TAG="cons-${RESOLVED_USE_CONSERVATISM}_beta2-${RESOLVED_CONSERVATISM_BETA}"

# ============================================================
# Temporary manifest for final summary
# ============================================================
MANIFEST="$(mktemp)"
export MANIFEST
trap 'rm -f "${MANIFEST}"' EXIT

# ============================================================
# Evaluate one benchmark
# ============================================================
run_one_benchmark() {
  local BENCH="$1"
  local PORT="$2"

  resolve_benchmark "${BENCH}"

  # IMPORTANT:
  # Do not hard-code 3513 or 1000 here.
  # Use exactly however many prefixes are in the existing MC-label
  # file for this benchmark, matching the NormalPRM protocol.
  EXPECTED_PREFIXES="$(count_jsonl_records "${MC_LABELS_THIS}")"

  if [ "${EXPECTED_PREFIXES}" -le 0 ]; then
    echo "ERROR: no MC labels in ${MC_LABELS_THIS}" >&2
    exit 1
  fi

  validate_annotation_against_mc_labels \
    "${ANNOTATION_THIS}" \
    "${MC_LABELS_THIS}" \
    "${EXPECTED_PREFIXES}"

  OUT_DIR="${CALIB_DIR}/bayesianprm/$(basename "${CHECKPOINT}")/${EVAL_TAG}"
  mkdir -p "${OUT_DIR}"

  FIXED_RAW_JSON="${OUT_DIR}/prm_output.json"
  FIXED_SCORE_JSON="${OUT_DIR}/prm_score.json"

  PRED_REL="${OUT_DIR}/predictions_rel.jsonl"
  PRED_FINAL="${OUT_DIR}/predictions_final.jsonl"
  PRED_SUMMARY="${OUT_DIR}/predictions.summary.json"

  METRICS_JSON="${OUT_DIR}/metrics.json"
  METRICS_CSV="${OUT_DIR}/metrics.csv"

  echo "============================================================"
  echo "BayesianPRM calibration"
  echo "Benchmark:         ${BENCH}"
  echo "Checkpoint:        ${CHECKPOINT}"
  echo "Annotation:        ${ANNOTATION_THIS}"
  echo "Image root:        ${IMAGE_ROOT}"
  echo "MC labels:         ${MC_LABELS_THIS}"
  echo "Expected prefixes: ${EXPECTED_PREFIXES}"
  echo "Conservatism:      ${RESOLVED_USE_CONSERVATISM}"
  echo "beta_2:            ${RESOLVED_CONSERVATISM_BETA}"
  echo "Output dir:        ${OUT_DIR}"
  echo "GPU(s):            ${CUDA_VISIBLE_DEVICES}"
  echo "Processes:         ${GPUS}"
  echo "Master port:       ${PORT}"
  echo "============================================================"

  # ----------------------------------------------------------
  # 1. BayesianPRM inference
  # ----------------------------------------------------------
  if [ "${REUSE_INFERENCE}" = "1" ] && [ -f "${FIXED_RAW_JSON}" ]; then
    echo "[reuse inference] ${FIXED_RAW_JSON}"
  else
    # Remove only old evaluator-generated timestamped files.
    rm -f \
      "${OUT_DIR}/mathvision_prm_"*.json \
      2>/dev/null || true

    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node="${GPUS}" \
      --master_port="${PORT}" \
      "${EVAL_SCRIPT}" \
      --checkpoint "${CHECKPOINT}" \
      --datasets mathvision_prm \
      --root "${IMAGE_ROOT}" \
      --annotation "${ANNOTATION_THIS}" \
      --mini-batch-size "${MINI_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --out-dir "${OUT_DIR}" \
      --belief-use-conservatism "${RESOLVED_USE_CONSERVATISM}" \
      --belief-conservatism-beta "${RESOLVED_CONSERVATISM_BETA}" \
      2>&1 | tee "${OUT_DIR}/inference.log"

    RAW_JSON="$(find_latest_raw_json "${OUT_DIR}")"
    SCORE_JSON="$(find_latest_score_json "${OUT_DIR}")"

    if [ -z "${RAW_JSON}" ] || [ ! -f "${RAW_JSON}" ]; then
      echo "ERROR: BayesianPRM evaluator did not produce raw output JSON." >&2
      exit 1
    fi

    mv -f "${RAW_JSON}" "${FIXED_RAW_JSON}"

    if [ -n "${SCORE_JSON}" ] && [ -f "${SCORE_JSON}" ]; then
      mv -f "${SCORE_JSON}" "${FIXED_SCORE_JSON}"
    fi

    # Remove any leftover timestamped evaluator files.
    rm -f \
      "${OUT_DIR}/mathvision_prm_"*.json \
      2>/dev/null || true
  fi

  printf '%s\n' "${CHECKPOINT}" \
    > "${OUT_DIR}/checkpoint_path.txt"

  # ----------------------------------------------------------
  # 2. Extract reliability-only + final predictions
  # ----------------------------------------------------------
  python "${EXTRACT_SCRIPT}" \
    --evaluator-output "${FIXED_RAW_JSON}" \
    --mc-labels "${MC_LABELS_THIS}" \
    --output-rel "${PRED_REL}" \
    --output-final "${PRED_FINAL}" \
    --summary-output "${PRED_SUMMARY}" \
    --model-name-rel "BayesianPRM-Rel-${MODEL_NAME}" \
    --model-name-final "BayesianPRM-Final-${MODEL_NAME}" \
    --checkpoint "${CHECKPOINT}" \
    --expected-prefixes "${EXPECTED_PREFIXES}" \
    --overwrite

  # ----------------------------------------------------------
  # 3. Compute KWDK-style calibration metrics
  # ----------------------------------------------------------
  python "${METRIC_SCRIPT}" \
    --input "${PRED_REL}" "${PRED_FINAL}" \
    --output-json "${METRICS_JSON}" \
    --output-csv "${METRICS_CSV}" \
    --num-bins "${NUM_BINS}" \
    --expected-prefixes "${EXPECTED_PREFIXES}" \
    --overwrite

  printf '%s\t%s\t%s\n' \
    "${BENCH}" \
    "${EXPECTED_PREFIXES}" \
    "${METRICS_JSON}" \
    >> "${MANIFEST}"
}

# ============================================================
# Run requested benchmarks
# ============================================================
idx=0

for BENCH in ${benchs}; do
  PORT=$((MASTER_PORT + idx))
  run_one_benchmark "${BENCH}" "${PORT}"
  idx=$((idx + 1))
done

# ============================================================
# Combined summary
# ============================================================
SUMMARY_DIR="outputs/calibration/bayesianprm/$(basename "${CHECKPOINT}")/${EVAL_TAG}"
mkdir -p "${SUMMARY_DIR}"

export SUMMARY_DIR
export CHECKPOINT
export MODEL_NAME
export EVAL_TAG
export RESOLVED_USE_CONSERVATISM
export RESOLVED_CONSERVATISM_BETA

python - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
summary_dir = Path(os.environ["SUMMARY_DIR"])

rows = []

with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        bench, expected_prefixes, metrics_path = (
            line.rstrip("\n").split("\t")
        )

        with open(metrics_path, "r", encoding="utf-8") as g:
            metrics = json.load(g)

        models = metrics.get("models", [])

        if len(models) != 2:
            raise ValueError(
                f"{metrics_path}: expected exactly two model "
                "entries (Reliability and Final)"
            )

        for variant, model in zip(
            ("Reliability", "Final"),
            models,
        ):
            rows.append(
                {
                    "variant": variant,
                    "benchmark": bench,
                    "num_prefixes": int(expected_prefixes),
                    "brier": float(model["brier"]),
                    "positive_brier": float(
                        model["positive_brier"]
                    ),
                    "adaptive_ce": float(
                        model["adaptive_ce"]
                    ),
                    "ece": float(model["ece"]),
                    "average_ce": float(
                        model["average_ce"]
                    ),
                }
            )

metric_keys = [
    "brier",
    "positive_brier",
    "adaptive_ce",
    "ece",
    "average_ce",
]

# Row-wise average across all five calibration metrics.
for row in rows:
    row["avg"] = sum(
        row[key] for key in metric_keys
    ) / len(metric_keys)

output = {
    "schema_version": 1,
    "checkpoint": os.environ["CHECKPOINT"],
    "model_name": os.environ["MODEL_NAME"],
    "eval_tag": os.environ["EVAL_TAG"],
    "belief_use_conservatism": (
        os.environ[
            "RESOLVED_USE_CONSERVATISM"
        ].lower()
        == "true"
    ),
    "belief_conservatism_beta": float(
        os.environ[
            "RESOLVED_CONSERVATISM_BETA"
        ]
    ),
    "benchmarks": rows,
}

json_path = summary_dir / "summary.json"
csv_path = summary_dir / "summary.csv"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2,
    )

fieldnames = [
    "variant",
    "benchmark",
    "num_prefixes",
    "brier",
    "positive_brier",
    "adaptive_ce",
    "ece",
    "average_ce",
    "avg",
]

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print(
    "=== BayesianPRM calibration summary ==="
)

header = (
    f"{'Variant':12s} "
    f"{'Benchmark':12s} "
    f"{'Prefixes':>9s} "
    f"{'Brier':>10s} "
    f"{'PosBrier':>10s} "
    f"{'AdaptiveCE':>10s} "
    f"{'ECE':>10s} "
    f"{'AverageCE':>10s} "
    f"{'Avg':>10s}"
)

print(header)
print("-" * len(header))

for row in rows:
    print(
        f"{row['variant']:12s} "
        f"{row['benchmark']:12s} "
        f"{row['num_prefixes']:9d} "
        f"{row['brier']:10.6f} "
        f"{row['positive_brier']:10.6f} "
        f"{row['adaptive_ce']:10.6f} "
        f"{row['ece']:10.6f} "
        f"{row['average_ce']:10.6f} "
        f"{row['avg']:10.6f}"
    )

print(f"\nJSON: {json_path}")
print(f"CSV:  {csv_path}")
PY

echo
echo "All requested BayesianPRM calibration evaluations finished: ${benchs}"
