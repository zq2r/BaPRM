#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script is intended to be copied to the BaPRM repo root.
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
MODEL_NAME=${MODEL_NAME:-"InternVL3-8B"}

# Write one or multiple benchmarks, executed sequentially.
# Supported: MathVision MathVerse MathVista MMStar
benchs=${benchs:-"MathVision MathVerse MathVista MMStar"}

CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/log/normal-${MODEL_NAME}-visualprm400k"}
CHECKPOINT=${CHECKPOINT:-}

GPUS=${GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
MASTER_PORT=${MASTER_PORT:-63900}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
NUM_BINS=${NUM_BINS:-10}

# 1: reuse fixed prm_output.json when it already exists.
# 0: rerun NormalPRM inference.
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

# Calibration annotations are grouped into the common schema consumed by
# evaluate_mathvision_prm_normal.py, so this evaluator can be reused for all
# four single-image benchmarks by overriding --root and --annotation.
EVAL_SCRIPT="eval/prm/evaluate_mathvision_prm_normal.py"
EXTRACT_SCRIPT="eval/calibration/extract_normalprm_calibration_predictions.py"
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
  echo "ERROR: NormalPRM checkpoint not found." >&2
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
  local EXPECTED_PREFIXES="$3"

  ANNOTATION_PATH="${ANNOTATION_PATH}" \
  MC_LABELS_PATH="${MC_LABELS_PATH}" \
  EXPECTED_PREFIXES="${EXPECTED_PREFIXES}" \
  python - <<'PY'
import json
import os
import sys

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
        raise ValueError(f"annotation item {i}: missing prefix_ids/solutions_splits lists")
    if len(prefix_ids) != len(solutions):
        raise ValueError(
            f"annotation item {i}: prefix_ids={len(prefix_ids)} "
            f"!= solutions_splits={len(solutions)}"
        )
    flat_ids.extend(prefix_ids)

if len(flat_ids) != expected:
    raise ValueError(
        f"annotation contains {len(flat_ids)} prefixes, MC labels contain {expected}"
    )
if len(set(flat_ids)) != len(flat_ids):
    raise ValueError("duplicate prefix_id in calibration annotation")

label_ids = []
with open(labels_path, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        pid = obj.get("prefix_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"invalid prefix_id at MC labels line {lineno}")
        label_ids.append(pid)

if set(flat_ids) != set(label_ids):
    missing = sorted(set(label_ids) - set(flat_ids))
    extra = sorted(set(flat_ids) - set(label_ids))
    raise ValueError(
        f"annotation/MC-label prefix mismatch: missing={len(missing)}, extra={len(extra)}"
    )

print(
    f"[calibration check] PASS: questions={len(annotation)}, "
    f"prefixes={expected}"
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

MANIFEST="$(mktemp)"
export MANIFEST
trap 'rm -f "${MANIFEST}"' EXIT

# ============================================================
# One benchmark
# ============================================================
run_one_benchmark() {
  local BENCH="$1"
  local PORT="$2"

  resolve_benchmark "${BENCH}"

  EXPECTED_PREFIXES="$(count_jsonl_records "${MC_LABELS_THIS}")"
  if [ "${EXPECTED_PREFIXES}" -le 0 ]; then
    echo "ERROR: no MC labels in ${MC_LABELS_THIS}" >&2
    exit 1
  fi

  validate_annotation_against_mc_labels \
    "${ANNOTATION_THIS}" \
    "${MC_LABELS_THIS}" \
    "${EXPECTED_PREFIXES}"

  OUT_DIR="${CALIB_DIR}/normalprm/$(basename "${CHECKPOINT}")"
  mkdir -p "${OUT_DIR}"

  FIXED_RAW_JSON="${OUT_DIR}/prm_output.json"
  FIXED_SCORE_JSON="${OUT_DIR}/prm_score.json"
  PRED_OUTPUT="${OUT_DIR}/predictions.jsonl"
  PRED_SUMMARY="${OUT_DIR}/predictions.summary.json"
  METRICS_JSON="${OUT_DIR}/metrics.json"
  METRICS_CSV="${OUT_DIR}/metrics.csv"

  echo "============================================================"
  echo "NormalPRM calibration"
  echo "Benchmark:         ${BENCH}"
  echo "Checkpoint:        ${CHECKPOINT}"
  echo "Annotation:        ${ANNOTATION_THIS}"
  echo "Image root:        ${IMAGE_ROOT}"
  echo "MC labels:         ${MC_LABELS_THIS}"
  echo "Expected prefixes: ${EXPECTED_PREFIXES}"
  echo "Output dir:        ${OUT_DIR}"
  echo "GPU(s):            ${CUDA_VISIBLE_DEVICES}"
  echo "Processes:         ${GPUS}"
  echo "Master port:       ${PORT}"
  echo "============================================================"

  if [ "${REUSE_INFERENCE}" = "1" ] && [ -f "${FIXED_RAW_JSON}" ]; then
    echo "[reuse inference] ${FIXED_RAW_JSON}"
  else
    # Remove previous evaluator-generated timestamped files only.
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
      2>&1 | tee "${OUT_DIR}/inference.log"

    RAW_JSON="$(find_latest_raw_json "${OUT_DIR}")"
    SCORE_JSON="$(find_latest_score_json "${OUT_DIR}")"

    if [ -z "${RAW_JSON}" ] || [ ! -f "${RAW_JSON}" ]; then
      echo "ERROR: NormalPRM evaluator did not produce raw output JSON." >&2
      exit 1
    fi

    mv -f "${RAW_JSON}" "${FIXED_RAW_JSON}"

    if [ -n "${SCORE_JSON}" ] && [ -f "${SCORE_JSON}" ]; then
      mv -f "${SCORE_JSON}" "${FIXED_SCORE_JSON}"
    fi

    # Clean up any leftover timestamped evaluator outputs.
    rm -f \
      "${OUT_DIR}/mathvision_prm_"*.json \
      2>/dev/null || true
  fi

  printf '%s\n' "${CHECKPOINT}" > "${OUT_DIR}/checkpoint_path.txt"

  # Align terminal NormalPRM probabilities to the Monte Carlo prefix labels.
  python "${EXTRACT_SCRIPT}" \
    --evaluator-output "${FIXED_RAW_JSON}" \
    --mc-labels "${MC_LABELS_THIS}" \
    --output "${PRED_OUTPUT}" \
    --summary-output "${PRED_SUMMARY}" \
    --model-name "NormalPRM-${MODEL_NAME}" \
    --checkpoint "${CHECKPOINT}" \
    --expected-prefixes "${EXPECTED_PREFIXES}" \
    --overwrite

  # KWDK-style calibration metrics:
  # Brier, Positive Brier, AdaptiveCE, ECE, AverageCE.
  python "${METRIC_SCRIPT}" \
    --input "${PRED_OUTPUT}" \
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
SUMMARY_DIR="outputs/calibration/normalprm/$(basename "${CHECKPOINT}")"
mkdir -p "${SUMMARY_DIR}"
export SUMMARY_DIR CHECKPOINT MODEL_NAME

python - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
summary_dir = Path(os.environ["SUMMARY_DIR"])
checkpoint = os.environ["CHECKPOINT"]
model_name = os.environ["MODEL_NAME"]

rows = []
with manifest.open("r", encoding="utf-8") as f:
    for line in f:
        bench, expected_prefixes, metrics_path = line.rstrip("\n").split("\t")
        with open(metrics_path, "r", encoding="utf-8") as g:
            metrics = json.load(g)
        models = metrics.get("models", [])
        if len(models) != 1:
            raise ValueError(f"{metrics_path}: expected exactly one model entry")
        m = models[0]
        rows.append({
            "benchmark": bench,
            "num_prefixes": int(expected_prefixes),
            "brier": float(m["brier"]),
            "positive_brier": float(m["positive_brier"]),
            "adaptive_ce": float(m["adaptive_ce"]),
            "ece": float(m["ece"]),
            "average_ce": float(m["average_ce"]),
        })

metric_keys = [
    "brier",
    "positive_brier",
    "adaptive_ce",
    "ece",
    "average_ce",
]

macro = {
    "benchmark": "MacroAvg",
    "num_prefixes": sum(r["num_prefixes"] for r in rows),
}
for key in metric_keys:
    macro[key] = sum(r[key] for r in rows) / len(rows)

output = {
    "schema_version": 1,
    "model_name": f"NormalPRM-{model_name}",
    "checkpoint": checkpoint,
    "benchmarks": rows,
    "macro_average": macro,
}

json_path = summary_dir / "summary.json"
csv_path = summary_dir / "summary.csv"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "benchmark",
            "num_prefixes",
            "brier",
            "positive_brier",
            "adaptive_ce",
            "ece",
            "average_ce",
        ],
    )
    writer.writeheader()
    writer.writerows(rows + [macro])

print()
print("=== NormalPRM calibration summary ===")
header = (
    f"{'Benchmark':12s} "
    f"{'Prefixes':>9s} "
    f"{'Brier':>10s} "
    f"{'PosBrier':>10s} "
    f"{'AdaptiveCE':>10s} "
    f"{'ECE':>10s} "
    f"{'AverageCE':>10s}"
)
print(header)
print("-" * len(header))

for r in rows + [macro]:
    print(
        f"{r['benchmark']:12s} "
        f"{r['num_prefixes']:9d} "
        f"{r['brier']:10.6f} "
        f"{r['positive_brier']:10.6f} "
        f"{r['adaptive_ce']:10.6f} "
        f"{r['ece']:10.6f} "
        f"{r['average_ce']:10.6f}"
    )

print(f"\nJSON: {json_path}")
print(f"CSV:  {csv_path}")
PY

echo "All requested NormalPRM calibration evaluations finished: ${benchs}"
