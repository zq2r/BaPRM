#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TTS_DATA_ROOT=${TTS_DATA_ROOT:-"${REPO_ROOT}/datasets/TTS_FINAL"}

GENERATOR_MODEL=${GENERATOR_MODEL:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model/InternVL3-8B"}
JUDGE_API_BASE=${JUDGE_API_BASE:-"http://127.0.0.1:8888/v1"}
JUDGE_MODEL=${JUDGE_MODEL:-"Qwen2.5-32B-Instruct"}

GENERATOR_CUDA_VISIBLE_DEVICES=${GENERATOR_CUDA_VISIBLE_DEVICES:-"0"}

NUM_ROLLOUTS=${NUM_ROLLOUTS:-16}
OVERSAMPLE=${OVERSAMPLE:-2.0}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-30}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

BENCHMARKS=(
  "MathVision|304|datasets/MathVision"
  "MathVerse|788|datasets/MathVerse"
  "MathVista|1000|datasets/MathVista"
  "MMStar|1500|datasets/MMStar"
)

echo "============================================================"
echo "TTS rollout repair"
echo "Data root : ${TTS_DATA_ROOT}"
echo "Generator : ${GENERATOR_MODEL}"
echo "Judge     : ${JUDGE_API_BASE} (${JUDGE_MODEL})"
echo "GPU       : ${GENERATOR_CUDA_VISIBLE_DEVICES}"
echo "============================================================"

if [[ "${SKIP_JUDGE_CHECK:-0}" != "1" ]]; then
  if ! curl -fsS --max-time 5 "${JUDGE_API_BASE}/models" >/dev/null; then
    echo "ERROR: judge server is not reachable at ${JUDGE_API_BASE}" >&2
    echo "Start the Qwen2.5-32B-Instruct judge server first, then rerun." >&2
    exit 1
  fi
fi

repair_one() {
  local BENCH="$1"
  local EXPECTED_ITEMS="$2"
  local IMAGE_ROOT="$3"

  local BENCH_DIR="${TTS_DATA_ROOT}/${BENCH}"
  local SEED="${BENCH_DIR}/seed_dataset.json"
  local FULL="${BENCH_DIR}/rollout_annotation_internvl3_8b_n16.json"

  local REPAIR_SEED="${BENCH_DIR}/repair_seed.json"
  local REPAIR_INDICES="${BENCH_DIR}/repair_indices.json"
  local REPAIR_OUTPUT="${BENCH_DIR}/repair_rollouts.json"

  echo
  echo "==================== ${BENCH} ===================="

  [[ -f "${SEED}" ]] || { echo "ERROR: missing ${SEED}" >&2; exit 1; }
  [[ -f "${FULL}" ]] || { echo "ERROR: missing ${FULL}" >&2; exit 1; }

  BENCH="${BENCH}" \
  EXPECTED_ITEMS="${EXPECTED_ITEMS}" \
  SEED="${SEED}" \
  FULL="${FULL}" \
  REPAIR_SEED="${REPAIR_SEED}" \
  REPAIR_INDICES="${REPAIR_INDICES}" \
  REPAIR_OUTPUT="${REPAIR_OUTPUT}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
  python - <<'PY'
import json
import os

bench = os.environ["BENCH"]
expected_items = int(os.environ["EXPECTED_ITEMS"])
seed_path = os.environ["SEED"]
full_path = os.environ["FULL"]
repair_seed_path = os.environ["REPAIR_SEED"]
repair_indices_path = os.environ["REPAIR_INDICES"]
repair_output_path = os.environ["REPAIR_OUTPUT"]
num_rollouts = int(os.environ["NUM_ROLLOUTS"])

with open(seed_path, "r", encoding="utf-8") as f:
    seed = json.load(f)
with open(full_path, "r", encoding="utf-8") as f:
    full = json.load(f)

if len(seed) != expected_items:
    raise SystemExit(
        f"ERROR: {bench} seed has {len(seed)} questions; expected {expected_items}."
    )
if len(full) != expected_items:
    raise SystemExit(
        f"ERROR: {bench} rollout file has {len(full)} questions; expected {expected_items}."
    )

bad = []
for i, item in enumerate(full):
    ns = len(item.get("solutions_splits", []))
    nl = len(item.get("labels", []))
    if ns != num_rollouts or nl != num_rollouts:
        bad.append(i)

print(f"{bench}: bad items = {len(bad)}")
if bad:
    print("indices:", bad)

old_indices = None
if os.path.exists(repair_indices_path):
    try:
        with open(repair_indices_path, "r", encoding="utf-8") as f:
            old_indices = json.load(f)
    except Exception:
        old_indices = None

if old_indices != bad and os.path.exists(repair_output_path):
    print("Bad-index set changed; removing stale repair_rollouts.json")
    os.remove(repair_output_path)

repair_seed = [seed[i] for i in bad]

with open(repair_seed_path, "w", encoding="utf-8") as f:
    json.dump(repair_seed, f, ensure_ascii=False, indent=2)
with open(repair_indices_path, "w", encoding="utf-8") as f:
    json.dump(bad, f, ensure_ascii=False, indent=2)
PY

  local BAD_COUNT
  BAD_COUNT="$(
    REPAIR_INDICES="${REPAIR_INDICES}" python - <<'PY'
import json, os
with open(os.environ["REPAIR_INDICES"], "r", encoding="utf-8") as f:
    print(len(json.load(f)))
PY
  )"

  if [[ "${BAD_COUNT}" == "0" ]]; then
    echo "[PASS] ${BENCH}: nothing to repair."
    rm -f "${REPAIR_SEED}" "${REPAIR_INDICES}" "${REPAIR_OUTPUT}"
    return 0
  fi

  echo "Repairing ${BAD_COUNT} item(s) in ${BENCH}..."

  CUDA_VISIBLE_DEVICES="${GENERATOR_CUDA_VISIBLE_DEVICES}" \
  python -m eval.build_eval_rollouts_annotation \
    --input "${REPAIR_SEED}" \
    --output "${REPAIR_OUTPUT}" \
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

  BENCH="${BENCH}" \
  FULL="${FULL}" \
  REPAIR_INDICES="${REPAIR_INDICES}" \
  REPAIR_OUTPUT="${REPAIR_OUTPUT}" \
  NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
  python - <<'PY'
import json
import os
import tempfile

bench = os.environ["BENCH"]
full_path = os.environ["FULL"]
indices_path = os.environ["REPAIR_INDICES"]
repair_path = os.environ["REPAIR_OUTPUT"]
num_rollouts = int(os.environ["NUM_ROLLOUTS"])

with open(full_path, "r", encoding="utf-8") as f:
    full = json.load(f)
with open(indices_path, "r", encoding="utf-8") as f:
    indices = json.load(f)
with open(repair_path, "r", encoding="utf-8") as f:
    repaired = json.load(f)

if len(repaired) != len(indices):
    raise SystemExit(
        f"ERROR: {bench}: repaired count {len(repaired)} != expected {len(indices)}."
    )

for j, (idx, item) in enumerate(zip(indices, repaired)):
    ns = len(item.get("solutions_splits", []))
    nl = len(item.get("labels", []))
    if ns != num_rollouts or nl != num_rollouts:
        raise SystemExit(
            f"ERROR: {bench}: repair #{j} for original index {idx} "
            f"is still incomplete: solutions_splits={ns}, labels={nl}."
        )
    full[idx] = item

directory = os.path.dirname(full_path)
fd, tmp_path = tempfile.mkstemp(prefix=".rollout_repair_", suffix=".json", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, full_path)
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

print(f"[PASS] {bench}: merged {len(indices)} repaired item(s).")
PY

  rm -f "${REPAIR_SEED}" "${REPAIR_INDICES}" "${REPAIR_OUTPUT}"
}

for spec in "${BENCHMARKS[@]}"; do
  IFS='|' read -r BENCH EXPECTED_ITEMS IMAGE_ROOT <<< "${spec}"
  repair_one "${BENCH}" "${EXPECTED_ITEMS}" "${IMAGE_ROOT}"
done

echo
echo "==================== Final validation ===================="

TTS_DATA_ROOT="${TTS_DATA_ROOT}" \
NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
python - <<'PY'
import json
import os
import sys

root = os.environ["TTS_DATA_ROOT"]
num_rollouts = int(os.environ["NUM_ROLLOUTS"])
expected = {
    "MathVision": 304,
    "MathVerse": 788,
    "MathVista": 1000,
    "MMStar": 1500,
}

all_ok = True

for name, n in expected.items():
    p = os.path.join(root, name, "rollout_annotation_internvl3_8b_n16.json")

    if not os.path.exists(p):
        print(f"{name:12s} MISSING")
        all_ok = False
        continue

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    bad = []
    for i, item in enumerate(data):
        ns = len(item.get("solutions_splits", []))
        nl = len(item.get("labels", []))
        if ns != num_rollouts or nl != num_rollouts:
            bad.append((i, ns, nl))

    ok = len(data) == n and not bad
    all_ok = all_ok and ok

    print(
        f"{name:12s} "
        f"questions={len(data):4d}/{n:<4d} "
        f"bad={len(bad):3d} "
        f"{'PASS' if ok else 'FAIL'}"
    )

    if bad:
        print("  first bad:", bad[:10])

if not all_ok:
    sys.exit(1)

print("\n[PASS] All final TTS rollout pools are complete.")
PY
