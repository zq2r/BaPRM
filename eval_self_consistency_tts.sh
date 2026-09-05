#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

TTS_DATA_ROOT=${TTS_DATA_ROOT:-"${REPO_ROOT}/datasets/TTS_FINAL"}
benchs=${benchs:-"MathVision MathVerse MathVista MMStar"}
N_GRID=${N_GRID:-"1,2,4,8,16"}
TTS_REPEATS=${TTS_REPEATS:-20}
TTS_SEED=${TTS_SEED:-42}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/outputs/tts/self_consistency"}

resolve_benchmark() {
  case "$1" in
    MathVision) EXPECTED_ITEMS=304 ;;
    MathVerse)  EXPECTED_ITEMS=788 ;;
    MathVista)  EXPECTED_ITEMS=1000 ;;
    MMStar)     EXPECTED_ITEMS=1500 ;;
    *) echo "ERROR: unknown benchmark: $1" >&2; exit 1 ;;
  esac
  ANNOTATION_THIS="${TTS_DATA_ROOT}/$1/rollout_annotation_internvl3_8b_n16.json"
  OUTPUT_THIS="${OUT_ROOT}/$1/tts_results.json"
}

validate_rollout() {
  ANNOTATION_PATH="$1" EXPECTED_ITEMS="$2" python - <<'PY'
import json, os, sys
p = os.environ['ANNOTATION_PATH']
expected = int(os.environ['EXPECTED_ITEMS'])
with open(p, 'r', encoding='utf-8') as f:
    x = json.load(f)
if len(x) != expected:
    raise SystemExit(f'ERROR: {p}: {len(x)} items, expected {expected}')
bad = [(i, len(a.get('solutions_splits', [])), len(a.get('labels', []))) for i, a in enumerate(x)
       if len(a.get('solutions_splits', [])) != 16 or len(a.get('labels', [])) != 16]
if bad:
    print('ERROR: incomplete rollout pool:', p)
    print('first bad:', bad[:10])
    sys.exit(1)
print(f'[rollout check] PASS: {len(x)}/{expected}, all have 16 candidates.')
PY
}

for BENCH in ${benchs}; do
  resolve_benchmark "${BENCH}"
  validate_rollout "${ANNOTATION_THIS}" "${EXPECTED_ITEMS}"
  mkdir -p "$(dirname "${OUTPUT_THIS}")"

  python eval/tts/evaluate_self_consistency_tts.py \
    --input-json "${ANNOTATION_THIS}" \
    --output-json "${OUTPUT_THIS}" \
    --n-grid "${N_GRID}" \
    --repeats "${TTS_REPEATS}" \
    --seed "${TTS_SEED}"
done

OUT_ROOT="${OUT_ROOT}" benchs="${benchs}" python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ['OUT_ROOT'])
requested = os.environ['benchs'].split()
canonical = ['MathVision', 'MathVerse', 'MathVista', 'MMStar']
benches = [b for b in canonical if b in requested] + [b for b in requested if b not in canonical]
data = {}
for b in benches:
    with open(root / b / 'tts_results.json', 'r', encoding='utf-8') as f:
        data[b] = json.load(f)

ns = sorted({int(n) for d in data.values() for n in d['results']})
w0, w = 10, 23
total = w0 + w * len(benches)

for metric_title, mean_key, std_key in [
    ('SELF-CONSISTENCY — Accuracy (%)', 'accuracy_mean', 'accuracy_std'),
    ('SELF-CONSISTENCY — Oracle Pass (%)', 'oracle_pass_mean', 'oracle_pass_std'),
]:
    print('\n' + '=' * total)
    print(metric_title)
    print('-' * total)
    print(f"{'Setting':<{w0}}", end='')
    for b in benches:
        print(f'{b:^{w}}', end='')
    print('\n' + '-' * total)
    for n in ns:
        print(f"{('N='+str(n)):<{w0}}", end='')
        for b in benches:
            r = data[b]['results'][str(n)]
            text = f"{100*r[mean_key]:.2f} ± {100*r[std_key]:.2f}"
            print(f'{text:^{w}}', end='')
        print()
    print('=' * total)

print('\nTie rate (%):')
for n in ns:
    vals = [f"{b}={100*data[b]['results'][str(n)]['tie_rate_mean']:.2f}" for b in benches]
    print(f'N={n:<2d}: ' + ', '.join(vals))
print(f'\nResults saved under: {root}')
PY

echo "Self-consistency TTS evaluation finished."
