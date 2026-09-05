from pathlib import Path

p = Path("eval_tts.sh")
s = p.read_text(encoding="utf-8")

start_marker = "# ============================================================\n# Final summary\n# ============================================================"
end_marker = '\nrm -f "${MANIFEST}"'

start = s.find(start_marker)
if start < 0:
    raise SystemExit("[FAIL] Final summary block start not found")

end = s.find(end_marker, start)
if end < 0:
    raise SystemExit("[FAIL] Final summary block end not found")

new_block = r'''# ============================================================
# Final summary
# ============================================================
# No timestamp. Different mode/benchmark combinations get different summary folders.
SUMMARY_TAG="$(printf '%s__%s' "${PRM_MODES}" "${benchs}" | tr '/ ,:' '_____')"
export SUMMARY_TAG

python - <<'PY'
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

manifest = os.environ["MANIFEST"]
summary_tag = os.environ["SUMMARY_TAG"]

CANONICAL_BENCHES = [
    "MathVision",
    "MathVerse",
    "MathVista",
    "MMStar",
]

data_by_mode = defaultdict(dict)
rows = []


def best_beta2(result):
    value = result.get("best_beta2_first_repeat")
    if value is not None:
        return float(value)

    repeat_results = result.get("repeat_results", [])
    if repeat_results:
        best = repeat_results[0].get("best", {})
        value = best.get("beta2")
        if value is not None:
            return float(value)

    return None


with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        mode, bench, path = line.rstrip("\n").split("\t")

        with open(path, "r", encoding="utf-8") as g:
            data = json.load(g)

        data_by_mode[mode][bench] = data

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
                "best_beta2": (
                    best_beta2(result)
                    if mode == "bayesian" and n_int > 1
                    else None
                ),
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
                "best_beta2": (
                    best_beta2(result)
                    if mode == "bayesian"
                    else None
                ),
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "oracle_pass_mean": result["oracle_pass_mean"],
                "oracle_pass_std": result["oracle_pass_std"],
            })


def row_order(r):
    if r["N"] == "IAS":
        n_key = 10**9
    else:
        n_key = int(r["N"])

    return (
        r["mode"],
        r["benchmark"],
        n_key,
    )


rows.sort(key=row_order)

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
            "best_beta2",
            "accuracy_mean",
            "accuracy_std",
            "oracle_pass_mean",
            "oracle_pass_std",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


def fmt_acc(result, mean_key, std_key):
    return (
        f"{100 * result[mean_key]:.2f}"
        f" ± {100 * result[std_key]:.2f}"
    )


def print_matrix(title, benches, row_labels, value_fn):
    first_width = 10
    col_width = 23
    total_width = first_width + col_width * len(benches)

    print()
    print("=" * total_width)
    print(title)
    print("-" * total_width)

    print(f"{'Setting':<{first_width}s}", end="")
    for bench in benches:
        print(f"{bench:^{col_width}s}", end="")
    print()

    print("-" * total_width)

    for label in row_labels:
        print(f"{label:<{first_width}s}", end="")

        for bench in benches:
            value = value_fn(bench, label)
            print(f"{value:^{col_width}s}", end="")

        print()

    print("=" * total_width)


for mode, bench_data in data_by_mode.items():
    benches = [
        b for b in CANONICAL_BENCHES
        if b in bench_data
    ]

    benches += [
        b for b in bench_data
        if b not in benches
    ]

    n_values = sorted({
        int(n)
        for data in bench_data.values()
        for n in data["results"].keys()
    })

    accuracy_rows = [f"N={n}" for n in n_values]

    if any("ias" in data for data in bench_data.values()):
        accuracy_rows.append("IAS")

    def accuracy_value(bench, label):
        data = bench_data[bench]

        if label == "IAS":
            result = data.get("ias")
        else:
            n = label.split("=", 1)[1]
            result = data["results"].get(n)

        if result is None:
            return "-"

        return fmt_acc(
            result,
            "accuracy_mean",
            "accuracy_std",
        )

    print_matrix(
        f"{mode.upper()} — Accuracy (%)",
        benches,
        accuracy_rows,
        accuracy_value,
    )

    def oracle_value(bench, label):
        data = bench_data[bench]

        if label == "IAS":
            result = data.get("ias")
        else:
            n = label.split("=", 1)[1]
            result = data["results"].get(n)

        if result is None:
            return "-"

        return fmt_acc(
            result,
            "oracle_pass_mean",
            "oracle_pass_std",
        )

    print_matrix(
        f"{mode.upper()} — Oracle Pass (%)",
        benches,
        accuracy_rows,
        oracle_value,
    )

    if any("ias" in data for data in bench_data.values()):
        def ias_budget_value(bench, label):
            result = bench_data[bench].get("ias")
            if result is None:
                return "-"

            if label == "Avg.N":
                return f"{result['average_n']:.3f}"

            if label == "Budget":
                return f"{100 * result['budget_ratio']:.2f}%"

            raise ValueError(label)

        print_matrix(
            f"{mode.upper()} — IAS Budget",
            benches,
            ["Avg.N", "Budget"],
            ias_budget_value,
        )

    if mode == "bayesian":
        beta_rows = [
            f"N={n}"
            for n in n_values
            if n > 1
        ]

        if any("ias" in data for data in bench_data.values()):
            beta_rows.append("IAS")

        def beta2_value(bench, label):
            data = bench_data[bench]

            if label == "IAS":
                result = data.get("ias")
            else:
                n = label.split("=", 1)[1]
                result = data["results"].get(n)

            if result is None:
                return "-"

            value = best_beta2(result)
            if value is None:
                return "-"

            return f"{value:.4g}"

        print_matrix(
            f"{mode.upper()} — Best beta2 (first repeat)",
            benches,
            beta_rows,
            beta2_value,
        )


print()
print(f"JSON: {json_path}")
print(f"CSV : {csv_path}")
PY'''

backup = Path("eval_tts.sh.bak_before_matrix_summary")
if not backup.exists():
    backup.write_text(s, encoding="utf-8")
    print(f"[PASS] backup -> {backup}")

s = s[:start] + new_block + s[end:]
p.write_text(s, encoding="utf-8")

print("[PASS] replaced Final summary with matrix summary")
