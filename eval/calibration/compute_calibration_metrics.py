#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")

    records: list[dict[str, Any]] = []
    seen_prefix_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{lineno}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object at {path}:{lineno}, "
                    f"got {type(record).__name__}"
                )

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {path}:{lineno}"
                )
            if prefix_id in seen_prefix_ids:
                raise ValueError(
                    f"Duplicate prefix_id in {path}: {prefix_id}"
                )

            seen_prefix_ids.add(prefix_id)
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def require_probability(
    value: Any,
    *,
    field: str,
    prefix_id: str,
) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{prefix_id}: {field} must be numeric, got {value!r}"
        )

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            f"{prefix_id}: {field} is not finite: {value}"
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{prefix_id}: {field} is outside [0, 1]: {value}"
        )
    return value


def fixed_width_bins(
    predictions: list[float],
    targets: list[float],
    num_bins: int,
) -> list[dict[str, Any]]:
    bins: list[list[int]] = [[] for _ in range(num_bins)]

    for index, prediction in enumerate(predictions):
        bin_index = min(int(prediction * num_bins), num_bins - 1)
        bins[bin_index].append(index)

    output: list[dict[str, Any]] = []

    for bin_index, indices in enumerate(bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins

        if indices:
            mean_prediction = sum(predictions[i] for i in indices) / len(indices)
            mean_target = sum(targets[i] for i in indices) / len(indices)
            gap = abs(mean_prediction - mean_target)
        else:
            mean_prediction = None
            mean_target = None
            gap = None

        output.append(
            {
                "bin_index": bin_index,
                "lower": lower,
                "upper": upper,
                "right_closed": bin_index == num_bins - 1,
                "count": len(indices),
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
                "absolute_gap": gap,
            }
        )

    return output


def adaptive_equal_count_bins(
    predictions: list[float],
    targets: list[float],
    num_bins: int,
) -> list[dict[str, Any]]:
    n = len(predictions)
    order = sorted(
        range(n),
        key=lambda i: (predictions[i], i),
    )

    base_size, remainder = divmod(n, num_bins)
    output: list[dict[str, Any]] = []
    cursor = 0

    for bin_index in range(num_bins):
        size = base_size + (1 if bin_index < remainder else 0)
        indices = order[cursor : cursor + size]
        cursor += size

        if not indices:
            output.append(
                {
                    "bin_index": bin_index,
                    "count": 0,
                    "min_prediction": None,
                    "max_prediction": None,
                    "mean_prediction": None,
                    "mean_target": None,
                    "absolute_gap": None,
                }
            )
            continue

        values = [predictions[i] for i in indices]
        target_values = [targets[i] for i in indices]
        mean_prediction = sum(values) / len(values)
        mean_target = sum(target_values) / len(target_values)

        output.append(
            {
                "bin_index": bin_index,
                "count": len(indices),
                "min_prediction": min(values),
                "max_prediction": max(values),
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
                "absolute_gap": abs(mean_prediction - mean_target),
            }
        )

    if cursor != n:
        raise AssertionError(
            f"Adaptive bins consumed {cursor} records, expected {n}"
        )

    return output


def weighted_calibration_error(
    bins: Iterable[dict[str, Any]],
    total_count: int,
) -> float:
    error = 0.0

    for item in bins:
        count = int(item["count"])
        gap = item["absolute_gap"]

        if count == 0:
            continue
        if not isinstance(gap, (int, float)):
            raise ValueError("Non-empty bin has no calibration gap")

        error += (count / total_count) * float(gap)

    return error


def equal_bin_calibration_error(
    bins: Iterable[dict[str, Any]],
) -> float:
    gaps = [
        float(item["absolute_gap"])
        for item in bins
        if int(item["count"]) > 0
    ]

    if not gaps:
        raise ValueError("No non-empty bins")

    return sum(gaps) / len(gaps)


def compute_metrics(
    predictions: list[float],
    targets: list[float],
    num_bins: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target lengths do not match")
    if not predictions:
        raise ValueError("No prediction-target pairs")

    n = len(predictions)
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets)
    ]

    # KWDK metrics:
    # Brier       = mean((prediction - target)^2)
    # Pos. Brier  = mean(max(prediction - target, 0)^2)
    brier = sum(error * error for error in errors) / n
    positive_brier = (
        sum(max(error, 0.0) ** 2 for error in errors) / n
    )

    fixed_bins = fixed_width_bins(
        predictions,
        targets,
        num_bins,
    )
    adaptive_bins = adaptive_equal_count_bins(
        predictions,
        targets,
        num_bins,
    )

    ece = weighted_calibration_error(fixed_bins, n)
    adaptive_ce = weighted_calibration_error(adaptive_bins, n)
    average_ce = equal_bin_calibration_error(fixed_bins)

    metrics = {
        "brier": brier,
        "positive_brier": positive_brier,
        "adaptive_ce": adaptive_ce,
        "ece": ece,
        "average_ce": average_ce,
    }

    diagnostics = {
        "fixed_width_bins": fixed_bins,
        "adaptive_equal_count_bins": adaptive_bins,
    }

    return metrics, diagnostics


def atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def atomic_write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    fieldnames = [
        "model_name",
        "input_file",
        "num_prefixes",
        "num_bins",
        "brier",
        "positive_brier",
        "adaptive_ce",
        "ece",
        "average_ce",
    ]

    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tmp.replace(path)


def infer_model_name(
    records: list[dict[str, Any]],
    input_path: Path,
) -> str:
    model_names = {
        str(record["model_name"]).strip()
        for record in records
        if isinstance(record.get("model_name"), str)
        and record["model_name"].strip()
    }

    if len(model_names) == 1:
        return next(iter(model_names))
    if len(model_names) > 1:
        raise ValueError(
            f"{input_path}: multiple model_name values: "
            f"{sorted(model_names)}"
        )

    return input_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the five calibration metrics reported by KWDK: "
            "Brier, Positive Brier, AdaptiveCE, ECE, and AverageCE."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more aligned prediction JSONL files containing "
            "prefix_id, pred_prob, and target_success_prob."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--expected-prefixes",
        type=int,
        default=3513,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_bins <= 1:
        raise ValueError("--num-bins must be greater than 1")
    if args.expected_prefixes <= 0:
        raise ValueError("--expected-prefixes must be positive")

    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    for path in (output_json, output_csv):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Use --overwrite to replace it."
            )

    result_rows: list[dict[str, Any]] = []
    detailed_models: list[dict[str, Any]] = []
    reference_prefix_ids: list[str] | None = None

    for raw_input_path in args.input:
        input_path = raw_input_path.expanduser().resolve()
        records = load_jsonl(input_path)

        if len(records) != args.expected_prefixes:
            raise ValueError(
                f"{input_path}: expected {args.expected_prefixes} "
                f"records, got {len(records)}"
            )

        prefix_ids = [record["prefix_id"] for record in records]

        if reference_prefix_ids is None:
            reference_prefix_ids = prefix_ids
        elif prefix_ids != reference_prefix_ids:
            raise ValueError(
                f"{input_path}: prefix order/set does not exactly match "
                "the first input file"
            )

        predictions: list[float] = []
        targets: list[float] = []

        for record in records:
            prefix_id = record["prefix_id"]
            predictions.append(
                require_probability(
                    record.get("pred_prob"),
                    field="pred_prob",
                    prefix_id=prefix_id,
                )
            )
            targets.append(
                require_probability(
                    record.get("target_success_prob"),
                    field="target_success_prob",
                    prefix_id=prefix_id,
                )
            )

        model_name = infer_model_name(records, input_path)
        metrics, diagnostics = compute_metrics(
            predictions,
            targets,
            args.num_bins,
        )

        row = {
            "model_name": model_name,
            "input_file": str(input_path),
            "num_prefixes": len(records),
            "num_bins": args.num_bins,
            **metrics,
        }
        result_rows.append(row)

        detailed_models.append(
            {
                **row,
                "metric_definitions": {
                    "brier": "mean((pred_prob - target_success_prob)^2)",
                    "positive_brier": (
                        "mean(max(pred_prob - "
                        "target_success_prob, 0)^2)"
                    ),
                    "ece": (
                        "fixed-width bins; sample-count weighted "
                        "absolute bin gap"
                    ),
                    "adaptive_ce": (
                        "equal-count bins; sample-count weighted "
                        "absolute bin gap"
                    ),
                    "average_ce": (
                        "fixed-width bins; equal weight over "
                        "non-empty bins"
                    ),
                },
                "bin_diagnostics": diagnostics,
            }
        )

    output_data = {
        "schema_version": 1,
        "num_bins": args.num_bins,
        "expected_prefixes": args.expected_prefixes,
        "models": detailed_models,
    }

    atomic_write_json(output_data, output_json)
    atomic_write_csv(result_rows, output_csv)

    print("=== Calibration metrics ===")
    header = (
        f"{'Model':32s} "
        f"{'Brier':>10s} "
        f"{'PosBrier':>10s} "
        f"{'AdaptiveCE':>10s} "
        f"{'ECE':>10s} "
        f"{'AverageCE':>10s}"
    )
    print(header)
    print("-" * len(header))

    for row in result_rows:
        print(
            f"{row['model_name'][:32]:32s} "
            f"{row['brier']:10.6f} "
            f"{row['positive_brier']:10.6f} "
            f"{row['adaptive_ce']:10.6f} "
            f"{row['ece']:10.6f} "
            f"{row['average_ce']:10.6f}"
        )

    print(f"\nBins: {args.num_bins}")
    print(f"JSON: {output_json}")
    print(f"CSV:  {output_csv}")
    print("\n[PASS] Calibration metrics computed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise