#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_by_prefix(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    records: dict[str, dict[str, Any]] = {}

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
                    f"Expected object at {path}:{lineno}, "
                    f"got {type(record).__name__}"
                )

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {path}:{lineno}"
                )
            if prefix_id in records:
                raise ValueError(
                    f"Duplicate prefix_id in MC labels: {prefix_id}"
                )

            records[prefix_id] = record

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
            f"{prefix_id}: {field} is outside [0,1]: {value}"
        )

    return value


def atomic_write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp.replace(path)


def atomic_write_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one terminal Normal PRM probability per calibration "
            "prefix and align it with Monte Carlo success labels."
        )
    )
    parser.add_argument("--evaluator-output", type=Path, required=True)
    parser.add_argument("--mc-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default="NormalPRM-InternVL3-8B",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--expected-prefixes", type=int, default=3513)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evaluator_path = args.evaluator_output.expanduser().resolve()
    labels_path = args.mc_labels.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else output_path.with_name(output_path.stem + ".summary.json")
    )

    for path in (output_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Use --overwrite to replace it."
            )

    evaluator_data = load_json(evaluator_path)
    if not isinstance(evaluator_data, list):
        raise TypeError(
            "Evaluator output must be a top-level JSON list"
        )

    labels_by_prefix = load_jsonl_by_prefix(labels_path)

    if len(labels_by_prefix) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} MC labels, "
            f"got {len(labels_by_prefix)}"
        )

    extracted: dict[str, dict[str, Any]] = {}

    for question_index, item in enumerate(evaluator_data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Evaluator item {question_index} is not an object"
            )

        prefix_ids = item.get("prefix_ids")
        solutions_splits = item.get("solutions_splits")
        prm_mu = item.get("prm_mu")

        for field_name, value in (
            ("prefix_ids", prefix_ids),
            ("solutions_splits", solutions_splits),
            ("prm_mu", prm_mu),
        ):
            if not isinstance(value, list):
                raise ValueError(
                    f"Evaluator item {question_index}: "
                    f"{field_name} is not a list"
                )

        lengths = {
            len(prefix_ids),
            len(solutions_splits),
            len(prm_mu),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"Evaluator item {question_index}: list lengths differ: "
                f"prefix_ids={len(prefix_ids)}, "
                f"solutions_splits={len(solutions_splits)}, "
                f"prm_mu={len(prm_mu)}"
            )

        for local_index, prefix_id in enumerate(prefix_ids):
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Evaluator item {question_index}, index "
                    f"{local_index}: invalid prefix_id"
                )
            if prefix_id in extracted:
                raise ValueError(
                    f"Duplicate prefix_id in evaluator output: {prefix_id}"
                )
            if prefix_id not in labels_by_prefix:
                raise ValueError(
                    f"Evaluator prefix is absent from MC labels: "
                    f"{prefix_id}"
                )

            steps = solutions_splits[local_index]
            mu_steps = prm_mu[local_index]

            if not isinstance(steps, list) or not steps:
                raise ValueError(
                    f"{prefix_id}: solutions_splits entry is empty"
                )
            if not isinstance(mu_steps, list) or not mu_steps:
                raise ValueError(
                    f"{prefix_id}: prm_mu entry is empty"
                )
            if len(mu_steps) != len(steps):
                raise ValueError(
                    f"{prefix_id}: {len(mu_steps)} PRM values for "
                    f"{len(steps)} reasoning steps"
                )

            pred_prob = require_probability(
                mu_steps[-1],
                field="terminal prm_mu",
                prefix_id=prefix_id,
            )

            label = labels_by_prefix[prefix_id]
            target_prob = require_probability(
                label.get("success_prob"),
                field="success_prob",
                prefix_id=prefix_id,
            )
            mc_correct = int(label["mc_correct"])
            mc_total = int(label["mc_total"])

            if mc_total <= 0 or not 0 <= mc_correct <= mc_total:
                raise ValueError(
                    f"{prefix_id}: invalid MC counts "
                    f"K={mc_correct}, N={mc_total}"
                )
            if not math.isclose(
                target_prob,
                mc_correct / mc_total,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{prefix_id}: target probability is not K/N"
                )
            if len(steps) != int(label["prefix_step_count"]):
                raise ValueError(
                    f"{prefix_id}: evaluator has {len(steps)} steps, "
                    f"MC label expects {label['prefix_step_count']}"
                )

            extracted[prefix_id] = {
                "schema_version": 1,
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "prefix_id": prefix_id,
                "question_id": label["question_id"],
                "source_trajectory_index": label[
                    "source_trajectory_index"
                ],
                "prefix_step_count": label["prefix_step_count"],
                "prefix_relative_position": label[
                    "prefix_relative_position"
                ],
                "mc_correct": mc_correct,
                "mc_total": mc_total,
                "target_success_prob": target_prob,
                "pred_prob": pred_prob,
                "pred_mu": pred_prob,
                "signed_error": pred_prob - target_prob,
                "absolute_error": abs(pred_prob - target_prob),
            }

    expected_ids = set(labels_by_prefix)
    actual_ids = set(extracted)

    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)

    if missing or extra:
        raise ValueError(
            "Evaluator/MC-label prefix mismatch: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing examples={missing[:5]}, "
            f"extra examples={extra[:5]}"
        )

    if len(extracted) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} predictions, "
            f"got {len(extracted)}"
        )

    # Preserve exactly the MC-label ordering so multiple model files are
    # directly comparable by the shared metric script.
    ordered_records = [
        extracted[prefix_id]
        for prefix_id in labels_by_prefix
    ]

    mean_target = sum(
        record["target_success_prob"]
        for record in ordered_records
    ) / len(ordered_records)
    mean_prediction = sum(
        record["pred_prob"]
        for record in ordered_records
    ) / len(ordered_records)
    mean_absolute_error = sum(
        record["absolute_error"]
        for record in ordered_records
    ) / len(ordered_records)

    atomic_write_jsonl(ordered_records, output_path)

    summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "evaluator_output": str(evaluator_path),
        "mc_labels": str(labels_path),
        "output": str(output_path),
        "counts": {
            "questions": len(evaluator_data),
            "prefixes": len(ordered_records),
        },
        "means": {
            "target_success_prob": mean_target,
            "pred_prob": mean_prediction,
            "signed_error": mean_prediction - mean_target,
            "absolute_error": mean_absolute_error,
        },
        "ranges": {
            "pred_prob": [
                min(record["pred_prob"] for record in ordered_records),
                max(record["pred_prob"] for record in ordered_records),
            ],
        },
    }

    atomic_write_json(summary, summary_path)

    print("=== Normal PRM calibration predictions ===")
    print(f"Questions: {len(evaluator_data)}")
    print(f"Prefixes: {len(ordered_records)}")
    print(f"Mean target: {mean_target:.6f}")
    print(f"Mean prediction: {mean_prediction:.6f}")
    print(
        "Mean signed error: "
        f"{mean_prediction - mean_target:+.6f}"
    )
    print(f"Mean absolute error: {mean_absolute_error:.6f}")
    print(
        "Prediction range: "
        f"[{summary['ranges']['pred_prob'][0]:.6f}, "
        f"{summary['ranges']['pred_prob'][1]:.6f}]"
    )
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print("\n[PASS] Normal PRM predictions extracted and aligned.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise
