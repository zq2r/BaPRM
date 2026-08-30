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
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(
                    f"Expected object at {path}:{lineno}, "
                    f"got {type(obj).__name__}"
                )

            prefix_id = obj.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {path}:{lineno}"
                )
            if prefix_id in records:
                raise ValueError(
                    f"Duplicate prefix_id in MC labels: {prefix_id}"
                )
            records[prefix_id] = obj

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


def make_record(
    *,
    label: dict[str, Any],
    prefix_id: str,
    pred_prob: float,
    model_name: str,
    checkpoint: str,
    variant: str,
) -> dict[str, Any]:
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
            f"{prefix_id}: success_prob is not K/N"
        )

    return {
        "schema_version": 1,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "prediction_variant": variant,
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


def summarize(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    checkpoint: str,
    evaluator_output: Path,
    mc_labels: Path,
    output: Path,
    questions: int,
    variant: str,
) -> dict[str, Any]:
    mean_target = sum(
        record["target_success_prob"]
        for record in records
    ) / len(records)
    mean_prediction = sum(
        record["pred_prob"]
        for record in records
    ) / len(records)
    mean_absolute_error = sum(
        record["absolute_error"]
        for record in records
    ) / len(records)

    return {
        "schema_version": 1,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "prediction_variant": variant,
        "evaluator_output": str(evaluator_output),
        "mc_labels": str(mc_labels),
        "output": str(output),
        "counts": {
            "questions": questions,
            "prefixes": len(records),
        },
        "means": {
            "target_success_prob": mean_target,
            "pred_prob": mean_prediction,
            "signed_error": mean_prediction - mean_target,
            "absolute_error": mean_absolute_error,
        },
        "ranges": {
            "pred_prob": [
                min(record["pred_prob"] for record in records),
                max(record["pred_prob"] for record in records),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract terminal BayesianPRM reliability-only and final "
            "conservatism-aware probabilities for calibration evaluation."
        )
    )
    parser.add_argument(
        "--evaluator-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mc-labels",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-rel",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-final",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-name-rel",
        type=str,
        default="BayesianPRM-Rel",
    )
    parser.add_argument(
        "--model-name-final",
        type=str,
        default="BayesianPRM-Final",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--expected-prefixes",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    evaluator_path = args.evaluator_output.expanduser().resolve()
    labels_path = args.mc_labels.expanduser().resolve()
    output_rel = args.output_rel.expanduser().resolve()
    output_final = args.output_final.expanduser().resolve()

    summary_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else output_final.with_name(
            output_final.stem + ".summary.json"
        )
    )

    for path in (output_rel, output_final, summary_path):
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

    extracted_rel: dict[str, dict[str, Any]] = {}
    extracted_final: dict[str, dict[str, Any]] = {}

    for question_index, item in enumerate(evaluator_data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Evaluator item {question_index} is not an object"
            )

        prefix_ids = item.get("prefix_ids")
        solutions_splits = item.get("solutions_splits")
        prm_mu_rel = item.get("prm_mu_rel")
        prm_mu_final = item.get("prm_mu")

        for field_name, value in (
            ("prefix_ids", prefix_ids),
            ("solutions_splits", solutions_splits),
            ("prm_mu_rel", prm_mu_rel),
            ("prm_mu", prm_mu_final),
        ):
            if not isinstance(value, list):
                raise ValueError(
                    f"Evaluator item {question_index}: "
                    f"{field_name} is not a list"
                )

        lengths = {
            len(prefix_ids),
            len(solutions_splits),
            len(prm_mu_rel),
            len(prm_mu_final),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"Evaluator item {question_index}: "
                "prefix/result list lengths differ"
            )

        for local_index, prefix_id in enumerate(prefix_ids):
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Evaluator item {question_index}, index "
                    f"{local_index}: invalid prefix_id"
                )
            if prefix_id in extracted_final:
                raise ValueError(
                    f"Duplicate prefix_id in evaluator output: "
                    f"{prefix_id}"
                )
            if prefix_id not in labels_by_prefix:
                raise ValueError(
                    f"Evaluator prefix is absent from MC labels: "
                    f"{prefix_id}"
                )

            steps = solutions_splits[local_index]
            rel_steps = prm_mu_rel[local_index]
            final_steps = prm_mu_final[local_index]

            if not isinstance(steps, list) or not steps:
                raise ValueError(
                    f"{prefix_id}: solutions_splits entry is empty"
                )

            for field_name, values in (
                ("prm_mu_rel", rel_steps),
                ("prm_mu", final_steps),
            ):
                if not isinstance(values, list) or not values:
                    raise ValueError(
                        f"{prefix_id}: {field_name} entry is empty"
                    )
                if len(values) != len(steps):
                    raise ValueError(
                        f"{prefix_id}: {len(values)} {field_name} "
                        f"values for {len(steps)} reasoning steps"
                    )

            pred_rel = require_probability(
                rel_steps[-1],
                field="terminal prm_mu_rel",
                prefix_id=prefix_id,
            )
            pred_final = require_probability(
                final_steps[-1],
                field="terminal prm_mu",
                prefix_id=prefix_id,
            )

            label = labels_by_prefix[prefix_id]

            if len(steps) != int(label["prefix_step_count"]):
                raise ValueError(
                    f"{prefix_id}: evaluator has {len(steps)} steps, "
                    f"MC label expects {label['prefix_step_count']}"
                )

            extracted_rel[prefix_id] = make_record(
                label=label,
                prefix_id=prefix_id,
                pred_prob=pred_rel,
                model_name=args.model_name_rel,
                checkpoint=args.checkpoint,
                variant="reliability",
            )
            extracted_final[prefix_id] = make_record(
                label=label,
                prefix_id=prefix_id,
                pred_prob=pred_final,
                model_name=args.model_name_final,
                checkpoint=args.checkpoint,
                variant="final",
            )

    expected_ids = set(labels_by_prefix)
    actual_ids = set(extracted_final)

    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)

    if missing or extra:
        raise ValueError(
            "Evaluator/MC-label prefix mismatch: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing examples={missing[:5]}, "
            f"extra examples={extra[:5]}"
        )

    if set(extracted_rel) != actual_ids:
        raise ValueError(
            "Reliability/final prefix sets do not match"
        )

    if len(actual_ids) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} predictions, "
            f"got {len(actual_ids)}"
        )

    # Preserve exactly the MC-label ordering so both variants can be
    # compared by compute_calibration_metrics.py in one invocation.
    ordered_rel = [
        extracted_rel[prefix_id]
        for prefix_id in labels_by_prefix
    ]
    ordered_final = [
        extracted_final[prefix_id]
        for prefix_id in labels_by_prefix
    ]

    atomic_write_jsonl(ordered_rel, output_rel)
    atomic_write_jsonl(ordered_final, output_final)

    rel_summary = summarize(
        ordered_rel,
        model_name=args.model_name_rel,
        checkpoint=args.checkpoint,
        evaluator_output=evaluator_path,
        mc_labels=labels_path,
        output=output_rel,
        questions=len(evaluator_data),
        variant="reliability",
    )
    final_summary = summarize(
        ordered_final,
        model_name=args.model_name_final,
        checkpoint=args.checkpoint,
        evaluator_output=evaluator_path,
        mc_labels=labels_path,
        output=output_final,
        questions=len(evaluator_data),
        variant="final",
    )

    combined_summary = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "expected_prefixes": args.expected_prefixes,
        "reliability": rel_summary,
        "final": final_summary,
    }
    atomic_write_json(combined_summary, summary_path)

    print("=== BayesianPRM calibration predictions ===")
    print(f"Questions: {len(evaluator_data)}")
    print(f"Prefixes: {len(ordered_final)}")
    print(
        "Reliability mean prediction: "
        f"{rel_summary['means']['pred_prob']:.6f}"
    )
    print(
        "Final mean prediction:       "
        f"{final_summary['means']['pred_prob']:.6f}"
    )
    print(
        "Mean target:                 "
        f"{final_summary['means']['target_success_prob']:.6f}"
    )
    print(f"Reliability output: {output_rel}")
    print(f"Final output:       {output_final}")
    print(f"Summary:            {summary_path}")
    print(
        "\n[PASS] BayesianPRM reliability/final predictions "
        "extracted and aligned."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise
