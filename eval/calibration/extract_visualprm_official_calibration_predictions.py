#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract terminal-step predictions from "
            "official VisualPRM-8B-v1_1 evaluator output "
            "and align them with MathVision MC labels."
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
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="VisualPRM-8B-v1_1",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--expected-prefixes",
        type=int,
        default=3513,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl_by_prefix(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    output = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for lineno, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            prefix_id = record.get(
                "prefix_id"
            )

            if (
                not isinstance(
                    prefix_id,
                    str,
                )
                or not prefix_id
            ):
                raise ValueError(
                    f"Missing prefix_id "
                    f"at line {lineno}"
                )

            if prefix_id in output:
                raise ValueError(
                    "Duplicate prefix_id: "
                    f"{prefix_id}"
                )

            output[prefix_id] = (
                record
            )

    return output


def require_probability(
    value: Any,
    *,
    name: str,
    prefix_id: str,
):
    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{prefix_id}: "
            f"{name} is not numeric."
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{prefix_id}: "
            f"{name} is not finite."
        )

    if not (
        0.0 <= value <= 1.0
    ):
        raise ValueError(
            f"{prefix_id}: "
            f"{name}={value} "
            "outside [0,1]."
        )

    return value


def atomic_write_jsonl(
    records,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(path)


def atomic_write_json(
    data,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp.replace(path)


def main():
    args = parse_args()

    evaluator_path = (
        args.evaluator_output
        .expanduser()
        .resolve()
    )

    labels_path = (
        args.mc_labels
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    summary_path = (
        args.summary_output
        .expanduser()
        .resolve()
    )

    for path in (
        output_path,
        summary_path,
    ):
        if (
            path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                f"Output exists: {path}"
            )

    evaluator_data = load_json(
        evaluator_path
    )

    if not isinstance(
        evaluator_data,
        list,
    ):
        raise TypeError(
            "Evaluator output must "
            "be a JSON list."
        )

    labels = load_jsonl_by_prefix(
        labels_path
    )

    if (
        len(labels)
        != args.expected_prefixes
    ):
        raise ValueError(
            f"Expected "
            f"{args.expected_prefixes} "
            f"MC labels, got "
            f"{len(labels)}."
        )

    extracted = {}

    for question_index, item in enumerate(
        evaluator_data
    ):
        prefix_ids = item.get(
            "prefix_ids"
        )

        solutions_splits = item.get(
            "solutions_splits"
        )

        scores = item.get(
            "visualprm_scores"
        )

        if not isinstance(
            prefix_ids,
            list,
        ):
            raise ValueError(
                f"Item {question_index}: "
                "missing prefix_ids."
            )

        if not isinstance(
            solutions_splits,
            list,
        ):
            raise ValueError(
                f"Item {question_index}: "
                "missing solutions_splits."
            )

        if not isinstance(
            scores,
            list,
        ):
            raise ValueError(
                f"Item {question_index}: "
                "missing visualprm_scores."
            )

        if not (
            len(prefix_ids)
            == len(solutions_splits)
            == len(scores)
        ):
            raise ValueError(
                f"Item {question_index}: "
                "per-prefix lengths differ."
            )

        for local_index, prefix_id in enumerate(
            prefix_ids
        ):
            if prefix_id in extracted:
                raise ValueError(
                    "Duplicate evaluator "
                    f"prefix_id: {prefix_id}"
                )

            if prefix_id not in labels:
                raise ValueError(
                    "Evaluator prefix absent "
                    f"from MC labels: "
                    f"{prefix_id}"
                )

            steps = (
                solutions_splits[
                    local_index
                ]
            )

            step_scores = (
                scores[
                    local_index
                ]
            )

            if (
                not isinstance(
                    steps,
                    list,
                )
                or not steps
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    "empty steps."
                )

            if (
                not isinstance(
                    step_scores,
                    list,
                )
                or not step_scores
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    "empty scores."
                )

            if (
                len(steps)
                != len(step_scores)
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    f"{len(steps)} steps "
                    f"but "
                    f"{len(step_scores)} scores."
                )

            pred_prob = (
                require_probability(
                    step_scores[-1],
                    name=(
                        "terminal "
                        "VisualPRM score"
                    ),
                    prefix_id=prefix_id,
                )
            )

            label = labels[
                prefix_id
            ]

            target_prob = (
                require_probability(
                    label.get(
                        "success_prob"
                    ),
                    name="success_prob",
                    prefix_id=prefix_id,
                )
            )

            mc_correct = int(
                label["mc_correct"]
            )

            mc_total = int(
                label["mc_total"]
            )

            if (
                mc_total <= 0
                or not (
                    0
                    <= mc_correct
                    <= mc_total
                )
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    "invalid MC counts "
                    f"{mc_correct}/"
                    f"{mc_total}."
                )

            expected_target = (
                mc_correct / mc_total
            )

            if not math.isclose(
                target_prob,
                expected_target,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    "success_prob != K/N."
                )

            expected_steps = (
                int(
                    label[
                        "prefix_step_count"
                    ]
                )
            )

            if (
                len(steps)
                != expected_steps
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    f"evaluator has "
                    f"{len(steps)} steps; "
                    f"MC label expects "
                    f"{expected_steps}."
                )

            error = (
                pred_prob
                - target_prob
            )

            extracted[prefix_id] = {
                "schema_version": 1,
                "model_name":
                    args.model_name,
                "checkpoint":
                    args.checkpoint,
                "prefix_id":
                    prefix_id,
                "question_id":
                    label.get(
                        "question_id"
                    ),
                "source_trajectory_index":
                    label.get(
                        "source_trajectory_index"
                    ),
                "prefix_step_count":
                    expected_steps,
                "prefix_relative_position":
                    label.get(
                        "prefix_relative_position"
                    ),
                "mc_correct":
                    mc_correct,
                "mc_total":
                    mc_total,
                "target_success_prob":
                    target_prob,
                "pred_prob":
                    pred_prob,
                "pred_visualprm_score":
                    pred_prob,
                "signed_error":
                    error,
                "absolute_error":
                    abs(error),
            }

    expected_ids = set(
        labels
    )

    actual_ids = set(
        extracted
    )

    missing = sorted(
        expected_ids
        - actual_ids
    )

    extra = sorted(
        actual_ids
        - expected_ids
    )

    if missing or extra:
        raise ValueError(
            "Evaluator/label prefix "
            "mismatch: "
            f"missing={len(missing)}, "
            f"extra={len(extra)}, "
            f"missing_examples="
            f"{missing[:5]}, "
            f"extra_examples="
            f"{extra[:5]}"
        )

    if (
        len(extracted)
        != args.expected_prefixes
    ):
        raise ValueError(
            f"Expected "
            f"{args.expected_prefixes} "
            "predictions, got "
            f"{len(extracted)}."
        )

    # Preserve MC label order, exactly like
    # BetaPRM extraction.
    records = [
        extracted[prefix_id]
        for prefix_id in labels
    ]

    mean_target = sum(
        x["target_success_prob"]
        for x in records
    ) / len(records)

    mean_prediction = sum(
        x["pred_prob"]
        for x in records
    ) / len(records)

    mean_signed_error = sum(
        x["signed_error"]
        for x in records
    ) / len(records)

    mean_absolute_error = sum(
        x["absolute_error"]
        for x in records
    ) / len(records)

    atomic_write_jsonl(
        records,
        output_path,
    )

    summary = {
        "schema_version": 1,
        "model_name":
            args.model_name,
        "checkpoint":
            args.checkpoint,
        "evaluator_output":
            str(evaluator_path),
        "mc_labels":
            str(labels_path),
        "output":
            str(output_path),
        "counts": {
            "questions":
                len(evaluator_data),
            "prefixes":
                len(records),
        },
        "means": {
            "target_success_prob":
                mean_target,
            "pred_prob":
                mean_prediction,
            "signed_error":
                mean_signed_error,
            "absolute_error":
                mean_absolute_error,
        },
        "ranges": {
            "pred_prob": [
                min(
                    x["pred_prob"]
                    for x in records
                ),
                max(
                    x["pred_prob"]
                    for x in records
                ),
            ]
        },
    }

    atomic_write_json(
        summary,
        summary_path,
    )

    print(
        "=== Official VisualPRM "
        "calibration predictions ==="
    )

    print(
        f"Questions: "
        f"{len(evaluator_data)}"
    )

    print(
        f"Prefixes: "
        f"{len(records)}"
    )

    print(
        f"Mean target: "
        f"{mean_target:.6f}"
    )

    print(
        f"Mean prediction: "
        f"{mean_prediction:.6f}"
    )

    print(
        "Mean signed error: "
        f"{mean_signed_error:+.6f}"
    )

    print(
        "Mean absolute error: "
        f"{mean_absolute_error:.6f}"
    )

    print(
        "Prediction range: "
        f"[{summary['ranges']['pred_prob'][0]:.6f}, "
        f"{summary['ranges']['pred_prob'][1]:.6f}]"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print()
    print(
        "[PASS] Official VisualPRM "
        "predictions extracted."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"\n[FAIL] {exc}",
            file=sys.stderr,
        )
        raise