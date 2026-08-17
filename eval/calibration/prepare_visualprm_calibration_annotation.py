#!/usr/bin/env python3

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert sampled VisualPRM prefix JSONL into the grouped "
            "annotation schema consumed by the existing PRM evaluator."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default="outputs/calibration/visualprm_val/geometry3k_val_prefixes_1000.jsonl",
        help="Prefix-level JSONL produced by sample_visualprm_validation_prefixes.py.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="outputs/calibration/visualprm_val/geometry3k_val_annotation_1000.json",
        help="Output grouped evaluator annotation JSON.",
    )

    parser.add_argument(
        "--expected-prefixes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="Geometry3K",
        help="Optional expected task name, e.g. Geometry3K.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []
    seen_prefix_ids = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}:{lineno}: expected JSON object"
                )

            prefix_id = record.get("prefix_id")

            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"{path}:{lineno}: missing prefix_id"
                )

            if prefix_id in seen_prefix_ids:
                raise ValueError(
                    f"Duplicate prefix_id: {prefix_id}"
                )

            seen_prefix_ids.add(prefix_id)
            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {path}"
        )

    return records


def normalize_image(record):
    """
    Geometry3K is single-image.

    We deliberately reject multi-image samples here because the existing
    MathVision PRM evaluator currently opens a single PIL image per item.
    """

    image = record.get("image")

    if isinstance(image, str):
        if not image.strip():
            raise ValueError(
                f"{record['prefix_id']}: empty image path"
            )
        return image.strip()

    if isinstance(image, list):
        if len(image) == 1 and isinstance(image[0], str):
            return image[0].strip()

        raise ValueError(
            f"{record['prefix_id']}: multi-image sample is not supported "
            f"by this evaluator path: {image}"
        )

    raise ValueError(
        f"{record['prefix_id']}: invalid image field"
    )


def validate_record(record, expected_n, expected_task):
    prefix_id = record["prefix_id"]

    required = (
        "task",
        "question_id",
        "question",
        "prefix_steps",
        "prefix_step_count",
        "prefix_relative_position",
        "source_trajectory_index",
        "mc_correct",
        "mc_total",
        "target_success_prob",
    )

    missing = [
        key
        for key in required
        if key not in record
    ]

    if missing:
        raise ValueError(
            f"{prefix_id}: missing fields: {missing}"
        )

    if expected_task is not None:
        if record["task"] != expected_task:
            raise ValueError(
                f"{prefix_id}: expected task={expected_task}, "
                f"got {record['task']}"
            )

    question = str(record["question"]).strip()

    if not question:
        raise ValueError(
            f"{prefix_id}: empty question"
        )

    prefix_steps = record["prefix_steps"]

    if (
        not isinstance(prefix_steps, list)
        or not prefix_steps
        or any(
            not isinstance(step, str)
            or not step.strip()
            for step in prefix_steps
        )
    ):
        raise ValueError(
            f"{prefix_id}: invalid prefix_steps"
        )

    prefix_step_count = int(
        record["prefix_step_count"]
    )

    if len(prefix_steps) != prefix_step_count:
        raise ValueError(
            f"{prefix_id}: prefix step mismatch: "
            f"len(prefix_steps)={len(prefix_steps)}, "
            f"prefix_step_count={prefix_step_count}"
        )

    k = int(record["mc_correct"])
    n = int(record["mc_total"])

    if n != expected_n:
        raise ValueError(
            f"{prefix_id}: expected N={expected_n}, got {n}"
        )

    if not 0 <= k <= n:
        raise ValueError(
            f"{prefix_id}: invalid K/N={k}/{n}"
        )

    target = float(
        record["target_success_prob"]
    )

    if abs(target - k / n) > 1e-12:
        raise ValueError(
            f"{prefix_id}: target_success_prob != K/N"
        )

    normalize_image(record)


def atomic_write_json(data, path):
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

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{output_path} already exists. "
            f"Use --overwrite."
        )

    records = load_jsonl(
        input_path
    )

    if len(records) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} prefixes, "
            f"got {len(records)}"
        )

    grouped = OrderedDict()

    for record in records:

        validate_record(
            record,
            expected_n=args.expected_n,
            expected_task=args.task,
        )

        prefix_id = record[
            "prefix_id"
        ]

        question_id = str(
            record["question_id"]
        )

        question = str(
            record["question"]
        ).strip()

        image_path = normalize_image(
            record
        )

        if question_id not in grouped:
            grouped[question_id] = {
                "id": question_id,

                "task": record["task"],

                # Evaluator chooses query_cot before question.
                "question": question,
                "query_cot": question,

                # Relative VisualPRM image path.
                "image_path": image_path,

                # Helpful basename fallback for the evaluator.
                "image": os.path.basename(
                    image_path
                ),

                "solutions_splits": [],

                # Keep all metadata required to align
                # evaluator outputs back to targets.
                "prefix_ids": [],
                "mc_correct": [],
                "mc_total": [],
                "success_probs": [],
                "prefix_step_counts": [],
                "prefix_relative_positions": [],
                "source_trajectory_indices": [],
            }

        item = grouped[
            question_id
        ]

        # All prefixes grouped under one trajectory/question
        # must share question and image.
        if item["question"] != question:
            raise ValueError(
                f"{prefix_id}: question mismatch "
                f"within question_id={question_id}"
            )

        if item["image_path"] != image_path:
            raise ValueError(
                f"{prefix_id}: image mismatch "
                f"within question_id={question_id}"
            )

        item["solutions_splits"].append(
            record["prefix_steps"]
        )

        item["prefix_ids"].append(
            prefix_id
        )

        item["mc_correct"].append(
            int(record["mc_correct"])
        )

        item["mc_total"].append(
            int(record["mc_total"])
        )

        item["success_probs"].append(
            float(record["target_success_prob"])
        )

        item[
            "prefix_step_counts"
        ].append(
            int(record["prefix_step_count"])
        )

        item[
            "prefix_relative_positions"
        ].append(
            float(
                record[
                    "prefix_relative_position"
                ]
            )
        )

        item[
            "source_trajectory_indices"
        ].append(
            int(
                record[
                    "source_trajectory_index"
                ]
            )
        )

    output = list(
        grouped.values()
    )

    flat_prefix_ids = [
        prefix_id
        for item in output
        for prefix_id in item["prefix_ids"]
    ]

    if len(flat_prefix_ids) != args.expected_prefixes:
        raise AssertionError(
            f"Expected {args.expected_prefixes} flattened prefixes, "
            f"got {len(flat_prefix_ids)}"
        )

    if len(flat_prefix_ids) != len(
        set(flat_prefix_ids)
    ):
        raise AssertionError(
            "Duplicate prefix IDs after grouping"
        )

    for item in output:
        lengths = {
            len(item["solutions_splits"]),
            len(item["prefix_ids"]),
            len(item["mc_correct"]),
            len(item["mc_total"]),
            len(item["success_probs"]),
            len(item["prefix_step_counts"]),
            len(item["prefix_relative_positions"]),
            len(item["source_trajectory_indices"]),
        }

        if len(lengths) != 1:
            raise AssertionError(
                f"Per-question list-length mismatch "
                f"for id={item['id']}"
            )

    atomic_write_json(
        output,
        output_path,
    )

    per_question = [
        len(item["prefix_ids"])
        for item in output
    ]

    print()
    print(
        "=== VisualPRM PRM calibration annotation ==="
    )
    print(
        f"Task                  : {args.task}"
    )
    print(
        f"Input prefixes        : {len(records)}"
    )
    print(
        f"Grouped trajectories  : {len(output)}"
    )
    print(
        f"Output prefixes       : {len(flat_prefix_ids)}"
    )
    print(
        "Prefixes per trajectory: "
        f"min={min(per_question)}, "
        f"max={max(per_question)}, "
        f"mean={sum(per_question) / len(per_question):.3f}"
    )
    print(
        f"Output                : {output_path}"
    )
    print()
    print(
        "[PASS] VisualPRM evaluator annotation generated."
    )


if __name__ == "__main__":
    main()