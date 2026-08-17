#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected object at {path}:{lineno}"
                )

            records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; use --overwrite"
        )

    records = load_jsonl(input_path)

    by_question: OrderedDict[str, list[dict[str, Any]]] = (
        OrderedDict()
    )

    for record in records:
        question_id = str(record["question_id"])
        by_question.setdefault(question_id, []).append(record)

    question_ids = list(by_question.keys())

    if args.num_questions > len(question_ids):
        raise ValueError(
            f"Requested {args.num_questions} questions, "
            f"but only {len(question_ids)} are available"
        )

    if args.num_questions == 1:
        selected_indices = [0]
    else:
        selected_indices = sorted({
            round(
                i * (len(question_ids) - 1)
                / (args.num_questions - 1)
            )
            for i in range(args.num_questions)
        })

    if len(selected_indices) != args.num_questions:
        raise RuntimeError(
            "Question index selection produced duplicate indices"
        )

    selected: list[dict[str, Any]] = []

    for question_index in selected_indices:
        question_id = question_ids[question_index]
        candidates = by_question[question_id]

        # This selection uses only prefix position and prefix ID.
        # It does not use source_outcome_label or PRM scores.
        candidates = sorted(
            candidates,
            key=lambda record: (
                float(record["prefix_relative_position"]),
                record["prefix_id"],
            ),
        )

        early = candidates[0]
        late = candidates[-1]

        selected.append(early)

        if late["prefix_id"] == early["prefix_id"]:
            raise RuntimeError(
                f"Question {question_id} has only one usable prefix"
            )

        selected.append(late)

    prefix_ids = [record["prefix_id"] for record in selected]

    if len(prefix_ids) != len(set(prefix_ids)):
        raise RuntimeError("Duplicate prefix IDs in pilot manifest")

    question_counts = Counter(
        str(record["question_id"])
        for record in selected
    )

    if any(count != 2 for count in question_counts.values()):
        raise RuntimeError(
            "Every selected question must have exactly two prefixes"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=== Pilot manifest ===")
    print(f"Source prefixes: {len(records)}")
    print(f"Source questions: {len(question_ids)}")
    print(f"Selected questions: {len(question_counts)}")
    print(f"Selected prefixes: {len(selected)}")
    print(f"Output: {output_path}")

    print("\n=== First five selected questions ===")
    shown = 0
    for question_id in question_counts:
        question_records = [
            record
            for record in selected
            if str(record["question_id"]) == question_id
        ]

        print(
            question_id,
            [
                (
                    record["prefix_id"],
                    record["prefix_relative_position"],
                )
                for record in question_records
            ],
        )

        shown += 1
        if shown >= 5:
            break

    print("\n[PASS] Pilot manifest generated.")


if __name__ == "__main__":
    main()
