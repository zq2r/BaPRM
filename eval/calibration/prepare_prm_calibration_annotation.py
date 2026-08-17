#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


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
                    f"Duplicate prefix_id: {prefix_id}"
                )

            seen_prefix_ids.add(prefix_id)
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def resolve_image_basename(record: dict[str, Any]) -> str:
    raw = normalize_text(record.get("image_path"))
    if not raw:
        raise ValueError(
            f"{record.get('prefix_id')}: missing image_path"
        )
    return os.path.basename(raw)


def validate_record(record: dict[str, Any], expected_n: int) -> None:
    prefix_id = record["prefix_id"]

    required = (
        "question_id",
        "question",
        "correct_answer",
        "image_path",
        "prefix_steps",
        "mc_correct",
        "mc_total",
        "success_prob",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            f"{prefix_id}: missing required fields {missing}"
        )

    prefix_steps = record["prefix_steps"]
    if (
        not isinstance(prefix_steps, list)
        or not prefix_steps
        or any(
            not isinstance(step, str) or not step.strip()
            for step in prefix_steps
        )
    ):
        raise ValueError(f"{prefix_id}: invalid prefix_steps")

    mc_total = int(record["mc_total"])
    mc_correct = int(record["mc_correct"])
    success_prob = float(record["success_prob"])

    if mc_total != expected_n:
        raise ValueError(
            f"{prefix_id}: expected mc_total={expected_n}, "
            f"got {mc_total}"
        )
    if not 0 <= mc_correct <= mc_total:
        raise ValueError(
            f"{prefix_id}: invalid mc_correct={mc_correct}"
        )
    if abs(success_prob - mc_correct / mc_total) > 1e-12:
        raise ValueError(
            f"{prefix_id}: success_prob is inconsistent with K/N"
        )


def atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert compact prefix-level MC labels into the grouped JSON "
            "schema consumed by the existing MathVision PRM evaluators."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prefixes", type=int, default=3513)
    parser.add_argument("--expected-n", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    records = load_jsonl(input_path)

    if len(records) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} prefixes, "
            f"got {len(records)}"
        )

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for record in records:
        validate_record(record, expected_n=args.expected_n)

        question_id = str(record["question_id"])
        question = normalize_text(record["question"])
        correct_answer = normalize_text(record["correct_answer"])
        image_path = normalize_text(record["image_path"])
        image_name = resolve_image_basename(record)

        if not question:
            raise ValueError(
                f"{record['prefix_id']}: empty question"
            )
        if not correct_answer:
            raise ValueError(
                f"{record['prefix_id']}: empty correct_answer"
            )

        if question_id not in grouped:
            grouped[question_id] = {
                "id": record["question_id"],
                "question": question,
                "query_cot": question,
                "correct_answer": correct_answer,

                # Existing evaluator path resolution is most robust when
                # both the original relative path and basename are present.
                "image_path": image_path,
                "image": image_name,

                "solutions_splits": [],
                "prefix_ids": [],
                "mc_correct": [],
                "mc_total": [],
                "success_probs": [],
                "prefix_step_counts": [],
                "prefix_relative_positions": [],
                "source_trajectory_indices": [],
            }

        item = grouped[question_id]

        if normalize_text(item["question"]) != question:
            raise ValueError(
                f"Question mismatch within question_id={question_id}"
            )
        if normalize_text(item["correct_answer"]) != correct_answer:
            raise ValueError(
                f"Correct-answer mismatch within question_id={question_id}"
            )
        if os.path.basename(str(item["image"])) != image_name:
            raise ValueError(
                f"Image mismatch within question_id={question_id}"
            )

        item["solutions_splits"].append(record["prefix_steps"])
        item["prefix_ids"].append(record["prefix_id"])
        item["mc_correct"].append(int(record["mc_correct"]))
        item["mc_total"].append(int(record["mc_total"]))
        item["success_probs"].append(float(record["success_prob"]))
        item["prefix_step_counts"].append(
            int(record["prefix_step_count"])
        )
        item["prefix_relative_positions"].append(
            float(record["prefix_relative_position"])
        )
        item["source_trajectory_indices"].append(
            int(record["source_trajectory_index"])
        )

    output = list(grouped.values())

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
    if len(set(flat_prefix_ids)) != len(flat_prefix_ids):
        raise AssertionError("Duplicate prefix IDs after grouping")

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
                f"Per-question list-length mismatch for id={item['id']}"
            )

    atomic_write_json(output, output_path)

    per_question = [len(item["prefix_ids"]) for item in output]

    print("=== PRM calibration annotation ===")
    print(f"Input prefixes: {len(records)}")
    print(f"Questions: {len(output)}")
    print(f"Output prefixes: {len(flat_prefix_ids)}")
    print(
        "Prefixes per question: "
        f"min={min(per_question)}, "
        f"max={max(per_question)}, "
        f"mean={sum(per_question) / len(per_question):.3f}"
    )
    print(f"Output: {output_path}")
    print("\n[PASS] Evaluator annotation generated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise