#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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
                    f"Missing or invalid prefix_id at {path}:{lineno}"
                )
            if prefix_id in seen_prefix_ids:
                raise ValueError(
                    f"Duplicate prefix_id in judgments: {prefix_id}"
                )

            seen_prefix_ids.add(prefix_id)
            records.append(record)

    if not records:
        raise ValueError(f"Input contains no records: {path}")

    return records


def atomic_write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(output_path) + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)


def atomic_write_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(output_path) + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(output_path)


def validate_binary_label(
    label: Any,
    *,
    prefix_id: str,
    continuation_id: str,
) -> int:
    if isinstance(label, bool):
        return int(label)

    if isinstance(label, (int, float)) and float(label) in (0.0, 1.0):
        return int(label)

    raise ValueError(
        f"{prefix_id}/{continuation_id}: expected binary label, got {label!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate full continuation judgments and create a compact "
            "prefix-level Monte Carlo calibration-label dataset."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--expected-prefixes", type=int, default=3513)
    parser.add_argument("--expected-n", type=int, default=16)
    parser.add_argument(
        "--policy-model",
        type=str,
        default="InternVL3-8B",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen2.5-32B-Instruct",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.expected_prefixes <= 0:
        raise ValueError("--expected-prefixes must be positive")
    if args.expected_n <= 0:
        raise ValueError("--expected-n must be positive")

    input_path = args.input.expanduser().resolve()
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

    records = load_jsonl(input_path)

    if len(records) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} prefixes, got {len(records)}"
        )

    compact_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    k_distribution: Counter[int] = Counter()
    answer_extraction_counts: Counter[str] = Counter()
    source_generation_fingerprints: set[str] = set()
    judge_fingerprints: set[str] = set()

    for record in records:
        prefix_id = record["prefix_id"]

        prefix_record = record.get("prefix_record")
        if not isinstance(prefix_record, dict):
            raise ValueError(f"{prefix_id}: missing prefix_record")

        judgments = record.get("judgments")
        if not isinstance(judgments, list):
            raise ValueError(f"{prefix_id}: judgments is not a list")
        if len(judgments) != args.expected_n:
            raise ValueError(
                f"{prefix_id}: expected {args.expected_n} judgments, "
                f"got {len(judgments)}"
            )

        summary = record.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{prefix_id}: missing summary")
        if summary.get("complete") is not True:
            raise ValueError(f"{prefix_id}: summary.complete is not true")
        if int(summary.get("judge_failed", -1)) != 0:
            raise ValueError(f"{prefix_id}: judge_failed is not zero")

        continuation_ids: set[str] = set()
        labels: list[int] = []

        for index, judgment in enumerate(judgments):
            if not isinstance(judgment, dict):
                raise TypeError(
                    f"{prefix_id}: judgment {index} is not an object"
                )

            continuation_id = judgment.get("continuation_id")
            if (
                not isinstance(continuation_id, str)
                or not continuation_id
            ):
                raise ValueError(
                    f"{prefix_id}: judgment {index} has no continuation_id"
                )
            if continuation_id in continuation_ids:
                raise ValueError(
                    f"{prefix_id}: duplicate continuation_id "
                    f"{continuation_id}"
                )
            continuation_ids.add(continuation_id)

            status = judgment.get("judge_status")
            if not isinstance(status, str) or not status:
                raise ValueError(
                    f"{prefix_id}/{continuation_id}: missing judge_status"
                )
            status_counts[status] += 1

            if status == "failed":
                raise ValueError(
                    f"{prefix_id}/{continuation_id}: judge status failed"
                )

            label = validate_binary_label(
                judgment.get("label"),
                prefix_id=prefix_id,
                continuation_id=continuation_id,
            )
            labels.append(label)
            label_counts[label] += 1

            method = judgment.get("answer_extraction_method")
            if isinstance(method, str):
                answer_extraction_counts[method] += 1

        correct = sum(labels)
        incorrect = len(labels) - correct
        success_prob = correct / len(labels)

        if int(summary.get("valid_judgments", -1)) != len(labels):
            raise ValueError(
                f"{prefix_id}: valid_judgments does not match labels"
            )
        if int(summary.get("correct", -1)) != correct:
            raise ValueError(
                f"{prefix_id}: summary.correct does not match labels"
            )
        if int(summary.get("incorrect", -1)) != incorrect:
            raise ValueError(
                f"{prefix_id}: summary.incorrect does not match labels"
            )

        stored_prob = summary.get("success_prob")
        if (
            not isinstance(stored_prob, (int, float))
            or not math.isclose(
                float(stored_prob),
                success_prob,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{prefix_id}: summary.success_prob mismatch"
            )

        source_generation_fingerprint = record.get(
            "source_generation_fingerprint"
        )
        judge_fingerprint = record.get("judge_fingerprint")

        if isinstance(source_generation_fingerprint, str):
            source_generation_fingerprints.add(
                source_generation_fingerprint
            )
        if isinstance(judge_fingerprint, str):
            judge_fingerprints.add(judge_fingerprint)

        required_prefix_fields = (
            "question_id",
            "image_path",
            "question",
            "correct_answer",
            "source_trajectory_index",
            "prefix_step_count",
            "prefix_relative_position",
            "prefix_steps",
        )
        missing_fields = [
            field
            for field in required_prefix_fields
            if field not in prefix_record
        ]
        if missing_fields:
            raise ValueError(
                f"{prefix_id}: prefix_record missing fields "
                f"{missing_fields}"
            )

        prefix_steps = prefix_record["prefix_steps"]
        if (
            not isinstance(prefix_steps, list)
            or not prefix_steps
            or any(
                not isinstance(step, str) or not step.strip()
                for step in prefix_steps
            )
        ):
            raise ValueError(f"{prefix_id}: invalid prefix_steps")

        compact_record = {
            "schema_version": 1,
            "benchmark": prefix_record.get(
                "benchmark",
                "MathVision",
            ),
            "policy_model": args.policy_model,
            "judge_model": args.judge_model,

            "prefix_id": prefix_id,
            "question_id": prefix_record["question_id"],
            "image_path": prefix_record["image_path"],
            "question": prefix_record["question"],
            "correct_answer": prefix_record["correct_answer"],

            "source_trajectory_index": prefix_record[
                "source_trajectory_index"
            ],
            "source_num_reasoning_steps": prefix_record.get(
                "source_num_reasoning_steps"
            ),
            "source_outcome_label": prefix_record.get(
                "source_outcome_label"
            ),

            "prefix_step_count": prefix_record["prefix_step_count"],
            "prefix_relative_position": prefix_record[
                "prefix_relative_position"
            ],
            "prefix_steps": prefix_steps,

            "mc_correct": correct,
            "mc_incorrect": incorrect,
            "mc_total": len(labels),
            "success_prob": success_prob,

            "source_generation_fingerprint": (
                source_generation_fingerprint
            ),
            "judge_fingerprint": judge_fingerprint,
        }

        compact_records.append(compact_record)
        k_distribution[correct] += 1

    compact_records.sort(
        key=lambda item: (
            str(item["question_id"]),
            int(item["source_trajectory_index"]),
            int(item["prefix_step_count"]),
            item["prefix_id"],
        )
    )

    atomic_write_jsonl(compact_records, output_path)

    total_judgments = sum(label_counts.values())
    total_correct = label_counts[1]
    total_incorrect = label_counts[0]

    summary_data = {
        "schema_version": 1,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "policy_model": args.policy_model,
        "judge_model": args.judge_model,
        "expected_prefixes": args.expected_prefixes,
        "expected_mc_rollouts": args.expected_n,

        "counts": {
            "prefixes": len(compact_records),
            "judgments": total_judgments,
            "correct": total_correct,
            "incorrect": total_incorrect,
            "overall_success_rate": (
                total_correct / total_judgments
                if total_judgments
                else None
            ),
            "zero_success_prefixes": k_distribution[0],
            "perfect_success_prefixes": k_distribution[
                args.expected_n
            ],
            "intermediate_success_prefixes": sum(
                count
                for k, count in k_distribution.items()
                if 0 < k < args.expected_n
            ),
        },

        "judge_status_counts": dict(status_counts),
        "label_counts": {
            str(key): value
            for key, value in sorted(label_counts.items())
        },
        "mc_correct_count_distribution": {
            str(key): value
            for key, value in sorted(k_distribution.items())
        },
        "answer_extraction_counts": dict(
            answer_extraction_counts
        ),
        "source_generation_fingerprints": sorted(
            source_generation_fingerprints
        ),
        "judge_fingerprints": sorted(judge_fingerprints),
    }

    atomic_write_json(summary_data, summary_path)

    print("=== MC calibration labels ===")
    print(f"Input: {input_path}")
    print(f"Prefixes: {len(compact_records)}")
    print(f"Judgments: {total_judgments}")
    print(
        f"Correct/incorrect: {total_correct}/{total_incorrect}"
    )
    print(
        "Overall continuation success rate: "
        f"{summary_data['counts']['overall_success_rate']:.6f}"
    )
    print(
        "K distribution: "
        f"{dict(sorted(k_distribution.items()))}"
    )
    print(
        "Zero/intermediate/perfect prefixes: "
        f"{summary_data['counts']['zero_success_prefixes']}/"
        f"{summary_data['counts']['intermediate_success_prefixes']}/"
        f"{summary_data['counts']['perfect_success_prefixes']}"
    )
    print(f"Judge statuses: {dict(status_counts)}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print("\n[PASS] Compact MC-label dataset generated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise