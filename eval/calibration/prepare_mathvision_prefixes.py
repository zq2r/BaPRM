#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MARKER_RE = re.compile(r"<\s*(step|answer)\s*>", flags=re.IGNORECASE)
CLOSING_MARKER_RE = re.compile(
    r"</\s*(step|answer)\s*>",
    flags=re.IGNORECASE,
)


@dataclass
class CanonicalTrajectory:
    reasoning_steps: list[str]
    final_answer: str
    parse_mode: str
    empty_entries_removed: int


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            f"Expected a JSON array in {path}, got {type(data).__name__}"
        )

    if not all(isinstance(item, dict) for item in data):
        raise TypeError(f"Not every record in {path} is a JSON object")

    return data


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = CLOSING_MARKER_RE.sub("", text)
    return text.strip()


def normalized_for_comparison(value: Any) -> str:
    return " ".join(clean_text(value).split())


def parse_marked_blob(
    raw_text: str,
) -> tuple[list[str], str, str]:
    """
    Parse a trajectory stored as one string containing <step>/<answer> markers.

    Rules:
    1. When <answer> exists, content before it is reasoning and content after it
       is the terminal answer.
    2. When only <step> exists, conservatively reserve the final marked segment
       as terminal content and use preceding segments as reasoning prefixes.
    3. An unmarked single string has no reliable step boundary and yields no
       reasoning steps.
    """
    text = clean_text(raw_text)
    if not text:
        return [], "", "empty_single_blob"

    matches = list(MARKER_RE.finditer(text))
    if not matches:
        return [], text, "unmarked_single_blob"

    chunks: list[tuple[str, str]] = []

    preamble = clean_text(text[: matches[0].start()])
    if preamble:
        chunks.append(("preamble", preamble))

    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        content = clean_text(text[start:end])
        if content:
            chunks.append((match.group(1).lower(), content))

    first_answer_index = next(
        (
            index
            for index, (kind, _) in enumerate(chunks)
            if kind == "answer"
        ),
        None,
    )

    if first_answer_index is not None:
        reasoning_steps = [
            content
            for kind, content in chunks[:first_answer_index]
            if kind in {"preamble", "step"} and content
        ]

        final_answer = "\n".join(
            content
            for _, content in chunks[first_answer_index:]
            if content
        ).strip()

        return reasoning_steps, final_answer, "marked_with_answer"

    step_like_parts = [
        content
        for kind, content in chunks
        if kind in {"preamble", "step"} and content
    ]

    if len(step_like_parts) >= 2:
        # No explicit <answer>: reserve the last segment as terminal content.
        return (
            step_like_parts[:-1],
            step_like_parts[-1],
            "marked_without_answer",
        )

    terminal = step_like_parts[0] if step_like_parts else text
    return [], terminal, "marked_without_usable_prefix"


def canonicalize_trajectory(raw_trajectory: Any) -> CanonicalTrajectory:
    if not isinstance(raw_trajectory, list):
        return CanonicalTrajectory(
            reasoning_steps=[],
            final_answer="",
            parse_mode="trajectory_not_list",
            empty_entries_removed=0,
        )

    cleaned_entries: list[str] = []
    empty_entries_removed = 0

    for value in raw_trajectory:
        text = clean_text(value)

        if not text:
            empty_entries_removed += 1
            continue

        cleaned_entries.append(text)

    if not cleaned_entries:
        return CanonicalTrajectory(
            reasoning_steps=[],
            final_answer="",
            parse_mode="empty_trajectory",
            empty_entries_removed=empty_entries_removed,
        )

    if len(cleaned_entries) == 1:
        steps, answer, mode = parse_marked_blob(cleaned_entries[0])
    else:
        # The annotation format defines all entries except the last as
        # reasoning steps and the final entry as the terminal answer.
        #
        # Reconstructing a marked string also handles the rare case where one
        # list entry itself still contains embedded <step>/<answer> markers.
        synthetic_blob = "".join(
            f"<step>\n{entry}\n"
            for entry in cleaned_entries[:-1]
        )
        synthetic_blob += f"<answer>\n{cleaned_entries[-1]}"

        steps, answer, _ = parse_marked_blob(synthetic_blob)
        mode = "multi_entry_list"

    reasoning_steps = [
        clean_text(step)
        for step in steps
        if clean_text(step)
    ]

    return CanonicalTrajectory(
        reasoning_steps=reasoning_steps,
        final_answer=clean_text(answer),
        parse_mode=mode,
        empty_entries_removed=empty_entries_removed,
    )


def stable_selection_key(
    global_seed: int,
    question_id: Any,
    trajectory_index: int,
) -> str:
    payload = (
        f"{global_seed}|{question_id}|{trajectory_index}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_id_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    text = text.strip("_")
    return text or "unknown"


def choose_prefix_lengths(
    num_reasoning_steps: int,
    num_prefixes: int,
) -> list[int]:
    """
    Select approximately evenly spaced prefix lengths.

    For num_prefixes=3 this corresponds to approximately:
      25%, 50%, and 75%.

    When a trajectory has fewer steps than requested prefixes, duplicate
    positions are removed.
    """
    if num_reasoning_steps <= 0:
        return []

    if num_prefixes <= 0:
        raise ValueError("num_prefixes must be positive")

    positions = {
        max(
            1,
            min(
                num_reasoning_steps,
                math.ceil(
                    (index + 1)
                    * num_reasoning_steps
                    / (num_prefixes + 1)
                ),
            ),
        )
        for index in range(num_prefixes)
    }

    return sorted(positions)


def atomic_write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a deterministic MathVision reasoning-prefix manifest "
            "from BetaPRM rollout annotations."
        )
    )
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)

    parser.add_argument(
        "--trajectories-per-question",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--prefixes-per-trajectory",
        type=int,
        default=3,
    )
    parser.add_argument("--selection-seed", type=int, default=42)

    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="0 means use all questions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.trajectories_per_question <= 0:
        raise ValueError("--trajectories-per-question must be positive")

    if args.prefixes_per_trajectory <= 0:
        raise ValueError("--prefixes-per-trajectory must be positive")

    seed_path = args.seed.resolve()
    annotation_path = args.annotation.resolve()
    output_path = args.output.resolve()

    summary_path = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output_path.with_name(
            output_path.stem + ".summary.json"
        )
    )

    for path in (output_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Use --overwrite to replace it."
            )

    seed_records = load_json_array(seed_path)
    annotation_records = load_json_array(annotation_path)

    if args.max_questions > 0:
        annotation_records = annotation_records[: args.max_questions]

    seed_by_id: dict[str, dict[str, Any]] = {}

    for item in seed_records:
        if "id" not in item:
            raise KeyError("A seed record is missing the 'id' field")

        key = str(item["id"])
        if key in seed_by_id:
            raise ValueError(f"Duplicate seed id: {key}")

        seed_by_id[key] = item

    manifest_records: list[dict[str, Any]] = []
    seen_prefix_ids: set[str] = set()

    parse_mode_counts: Counter[str] = Counter()
    skipped_trajectory_reasons: Counter[str] = Counter()
    selected_label_counts: Counter[str] = Counter()
    selected_trajectories_per_question: list[int] = []
    prefixes_per_question: list[int] = []

    empty_entries_removed_total = 0
    valid_trajectory_count = 0
    selected_trajectory_count = 0

    skipped_examples: list[dict[str, Any]] = []

    for question_index, annotation_item in enumerate(annotation_records):
        if "id" not in annotation_item:
            raise KeyError(
                f"Annotation record {question_index} is missing 'id'"
            )

        question_id = annotation_item["id"]
        question_key = str(question_id)

        if question_key not in seed_by_id:
            raise KeyError(
                f"Question id {question_id} exists in annotation "
                "but not in seed dataset"
            )

        seed_item = seed_by_id[question_key]

        seed_question = clean_text(seed_item.get("question"))
        annotation_question = clean_text(
            annotation_item.get("question")
        )

        if (
            normalized_for_comparison(seed_question)
            != normalized_for_comparison(annotation_question)
        ):
            raise ValueError(
                f"Question mismatch for id={question_id}"
            )

        correct_answer = clean_text(
            seed_item.get("correct_answer")
        )
        if not correct_answer:
            raise ValueError(
                f"Missing correct_answer for id={question_id}"
            )

        image_path = (
            seed_item.get("image_path")
            or seed_item.get("image")
            or annotation_item.get("image_path")
            or annotation_item.get("image")
        )

        if not image_path:
            raise ValueError(
                f"Missing image path for id={question_id}"
            )

        solutions = annotation_item.get("solutions_splits")
        labels = annotation_item.get("labels")

        if not isinstance(solutions, list):
            raise TypeError(
                f"solutions_splits is not a list for id={question_id}"
            )

        if not isinstance(labels, list):
            raise TypeError(
                f"labels is not a list for id={question_id}"
            )

        if len(solutions) != len(labels):
            raise ValueError(
                f"Trajectory/label count mismatch for id={question_id}: "
                f"{len(solutions)} != {len(labels)}"
            )

        candidates: list[dict[str, Any]] = []

        for trajectory_index, raw_trajectory in enumerate(solutions):
            canonical = canonicalize_trajectory(raw_trajectory)

            parse_mode_counts[canonical.parse_mode] += 1
            empty_entries_removed_total += (
                canonical.empty_entries_removed
            )

            if not canonical.reasoning_steps:
                skipped_trajectory_reasons[
                    canonical.parse_mode
                ] += 1

                if len(skipped_examples) < 20:
                    skipped_examples.append(
                        {
                            "question_id": question_id,
                            "trajectory_index": trajectory_index,
                            "parse_mode": canonical.parse_mode,
                            "raw_trajectory": raw_trajectory,
                        }
                    )
                continue

            valid_trajectory_count += 1

            candidates.append(
                {
                    "trajectory_index": trajectory_index,
                    "reasoning_steps": canonical.reasoning_steps,
                    "final_answer": canonical.final_answer,
                    "parse_mode": canonical.parse_mode,
                    "selection_key": stable_selection_key(
                        args.selection_seed,
                        question_id,
                        trajectory_index,
                    ),
                }
            )

        # Selection uses only question id, trajectory index, and global seed.
        # It does not use labels, correctness, PRM scores, or trajectory length.
        candidates.sort(key=lambda item: item["selection_key"])

        selected_candidates = candidates[
            : args.trajectories_per_question
        ]

        selected_trajectories_per_question.append(
            len(selected_candidates)
        )
        selected_trajectory_count += len(selected_candidates)

        question_prefix_count = 0

        for candidate in selected_candidates:
            trajectory_index = candidate["trajectory_index"]
            reasoning_steps = candidate["reasoning_steps"]
            num_reasoning_steps = len(reasoning_steps)

            prefix_lengths = choose_prefix_lengths(
                num_reasoning_steps,
                args.prefixes_per_trajectory,
            )

            source_label = labels[trajectory_index]

            if isinstance(source_label, bool):
                source_label = int(source_label)
            elif isinstance(source_label, (int, float)):
                source_label = int(source_label)
            else:
                raise TypeError(
                    f"Invalid label type for id={question_id}, "
                    f"trajectory={trajectory_index}: "
                    f"{type(source_label).__name__}"
                )

            if source_label not in (0, 1):
                raise ValueError(
                    f"Non-binary label for id={question_id}, "
                    f"trajectory={trajectory_index}: {source_label}"
                )

            selected_label_counts[str(source_label)] += 1

            for prefix_step_count in prefix_lengths:
                prefix_id = (
                    f"mathvision_"
                    f"{safe_id_component(question_id)}_"
                    f"traj{trajectory_index:02d}_"
                    f"prefix{prefix_step_count:02d}"
                )

                if prefix_id in seen_prefix_ids:
                    raise ValueError(
                        f"Duplicate prefix_id generated: {prefix_id}"
                    )
                seen_prefix_ids.add(prefix_id)

                prefix_steps = reasoning_steps[:prefix_step_count]

                if not prefix_steps:
                    raise AssertionError(
                        f"Empty prefix generated: {prefix_id}"
                    )

                record = {
                    "schema_version": 1,
                    "benchmark": "MathVision",

                    "question_index": question_index,
                    "question_id": question_id,
                    "image_path": str(image_path),
                    "question": seed_question,
                    "correct_answer": correct_answer,

                    "source_trajectory_index": trajectory_index,
                    "source_outcome_label": source_label,
                    "source_parse_mode": candidate["parse_mode"],
                    "source_num_reasoning_steps": (
                        num_reasoning_steps
                    ),

                    "prefix_id": prefix_id,
                    "prefix_step_count": prefix_step_count,
                    "prefix_relative_position": round(
                        prefix_step_count / num_reasoning_steps,
                        6,
                    ),
                    "prefix_steps": prefix_steps,

                    "source_trajectory_quality_filtered": True,
                    "source_trajectory_selection": (
                        "beta_prm_quality_filtered_pool_"
                        "then_stable_hash_subsample"
                    ),
                }

                manifest_records.append(record)
                question_prefix_count += 1

        prefixes_per_question.append(question_prefix_count)

    manifest_records.sort(
        key=lambda item: (
            item["question_index"],
            item["source_trajectory_index"],
            item["prefix_step_count"],
        )
    )

    if not manifest_records:
        raise RuntimeError("No prefix records were generated")

    for record in manifest_records:
        steps = record["prefix_steps"]

        if not isinstance(steps, list) or not steps:
            raise AssertionError(
                f"Invalid prefix_steps in {record['prefix_id']}"
            )

        if len(steps) != record["prefix_step_count"]:
            raise AssertionError(
                f"prefix_step_count mismatch in "
                f"{record['prefix_id']}"
            )

        if any(
            not isinstance(step, str) or not step.strip()
            for step in steps
        ):
            raise AssertionError(
                f"Empty/non-string step in {record['prefix_id']}"
            )

    atomic_write_jsonl(manifest_records, output_path)

    summary = {
        "schema_version": 1,
        "seed_file": str(seed_path),
        "annotation_file": str(annotation_path),
        "output_file": str(output_path),

        "config": {
            "trajectories_per_question": (
                args.trajectories_per_question
            ),
            "prefixes_per_trajectory": (
                args.prefixes_per_trajectory
            ),
            "selection_seed": args.selection_seed,
            "max_questions": args.max_questions,
        },

        "counts": {
            "questions_processed": len(annotation_records),
            "raw_trajectories": sum(
                len(item["solutions_splits"])
                for item in annotation_records
            ),
            "valid_trajectories": valid_trajectory_count,
            "selected_trajectories": selected_trajectory_count,
            "prefixes_generated": len(manifest_records),
            "empty_entries_removed": (
                empty_entries_removed_total
            ),
        },

        "parse_mode_counts": dict(parse_mode_counts),
        "skipped_trajectory_reasons": dict(
            skipped_trajectory_reasons
        ),
        "selected_source_label_counts": dict(
            selected_label_counts
        ),

        "selected_trajectories_per_question": {
            "min": min(selected_trajectories_per_question),
            "max": max(selected_trajectories_per_question),
        },
        "prefixes_per_question": {
            "min": min(prefixes_per_question),
            "max": max(prefixes_per_question),
        },

        "skipped_examples": skipped_examples,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== MathVision prefix manifest ===")
    print(f"Questions processed: {len(annotation_records)}")
    print(
        "Raw trajectories: "
        f"{summary['counts']['raw_trajectories']}"
    )
    print(f"Valid trajectories: {valid_trajectory_count}")
    print(f"Selected trajectories: {selected_trajectory_count}")
    print(f"Prefixes generated: {len(manifest_records)}")
    print(
        "Empty entries removed: "
        f"{empty_entries_removed_total}"
    )
    print(f"Parse modes: {dict(parse_mode_counts)}")
    print(
        "Skipped trajectories: "
        f"{dict(skipped_trajectory_reasons)}"
    )
    print(
        "Selected source labels: "
        f"{dict(selected_label_counts)}"
    )
    print(f"Manifest: {output_path}")
    print(f"Summary: {summary_path}")

    show_count = min(max(0, args.show), len(manifest_records))
    if show_count:
        print(f"\n=== First {show_count} prefixes ===")
        for record in manifest_records[:show_count]:
            preview = {
                "prefix_id": record["prefix_id"],
                "question_id": record["question_id"],
                "source_trajectory_index": (
                    record["source_trajectory_index"]
                ),
                "source_num_reasoning_steps": (
                    record["source_num_reasoning_steps"]
                ),
                "prefix_step_count": (
                    record["prefix_step_count"]
                ),
                "prefix_relative_position": (
                    record["prefix_relative_position"]
                ),
                "prefix_steps": record["prefix_steps"],
            }
            print(json.dumps(
                preview,
                ensure_ascii=False,
                indent=2,
            ))

    print("\n[PASS] Prefix manifest generated successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise
