#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed MMStar prefix manifest from full source responses."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/calibration/mmstar/"
            "source_responses_internvl3_8b.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/calibration/mmstar/prefixes_1000.jsonl"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/calibration/mmstar/"
            "prefixes_1000.summary.json"
        ),
    )

    parser.add_argument(
        "--num-prefixes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{lineno}"
                ) from exc

            if not isinstance(item, dict):
                raise TypeError(
                    f"{path}:{lineno} is not a JSON object"
                )

            records.append(item)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def clean_segments(
    segments: Any,
) -> list[str]:
    if not isinstance(segments, list):
        return []

    result = []

    for x in segments:
        if x is None:
            continue

        text = str(x).strip()

        if text:
            result.append(text)

    return result


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = args.summary_output.expanduser().resolve()

    if args.num_prefixes <= 0:
        raise ValueError("--num-prefixes must be > 0")

    for ratio in args.ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError(
                f"Prefix ratio must be in (0,1): {ratio}"
            )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Use --overwrite if intentional."
        )

    source_records = load_jsonl(input_path)

    candidates: list[dict[str, Any]] = []

    skipped_empty_response = 0
    skipped_no_reasoning = 0
    total_responses = 0

    seen_prefix_ids: set[str] = set()

    for record in source_records:
        sample_id = str(record["id"])
        question = str(record["question"]).strip()
        correct_answer = str(
            record["correct_answer"]
        ).strip()
        image_path = str(record["image_path"]).strip()

        responses = record.get("responses", [])

        if not isinstance(responses, list):
            raise TypeError(
                f"{sample_id}: responses must be a list"
            )

        for response_pos, response in enumerate(responses):
            total_responses += 1

            if not isinstance(response, dict):
                skipped_empty_response += 1
                continue

            response_index = int(
                response.get(
                    "response_index",
                    response_pos,
                )
            )

            segments = clean_segments(
                response.get("segments")
            )

            if not segments:
                skipped_empty_response += 1
                continue

            # generate_mmstar_source_responses.py stores
            # [reasoning step 1, ..., reasoning step N, final answer].
            #
            # The terminal answer MUST NOT appear in the prefix.
            if len(segments) < 2:
                skipped_no_reasoning += 1
                continue

            reasoning_steps = segments[:-1]
            terminal_answer = segments[-1]

            num_reasoning_steps = len(reasoning_steps)

            if num_reasoning_steps == 0:
                skipped_no_reasoning += 1
                continue

            # Multiple ratios may map to the same step count for
            # short trajectories. Keep only one copy.
            used_step_counts: set[int] = set()

            for requested_ratio in args.ratios:
                prefix_step_count = math.ceil(
                    num_reasoning_steps * requested_ratio
                )

                prefix_step_count = max(
                    1,
                    min(
                        prefix_step_count,
                        num_reasoning_steps,
                    ),
                )

                if prefix_step_count in used_step_counts:
                    continue

                used_step_counts.add(prefix_step_count)

                prefix_steps = reasoning_steps[
                    :prefix_step_count
                ]

                prefix_id = (
                    f"{sample_id}"
                    f"_traj{response_index:02d}"
                    f"_step{prefix_step_count:02d}"
                )

                if prefix_id in seen_prefix_ids:
                    raise ValueError(
                        f"Duplicate prefix_id: {prefix_id}"
                    )

                seen_prefix_ids.add(prefix_id)

                candidate = {
                    "prefix_id": prefix_id,
                    "benchmark": "MMStar",
                    "task": "MMStar",
                    "question_id": sample_id,
                    "source_trajectory_index":
                        response_index,
                    "question": question,
                    "correct_answer":
                        correct_answer,
                    "image_path": image_path,
                    "prefix_steps": prefix_steps,
                    "prefix_process_text":
                        "\n".join(prefix_steps),
                    "prefix_step_count":
                        prefix_step_count,
                    "total_step_count":
                        num_reasoning_steps,
                    "prefix_relative_position":
                        (
                            prefix_step_count
                            / num_reasoning_steps
                        ),
                    "requested_ratio":
                        requested_ratio,
                    "source_terminal_answer":
                        terminal_answer,
                    "category":
                        record.get("category"),
                    "l2_category":
                        record.get("l2_category"),
                }

                candidates.append(candidate)

    if len(candidates) < args.num_prefixes:
        raise RuntimeError(
            f"Only {len(candidates)} valid candidate prefixes, "
            f"but requested {args.num_prefixes}"
        )

    rng = random.Random(args.seed)

    selected = rng.sample(
        candidates,
        args.num_prefixes,
    )

    # Stable file order after deterministic random selection.
    selected.sort(
        key=lambda x: x["prefix_id"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in selected:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    unique_questions = len(
        {
            item["question_id"]
            for item in selected
        }
    )

    position_counts: dict[str, int] = {}

    for item in selected:
        key = (
            f"{item['prefix_relative_position']:.3f}"
        )
        position_counts[key] = (
            position_counts.get(key, 0) + 1
        )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "source_questions": len(source_records),
        "source_responses": total_responses,
        "candidate_prefixes": len(candidates),
        "selected_prefixes": len(selected),
        "unique_selected_questions":
            unique_questions,
        "seed": args.seed,
        "requested_ratios": args.ratios,
        "skipped_empty_response":
            skipped_empty_response,
        "skipped_no_reasoning":
            skipped_no_reasoning,
        "selected_relative_position_counts":
            position_counts,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("===== MMStar prefix sampling =====")
    print(
        f"Source questions    : "
        f"{len(source_records)}"
    )
    print(
        f"Source responses    : "
        f"{total_responses}"
    )
    print(
        f"Candidate prefixes  : "
        f"{len(candidates)}"
    )
    print(
        f"Selected prefixes   : "
        f"{len(selected)}"
    )
    print(
        f"Unique questions    : "
        f"{unique_questions}"
    )
    print(
        f"Skipped empty       : "
        f"{skipped_empty_response}"
    )
    print(
        f"Skipped no reasoning: "
        f"{skipped_no_reasoning}"
    )
    print(f"Output              : {output_path}")
    print(f"Summary             : {summary_path}")


if __name__ == "__main__":
    main()