#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


PREFIX_FRACTIONS = (0.25, 0.50, 0.75)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(
            "datasets/MathVerse/"
            "MathVerse_rollout_annotation_InternVL8B_oversample.json"
        ),
    )

    parser.add_argument(
        "--seed-dataset",
        type=Path,
        default=Path(
            "datasets/MathVerse/seed_dataset.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/calibration/mathverse/prefixes_1000.jsonl"
        ),
    )

    parser.add_argument(
        "--num-prefixes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--trajectories-per-question",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_jsonl(records, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(path)


def stable_seed(
    global_seed: int,
    identifier: str,
) -> int:

    raw = (
        f"{global_seed}:{identifier}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        raw
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def normalize_steps(
    trajectory: Any,
):
    if not isinstance(
        trajectory,
        list,
    ):
        return None

    steps = []

    for step in trajectory:
        if not isinstance(
            step,
            str,
        ):
            return None

        step = step.strip()

        if step:
            steps.append(step)

    if len(steps) < 2:
        return None

    return steps


def get_prefix_lengths(
    num_steps: int,
):
    max_prefix_len = num_steps - 1

    lengths = []

    for frac in PREFIX_FRACTIONS:

        prefix_len = int(
            math.ceil(
                frac * num_steps
            )
        )

        prefix_len = max(
            1,
            min(
                max_prefix_len,
                prefix_len,
            ),
        )

        lengths.append(
            prefix_len
        )

    return sorted(
        set(lengths)
    )


def main():
    args = parse_args()

    annotation_path = (
        args.annotation
        .expanduser()
        .resolve()
    )

    seed_path = (
        args.seed_dataset
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
            f"Output exists: {output_path}\n"
            f"Use --overwrite."
        )

    rollout_data = load_json(
        annotation_path
    )

    seed_data = load_json(
        seed_path
    )

    if not isinstance(
        rollout_data,
        list,
    ):
        raise TypeError(
            "Rollout annotation must be a list"
        )

    if not isinstance(
        seed_data,
        list,
    ):
        raise TypeError(
            "Seed dataset must be a list"
        )

    # ---------------------------------------
    # Exact ID-based lookup
    # ---------------------------------------

    seed_by_id = {}

    for item in seed_data:

        if not isinstance(
            item,
            dict,
        ):
            continue

        item_id = item.get("id")

        if item_id is None:
            continue

        if item_id in seed_by_id:
            raise ValueError(
                f"Duplicate seed id: {item_id}"
            )

        seed_by_id[item_id] = item

    candidates = []

    valid_trajectories_total = 0
    invalid_trajectories_total = 0
    questions_with_candidates = 0

    for question_index, item in enumerate(
        rollout_data
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        item_id = item.get("id")

        if item_id not in seed_by_id:
            raise ValueError(
                f"Rollout id {item_id} "
                f"missing from seed dataset"
            )

        seed_item = seed_by_id[
            item_id
        ]

        question = (
            item.get("question")
            or item.get("query_cot")
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                f"id={item_id}: missing question"
            )

        question = question.strip()

        seed_question = seed_item.get(
            "question"
        )

        if (
            isinstance(seed_question, str)
            and seed_question.strip()
            != question
        ):
            raise ValueError(
                f"id={item_id}: rollout/seed "
                f"question mismatch"
            )

        correct_answer = seed_item.get(
            "correct_answer"
        )

        if correct_answer is None:
            raise ValueError(
                f"id={item_id}: "
                f"missing correct_answer"
            )

        correct_answer = str(
            correct_answer
        ).strip()

        image_path = item.get(
            "image_path"
        )

        if not isinstance(
            image_path,
            str,
        ):
            raise ValueError(
                f"id={item_id}: "
                f"invalid image_path"
            )

        raw_trajectories = item.get(
            "solutions_splits"
        )

        if not isinstance(
            raw_trajectories,
            list,
        ):
            raise ValueError(
                f"id={item_id}: "
                f"solutions_splits is not list"
            )

        valid_trajectories = []

        for (
            trajectory_index,
            trajectory,
        ) in enumerate(
            raw_trajectories
        ):

            steps = normalize_steps(
                trajectory
            )

            if steps is None:
                invalid_trajectories_total += 1
                continue

            valid_trajectories.append(
                (
                    trajectory_index,
                    steps,
                )
            )

        if not valid_trajectories:
            continue

        valid_trajectories_total += len(
            valid_trajectories
        )

        # Deterministic trajectory sampling
        q_rng = random.Random(
            stable_seed(
                args.sample_seed,
                f"mathverse:{item_id}",
            )
        )

        if (
            len(valid_trajectories)
            > args.trajectories_per_question
        ):
            chosen = q_rng.sample(
                valid_trajectories,
                args.trajectories_per_question,
            )
        else:
            chosen = valid_trajectories

        before = len(candidates)

        for (
            trajectory_index,
            steps,
        ) in chosen:

            prefix_lengths = (
                get_prefix_lengths(
                    len(steps)
                )
            )

            for prefix_len in (
                prefix_lengths
            ):

                prefix_steps = (
                    steps[:prefix_len]
                )

                prefix_id = (
                    f"mathverse_"
                    f"q{int(item_id):04d}_"
                    f"traj{trajectory_index:03d}_"
                    f"step{prefix_len:03d}"
                )

                candidates.append(
                    {
                        "schema_version": 1,

                        "benchmark": (
                            "MathVerse"
                        ),

                        "prefix_id": (
                            prefix_id
                        ),

                        "question_id": (
                            f"mathverse_"
                            f"{int(item_id):04d}"
                        ),

                        "question_index": (
                            question_index
                        ),

                        "original_id": (
                            item_id
                        ),

                        "question": (
                            question
                        ),

                        "correct_answer": (
                            correct_answer
                        ),

                        "image_path": (
                            image_path
                        ),

                        "image": (
                            item.get(
                                "image",
                                Path(
                                    image_path
                                ).name,
                            )
                        ),

                        "source_trajectory_index": (
                            trajectory_index
                        ),

                        "source_trajectory_step_count": (
                            len(steps)
                        ),

                        "prefix_steps": (
                            prefix_steps
                        ),

                        "prefix_step_count": (
                            prefix_len
                        ),

                        "prefix_relative_position": (
                            prefix_len
                            / len(steps)
                        ),
                    }
                )

        if len(candidates) > before:
            questions_with_candidates += 1

    if len(candidates) < (
        args.num_prefixes
    ):
        raise ValueError(
            f"Only {len(candidates)} "
            f"eligible prefixes"
        )

    # ---------------------------------------
    # Global uniform prefix sampling
    # ---------------------------------------

    rng = random.Random(
        args.sample_seed
    )

    selected = rng.sample(
        candidates,
        args.num_prefixes,
    )

    selected.sort(
        key=lambda x: (
            x["question_index"],
            x[
                "source_trajectory_index"
            ],
            x["prefix_step_count"],
        )
    )

    prefix_ids = [
        x["prefix_id"]
        for x in selected
    ]

    if len(prefix_ids) != len(
        set(prefix_ids)
    ):
        raise AssertionError(
            "Duplicate prefix IDs"
        )

    positions = [
        x["prefix_relative_position"]
        for x in selected
    ]

    atomic_write_jsonl(
        selected,
        output_path,
    )

    print()
    print(
        "=== MathVerse prefix sampling ==="
    )
    print(
        f"Questions in annotation       : "
        f"{len(rollout_data)}"
    )
    print(
        f"Questions with candidates     : "
        f"{questions_with_candidates}"
    )
    print(
        f"Valid trajectories            : "
        f"{valid_trajectories_total}"
    )
    print(
        f"Invalid trajectories          : "
        f"{invalid_trajectories_total}"
    )
    print(
        f"Candidate prefixes            : "
        f"{len(candidates)}"
    )
    print(
        f"Selected prefixes             : "
        f"{len(selected)}"
    )
    print(
        f"Unique selected questions     : "
        f"{len(set(x['question_id'] for x in selected))}"
    )
    print(
        f"Mean relative position        : "
        f"{sum(positions)/len(positions):.4f}"
    )
    print(
        f"Output                        : "
        f"{output_path}"
    )
    print()
    print(
        "[PASS] MathVerse prefixes generated."
    )


if __name__ == "__main__":
    main()