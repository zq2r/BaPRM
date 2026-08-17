#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


DEFAULT_PREFIX_FRACTIONS = (0.25, 0.50, 0.75)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct a sampled MathVista prefix manifest for "
            "PRM calibration evaluation."
        )
    )

    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(
            "datasets/MathVista/"
            "MathVista_rollout_annotation_InternVL8B_oversample.json"
        ),
    )

    parser.add_argument(
        "--seed-dataset",
        type=Path,
        default=Path(
            "datasets/MathVista/seed_dataset.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/calibration/mathvista/prefixes_1000.jsonl"
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


def atomic_write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(path) + ".tmp")

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


def stable_seed(
    global_seed: int,
    identifier: str,
) -> int:
    """
    Do NOT use Python hash(), because it is not stable
    across interpreter processes.
    """

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


def extract_answer(record: dict[str, Any]):
    """
    Support common MathVista answer field names.
    """

    candidate_keys = (
        "correct_answer",
        "answer",
        "gt_answer",
        "ground_truth",
        "target",
        "label",
    )

    for key in candidate_keys:
        if key not in record:
            continue

        value = record[key]

        if value is None:
            continue

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            value = str(value).strip()

            if value:
                return value

    return None


def build_seed_lookup(
    seed_data: Any,
):
    """
    Construct several lookup tables so that the code
    does not rely solely on list-index alignment.
    """

    by_image = {}
    by_question = {}
    by_index = {}

    if isinstance(seed_data, list):
        iterable = enumerate(seed_data)

    elif isinstance(seed_data, dict):
        iterable = enumerate(
            seed_data.values()
        )

    else:
        raise TypeError(
            "seed_dataset.json must be list or dict"
        )

    for idx, item in iterable:
        if not isinstance(item, dict):
            continue

        by_index[idx] = item

        image = (
            item.get("image")
            or item.get("image_path")
        )

        if isinstance(image, str):
            by_image[
                Path(image).name
            ] = item

        question = (
            item.get("question")
            or item.get("query_cot")
        )

        if isinstance(question, str):
            question = question.strip()

            if question:
                by_question[
                    question
                ] = item

    return (
        by_index,
        by_image,
        by_question,
    )


def resolve_correct_answer(
    rollout_item: dict[str, Any],
    question_index: int,
    seed_lookup,
):
    # First try the rollout annotation itself.
    answer = extract_answer(
        rollout_item
    )

    if answer is not None:
        return answer

    by_index, by_image, by_question = (
        seed_lookup
    )

    candidates = []

    # Prefer image matching.
    image = (
        rollout_item.get("image")
        or rollout_item.get("image_path")
    )

    if isinstance(image, str):
        basename = Path(image).name

        if basename in by_image:
            candidates.append(
                by_image[basename]
            )

    # Then exact question matching.
    question = (
        rollout_item.get("question")
        or rollout_item.get("query_cot")
    )

    if isinstance(question, str):
        question = question.strip()

        if question in by_question:
            candidates.append(
                by_question[question]
            )

    # Finally use index alignment as fallback.
    if question_index in by_index:
        candidates.append(
            by_index[question_index]
        )

    for candidate in candidates:
        answer = extract_answer(
            candidate
        )

        if answer is not None:
            return answer

    raise ValueError(
        f"Cannot resolve correct answer for "
        f"MathVista item index={question_index}"
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

        if not step:
            continue

        steps.append(step)

    # Need at least two steps:
    # we should not use the entire one-step answer
    # as a continuation prefix.
    if len(steps) < 2:
        return None

    return steps


def get_prefix_lengths(
    num_steps: int,
):
    """
    Select roughly 25%, 50%, 75% positions.

    Crucially, never include the full trajectory:
        prefix_length <= num_steps - 1
    because later we need to continue generating.
    """

    max_prefix_len = num_steps - 1

    lengths = []

    for frac in DEFAULT_PREFIX_FRACTIONS:

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

    # Short trajectories can map multiple fractions
    # to the same prefix length.
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
            f"Output already exists: "
            f"{output_path}\n"
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
            "MathVista rollout annotation "
            "must be a JSON list"
        )

    seed_lookup = build_seed_lookup(
        seed_data
    )

    candidates = []

    num_questions_with_candidates = 0
    num_valid_trajectories = 0
    num_invalid_trajectories = 0

    for question_index, item in enumerate(
        rollout_data
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        question = (
            item.get("question")
            or item.get("query_cot")
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                f"Item {question_index}: "
                f"missing question"
            )

        question = question.strip()

        image_path = (
            item.get("image_path")
            or item.get("image")
        )

        if not isinstance(
            image_path,
            str,
        ):
            raise ValueError(
                f"Item {question_index}: "
                f"invalid image path"
            )

        correct_answer = (
            resolve_correct_answer(
                rollout_item=item,
                question_index=question_index,
                seed_lookup=seed_lookup,
            )
        )

        raw_trajectories = (
            item.get(
                "solutions_splits"
            )
        )

        if not isinstance(
            raw_trajectories,
            list,
        ):
            raise ValueError(
                f"Item {question_index}: "
                f"solutions_splits is not a list"
            )

        valid_trajectories = []

        for trajectory_index, trajectory in enumerate(
            raw_trajectories
        ):
            steps = normalize_steps(
                trajectory
            )

            if steps is None:
                num_invalid_trajectories += 1
                continue

            valid_trajectories.append(
                (
                    trajectory_index,
                    steps,
                )
            )

        if not valid_trajectories:
            continue

        num_valid_trajectories += len(
            valid_trajectories
        )

        # Stable per-question trajectory selection.
        question_rng = random.Random(
            stable_seed(
                args.sample_seed,
                f"mathvista:{question_index}",
            )
        )

        if (
            len(valid_trajectories)
            > args.trajectories_per_question
        ):
            chosen_trajectories = (
                question_rng.sample(
                    valid_trajectories,
                    args.trajectories_per_question,
                )
            )
        else:
            chosen_trajectories = (
                valid_trajectories
            )

        before = len(candidates)

        for (
            trajectory_index,
            steps,
        ) in chosen_trajectories:

            prefix_lengths = (
                get_prefix_lengths(
                    len(steps)
                )
            )

            for prefix_len in prefix_lengths:

                prefix_steps = (
                    steps[:prefix_len]
                )

                prefix_relative_position = (
                    prefix_len
                    / len(steps)
                )

                prefix_id = (
                    f"mathvista_"
                    f"q{question_index:04d}_"
                    f"traj{trajectory_index:03d}_"
                    f"step{prefix_len:03d}"
                )

                record = {
                    "schema_version": 1,

                    "benchmark": "MathVista",

                    "prefix_id": prefix_id,

                    "question_id": (
                        f"mathvista_"
                        f"{question_index:04d}"
                    ),

                    "question_index": (
                        question_index
                    ),

                    "question": question,

                    "correct_answer": (
                        correct_answer
                    ),

                    "image_path": (
                        image_path
                    ),

                    "image": (
                        Path(
                            image_path
                        ).name
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
                        prefix_relative_position
                    ),
                }

                candidates.append(
                    record
                )

        if len(candidates) > before:
            num_questions_with_candidates += 1

    if len(candidates) < args.num_prefixes:
        raise ValueError(
            f"Only {len(candidates)} eligible "
            f"candidate prefixes, cannot sample "
            f"{args.num_prefixes}."
        )

    # ------------------------------------------------
    # Global prefix sampling
    # ------------------------------------------------

    rng = random.Random(
        args.sample_seed
    )

    selected = rng.sample(
        candidates,
        args.num_prefixes,
    )

    # Stable output ordering.
    selected.sort(
        key=lambda x: (
            x["question_index"],
            x["source_trajectory_index"],
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
            "Duplicate prefix IDs detected"
        )

    unique_questions = len(
        {
            x["question_id"]
            for x in selected
        }
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
        "=== MathVista prefix sampling ==="
    )
    print(
        f"Questions in annotation        : "
        f"{len(rollout_data)}"
    )
    print(
        f"Questions with candidates      : "
        f"{num_questions_with_candidates}"
    )
    print(
        f"Valid source trajectories      : "
        f"{num_valid_trajectories}"
    )
    print(
        f"Invalid source trajectories    : "
        f"{num_invalid_trajectories}"
    )
    print(
        f"Candidate prefixes             : "
        f"{len(candidates)}"
    )
    print(
        f"Selected prefixes              : "
        f"{len(selected)}"
    )
    print(
        f"Unique selected questions      : "
        f"{unique_questions}"
    )
    print(
        f"Mean prefix relative position  : "
        f"{sum(positions) / len(positions):.4f}"
    )
    print(
        f"Sample seed                    : "
        f"{args.sample_seed}"
    )
    print(
        f"Output                         : "
        f"{output_path}"
    )
    print()
    print(
        "[PASS] MathVista prefix manifest generated."
    )


if __name__ == "__main__":
    main()