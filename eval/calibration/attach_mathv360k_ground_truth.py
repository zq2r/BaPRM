#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach MathV360K ground-truth answers to a sampled "
            "VisualPRM policy-shift prefix manifest."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input policy-shift prefix JSONL.",
    )

    parser.add_argument(
        "--mathv360k-annotation",
        type=Path,
        required=True,
        help="MathV360K train_samples_all_tuning.json.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL with correct_answer attached.",
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

    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    Normalize only formatting differences.

    Do NOT aggressively modify semantic content because the
    image-question pair is used as the exact GT join key.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    # Remove redundant empty lines.
    output = []
    previous_empty = False

    for line in lines:
        is_empty = not line

        if is_empty and previous_empty:
            continue

        output.append(line)
        previous_empty = is_empty

    return "\n".join(output).strip()


def normalize_image_path(path: str) -> str:
    """
    Convert all possible local/absolute paths into the canonical
    MathV360K relative image path.

    Examples:

      datasets/VisualPRM400K-v1.1-Raw/
          MathV360K/A-OKVQA/images/420634.jpg

    becomes

      A-OKVQA/images/420634.jpg
    """

    path = path.replace("\\", "/")

    markers = [
        "VisualPRM400K-v1.1-Raw/MathV360K/",
        "MathV360K/",
    ]

    for marker in markers:
        if marker in path:
            path = path.split(
                marker,
                1,
            )[1]
            break

    return path.lstrip("/")


def extract_gt_answer(value: str) -> str:
    """
    Extract MathV360K answer.

    Typical annotation formats include:

        The answer is B
        The answer is 100
        The answer is 10^4

    If no wrapper exists, preserve the original answer text.
    """

    value = value.strip()

    patterns = [
        r"(?is)^\s*the\s+answer\s+is\s*:?\s*(.*?)\s*$",
        r"(?is)^\s*answer\s*:?\s*(.*?)\s*$",
    ]

    answer = None

    for pattern in patterns:
        match = re.match(
            pattern,
            value,
        )

        if match:
            answer = match.group(1).strip()
            break

    if answer is None:
        answer = value

    # Remove only superficial final punctuation.
    answer = answer.strip()

    while (
        len(answer) > 1
        and answer[-1] in ".。"
    ):
        answer = answer[:-1].rstrip()

    if not answer:
        raise ValueError(
            f"Could not extract answer from "
            f"{value!r}"
        )

    return answer


def parse_mathv360k_record(
    item: dict[str, Any],
    index: int,
):
    image = item.get("image")

    conversations = item.get(
        "conversations"
    )

    if not isinstance(image, str):
        return None

    if (
        not isinstance(
            conversations,
            list,
        )
        or len(conversations) < 2
    ):
        return None

    human = conversations[0]
    assistant = conversations[1]

    if not isinstance(
        human,
        dict,
    ):
        return None

    if not isinstance(
        assistant,
        dict,
    ):
        return None

    question = human.get("value")
    answer_text = assistant.get("value")

    if not isinstance(
        question,
        str,
    ):
        return None

    if not isinstance(
        answer_text,
        str,
    ):
        return None

    image_key = normalize_image_path(
        image
    )

    question_key = normalize_text(
        question
    )

    answer = extract_gt_answer(
        answer_text
    )

    return (
        image_key,
        question_key,
        answer,
    )


def build_gt_lookup(
    annotation_path: Path,
):
    data = load_json(
        annotation_path
    )

    if not isinstance(
        data,
        list,
    ):
        raise TypeError(
            "MathV360K annotation must be "
            "a top-level JSON list"
        )

    # Exact lookup:
    #
    # (image, question) -> set(answers)
    #
    # We intentionally retain conflicting annotations rather
    # than failing while loading the entire dataset.
    lookup: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    # Used only for diagnostics when exact question matching fails.
    by_image: dict[
        str,
        list[tuple[str, str]],
    ] = defaultdict(list)

    valid_records = 0

    for index, item in enumerate(data):
        if not isinstance(
            item,
            dict,
        ):
            continue

        parsed = parse_mathv360k_record(
            item,
            index,
        )

        if parsed is None:
            continue

        (
            image_key,
            question_key,
            answer,
        ) = parsed

        lookup[
            (
                image_key,
                question_key,
            )
        ].add(answer)

        by_image[
            image_key
        ].append(
            (
                question_key,
                answer,
            )
        )

        valid_records += 1

    if not lookup:
        raise ValueError(
            "No valid MathV360K ground-truth "
            "records found"
        )

    num_conflicting_keys = sum(
        len(answers) > 1
        for answers in lookup.values()
    )

    print(
        "=== MathV360K GT database ==="
    )
    print(
        f"Raw records          : "
        f"{len(data)}"
    )
    print(
        f"Valid records        : "
        f"{valid_records}"
    )
    print(
        f"Unique image/question: "
        f"{len(lookup)}"
    )
    print(
        f"Conflicting GT keys  : "
        f"{num_conflicting_keys}"
    )

    return lookup, by_image


def atomic_write_jsonl(
    records: list[dict[str, Any]],
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


def main():
    args = parse_args()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    annotation_path = (
        args.mathv360k_annotation
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input does not exist: "
            f"{input_path}"
        )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"MathV360K annotation does not "
            f"exist: {annotation_path}"
        )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: "
            f"{output_path}\n"
            f"Use --overwrite to replace it."
        )

    gt_lookup, by_image = (
        build_gt_lookup(
            annotation_path
        )
    )

    input_records = []

    with input_path.open(
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

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at "
                    f"{input_path}:{lineno}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise TypeError(
                    f"{input_path}:{lineno}: "
                    f"record is not an object"
                )

            input_records.append(
                record
            )

    if (
        len(input_records)
        != args.expected_prefixes
    ):
        raise ValueError(
            f"Expected "
            f"{args.expected_prefixes} "
            f"input prefixes, got "
            f"{len(input_records)}"
        )

    output_records = []

    missing = []
    ambiguous = []

    for record in input_records:

        prefix_id = record.get(
            "prefix_id"
        )

        if not isinstance(
            prefix_id,
            str,
        ):
            raise ValueError(
                "Prefix record missing "
                "valid prefix_id"
            )

        question = record.get(
            "question"
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                f"{prefix_id}: "
                f"missing question"
            )

        image = record.get(
            "image"
        )

        if not isinstance(
            image,
            str,
        ):
            # Fall back to resolved image_path.
            image = record.get(
                "image_path"
            )

        if not isinstance(
            image,
            str,
        ):
            raise ValueError(
                f"{prefix_id}: "
                f"missing image/image_path"
            )

        image_key = (
            normalize_image_path(
                image
            )
        )

        question_key = (
            normalize_text(
                question
            )
        )

        key = (
            image_key,
            question_key,
        )

        answers = gt_lookup.get(
            key
        )

        if answers is None:
            same_image = by_image.get(
                image_key,
                [],
            )

            missing.append(
                {
                    "prefix_id":
                        prefix_id,
                    "image":
                        image_key,
                    "question":
                        question_key,
                    "same_image_count":
                        len(same_image),
                    "candidate_questions": [
                        q
                        for q, _ in
                        same_image[:3]
                    ],
                    "candidate_answers": [
                        a
                        for _, a in
                        same_image[:3]
                    ],
                }
            )

            continue

        if len(answers) != 1:
            ambiguous.append(
                {
                    "prefix_id":
                        prefix_id,
                    "image":
                        image_key,
                    "question":
                        question_key,
                    "answers":
                        sorted(answers),
                }
            )

            continue

        answer = next(
            iter(answers)
        )

        output_record = dict(
            record
        )

        output_record[
            "correct_answer"
        ] = answer

        output_records.append(
            output_record
        )

    print()
    print(
        "=== Sampled prefix GT matching ==="
    )
    print(
        f"Input prefixes       : "
        f"{len(input_records)}"
    )
    print(
        f"Successfully matched : "
        f"{len(output_records)}"
    )
    print(
        f"Missing GT           : "
        f"{len(missing)}"
    )
    print(
        f"Ambiguous GT         : "
        f"{len(ambiguous)}"
    )

    if missing:
        print()
        print(
            "=== Missing GT examples ==="
        )

        for item in missing[:10]:
            print()
            print(
                "prefix_id:",
                item["prefix_id"],
            )
            print(
                "image:",
                item["image"],
            )
            print(
                "same-image candidates:",
                item[
                    "same_image_count"
                ],
            )
            print(
                "question:",
                item[
                    "question"
                ][:500],
            )

            if item[
                "candidate_questions"
            ]:
                print(
                    "first candidate question:",
                    item[
                        "candidate_questions"
                    ][0][:500],
                )
                print(
                    "first candidate answer:",
                    item[
                        "candidate_answers"
                    ][0],
                )

    if ambiguous:
        print()
        print(
            "=== Ambiguous GT examples ==="
        )

        for item in ambiguous[:10]:
            print()
            print(
                "prefix_id:",
                item["prefix_id"],
            )
            print(
                "image:",
                item["image"],
            )
            print(
                "answers:",
                item["answers"],
            )
            print(
                "question:",
                item[
                    "question"
                ][:500],
            )

    # Strict mode:
    # every sampled prefix must have one unique GT.
    if missing or ambiguous:
        raise ValueError(
            "Ground-truth matching failed: "
            f"missing={len(missing)}, "
            f"ambiguous={len(ambiguous)}"
        )

    if (
        len(output_records)
        != args.expected_prefixes
    ):
        raise ValueError(
            f"Expected "
            f"{args.expected_prefixes} "
            f"matched prefixes, got "
            f"{len(output_records)}"
        )

    prefix_ids = [
        record["prefix_id"]
        for record in output_records
    ]

    if (
        len(prefix_ids)
        != len(set(prefix_ids))
    ):
        raise ValueError(
            "Duplicate prefix_id detected "
            "in output"
        )

    atomic_write_jsonl(
        output_records,
        output_path,
    )

    print()
    print(
        "=== MathV360K GT attachment ==="
    )
    print(
        f"Prefixes       : "
        f"{len(output_records)}"
    )
    print(
        f"GT attached    : "
        f"{len(output_records)}"
    )
    print(
        f"Output         : "
        f"{output_path}"
    )
    print()
    print(
        "[PASS] MathV360K ground truth "
        "attached successfully."
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