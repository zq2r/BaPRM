#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{lineno}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(
                    f"Expected an object at {path}:{lineno}, "
                    f"got {type(item).__name__}"
                )
            records.append(item)
        return records

    if isinstance(obj, list):
        if not all(isinstance(item, dict) for item in obj):
            raise TypeError(f"Not every record in {path} is a JSON object")
        return obj

    if isinstance(obj, dict):
        for key in ("data", "items", "records"):
            value = obj.get(key)
            if isinstance(value, list) and all(
                isinstance(item, dict) for item in value
            ):
                return value

    raise TypeError(
        f"Expected a JSON array or JSONL objects in {path}, "
        f"got {type(obj).__name__}"
    )


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def preview(value: Any, limit: int = 120) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def resolve_image(
    raw_path: Any,
    data_file: Path,
    image_root: Path | None,
) -> Path | None:
    if raw_path is None or not str(raw_path).strip():
        return None

    path = Path(str(raw_path))
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                Path.cwd() / path,
                data_file.parent / path,
            ]
        )
        if image_root is not None:
            candidates.append(image_root / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def describe(values: list[int]) -> str:
    if not values:
        return "empty"

    values = sorted(values)

    def percentile(q: float) -> int:
        index = round((len(values) - 1) * q)
        return values[index]

    return (
        f"min={values[0]}, "
        f"median={statistics.median(values)}, "
        f"p90={percentile(0.90)}, "
        f"max={values[-1]}"
    )


def audit_seed(
    records: list[dict[str, Any]],
    path: Path,
    image_root: Path | None,
) -> dict[str, int]:
    errors = Counter()
    key_counts = Counter()
    question_counts = Counter()

    for item in records:
        key_counts.update(item.keys())

        question = normalize_text(item.get("question"))
        if not question:
            errors["missing_question"] += 1
        else:
            question_counts[question] += 1

        if not normalize_text(item.get("correct_answer")):
            errors["missing_correct_answer"] += 1

        raw_image = item.get("image_path") or item.get("image")
        if not raw_image:
            errors["missing_image_field"] += 1
        elif resolve_image(raw_image, path, image_root) is None:
            errors["unresolved_image_path"] += 1

    print("\n=== Seed dataset ===")
    print(f"File: {path}")
    print(f"Records: {len(records)}")
    print(f"Common keys: {key_counts.most_common(15)}")
    print(
        "Duplicate normalized questions: "
        f"{sum(count - 1 for count in question_counts.values() if count > 1)}"
    )
    print(f"Errors: {dict(errors)}")

    return dict(errors)


def is_binary_label(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return float(value) in (0.0, 1.0)
    return False


def audit_annotation(
    records: list[dict[str, Any]],
    path: Path,
    image_root: Path | None,
    show: int,
) -> tuple[dict[str, int], Counter[str]]:
    errors = Counter()
    key_counts = Counter()
    question_counts: Counter[str] = Counter()
    trajectory_counts: list[int] = []
    step_counts: list[int] = []
    label_values = Counter()

    for item in records:
        key_counts.update(item.keys())

        question = normalize_text(item.get("question"))
        if not question:
            errors["missing_question"] += 1
        else:
            question_counts[question] += 1

        raw_image = item.get("image_path") or item.get("image")
        if not raw_image:
            errors["missing_image_field"] += 1
        elif resolve_image(raw_image, path, image_root) is None:
            errors["unresolved_image_path"] += 1

        solutions = item.get("solutions_splits")
        labels = item.get("labels")

        if not isinstance(solutions, list):
            errors["solutions_splits_not_list"] += 1
            continue

        if not isinstance(labels, list):
            errors["labels_not_list"] += 1
            continue

        trajectory_counts.append(len(solutions))

        if len(solutions) != len(labels):
            errors["trajectory_label_length_mismatch"] += 1

        for label in labels:
            label_values[str(label)] += 1
            if not is_binary_label(label):
                errors["non_binary_label"] += 1

        for trajectory in solutions:
            if not isinstance(trajectory, list):
                errors["trajectory_not_step_list"] += 1
                continue

            step_counts.append(len(trajectory))

            if not trajectory:
                errors["empty_trajectory"] += 1
                continue

            for step in trajectory:
                if not isinstance(step, str):
                    errors["non_string_step"] += 1
                elif not step.strip():
                    errors["empty_step"] += 1

    print("\n=== Rollout annotation ===")
    print(f"File: {path}")
    print(f"Records: {len(records)}")
    print(f"Common keys: {key_counts.most_common(15)}")
    print(f"Trajectories per question: {describe(trajectory_counts)}")
    print(f"Elements per trajectory: {describe(step_counts)}")
    print(f"Label values: {dict(label_values)}")
    print(
        "Duplicate normalized questions: "
        f"{sum(count - 1 for count in question_counts.values() if count > 1)}"
    )
    print(f"Errors: {dict(errors)}")

    print(f"\n=== First {min(show, len(records))} annotation samples ===")
    for index, item in enumerate(records[:show]):
        solutions = item.get("solutions_splits")
        labels = item.get("labels")

        first_trajectory = (
            solutions[0]
            if isinstance(solutions, list)
            and solutions
            and isinstance(solutions[0], list)
            else []
        )

        print(f"\nSample {index}")
        print(f"  id: {item.get('id')}")
        print(f"  question: {preview(item.get('question'))}")
        print(f"  image_path: {item.get('image_path') or item.get('image')}")
        print(
            "  num_trajectories: "
            f"{len(solutions) if isinstance(solutions, list) else 'invalid'}"
        )
        print(
            "  num_labels: "
            f"{len(labels) if isinstance(labels, list) else 'invalid'}"
        )
        print(f"  labels[:8]: {labels[:8] if isinstance(labels, list) else labels}")
        print(f"  first_trajectory_length: {len(first_trajectory)}")
        if first_trajectory:
            print(f"  first_element: {preview(first_trajectory[0])}")
            print(f"  last_element: {preview(first_trajectory[-1])}")

    return dict(errors), question_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local MathVision seed and rollout annotation files."
    )
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    seed_path = args.seed.resolve()
    annotation_path = args.annotation.resolve()
    image_root = args.image_root.resolve() if args.image_root else None

    seed_records = load_records(seed_path)
    annotation_records = load_records(annotation_path)

    seed_errors = audit_seed(seed_records, seed_path, image_root)
    annotation_errors, annotation_questions = audit_annotation(
        annotation_records,
        annotation_path,
        image_root,
        max(0, args.show),
    )

    seed_questions = Counter(
        normalize_text(item.get("question"))
        for item in seed_records
        if normalize_text(item.get("question"))
    )

    unmatched_annotation = sum(
        count
        for question, count in annotation_questions.items()
        if question not in seed_questions
    )
    seed_without_annotation = sum(
        count
        for question, count in seed_questions.items()
        if question not in annotation_questions
    )

    print("\n=== Cross-file alignment ===")
    print(f"Seed questions absent from annotation: {seed_without_annotation}")
    print(f"Annotation questions absent from seed: {unmatched_annotation}")

    critical_errors = (
        sum(seed_errors.values())
        + sum(annotation_errors.values())
        + unmatched_annotation
    )

    if critical_errors:
        print(
            f"\n[FAIL] Found {critical_errors} structural/path issues. "
            "Resolve them before generating calibration prefixes."
        )
        sys.exit(1)

    print(
        "\n[PASS] Seed and rollout annotation are structurally consistent. "
        "The next step is constructing the fixed prefix manifest."
    )


if __name__ == "__main__":
    main()
