#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


EXPECTED_SIZES = {
    "MathVision": 304,
    "MathVerse": 788,
    "MathVista": 1000,
    "MMStar": 1500,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_samples(data, name):
    if not data:
        raise ValueError(f"{name}: empty dataset.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{i}] is not a dict.")
        if "question" not in item:
            raise ValueError(f"{name}[{i}] missing `question`.")
        if "correct_answer" not in item:
            raise ValueError(f"{name}[{i}] missing `correct_answer`.")
        if "image_path" not in item and "image" not in item:
            raise ValueError(
                f"{name}[{i}] missing both `image_path` and `image`."
            )


def normalize_version(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def select_dataset(name, data):
    if name == "MathVerse":
        # The local standardized MathVerse seed contains the complete
        # official testmini set (788 problems x 5 versions = 3940),
        # but problem_version was removed during conversion.
        #
        # The original MathVerse ordering is preserved:
        # every problem occupies five consecutive sample IDs, and
        # Vision Only is the 5th version:
        #   5, 10, 15, ..., 3940.

        if len(data) != 3940:
            raise ValueError(
                f"MathVerse source has {len(data)} samples, "
                "expected the complete 3940-sample testmini set."
            )

        ids = [int(item["id"]) for item in data]

        expected_ids = list(range(1, 3941))

        if ids != expected_ids:
            raise ValueError(
                "MathVerse IDs are not exactly 1..3940 in order. "
                "Cannot safely recover the Vision Only split by ID."
            )

        selected = [
            item
            for item in data
            if int(item["id"]) % 5 == 0
        ]

        if len(selected) != 788:
            raise ValueError(
                f"Recovered {len(selected)} Vision Only samples, "
                "expected 788."
            )

    else:
        selected = list(data)

    expected = EXPECTED_SIZES[name]
    if len(selected) != expected:
        raise ValueError(
            f"{name}: selected {len(selected)} samples, "
            f"expected {expected}. "
            "Please check that the source seed dataset is the intended "
            "official split."
        )

    return selected


def process_dataset(name, input_path, output_root, meta_dir):
    data = load_json(input_path)
    validate_samples(data, name)

    selected = select_dataset(name, data)
    validate_samples(selected, name)

    output_path = Path(output_root) / name / "seed_dataset.json"
    save_json(selected, output_path)

    selected_indices = []
    selected_ids = {id(item) for item in selected}
    for i, item in enumerate(data):
        if id(item) in selected_ids:
            selected_indices.append(i)

    meta = {
        "benchmark": name,
        "source": os.path.abspath(input_path),
        "source_size": len(data),
        "sample_size": len(selected),
        "protocol": (
            "vision_only"
            if name == "MathVerse"
            else "full_supplied_official_split"
        ),
        "selected_indices": selected_indices,
    }

    meta_path = Path(meta_dir) / f"{name.lower()}_selection_meta.json"
    save_json(meta, meta_path)

    print(
        f"{name:12s}: source={len(data):4d} "
        f"-> selected={len(selected):4d} "
        f"-> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the final TTS evaluation sets: "
            "MathVision testmini (304), MathVerse Vision Only (788), "
            "MathVista testmini (1000), and MMStar public set (1500)."
        )
    )
    parser.add_argument(
        "--mathvision-input",
        default="datasets/MathVision/seed_dataset.json",
    )
    parser.add_argument(
        "--mathverse-input",
        default="datasets/MathVerse/seed_dataset.json",
    )
    parser.add_argument(
        "--mathvista-input",
        default="datasets/MathVista/seed_dataset.json",
    )
    parser.add_argument(
        "--mmstar-input",
        default="datasets/MMStar/seed_dataset.json",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/TTS_FINAL",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    meta_dir = output_root / "selection_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    process_dataset(
        "MathVision",
        args.mathvision_input,
        output_root,
        meta_dir,
    )
    process_dataset(
        "MathVerse",
        args.mathverse_input,
        output_root,
        meta_dir,
    )
    process_dataset(
        "MathVista",
        args.mathvista_input,
        output_root,
        meta_dir,
    )
    process_dataset(
        "MMStar",
        args.mmstar_input,
        output_root,
        meta_dir,
    )

    print("\n[PASS] Final TTS seed datasets prepared.")


if __name__ == "__main__":
    main()
