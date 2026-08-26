#!/usr/bin/env python3

import argparse
import json
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MMStar parquet into BetaPRM-compatible seed_dataset.json."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="datasets/MMStar/mmstar.parquet",
        help="Path to MMStar parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/MMStar",
        help="Output MMStar directory.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite already extracted images.",
    )
    return parser.parse_args()


def decode_image(value):
    """
    Decode the MMStar image field into a PIL RGB image.

    Supports:
      - PIL.Image.Image
      - bytes / bytearray / memoryview
      - HuggingFace Image-style dict: {"bytes": ..., "path": ...}
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(value))) as img:
            return img.convert("RGB")

    if isinstance(value, dict):
        image_bytes = value.get("bytes", None)
        image_path = value.get("path", None)

        if image_bytes is not None:
            with Image.open(BytesIO(bytes(image_bytes))) as img:
                return img.convert("RGB")

        if image_path:
            with Image.open(image_path) as img:
                return img.convert("RGB")

    raise TypeError(
        f"Unsupported MMStar image type: {type(value)}; "
        f"value preview: {repr(value)[:200]}"
    )


def normalize_answer(answer):
    answer = str(answer).strip().upper()

    if answer in {"A", "B", "C", "D"}:
        return answer

    # Be slightly robust to values such as "A." or "(A)".
    cleaned = (
        answer.replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace(":", "")
        .strip()
    )

    if cleaned in {"A", "B", "C", "D"}:
        return cleaned

    raise ValueError(f"Unexpected MMStar answer: {answer!r}")


def main():
    args = parse_args()

    parquet_path = Path(args.parquet).resolve()
    output_dir = Path(args.output_dir).resolve()

    image_dir = output_dir / "extracted_images"
    output_path = output_dir / "seed_dataset.json"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"MMStar parquet not found: {parquet_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MMStar from: {parquet_path}")

    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )

    print(f"Loaded MMStar: {len(dataset)} samples")

    if len(dataset) != 1500:
        print(
            f"[WARNING] Official MMStar contains 1500 samples, "
            f"but loaded {len(dataset)}."
        )

    samples = []
    seen_ids = set()

    for row_idx, row in enumerate(dataset):
        if "index" not in row:
            raise KeyError(
                f"Missing 'index' field at dataset row {row_idx}"
            )

        if "question" not in row:
            raise KeyError(
                f"Missing 'question' field at dataset row {row_idx}"
            )

        if "answer" not in row:
            raise KeyError(
                f"Missing 'answer' field at dataset row {row_idx}"
            )

        if "image" not in row:
            raise KeyError(
                f"Missing 'image' field at dataset row {row_idx}"
            )

        raw_idx = row["index"]

        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            idx = row_idx

        sample_id = f"mmstar-{idx:04d}"

        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate MMStar sample id: {sample_id}"
            )
        seen_ids.add(sample_id)

        question = str(row["question"]).strip()

        if not question:
            raise ValueError(
                f"Empty question at index={idx}"
            )

        correct_answer = normalize_answer(row["answer"])

        image_raw = row["image"]

        if image_raw is None:
            raise ValueError(
                f"Missing image at index={idx}"
            )

        image_name = f"{idx:04d}.png"
        image_path = image_dir / image_name

        if args.overwrite_images or not image_path.exists():
            try:
                image = decode_image(image_raw)
                image.save(image_path, format="PNG")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to decode/save image for "
                    f"dataset row={row_idx}, index={idx}"
                ) from exc

        # image_path is relative to datasets/MMStar.
        relative_image_path = f"extracted_images/{image_name}"

        item = {
            "id": sample_id,
            "index": idx,
            "question": question,
            "correct_answer": correct_answer,
            "image_path": relative_image_path,
        }

        # Preserve useful MMStar metadata when available.
        if row.get("category") is not None:
            item["category"] = row["category"]

        if row.get("l2_category") is not None:
            item["l2_category"] = row["l2_category"]

        if row.get("meta_info") is not None:
            item["meta_info"] = row["meta_info"]

        samples.append(item)

        if (row_idx + 1) % 100 == 0:
            print(
                f"Processed {row_idx + 1}/{len(dataset)} samples"
            )

    num_images = len(list(image_dir.glob("*.png")))

    if len(samples) != len(dataset):
        raise RuntimeError(
            f"Sample count mismatch: "
            f"{len(samples)} != {len(dataset)}"
        )

    missing_images = []

    for item in samples:
        p = output_dir / item["image_path"]
        if not p.exists():
            missing_images.append(str(p))

    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} extracted images are missing. "
            f"First missing file: {missing_images[0]}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            samples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("===== MMStar preparation complete =====")
    print(f"Samples       : {len(samples)}")
    print(f"PNG images    : {num_images}")
    print(f"Image dir     : {image_dir}")
    print(f"Seed dataset  : {output_path}")
    print()

    print("First sample:")
    print(
        json.dumps(
            samples[0],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()