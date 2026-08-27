import argparse
import json
import os
import random
from pathlib import Path


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


def sample_dataset(data, n, seed):
    if n is None:
        # Keep the whole dataset.
        indices = list(range(len(data)))
    else:
        if len(data) < n:
            raise ValueError(
                f"Requested {n} samples, but dataset only has {len(data)}."
            )

        rng = random.Random(seed)

        # Sample indices first, then restore original dataset order.
        indices = sorted(rng.sample(range(len(data)), n))

    sampled = [data[i] for i in indices]

    return sampled, indices


def process_dataset(
    name,
    input_path,
    output_path,
    n,
    seed,
    meta_dir,
):
    data = load_json(input_path)
    validate_samples(data, name)

    sampled, indices = sample_dataset(
        data=data,
        n=n,
        seed=seed,
    )

    save_json(sampled, output_path)

    meta = {
        "benchmark": name,
        "source": os.path.abspath(input_path),
        "source_size": len(data),
        "sample_size": len(sampled),
        "seed": seed,
        "selected_indices": indices,
    }

    meta_path = Path(meta_dir) / f"{name.lower()}_selection_meta.json"
    save_json(meta, meta_path)

    print(
        f"{name:12s}: "
        f"source={len(data):4d} -> selected={len(sampled):4d} "
        f"-> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

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
        default="datasets/TTS",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    meta_dir = output_root / "selection_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # MathVision: use all examples.
    process_dataset(
        name="MathVision",
        input_path=args.mathvision_input,
        output_path=output_root / "MathVision" / "seed_dataset.json",
        n=None,
        seed=args.seed,
        meta_dir=meta_dir,
    )

    # Other benchmarks: fixed random sample of 300.
    process_dataset(
        name="MathVerse",
        input_path=args.mathverse_input,
        output_path=output_root / "MathVerse" / "seed_dataset.json",
        n=300,
        seed=args.seed,
        meta_dir=meta_dir,
    )

    process_dataset(
        name="MathVista",
        input_path=args.mathvista_input,
        output_path=output_root / "MathVista" / "seed_dataset.json",
        n=300,
        seed=args.seed,
        meta_dir=meta_dir,
    )

    process_dataset(
        name="MMStar",
        input_path=args.mmstar_input,
        output_path=output_root / "MMStar" / "seed_dataset.json",
        n=300,
        seed=args.seed,
        meta_dir=meta_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()