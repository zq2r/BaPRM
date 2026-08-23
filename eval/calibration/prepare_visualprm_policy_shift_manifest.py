#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("datasets"),
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

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            rec = json.loads(line)

            prefix_id = rec["prefix_id"]
            question = rec["question"]
            prefix_steps = rec["prefix_steps"]

            image = rec["image"]

            if not isinstance(image, str):
                raise ValueError(
                    f"{prefix_id}: multi-image sample is not supported"
                )

            # Original VisualPRM image path:
            # VisualPRM400K-v1.1-Raw/MathV360K/...
            #
            # Actual local path:
            # datasets/VisualPRM400K-v1.1-Raw/MathV360K/...
            image_path = image_root / image

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{prefix_id}: image not found: {image_path}"
                )

            mc_correct = int(rec["mc_correct"])
            mc_total = int(rec["mc_total"])
            source_prob = float(
                rec["target_success_prob"]
            )

            if mc_total <= 0:
                raise ValueError(
                    f"{prefix_id}: invalid mc_total={mc_total}"
                )

            expected_prob = mc_correct / mc_total

            if abs(source_prob - expected_prob) > 1e-12:
                raise ValueError(
                    f"{prefix_id}: source probability != K/N"
                )

            out = dict(rec)

            # Fields required by continuation generation.
            out["benchmark"] = "MathV360K"
            out["image_path"] = str(image_path)

            # Explicitly preserve the original VisualPRM MC target.
            out["source_mc_correct"] = mc_correct
            out["source_mc_total"] = mc_total
            out["source_success_prob"] = source_prob

            records.append(out)

    if len(records) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} prefixes, "
            f"got {len(records)}"
        )

    prefix_ids = [
        x["prefix_id"]
        for x in records
    ]

    if len(prefix_ids) != len(set(prefix_ids)):
        raise ValueError(
            "Duplicate prefix_id detected"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(output_path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(output_path)

    print("=== MathV360K policy-shift manifest ===")
    print(f"Prefixes        : {len(records)}")
    print(
        "Mean source true: "
        f"{sum(x['source_success_prob'] for x in records) / len(records):.6f}"
    )
    print(f"Output          : {output_path}")
    print(
        "[PASS] Policy-shift rollout manifest generated."
    )


if __name__ == "__main__":
    main()