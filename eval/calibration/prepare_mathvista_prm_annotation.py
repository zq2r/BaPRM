#!/usr/bin/env python3

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prefixes", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; use --overwrite"
        )

    records = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} prefixes, "
            f"got {len(records)}"
        )

    # Group sampled prefixes belonging to the same original MathVista question.
    groups = OrderedDict()

    for rec in records:
        qid = rec["question_id"]

        if qid not in groups:
            groups[qid] = {
                "id": rec.get("question_index"),
                "question_id": qid,
                "question": rec["question"],
                "query_cot": rec["question"],
                "image_path": rec["image_path"],
                "image": rec["image"],
                "solutions_splits": [],
                "prefix_ids": [],
            }

        g = groups[qid]

        # Basic consistency checks.
        if g["question"] != rec["question"]:
            raise ValueError(f"{qid}: inconsistent question")
        if g["image_path"] != rec["image_path"]:
            raise ValueError(f"{qid}: inconsistent image_path")

        g["solutions_splits"].append(rec["prefix_steps"])
        g["prefix_ids"].append(rec["prefix_id"])

    output = list(groups.values())

    total_prefixes = sum(
        len(x["prefix_ids"])
        for x in output
    )

    if total_prefixes != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} grouped prefixes, "
            f"got {total_prefixes}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(args.output) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    tmp.replace(args.output)

    sizes = [len(x["prefix_ids"]) for x in output]

    print("=== MathVista PRM annotation ===")
    print(f"Input prefixes       : {len(records)}")
    print(f"Grouped questions    : {len(output)}")
    print(f"Output prefixes      : {total_prefixes}")
    print(
        "Prefixes/question    : "
        f"min={min(sizes)}, max={max(sizes)}, "
        f"mean={sum(sizes)/len(sizes):.3f}"
    )
    print(f"Output               : {args.output}")
    print("[PASS] MathVista PRM annotation generated.")


if __name__ == "__main__":
    main()