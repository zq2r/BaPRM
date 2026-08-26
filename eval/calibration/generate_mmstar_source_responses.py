#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full source-policy reasoning responses for MMStar. "
            "No judging is performed in this stage."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/MMStar/seed_dataset.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/calibration/mmstar/"
            "source_responses_internvl3_8b.jsonl"
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("datasets/MMStar"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../../model/InternVL3-8B",
    )

    parser.add_argument(
        "--num-responses",
        type=int,
        default=4,
        help="Number of independent full source responses per question.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Exclusive end index; 0 means all remaining samples.",
    )

    parser.add_argument(
        "--fsync-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def make_cot_prompt(question: str) -> str:
    """
    Keep the same response protocol used by the existing
    BetaPRM rollout-construction code.
    """
    return f"""
You are an AI assistant that must strictly follow the response protocol below.

1) Reasoning phase:
- Analyze the question thoroughly and consider plausible solution strategies.
- Proceed step by step.
- Wrap each reasoning step in <step>...</step>.
- Keep each step concise, factual, and focused on the question and the image.
- Do NOT include any text outside <step> tags in this phase.

2) Final answer:
- After the reasoning steps, provide the final answer wrapped in exactly one
  <answer>...</answer> tag.
- The final answer must be self-contained.
- Output nothing outside the required tags.

Question:
{question}
""".strip()


def load_seed_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            f"Expected JSON list in {path}, got {type(data).__name__}"
        )

    if not data:
        raise ValueError(f"Dataset is empty: {path}")

    seen_ids: set[str] = set()

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Dataset item {i} is not an object."
            )

        sample_id = item.get("id")
        question = item.get("question")
        correct_answer = item.get("correct_answer")
        image_path = item.get("image_path")

        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Invalid id at item {i}")

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate id: {sample_id}")

        seen_ids.add(sample_id)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Invalid question for {sample_id}"
            )

        if not isinstance(correct_answer, str):
            raise ValueError(
                f"Invalid correct_answer for {sample_id}"
            )

        if not isinstance(image_path, str) or not image_path:
            raise ValueError(
                f"Invalid image_path for {sample_id}"
            )

    return data


def resolve_image_path(
    raw_path: str,
    image_root: Path,
) -> Path:
    raw = Path(raw_path).expanduser()

    if raw.is_absolute():
        path = raw
    else:
        path = image_root / raw

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    return path


def normalize_segments(
    result: Any,
) -> list[list[str]]:
    """
    LanguageModel.generate_results() should return:
        list[list[str]]

    Each inner list corresponds to:
        [reasoning step 1, ..., final answer]
    """

    if not isinstance(result, list):
        raise TypeError(
            "generate_results() returned "
            f"{type(result).__name__}, expected list"
        )

    normalized: list[list[str]] = []

    for response_idx, segments in enumerate(result):
        if segments is None:
            segments = []

        if not isinstance(segments, list):
            raise TypeError(
                f"Response {response_idx} is "
                f"{type(segments).__name__}, expected list"
            )

        cleaned: list[str] = []

        for segment in segments:
            if segment is None:
                continue

            text = str(segment).strip()

            if text:
                cleaned.append(text)

        normalized.append(cleaned)

    return normalized


def load_completed(
    output_path: Path,
    expected_num_responses: int,
) -> set[str]:
    """
    Safe JSONL resume:
    only records with the expected number of source responses
    are considered complete.
    """
    completed: set[str] = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid existing output at "
                    f"{output_path}:{lineno}"
                ) from exc

            sample_id = record.get("id")
            responses = record.get("responses")

            if not isinstance(sample_id, str):
                raise ValueError(
                    f"Missing id at {output_path}:{lineno}"
                )

            if sample_id in completed:
                raise ValueError(
                    f"Duplicate existing id: {sample_id}"
                )

            if (
                not isinstance(responses, list)
                or len(responses) != expected_num_responses
            ):
                raise ValueError(
                    f"Incomplete record for {sample_id}: "
                    f"expected {expected_num_responses} responses"
                )

            completed.add(sample_id)

    return completed


def main() -> None:
    args = parse_args()

    if args.num_responses <= 0:
        raise ValueError("--num-responses must be > 0")

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    if args.end_index < 0:
        raise ValueError("--end-index must be >= 0")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()

    model_path = str(
        Path(args.model).expanduser().resolve()
    )

    data = load_seed_dataset(input_path)

    end_index = (
        len(data)
        if args.end_index == 0
        else min(args.end_index, len(data))
    )

    if args.start_index >= end_index:
        raise ValueError(
            f"Invalid range [{args.start_index}, {end_index})"
        )

    selected = data[args.start_index:end_index]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed = load_completed(
        output_path,
        expected_num_responses=args.num_responses,
    )

    pending = [
        item
        for item in selected
        if item["id"] not in completed
    ]

    print("===== MMStar source response generation =====")
    print(f"Input             : {input_path}")
    print(f"Dataset samples   : {len(data)}")
    print(
        f"Selected range    : "
        f"[{args.start_index}, {end_index})"
    )
    print(f"Target samples    : {len(selected)}")
    print(f"Already completed : {len(completed)}")
    print(f"Pending           : {len(pending)}")
    print(f"Responses/question: {args.num_responses}")
    print(f"Model             : {model_path}")
    print(f"Image root        : {image_root}")
    print(f"Output            : {output_path}")
    print()

    if not pending:
        print("[PASS] No pending samples.")
        return

    # Same local vLLM generator used by the existing
    # BetaPRM rollout builder.
    from data_pipeline.llm_utils import LanguageModel

    generator = LanguageModel(
        model=model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    newly_written = 0

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for item in tqdm(
            pending,
            desc="Generating MMStar source responses",
        ):
            sample_id = item["id"]
            question = item["question"]

            image_path = resolve_image_path(
                item["image_path"],
                image_root=image_root,
            )

            prompt = make_cot_prompt(question)

            try:
                raw_results = generator.generate_results(
                    prompt,
                    image_path=str(image_path),
                    num_copies=args.num_responses,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Generation failed for {sample_id}"
                ) from exc

            responses = normalize_segments(
                raw_results
            )

            if len(responses) != args.num_responses:
                raise RuntimeError(
                    f"{sample_id}: expected "
                    f"{args.num_responses} responses, "
                    f"got {len(responses)}"
                )

            empty_responses = sum(
                int(len(x) == 0)
                for x in responses
            )

            record = {
                "schema_version": 1,
                "id": sample_id,
                "index": item.get("index"),
                "question": question,
                "correct_answer": item[
                    "correct_answer"
                ],
                "image_path": item["image_path"],
                "category": item.get("category"),
                "l2_category": item.get(
                    "l2_category"
                ),
                "generation_config": {
                    "model": model_path,
                    "num_responses": args.num_responses,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_new_tokens":
                        args.max_new_tokens,
                },
                "responses": [
                    {
                        "response_index": response_idx,
                        "segments": segments,
                        "num_segments": len(segments),
                        "is_empty":
                            len(segments) == 0,
                    }
                    for response_idx, segments
                    in enumerate(responses)
                ],
            }

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()

            newly_written += 1

            if (
                args.fsync_every > 0
                and newly_written
                % args.fsync_every
                == 0
            ):
                os.fsync(fout.fileno())

            if empty_responses:
                print(
                    f"\n[WARN] {sample_id}: "
                    f"{empty_responses}/"
                    f"{args.num_responses} "
                    "empty responses",
                    flush=True,
                )

        fout.flush()
        os.fsync(fout.fileno())

    print()
    print(
        f"[PASS] Generated source responses for "
        f"{newly_written} MMStar samples."
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.")
        raise SystemExit(130)