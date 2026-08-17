#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
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
                    f"Expected object at {path}:{lineno}, "
                    f"got {type(item).__name__}"
                )
            records.append(item)

    return records


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def preview(value: Any, limit: int = 500) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a prefix-continuation smoke-test JSONL file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-prefixes", type=int, default=10)
    parser.add_argument("--expected-n", type=int, default=4)
    parser.add_argument("--show", type=int, default=2)
    args = parser.parse_args()

    records = load_jsonl(args.input.resolve())
    errors: list[str] = []
    warnings: list[str] = []

    prefix_ids: set[str] = set()
    continuation_ids: set[str] = set()
    models: set[str] = set()
    fingerprints: set[str] = set()
    continuation_counts: list[int] = []
    token_lengths: list[int] = []
    empty_count = 0
    duplicate_raw_count = 0
    missing_prompt_count = 0
    selected_ignored_count = 0

    for record_index, record in enumerate(records):
        prefix_id = record.get("prefix_id")
        if not isinstance(prefix_id, str) or not prefix_id:
            errors.append(f"record {record_index}: invalid prefix_id")
            continue

        if prefix_id in prefix_ids:
            errors.append(f"duplicate prefix_id: {prefix_id}")
        prefix_ids.add(prefix_id)

        prefix_record = record.get("prefix_record")
        if not isinstance(prefix_record, dict):
            errors.append(f"{prefix_id}: prefix_record is not an object")
            continue

        if prefix_record.get("prefix_id") != prefix_id:
            errors.append(
                f"{prefix_id}: outer and inner prefix_id do not match"
            )

        question = prefix_record.get("question")
        prefix_steps = prefix_record.get("prefix_steps")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{prefix_id}: invalid question")
        if (
            not isinstance(prefix_steps, list)
            or not prefix_steps
            or any(
                not isinstance(step, str) or not step.strip()
                for step in prefix_steps
            )
        ):
            errors.append(f"{prefix_id}: invalid prefix_steps")

        config = record.get("generation_config")
        if not isinstance(config, dict):
            errors.append(f"{prefix_id}: missing generation_config")
        else:
            model = config.get("policy_model")
            if model is not None:
                models.add(str(model))

            if config.get("consume_response_field") != "raw":
                errors.append(
                    f"{prefix_id}: consume_response_field is not raw"
                )
            if float(config.get("oversample_factor", -1)) != 1.0:
                errors.append(
                    f"{prefix_id}: oversample_factor is not 1.0"
                )

        fingerprint = record.get("generation_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            errors.append(f"{prefix_id}: missing generation_fingerprint")
        else:
            fingerprints.add(fingerprint)

        diagnostics = record.get("server_diagnostics")
        if not isinstance(diagnostics, dict):
            errors.append(f"{prefix_id}: missing server_diagnostics")
        elif diagnostics.get("selected_outputs_ignored") is True:
            selected_ignored_count += 1
        else:
            errors.append(
                f"{prefix_id}: selected_outputs_ignored is not true"
            )

        prompt = record.get("user_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            missing_prompt_count += 1
        else:
            if isinstance(question, str):
                short_question = normalize_text(question)[:80]
                if short_question and short_question not in normalize_text(prompt):
                    warnings.append(
                        f"{prefix_id}: question text was not found verbatim "
                        "in user_prompt"
                    )

            if isinstance(prefix_steps, list):
                for step_index, step in enumerate(prefix_steps):
                    normalized_step = normalize_text(step)
                    if (
                        normalized_step
                        and normalized_step[:80]
                        not in normalize_text(prompt)
                    ):
                        warnings.append(
                            f"{prefix_id}: prefix step {step_index + 1} "
                            "was not found verbatim in user_prompt"
                        )

        continuations = record.get("continuations")
        if not isinstance(continuations, list):
            errors.append(f"{prefix_id}: continuations is not a list")
            continue

        continuation_counts.append(len(continuations))
        if len(continuations) != args.expected_n:
            errors.append(
                f"{prefix_id}: expected {args.expected_n} continuations, "
                f"got {len(continuations)}"
            )

        raw_counter: Counter[str] = Counter()

        for continuation_index, continuation in enumerate(continuations):
            if not isinstance(continuation, dict):
                errors.append(
                    f"{prefix_id}: continuation {continuation_index} "
                    "is not an object"
                )
                continue

            continuation_id = continuation.get("continuation_id")
            if not isinstance(continuation_id, str) or not continuation_id:
                errors.append(
                    f"{prefix_id}: continuation {continuation_index} "
                    "has no valid continuation_id"
                )
            elif continuation_id in continuation_ids:
                errors.append(
                    f"duplicate continuation_id: {continuation_id}"
                )
            else:
                continuation_ids.add(continuation_id)

            raw_text = continuation.get("raw_text", "")
            segments = continuation.get("segments", [])
            token_len = continuation.get("token_len", 0)

            if not isinstance(raw_text, str):
                errors.append(
                    f"{prefix_id}: {continuation_id} raw_text is not a string"
                )
                raw_text = str(raw_text)

            if not isinstance(segments, list) or any(
                not isinstance(segment, str) for segment in segments
            ):
                errors.append(
                    f"{prefix_id}: {continuation_id} has invalid segments"
                )

            try:
                token_len = int(token_len)
            except (TypeError, ValueError):
                errors.append(
                    f"{prefix_id}: {continuation_id} has invalid token_len"
                )
                token_len = 0

            token_lengths.append(token_len)
            raw_counter[normalize_text(raw_text)] += 1

            is_empty = bool(continuation.get("is_empty_generation"))
            actual_empty = (
                not raw_text.strip()
                and (
                    not isinstance(segments, list)
                    or not any(
                        isinstance(segment, str) and segment.strip()
                        for segment in segments
                    )
                )
            )
            if is_empty != actual_empty:
                errors.append(
                    f"{prefix_id}: {continuation_id} empty flag mismatch"
                )
            if actual_empty:
                empty_count += 1

        duplicate_raw_count += sum(
            count - 1
            for text, count in raw_counter.items()
            if text and count > 1
        )

    if len(records) != args.expected_prefixes:
        errors.append(
            f"expected {args.expected_prefixes} prefix records, "
            f"got {len(records)}"
        )

    if len(models) != 1:
        errors.append(f"expected one policy model, got {sorted(models)}")
    if len(fingerprints) != 1:
        errors.append(
            f"expected one generation fingerprint, got {len(fingerprints)}"
        )
    if selected_ignored_count != len(records):
        errors.append(
            "not every record confirms that selected outputs were ignored"
        )

    print("=== Continuation audit ===")
    print(f"Input: {args.input.resolve()}")
    print(f"Prefix records: {len(records)}")
    print(f"Continuation records: {len(continuation_ids)}")
    print(f"Continuation counts: {Counter(continuation_counts)}")
    print(f"Models: {sorted(models)}")
    print(f"Fingerprint count: {len(fingerprints)}")
    print(f"Empty generations: {empty_count}")
    print(f"Duplicate raw samples: {duplicate_raw_count}")
    print(f"Missing user_prompt: {missing_prompt_count}")
    if token_lengths:
        print(
            "Token lengths: "
            f"min={min(token_lengths)}, "
            f"max={max(token_lengths)}, "
            f"mean={sum(token_lengths) / len(token_lengths):.2f}"
        )

    if warnings:
        print(f"\n=== Warnings ({len(warnings)}) ===")
        for warning in warnings[:20]:
            print(f"- {warning}")
        if len(warnings) > 20:
            print(f"... and {len(warnings) - 20} more")

    show_count = min(max(0, args.show), len(records))
    print(f"\n=== First {show_count} records ===")
    for record in records[:show_count]:
        prefix = record["prefix_record"]
        print("\n" + "=" * 100)
        print(f"PREFIX ID: {record['prefix_id']}")
        print(f"QUESTION:\n{prefix.get('question')}")
        print("\nPREFIX STEPS:")
        for index, step in enumerate(prefix.get("prefix_steps", []), start=1):
            print(f"[{index}] {step}")

        print("\nUSER PROMPT:")
        print(preview(record.get("user_prompt"), 1500))

        for continuation in record.get("continuations", []):
            print(
                f"\n--- {continuation.get('continuation_id')} "
                f"(tokens={continuation.get('token_len')}) ---"
            )
            print("segments:")
            print(json.dumps(
                continuation.get("segments"),
                ensure_ascii=False,
                indent=2,
            ))
            print("raw_text:")
            print(preview(continuation.get("raw_text"), 2000))

    if errors:
        print(f"\n[FAIL] Found {len(errors)} issue(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\n[PASS] Continuation output is structurally valid.")
    print(
        "Manually confirm that the displayed outputs genuinely continue "
        "from the provided prefix rather than restarting the solution."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise