#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


ANSWER_RE = re.compile(
    r"<\s*answer\b[^>]*>(.*?)</\s*answer\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
LEADING_DECISION_RE = re.compile(
    r"^\s*(yes|no)\b",
    flags=re.IGNORECASE,
)

JUDGE_PROMPT_VERSION = "beta_prm_mathvision_yes_no_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")

    records: list[dict[str, Any]] = []
    seen_prefix_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{lineno}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected object at {path}:{lineno}, "
                    f"got {type(record).__name__}"
                )

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {path}:{lineno}"
                )
            if prefix_id in seen_prefix_ids:
                raise ValueError(
                    f"Duplicate prefix_id in input: {prefix_id}"
                )

            seen_prefix_ids.add(prefix_id)
            records.append(record)

    if not records:
        raise ValueError(f"Input contains no records: {path}")

    return records


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_text(value: Any) -> str:
    text = str(value or "")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_final_answer(
    continuation: dict[str, Any],
) -> tuple[str, str]:
    """
    Extract the terminal answer without using correctness information.

    Priority:
      1. Last complete <answer>...</answer> block in raw_text.
      2. Last parsed segment returned by the generator.
      3. No answer: return an empty answer. This is a generator failure and
         is labeled incorrect without calling the judge.
    """
    raw_text = normalize_text(continuation.get("raw_text"))
    matches = ANSWER_RE.findall(raw_text)

    if matches:
        answer = normalize_text(matches[-1])
        if answer:
            return answer, "xml_answer"

    segments = continuation.get("segments")
    if isinstance(segments, list) and segments:
        last_segment = segments[-1]
        if isinstance(last_segment, str):
            answer = normalize_text(last_segment)
            if answer:
                return answer, "last_segment"

    return "", "missing_final_answer"


def build_judge_prompt(
    *,
    question: str,
    correct_answer: str,
    model_answer: str,
) -> str:
    # This follows the BetaPRM MathVision correctness-judge prompt.
    return f"""You are given a question, the correct answer and a model's answer.
Please determine if the model's answer matches the correct answer. Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the model's answer and the correct answer. Only the correctness of the model's answer matters. Return only "Yes" if the model's answer is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question: {question}
--------------------------------
Correct Answer: {correct_answer}
--------------------------------
Model's Answer: {model_answer}
--------------------------------"""


def parse_yes_no(content: Any) -> int | None:
    text = normalize_text(content)
    match = LEADING_DECISION_RE.match(text)
    if not match:
        return None

    decision = match.group(1).lower()
    return 1 if decision == "yes" else 0


class JudgeClient:
    def __init__(
        self,
        *,
        api_base: str,
        model_name: str,
        timeout: float,
        retries: int,
        max_tokens: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            # Do not route localhost requests through proxy variables.
            session.trust_env = False
            self._local.session = session
        return session

    def list_models(self) -> dict[str, Any]:
        response = self._session().get(
            f"{self.api_base}/models",
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(
                f"/models returned {type(data).__name__}, expected object"
            )
        return data

    def judge(
        self,
        *,
        question: str,
        correct_answer: str,
        model_answer: str,
    ) -> dict[str, Any]:
        prompt = build_judge_prompt(
            question=question,
            correct_answer=correct_answer,
            model_answer=model_answer,
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        last_error: str | None = None
        raw_responses: list[str] = []

        for attempt in range(1, self.retries + 1):
            try:
                response = self._session().post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: "
                        f"{response.text[:1000]}"
                    )

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                content_text = normalize_text(content)
                raw_responses.append(content_text)

                label = parse_yes_no(content_text)
                if label is not None:
                    return {
                        "label": label,
                        "judge_status": "ok",
                        "judge_raw_response": content_text,
                        "judge_attempts": attempt,
                        "judge_error": None,
                    }

                last_error = (
                    "Judge response did not begin with Yes or No: "
                    f"{content_text!r}"
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.retries:
                time.sleep(min(10.0, 2.0 ** (attempt - 1)))

        return {
            "label": None,
            "judge_status": "failed",
            "judge_raw_response": (
                raw_responses[-1] if raw_responses else None
            ),
            "judge_attempts": self.retries,
            "judge_error": last_error,
        }


def load_completed_prefixes(
    *,
    output_path: Path,
    expected_fingerprint: str,
) -> set[str]:
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
                    f"{output_path}:{lineno}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected object at {output_path}:{lineno}"
                )

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {output_path}:{lineno}"
                )

            if prefix_id in completed:
                raise ValueError(
                    f"Duplicate prefix_id in output: {prefix_id}"
                )

            fingerprint = record.get("judge_fingerprint")
            if fingerprint != expected_fingerprint:
                raise ValueError(
                    "Existing output uses a different judge configuration.\n"
                    f"File: {output_path}\n"
                    f"Line: {lineno}\n"
                    f"Expected: {expected_fingerprint}\n"
                    f"Found: {fingerprint}\n"
                    "Use a new output path or pass --overwrite."
                )

            judgments = record.get("judgments")
            if not isinstance(judgments, list):
                raise ValueError(
                    f"Missing judgments at {output_path}:{lineno}"
                )

            completed.add(prefix_id)

    return completed


def validate_prefix_record(record: dict[str, Any]) -> None:
    prefix_id = record.get("prefix_id")
    prefix_record = record.get("prefix_record")
    continuations = record.get("continuations")

    if not isinstance(prefix_id, str) or not prefix_id:
        raise ValueError("Input record has no valid prefix_id")

    if not isinstance(prefix_record, dict):
        raise ValueError(f"{prefix_id}: missing prefix_record")

    question = prefix_record.get("question")
    correct_answer = prefix_record.get("correct_answer")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{prefix_id}: missing question")

    if (
        not isinstance(correct_answer, str)
        or not correct_answer.strip()
    ):
        raise ValueError(f"{prefix_id}: missing correct_answer")

    if not isinstance(continuations, list) or not continuations:
        raise ValueError(f"{prefix_id}: no continuations")


def judge_one_prefix(
    *,
    record: dict[str, Any],
    client: JudgeClient,
    judge_fingerprint: str,
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    validate_prefix_record(record)

    prefix_id = record["prefix_id"]
    prefix_record = record["prefix_record"]
    question = normalize_text(prefix_record["question"])
    correct_answer = normalize_text(prefix_record["correct_answer"])
    continuations = record["continuations"]

    judgments: list[dict[str, Any]] = []

    for continuation_index, continuation in enumerate(continuations):
        if not isinstance(continuation, dict):
            raise TypeError(
                f"{prefix_id}: continuation {continuation_index} "
                "is not an object"
            )

        continuation_id = continuation.get("continuation_id")
        if (
            not isinstance(continuation_id, str)
            or not continuation_id
        ):
            continuation_id = (
                f"{prefix_id}__mc{continuation_index:02d}"
            )

        model_answer, extraction_method = extract_final_answer(
            continuation
        )

        if not model_answer:
            # A normally completed generation with no extractable answer is a
            # policy failure, not a judge infrastructure failure.
            result = {
                "label": 0,
                "judge_status": "not_called_missing_answer",
                "judge_raw_response": None,
                "judge_attempts": 0,
                "judge_error": None,
            }
        else:
            result = client.judge(
                question=question,
                correct_answer=correct_answer,
                model_answer=model_answer,
            )

        judgments.append(
            {
                "continuation_id": continuation_id,
                "continuation_index": continuation_index,
                "model_answer": model_answer,
                "answer_extraction_method": extraction_method,
                **result,
            }
        )

    valid_labels = [
        item["label"]
        for item in judgments
        if item["label"] in (0, 1)
    ]
    judge_failed = sum(
        item["judge_status"] == "failed"
        for item in judgments
    )
    correct_count = sum(label == 1 for label in valid_labels)
    incorrect_count = sum(label == 0 for label in valid_labels)

    return {
        "schema_version": 1,
        "prefix_id": prefix_id,
        "prefix_record": prefix_record,
        "source_generation_fingerprint": record.get(
            "generation_fingerprint"
        ),
        "judge_fingerprint": judge_fingerprint,
        "judge_config": judge_config,
        "judgments": judgments,
        "summary": {
            "expected_continuations": len(continuations),
            "valid_judgments": len(valid_labels),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "judge_failed": judge_failed,
            "success_prob": (
                correct_count / len(valid_labels)
                if valid_labels
                else None
            ),
            "complete": (
                len(valid_labels) == len(continuations)
                and judge_failed == 0
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge generated MathVision prefix continuations using an "
            "OpenAI-compatible Qwen judge server."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default="http://127.0.0.1:8888/v1",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen2.5-32B-Instruct",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of prefixes judged concurrently.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Exclusive end index; 0 means end of input.",
    )
    parser.add_argument(
        "--max-prefixes",
        type=int,
        default=0,
        help="Additional limit after slicing; 0 means none.",
    )
    parser.add_argument("--fsync-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.retries <= 0:
        raise ValueError("--retries must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.end_index < 0:
        raise ValueError("--end-index cannot be negative")
    if args.max_prefixes < 0:
        raise ValueError("--max-prefixes cannot be negative")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    records = load_jsonl(input_path)

    end_index = (
        len(records)
        if args.end_index == 0
        else min(args.end_index, len(records))
    )
    if args.start_index >= end_index:
        raise ValueError(
            f"Invalid range [{args.start_index}, {end_index}) "
            f"for {len(records)} input records"
        )

    target_records = records[args.start_index:end_index]
    if args.max_prefixes > 0:
        target_records = target_records[: args.max_prefixes]

    client = JudgeClient(
        api_base=args.judge_api_base,
        model_name=args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
    )

    models_response = client.list_models()
    served_model_ids = {
        str(item.get("id"))
        for item in models_response.get("data", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    if (
        served_model_ids
        and args.judge_model not in served_model_ids
    ):
        raise RuntimeError(
            f"Requested judge model {args.judge_model!r} is not served. "
            f"Available models: {sorted(served_model_ids)}"
        )

    judge_config: dict[str, Any] = {
        "judge_api_base": args.judge_api_base.rstrip("/"),
        "judge_model": args.judge_model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
        "answer_extraction": (
            "last_xml_answer_then_last_generator_segment"
        ),
        "missing_answer_policy": "incorrect_without_judge_call",
    }
    judge_fingerprint = config_fingerprint(judge_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed = load_completed_prefixes(
        output_path=output_path,
        expected_fingerprint=judge_fingerprint,
    )
    pending = [
        record
        for record in target_records
        if record["prefix_id"] not in completed
    ]

    print("=== Judge server ===")
    print(f"API base: {args.judge_api_base}")
    print(f"Judge model: {args.judge_model}")
    print(f"Served models: {sorted(served_model_ids)}")

    print("\n=== Judgment job ===")
    print(f"Input prefix records: {len(records)}")
    print(f"Target records: {len(target_records)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Workers: {args.workers}")
    print(f"Judge fingerprint: {judge_fingerprint}")
    print(f"Output: {output_path}")

    if not pending:
        print("\n[PASS] No pending prefixes.")
        return

    started = time.time()
    newly_written = 0
    completed_jobs = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures: dict[Future[dict[str, Any]], str] = {}

            for record in pending:
                future = executor.submit(
                    judge_one_prefix,
                    record=record,
                    client=client,
                    judge_fingerprint=judge_fingerprint,
                    judge_config=judge_config,
                )
                futures[future] = record["prefix_id"]

            for future in as_completed(futures):
                prefix_id = futures[future]

                try:
                    output_record = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed while judging prefix {prefix_id}"
                    ) from exc

                output_file.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()

                newly_written += 1
                completed_jobs += 1

                if (
                    args.fsync_every > 0
                    and newly_written % args.fsync_every == 0
                ):
                    os.fsync(output_file.fileno())

                summary = output_record["summary"]
                elapsed = time.time() - started
                avg_seconds = elapsed / completed_jobs

                print(
                    f"[{completed_jobs}/{len(pending)}] "
                    f"{prefix_id}: "
                    f"K={summary['correct']}/"
                    f"{summary['valid_judgments']}, "
                    f"failed={summary['judge_failed']}, "
                    f"avg={avg_seconds:.2f}s/prefix",
                    flush=True,
                )

        output_file.flush()
        os.fsync(output_file.fileno())

    print(
        f"\n[PASS] Wrote judgments for "
        f"{newly_written} prefixes."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise