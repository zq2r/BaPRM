#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

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
                    f"Expected a JSON object at {path}:{lineno}, "
                    f"got {type(record).__name__}"
                )

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing or invalid prefix_id at {path}:{lineno}"
                )
            if prefix_id in seen_prefix_ids:
                raise ValueError(
                    f"Duplicate prefix_id in manifest: {prefix_id}"
                )

            seen_prefix_ids.add(prefix_id)
            records.append(record)

    if not records:
        raise ValueError(f"Manifest contains no records: {path}")

    return records


def stable_seed(global_seed: int, prefix_id: str) -> int:
    """Derive a deterministic positive 31-bit seed for one prefix."""
    payload = f"{global_seed}|{prefix_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)

    return unique


def resolve_image_path(
    raw_image_path: Any,
    *,
    repo_root: Path,
    image_root: Path | None,
    manifest_path: Path,
) -> Path:
    """
    Resolve an image path from the prefix manifest.

    Supported manifest formats include:
      - extracted_images/4.png
      - datasets/MathVision/extracted_images/4.png
      - an absolute path
    """
    if raw_image_path is None or not str(raw_image_path).strip():
        raise ValueError("Manifest record has no image_path")

    raw = Path(str(raw_image_path).strip()).expanduser()

    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates: list[Path] = []

        # Explicit dataset root has highest priority.
        if image_root is not None:
            candidates.append(image_root / raw)

            try:
                relative_to_mathvision = raw.relative_to(
                    Path("datasets") / "MathVision"
                )
            except ValueError:
                pass
            else:
                candidates.append(
                    image_root / relative_to_mathvision
                )

        candidates.extend(
            [
                Path.cwd() / raw,
                repo_root / raw,
                manifest_path.parent / raw,
                repo_root / "datasets" / "MathVision" / raw,
            ]
        )

    checked = _deduplicate_paths(candidates)

    for candidate in checked:
        if candidate.is_file():
            return candidate

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Image does not exist. Checked:\n"
        f"{checked_text}"
    )


def get_server_health(
    session: requests.Session,
    server_url: str,
    timeout: float,
) -> dict[str, Any]:
    url = f"{server_url.rstrip('/')}/healthz"
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(
            f"Health endpoint returned {type(data).__name__}, "
            "expected an object"
        )
    if data.get("ok") is not True:
        raise RuntimeError(f"Generator health check failed: {data}")
    if not data.get("model"):
        raise RuntimeError(f"Health response has no model field: {data}")

    return data


def normalize_raw_outputs(
    response_data: Any,
    expected_count: int,
) -> list[dict[str, Any]]:
    """
    Validate /generate_continue output and return only raw model samples.

    response.selected is deliberately ignored because it is produced by
    post-generation selection/deduplication.
    """
    if not isinstance(response_data, dict):
        raise TypeError(
            "Generator response must be an object, got "
            f"{type(response_data).__name__}"
        )

    raw = response_data.get("raw")
    raw_count = response_data.get("raw_count")

    if not isinstance(raw, list):
        raise ValueError(
            "Generator response has no raw list. "
            "The request must set return_raw=true."
        )
    if raw_count is not None and int(raw_count) != expected_count:
        raise ValueError(
            f"raw_count mismatch: expected {expected_count}, "
            f"got {raw_count}"
        )
    if len(raw) != expected_count:
        raise ValueError(
            f"Raw output length mismatch: expected {expected_count}, "
            f"got {len(raw)}"
        )

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(
                f"raw[{index}] is {type(item).__name__}, "
                "expected an object"
            )

        raw_text = item.get("raw_text", "")
        if raw_text is None:
            raw_text = ""
        elif not isinstance(raw_text, str):
            raw_text = str(raw_text)

        segments = item.get("segments", [])
        if segments is None:
            segments = []
        if not isinstance(segments, list):
            raise TypeError(f"raw[{index}].segments is not a list")
        if any(not isinstance(segment, str) for segment in segments):
            raise TypeError(
                f"raw[{index}].segments contains a non-string value"
            )

        token_len = item.get("token_len", 0)
        try:
            token_len = int(token_len)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"raw[{index}].token_len is invalid: {token_len!r}"
            ) from exc

        normalized.append(
            {
                "raw_text": raw_text,
                "segments": segments,
                "token_len": token_len,
                "is_empty_generation": (
                    not raw_text.strip()
                    and not any(segment.strip() for segment in segments)
                ),
            }
        )

    return normalized


def request_continuations(
    *,
    session: requests.Session,
    server_url: str,
    payload: dict[str, Any],
    expected_count: int,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    url = f"{server_url.rstrip('/')}/generate_continue"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                url,
                json=payload,
                timeout=timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Generator returned HTTP {response.status_code}: "
                    f"{response.text[:2000]}"
                )

            response_data = response.json()
            raw_outputs = normalize_raw_outputs(
                response_data,
                expected_count=expected_count,
            )
            return response_data, raw_outputs

        except Exception as exc:
            last_error = exc

            if attempt == retries:
                break

            delay = min(30.0, 2.0 ** (attempt - 1))
            print(
                f"[WARN] Request attempt {attempt}/{retries} failed: "
                f"{exc}\nRetrying in {delay:.1f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    assert last_error is not None
    raise RuntimeError(
        f"Generation failed after {retries} attempts"
    ) from last_error


def load_completed_prefixes(
    *,
    output_path: Path,
    expected_fingerprint: str,
    expected_n_generations: int,
) -> set[str]:
    """Load completed prefixes for safe resume."""
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
                    f"Existing output is invalid at "
                    f"{output_path}:{lineno}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected an object at {output_path}:{lineno}"
                )

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {output_path}:{lineno}"
                )

            if prefix_id in completed:
                raise ValueError(
                    f"Duplicate prefix_id in existing output: "
                    f"{prefix_id}"
                )

            fingerprint = record.get("generation_fingerprint")
            if fingerprint != expected_fingerprint:
                raise ValueError(
                    "Existing output was generated with a different "
                    "configuration.\n"
                    f"File: {output_path}\n"
                    f"Line: {lineno}\n"
                    f"Expected fingerprint: {expected_fingerprint}\n"
                    f"Found fingerprint: {fingerprint}\n"
                    "Use a new output path or pass --overwrite."
                )

            continuations = record.get("continuations")
            if (
                not isinstance(continuations, list)
                or len(continuations) != expected_n_generations
            ):
                raise ValueError(
                    f"Incomplete continuation record at "
                    f"{output_path}:{lineno}; expected "
                    f"{expected_n_generations} outputs"
                )

            completed.add(prefix_id)

    return completed


def validate_manifest_record(record: dict[str, Any]) -> None:
    prefix_id = record.get("prefix_id")
    question = record.get("question")
    prefix_steps = record.get("prefix_steps")

    if not isinstance(prefix_id, str) or not prefix_id:
        raise ValueError("Manifest record has no valid prefix_id")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Invalid question in {prefix_id}")

    if not isinstance(prefix_steps, list) or not prefix_steps:
        raise ValueError(f"Invalid prefix_steps in {prefix_id}")

    if any(
        not isinstance(step, str) or not step.strip()
        for step in prefix_steps
    ):
        raise ValueError(
            f"prefix_steps contains an empty/non-string step "
            f"in {prefix_id}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate unfiltered raw Monte Carlo continuations for a "
            "MathVision prefix manifest using the ACA InternVL server."
        )
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Normally datasets/MathVision.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:18080",
    )

    parser.add_argument("--n-generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--global-seed", type=int, default=42)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Exclusive end index; 0 means end of manifest.",
    )
    parser.add_argument(
        "--max-prefixes",
        type=int,
        default=0,
        help="Additional limit after start/end slicing; 0 means none.",
    )

    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--fsync-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_generations <= 0:
        raise ValueError("--n-generations must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.end_index < 0:
        raise ValueError("--end-index cannot be negative")
    if args.max_prefixes < 0:
        raise ValueError("--max-prefixes cannot be negative")
    if args.retries <= 0:
        raise ValueError("--retries must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("--top-k must be -1 or a positive integer")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root is not None
        else None
    )

    manifest = load_jsonl(manifest_path)

    end_index = (
        len(manifest)
        if args.end_index == 0
        else min(args.end_index, len(manifest))
    )
    if args.start_index >= end_index:
        raise ValueError(
            f"Invalid range [{args.start_index}, {end_index}) "
            f"for a manifest with {len(manifest)} records"
        )

    target_records = manifest[args.start_index:end_index]
    if args.max_prefixes > 0:
        target_records = target_records[: args.max_prefixes]

    session = requests.Session()
    session.trust_env = False

    health = get_server_health(
        session=session,
        server_url=args.server_url,
        timeout=args.health_timeout,
    )

    policy_model = str(health["model"])
    tensor_parallel_size = health.get("tp")

    generation_config: dict[str, Any] = {
        "policy_model": policy_model,
        "tensor_parallel_size": tensor_parallel_size,
        "endpoint": "/generate_continue",
        "n_generations": args.n_generations,
        "oversample_factor": 1.0,
        "consume_response_field": "raw",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "global_seed": args.global_seed,
    }
    fingerprint = config_fingerprint(generation_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed = load_completed_prefixes(
        output_path=output_path,
        expected_fingerprint=fingerprint,
        expected_n_generations=args.n_generations,
    )

    pending = [
        record
        for record in target_records
        if record["prefix_id"] not in completed
    ]

    print("\n=== Generator server ===")
    print(f"Server: {args.server_url}")
    print(f"Health: {health}")
    print(f"Policy model: {policy_model}")

    print("\n=== Continuation job ===")
    print(f"Manifest records: {len(manifest)}")
    print(f"Selected range: [{args.start_index}, {end_index})")
    print(f"Target records: {len(target_records)}")
    print(f"Already completed in output: {len(completed)}")
    print(f"Pending in target range: {len(pending)}")
    print(f"N_MC: {args.n_generations}")
    print(f"Image root: {image_root}")
    print(f"Generation fingerprint: {fingerprint}")
    print(f"Output: {output_path}")
    print(
        "Continuation source: response.raw "
        "(response.selected is deliberately ignored)"
    )

    if not pending:
        print("\n[PASS] No pending prefixes.")
        return

    started = time.time()
    newly_written = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for job_index, prefix_record in enumerate(pending, start=1):
            validate_manifest_record(prefix_record)

            prefix_id = prefix_record["prefix_id"]
            question = prefix_record["question"]
            prefix_steps = prefix_record["prefix_steps"]

            try:
                resolved_image = resolve_image_path(
                    prefix_record.get("image_path"),
                    repo_root=repo_root,
                    image_root=image_root,
                    manifest_path=manifest_path,
                )

                request_seed = stable_seed(
                    global_seed=args.global_seed,
                    prefix_id=prefix_id,
                )

                payload: dict[str, Any] = {
                    "question": question,
                    "image_path": str(resolved_image),
                    "prefix_steps": prefix_steps,
                    "extra_instruction": None,
                    "m": args.n_generations,
                    "oversample_factor": 1.0,
                    "sampling": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                        "repetition_penalty": args.repetition_penalty,
                    },
                    "seed": request_seed,
                    "return_raw": True,
                }

                response_data, raw_outputs = request_continuations(
                    session=session,
                    server_url=args.server_url,
                    payload=payload,
                    expected_count=args.n_generations,
                    timeout=args.request_timeout,
                    retries=args.retries,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Failed while processing prefix {prefix_id}"
                ) from exc

            continuations: list[dict[str, Any]] = []
            for continuation_index, raw_item in enumerate(raw_outputs):
                continuations.append(
                    {
                        "continuation_id": (
                            f"{prefix_id}__mc{continuation_index:02d}"
                        ),
                        "continuation_index": continuation_index,
                        "raw_text": raw_item["raw_text"],
                        "segments": raw_item["segments"],
                        "token_len": raw_item["token_len"],
                        "is_empty_generation": (
                            raw_item["is_empty_generation"]
                        ),
                    }
                )

            output_record: dict[str, Any] = {
                "schema_version": 2,
                "prefix_id": prefix_id,
                "prefix_record": prefix_record,
                "generation_fingerprint": fingerprint,
                "generation_config": generation_config,
                "request_seed": request_seed,
                "resolved_image_path": str(resolved_image),
                "user_prompt": response_data.get("user_prompt"),
                "server_diagnostics": {
                    "raw_count": response_data.get("raw_count"),
                    "kept_count": response_data.get("kept_count"),
                    "effective_completion_tokens": response_data.get(
                        "effective_completion_tokens"
                    ),
                    "actual_completion_tokens": response_data.get(
                        "actual_completion_tokens"
                    ),
                    "kept_indices": response_data.get("kept_indices"),
                    "selected_outputs_ignored": True,
                },
                "continuations": continuations,
            }

            output_file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            newly_written += 1
            if (
                args.fsync_every > 0
                and newly_written % args.fsync_every == 0
            ):
                os.fsync(output_file.fileno())

            elapsed = time.time() - started
            avg_seconds = elapsed / job_index
            total_tokens = sum(
                item["token_len"] for item in continuations
            )
            empty_count = sum(
                int(item["is_empty_generation"])
                for item in continuations
            )

            print(
                f"[{job_index}/{len(pending)}] {prefix_id}: "
                f"{len(continuations)} raw continuations, "
                f"{total_tokens} tokens, "
                f"{empty_count} empty, "
                f"avg {avg_seconds:.2f}s/prefix",
                flush=True,
            )

        output_file.flush()
        os.fsync(output_file.fileno())

    print(
        f"\n[PASS] Generated raw continuations for "
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