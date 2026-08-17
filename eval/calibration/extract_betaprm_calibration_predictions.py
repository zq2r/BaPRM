#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_by_prefix(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    output: dict[str, dict[str, Any]] = {}
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

            prefix_id = record.get("prefix_id")
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Missing prefix_id at {path}:{lineno}"
                )
            if prefix_id in output:
                raise ValueError(
                    f"Duplicate prefix_id in MC labels: {prefix_id}"
                )
            output[prefix_id] = record

    if not output:
        raise ValueError(f"No records found in {path}")

    return output


def atomic_write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def require_probability(
    value: Any,
    *,
    name: str,
    prefix_id: str,
) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{prefix_id}: {name} must be numeric, got {value!r}"
        )

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            f"{prefix_id}: {name} is not finite: {value}"
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{prefix_id}: {name} is outside [0, 1]: {value}"
        )
    return value


def require_positive(
    value: Any,
    *,
    name: str,
    prefix_id: str,
) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{prefix_id}: {name} must be numeric, got {value!r}"
        )

    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{prefix_id}: {name} must be finite and positive, "
            f"got {value}"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one terminal BetaPRM prediction per calibration "
            "prefix and align it with compact Monte Carlo labels."
        )
    )
    parser.add_argument(
        "--evaluator-output",
        type=Path,
        default="/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/outputs/calibration/mathvision/betaprm_checkpoint1103/mathvision_prm_260719083116.json",
    )
    parser.add_argument(
        "--mc-labels",
        type=Path,
        default="outputs/calibration/mathvision/mc_labels_full_internvl3_n16.jsonl",
    )
    parser.add_argument("--output", type=Path, default="outputs/calibration/mathvision/predictions_betaprm_checkpoint1103.jsonl")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="BetaPRM-InternVL3-8B")
    parser.add_argument("--checkpoint", type=str, default="/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/log/beta-InternVL3-8B-visualprm400k/checkpoint-1103")
    parser.add_argument("--expected-prefixes", type=int, default=3513)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evaluator_path = args.evaluator_output.expanduser().resolve()
    labels_path = args.mc_labels.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else output_path.with_name(output_path.stem + ".summary.json")
    )

    for path in (output_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Use --overwrite to replace it."
            )

    evaluator_data = load_json(evaluator_path)
    if not isinstance(evaluator_data, list):
        raise TypeError(
            "Evaluator output must be a top-level JSON list"
        )

    labels_by_prefix = load_jsonl_by_prefix(labels_path)
    if len(labels_by_prefix) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} MC labels, "
            f"got {len(labels_by_prefix)}"
        )

    extracted: dict[str, dict[str, Any]] = {}

    for question_index, item in enumerate(evaluator_data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Evaluator item {question_index} is not an object"
            )

        prefix_ids = item.get("prefix_ids")
        solutions_splits = item.get("solutions_splits")
        prm_mu = item.get("prm_mu")
        prm_kappa = item.get("prm_kappa")
        prm_sigma = item.get("prm_sigma")

        fields = {
            "prefix_ids": prefix_ids,
            "solutions_splits": solutions_splits,
            "prm_mu": prm_mu,
            "prm_kappa": prm_kappa,
            "prm_sigma": prm_sigma,
        }
        for name, value in fields.items():
            if not isinstance(value, list):
                raise ValueError(
                    f"Evaluator item {question_index}: "
                    f"{name} is not a list"
                )

        lengths = {len(value) for value in fields.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Evaluator item {question_index}: per-prefix list "
                f"lengths do not match: "
                f"{ {name: len(value) for name, value in fields.items()} }"
            )

        for local_index, prefix_id in enumerate(prefix_ids):
            if not isinstance(prefix_id, str) or not prefix_id:
                raise ValueError(
                    f"Evaluator item {question_index}, prefix "
                    f"{local_index}: invalid prefix_id"
                )
            if prefix_id in extracted:
                raise ValueError(
                    f"Duplicate prefix_id in evaluator output: {prefix_id}"
                )
            if prefix_id not in labels_by_prefix:
                raise ValueError(
                    f"Evaluator prefix missing from MC labels: {prefix_id}"
                )

            steps = solutions_splits[local_index]
            mu_steps = prm_mu[local_index]
            kappa_steps = prm_kappa[local_index]
            sigma_steps = prm_sigma[local_index]

            if not isinstance(steps, list) or not steps:
                raise ValueError(
                    f"{prefix_id}: empty solutions_splits entry"
                )
            for name, values in (
                ("prm_mu", mu_steps),
                ("prm_kappa", kappa_steps),
                ("prm_sigma", sigma_steps),
            ):
                if not isinstance(values, list):
                    raise ValueError(
                        f"{prefix_id}: {name} entry is not a list"
                    )
                if len(values) != len(steps):
                    raise ValueError(
                        f"{prefix_id}: {name} has {len(values)} values "
                        f"for {len(steps)} steps"
                    )
                if not values:
                    raise ValueError(
                        f"{prefix_id}: {name} is empty"
                    )

            pred_mu = require_probability(
                mu_steps[-1],
                name="terminal prm_mu",
                prefix_id=prefix_id,
            )
            pred_kappa = require_positive(
                kappa_steps[-1],
                name="terminal prm_kappa",
                prefix_id=prefix_id,
            )
            pred_sigma = require_probability(
                sigma_steps[-1],
                name="terminal prm_sigma",
                prefix_id=prefix_id,
            )

            expected_sigma = math.sqrt(
                pred_mu * (1.0 - pred_mu)
                / (pred_kappa + 1.0)
            )
            if not math.isclose(
                pred_sigma,
                expected_sigma,
                # Evaluator tensors are stored from bfloat16, so allow
                # normal quantization error when recomputing in float64.
                rel_tol=2e-2,
                abs_tol=2e-3,
            ):
                raise ValueError(
                    f"{prefix_id}: stored sigma={pred_sigma} does not "
                    f"match sqrt(mu*(1-mu)/(kappa+1))="
                    f"{expected_sigma}"
                )

            label = labels_by_prefix[prefix_id]
            target_prob = require_probability(
                label.get("success_prob"),
                name="success_prob",
                prefix_id=prefix_id,
            )
            mc_correct = int(label["mc_correct"])
            mc_total = int(label["mc_total"])

            if mc_total <= 0 or not 0 <= mc_correct <= mc_total:
                raise ValueError(
                    f"{prefix_id}: invalid MC counts "
                    f"K={mc_correct}, N={mc_total}"
                )
            if not math.isclose(
                target_prob,
                mc_correct / mc_total,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{prefix_id}: target probability is not K/N"
                )

            if len(steps) != int(label["prefix_step_count"]):
                raise ValueError(
                    f"{prefix_id}: evaluator has {len(steps)} steps, "
                    f"MC label expects {label['prefix_step_count']}"
                )

            extracted[prefix_id] = {
                "schema_version": 1,
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "prefix_id": prefix_id,
                "question_id": label["question_id"],
                "source_trajectory_index": label[
                    "source_trajectory_index"
                ],
                "prefix_step_count": label["prefix_step_count"],
                "prefix_relative_position": label[
                    "prefix_relative_position"
                ],
                "mc_correct": mc_correct,
                "mc_total": mc_total,
                "target_success_prob": target_prob,
                "pred_prob": pred_mu,
                "pred_mu": pred_mu,
                "pred_kappa": pred_kappa,
                "pred_sigma": pred_sigma,
                "signed_error": pred_mu - target_prob,
                "absolute_error": abs(pred_mu - target_prob),
            }

    expected_ids = set(labels_by_prefix)
    actual_ids = set(extracted)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)

    if missing or extra:
        raise ValueError(
            "Evaluator/label prefix mismatch: "
            f"missing={len(missing)}, extra={len(extra)}; "
            f"missing examples={missing[:5]}, extra examples={extra[:5]}"
        )
    if len(extracted) != args.expected_prefixes:
        raise ValueError(
            f"Expected {args.expected_prefixes} extracted predictions, "
            f"got {len(extracted)}"
        )

    ordered_records = [
        extracted[prefix_id]
        for prefix_id in labels_by_prefix
    ]

    mean_target = sum(
        record["target_success_prob"]
        for record in ordered_records
    ) / len(ordered_records)
    mean_prediction = sum(
        record["pred_prob"]
        for record in ordered_records
    ) / len(ordered_records)
    mean_abs_error = sum(
        record["absolute_error"]
        for record in ordered_records
    ) / len(ordered_records)
    mean_kappa = sum(
        record["pred_kappa"]
        for record in ordered_records
    ) / len(ordered_records)
    mean_sigma = sum(
        record["pred_sigma"]
        for record in ordered_records
    ) / len(ordered_records)

    atomic_write_jsonl(ordered_records, output_path)

    summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "evaluator_output": str(evaluator_path),
        "mc_labels": str(labels_path),
        "output": str(output_path),
        "counts": {
            "questions": len(evaluator_data),
            "prefixes": len(ordered_records),
        },
        "means": {
            "target_success_prob": mean_target,
            "pred_prob": mean_prediction,
            "signed_error": mean_prediction - mean_target,
            "absolute_error": mean_abs_error,
            "pred_kappa": mean_kappa,
            "pred_sigma": mean_sigma,
        },
        "ranges": {
            "pred_prob": [
                min(record["pred_prob"] for record in ordered_records),
                max(record["pred_prob"] for record in ordered_records),
            ],
            "pred_kappa": [
                min(record["pred_kappa"] for record in ordered_records),
                max(record["pred_kappa"] for record in ordered_records),
            ],
            "pred_sigma": [
                min(record["pred_sigma"] for record in ordered_records),
                max(record["pred_sigma"] for record in ordered_records),
            ],
        },
    }
    atomic_write_json(summary, summary_path)

    print("=== BetaPRM calibration predictions ===")
    print(f"Questions: {len(evaluator_data)}")
    print(f"Prefixes: {len(ordered_records)}")
    print(f"Mean target: {mean_target:.6f}")
    print(f"Mean prediction: {mean_prediction:.6f}")
    print(
        "Mean signed error: "
        f"{mean_prediction - mean_target:+.6f}"
    )
    print(f"Mean absolute error: {mean_abs_error:.6f}")
    print(f"Mean kappa: {mean_kappa:.6f}")
    print(f"Mean sigma: {mean_sigma:.6f}")
    print(
        "Prediction range: "
        f"[{summary['ranges']['pred_prob'][0]:.6f}, "
        f"{summary['ranges']['pred_prob'][1]:.6f}]"
    )
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print("\n[PASS] BetaPRM predictions extracted and aligned.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise