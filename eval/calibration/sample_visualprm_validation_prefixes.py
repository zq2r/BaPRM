#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import torch


PRM_TOKEN = "<prm>"
VISUALPRM_ROOT = "VisualPRM400K-v1.1-Raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample task-specific prefix-level evaluation data from "
            "the held-out split of VisualPRM400K."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default="datasets/Beta-Binomial-project/all_combined_beta_binom.jsonl",
        help="Input all_combined_beta_binom.jsonl.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="outputs/calibration/visualprm_val/geometry3k_val_prefixes_1000.jsonl",
        help="Output prefix-level JSONL.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="Geometry3K",
        help="VisualPRM source task, e.g. Geometry3K, ai2d, UniGeo.",
    )

    parser.add_argument(
        "--num-prefixes",
        type=int,
        default=1000,
        help="Number of prefixes to sample.",
    )

    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help=(
            "Fraction assigned to the first training split. "
            "Must match the training configuration."
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used to reproduce the original train/belief split.",
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used to sample prefixes within the requested task.",
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        default=16,
        help="Expected number of MC rollouts per prefix.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    return parser.parse_args()


def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            records.append(
                {
                    "source_index": line_idx,
                    "item": item,
                }
            )

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def reproduce_split(
    n_total: int,
    split_ratio: float,
    split_seed: int,
):
    """
    Reproduce the deterministic split used during training:

        perm = torch.randperm(N, seed)
        first round(ratio * N) -> ensemble/train split
        remaining             -> belief/held-out split
    """

    if not 0.0 < split_ratio < 1.0:
        raise ValueError(
            f"split_ratio must be in (0, 1), got {split_ratio}"
        )

    n_first = int(round(n_total * split_ratio))
    n_first = max(1, min(n_total - 1, n_first))

    generator = torch.Generator()
    generator.manual_seed(split_seed)

    permutation = torch.randperm(
        n_total,
        generator=generator,
    ).tolist()

    first_indices = permutation[:n_first]
    second_indices = permutation[n_first:]

    return first_indices, second_indices


def normalize_images(item):
    """
    image can be either:
        str
        list[str]
    """

    image = item.get("image")

    if isinstance(image, str):
        return [image]

    if isinstance(image, list):
        return [
            x
            for x in image
            if isinstance(x, str)
        ]

    return []


def extract_task(item):
    """
    Extract task name from canonical VisualPRM path.

    Example:

        VisualPRM400K-v1.1-Raw/Geometry3K/train/1832/img_diagram.png

    returns:

        Geometry3K

    For ambiguous relative paths such as:

        images/train-xxx-img0.png

    return None instead of guessing.
    """

    images = normalize_images(item)

    if not images:
        return None

    detected_tasks = set()

    for image_path in images:
        parts = [
            x
            for x in image_path.replace("\\", "/").split("/")
            if x
        ]

        if (
            len(parts) >= 2
            and parts[0] == VISUALPRM_ROOT
        ):
            detected_tasks.add(parts[1])

    if not detected_tasks:
        return None

    if len(detected_tasks) > 1:
        raise ValueError(
            f"One record contains multiple source tasks: "
            f"{sorted(detected_tasks)}"
        )

    return next(iter(detected_tasks))


def get_human_text(item):
    conversations = item.get("conversations")

    if not isinstance(conversations, list):
        raise ValueError("Missing or invalid conversations")

    for message in conversations:
        if (
            isinstance(message, dict)
            and message.get("from") == "human"
        ):
            value = message.get("value")

            if not isinstance(value, str):
                raise ValueError(
                    "Human conversation value must be a string"
                )

            return value

    raise ValueError("Cannot find human conversation")


def get_ratio_targets(item):
    conversations = item.get("conversations")

    if not isinstance(conversations, list):
        raise ValueError("Missing or invalid conversations")

    for message in conversations:
        if (
            isinstance(message, dict)
            and message.get("from") == "gpt"
        ):
            value = message.get("value")

            if not isinstance(value, list):
                raise ValueError(
                    "GPT conversation value must be a list"
                )

            return [float(x) for x in value]

    raise ValueError("Cannot find GPT target conversation")


def split_question_and_process(human_text):
    """
    Expected format:

        Question: ...
        ...
        Process: step 1<prm>

        step 2<prm>
        ...
    """

    marker = "Process:"

    if marker not in human_text:
        raise ValueError(
            "Cannot find 'Process:' in human prompt"
        )

    question_text, process_text = human_text.split(
        marker,
        1,
    )

    question_text = question_text.strip()

    if question_text.startswith("Question:"):
        question_text = question_text[
            len("Question:"):
        ].strip()

    process_text = process_text.strip()

    if not question_text:
        raise ValueError("Empty question")

    if not process_text:
        raise ValueError("Empty process")

    return question_text, process_text


def extract_process_steps(process_text):
    """
    Every <prm> corresponds to one supervised process position.
    """

    chunks = process_text.split(PRM_TOKEN)

    steps = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    if not steps:
        raise ValueError(
            "No process steps extracted from prompt"
        )

    return steps


def build_prefix_records(
    source_index,
    item,
    task,
    expected_n,
):
    human_text = get_human_text(item)

    question, process_text = split_question_and_process(
        human_text
    )

    process_steps = extract_process_steps(
        process_text
    )

    ratio_targets = get_ratio_targets(item)

    prm_counts = item.get("prm_counts")

    if not isinstance(prm_counts, dict):
        raise ValueError("Missing prm_counts")

    k_values = prm_counts.get("k")
    n_values = prm_counts.get("n")

    if not isinstance(k_values, list):
        raise ValueError("prm_counts.k must be a list")

    if not isinstance(n_values, list):
        raise ValueError("prm_counts.n must be a list")

    num_steps = len(process_steps)

    if len(k_values) != num_steps:
        raise ValueError(
            f"Number of K labels does not match number of "
            f"<prm> positions: "
            f"K={len(k_values)}, steps={num_steps}"
        )

    if len(n_values) != num_steps:
        raise ValueError(
            f"Number of N labels does not match number of "
            f"<prm> positions: "
            f"N={len(n_values)}, steps={num_steps}"
        )

    if len(ratio_targets) != num_steps:
        raise ValueError(
            f"Number of soft targets does not match number "
            f"of <prm> positions: "
            f"targets={len(ratio_targets)}, steps={num_steps}"
        )

    images = normalize_images(item)

    prefix_records = []

    for step_idx in range(num_steps):

        k = int(k_values[step_idx])
        n = int(n_values[step_idx])

        if n <= 0:
            raise ValueError(
                f"Invalid n={n} at step {step_idx}"
            )

        if not 0 <= k <= n:
            raise ValueError(
                f"Invalid K/N at step {step_idx}: {k}/{n}"
            )

        if expected_n is not None and n != expected_n:
            raise ValueError(
                f"Unexpected MC rollout number at "
                f"step {step_idx}: "
                f"expected {expected_n}, got {n}"
            )

        target_success_prob = k / n

        stored_ratio = float(
            ratio_targets[step_idx]
        )

        # Verify that original VisualPRM soft target
        # exactly corresponds to K/N.
        if abs(
            target_success_prob - stored_ratio
        ) > 1e-8:
            raise ValueError(
                f"K/N mismatch at source={source_index}, "
                f"step={step_idx}: "
                f"K/N={target_success_prob}, "
                f"stored={stored_ratio}"
            )

        prefix_steps = process_steps[
            : step_idx + 1
        ]

        # Keep a reconstructed text version as well.
        prefix_process_text = (
            f"{PRM_TOKEN}\n\n".join(prefix_steps)
            + PRM_TOKEN
        )

        prefix_id = (
            f"{task}_"
            f"{source_index:07d}_"
            f"step_{step_idx + 1:03d}"
        )

        record = {
            "prefix_id": prefix_id,

            "task": task,

            "source_trajectory_index": source_index,

            "question_id": (
                f"{task}_{source_index:07d}"
            ),

            "question": question,

            "image": (
                images[0]
                if len(images) == 1
                else images
            ),

            "prefix_steps": prefix_steps,

            "prefix_process_text": prefix_process_text,

            "prefix_step_count": step_idx + 1,

            "total_step_count": num_steps,

            "prefix_relative_position": (
                (step_idx + 1) / num_steps
            ),

            # Monte Carlo supervision
            "mc_correct": k,
            "mc_total": n,

            # This is the "real" value used later.
            "target_success_prob": (
                target_success_prob
            ),

            # Retain original label for audit.
            "original_ratio_target": (
                stored_ratio
            ),
        }

        prefix_records.append(record)

    return prefix_records


def atomic_write_jsonl(
    records,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = Path(
        str(output_path) + ".tmp"
    )

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp_path.replace(output_path)


def main():
    args = parse_args()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file does not exist: "
            f"{input_path}"
        )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            f"Use --overwrite to replace it."
        )

    if args.num_prefixes <= 0:
        raise ValueError(
            "--num-prefixes must be positive"
        )

    # -------------------------------------------------
    # 1. Load full VisualPRM dataset
    # -------------------------------------------------

    raw_records = load_jsonl(
        input_path
    )

    n_total = len(raw_records)

    # -------------------------------------------------
    # 2. Reproduce global 80/20 split FIRST
    # -------------------------------------------------

    first_indices, heldout_indices = (
        reproduce_split(
            n_total=n_total,
            split_ratio=args.split_ratio,
            split_seed=args.split_seed,
        )
    )

    heldout_set = set(
        heldout_indices
    )

    # -------------------------------------------------
    # 3. Filter requested task ONLY inside held-out set
    # -------------------------------------------------

    task_total_trajectories = 0
    task_heldout_trajectories = 0

    unknown_task_trajectories = 0
    skipped_trajectories = 0

    all_task_prefixes = []

    for dataset_idx, raw_record in enumerate(
        raw_records
    ):
        source_index = raw_record[
            "source_index"
        ]

        item = raw_record[
            "item"
        ]

        try:
            task = extract_task(item)
        except Exception as exc:
            unknown_task_trajectories += 1

            print(
                f"[WARN] Cannot identify task for "
                f"source_index={source_index}: {exc}"
            )
            continue

        if task is None:
            unknown_task_trajectories += 1
            continue

        if task == args.task:
            task_total_trajectories += 1

        # Important:
        # global split must be applied before task filtering.
        if dataset_idx not in heldout_set:
            continue

        if task != args.task:
            continue

        task_heldout_trajectories += 1

        try:
            prefix_records = (
                build_prefix_records(
                    source_index=source_index,
                    item=item,
                    task=task,
                    expected_n=args.expected_n,
                )
            )

        except Exception as exc:
            skipped_trajectories += 1

            print(
                f"[WARN] Skip "
                f"source_index={source_index}: "
                f"{exc}"
            )

            continue

        all_task_prefixes.extend(
            prefix_records
        )

    # -------------------------------------------------
    # 4. Check enough prefixes exist
    # -------------------------------------------------

    num_eligible = len(
        all_task_prefixes
    )

    if num_eligible < args.num_prefixes:
        raise ValueError(
            f"Task '{args.task}' has only "
            f"{num_eligible} eligible prefixes "
            f"in held-out split, but "
            f"{args.num_prefixes} were requested."
        )

    # -------------------------------------------------
    # 5. Uniformly sample PREFIXES
    # -------------------------------------------------

    rng = random.Random(
        args.sample_seed
    )

    selected = rng.sample(
        all_task_prefixes,
        args.num_prefixes,
    )

    # Stable ordering for reproducibility/debugging.
    selected.sort(
        key=lambda x: (
            x["source_trajectory_index"],
            x["prefix_step_count"],
        )
    )

    # -------------------------------------------------
    # 6. Sanity checks
    # -------------------------------------------------

    if len(selected) != args.num_prefixes:
        raise AssertionError(
            "Unexpected number of selected prefixes"
        )

    prefix_ids = [
        x["prefix_id"]
        for x in selected
    ]

    if len(prefix_ids) != len(
        set(prefix_ids)
    ):
        raise AssertionError(
            "Duplicate prefix IDs detected"
        )

    selected_tasks = {
        x["task"]
        for x in selected
    }

    if selected_tasks != {args.task}:
        raise AssertionError(
            f"Unexpected tasks in output: "
            f"{selected_tasks}"
        )

    for x in selected:
        expected = (
            x["mc_correct"]
            / x["mc_total"]
        )

        if abs(
            expected
            - x["target_success_prob"]
        ) > 1e-8:
            raise AssertionError(
                f"Target mismatch in "
                f"{x['prefix_id']}"
            )

    # -------------------------------------------------
    # 7. Save
    # -------------------------------------------------

    atomic_write_jsonl(
        selected,
        output_path,
    )

    # -------------------------------------------------
    # 8. Statistics
    # -------------------------------------------------

    target_values = [
        x["target_success_prob"]
        for x in selected
    ]

    unique_trajectories = len(
        {
            x["source_trajectory_index"]
            for x in selected
        }
    )

    mean_target = (
        sum(target_values)
        / len(target_values)
    )

    zero_target_rate = (
        sum(
            x == 0.0
            for x in target_values
        )
        / len(target_values)
    )

    one_target_rate = (
        sum(
            x == 1.0
            for x in target_values
        )
        / len(target_values)
    )

    print()
    print(
        "=== VisualPRM task-specific "
        "prefix sampling ==="
    )

    print(
        f"Requested task                  : "
        f"{args.task}"
    )

    print(
        f"Raw trajectories                : "
        f"{n_total}"
    )

    print(
        f"First split trajectories        : "
        f"{len(first_indices)}"
    )

    print(
        f"Held-out split trajectories     : "
        f"{len(heldout_indices)}"
    )

    print(
        f"Split ratio                     : "
        f"{args.split_ratio}"
    )

    print(
        f"Split seed                      : "
        f"{args.split_seed}"
    )

    print(
        f"Task trajectories (full data)   : "
        f"{task_total_trajectories}"
    )

    print(
        f"Task trajectories (held-out)    : "
        f"{task_heldout_trajectories}"
    )

    print(
        f"Unknown-task trajectories       : "
        f"{unknown_task_trajectories}"
    )

    print(
        f"Skipped task trajectories       : "
        f"{skipped_trajectories}"
    )

    print(
        f"Eligible task prefixes          : "
        f"{num_eligible}"
    )

    print(
        f"Selected prefixes               : "
        f"{len(selected)}"
    )

    print(
        f"Unique source trajectories      : "
        f"{unique_trajectories}"
    )

    print(
        f"Sample seed                     : "
        f"{args.sample_seed}"
    )

    print(
        f"Mean target success probability : "
        f"{mean_target:.6f}"
    )

    print(
        f"Target == 0 rate                : "
        f"{100.0 * zero_target_rate:.2f}%"
    )

    print(
        f"Target == 1 rate                : "
        f"{100.0 * one_target_rate:.2f}%"
    )

    print(
        f"Output                           : "
        f"{output_path}"
    )

    print()
    print(
        "[PASS] Task-specific prefix "
        "dataset generated successfully."
    )


if __name__ == "__main__":
    main()