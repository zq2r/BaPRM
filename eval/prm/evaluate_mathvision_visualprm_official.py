#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate official OpenGVLab VisualPRM-8B-v1_1 "
            "on the existing MathVision calibration prefixes."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="mathvision_visualprm_official.json",
    )

    parser.add_argument(
        "--expected-prefixes",
        type=int,
        default=3513,
    )

    parser.add_argument(
        "--max-num",
        type=int,
        default=12,
        help="Maximum number of dynamic image tiles.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(
                lambda img: (
                    img.convert("RGB")
                    if img.mode != "RGB"
                    else img
                )
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = (
            ratio[0] / ratio[1]
        )

        ratio_diff = abs(
            aspect_ratio
            - target_aspect_ratio
        )

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio

        elif ratio_diff == best_ratio_diff:
            if (
                area
                > 0.5
                * image_size
                * image_size
                * ratio[0]
                * ratio[1]
            ):
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=12,
    image_size=448,
    use_thumbnail=True,
):
    orig_width, orig_height = image.size

    aspect_ratio = (
        orig_width / orig_height
    )

    target_ratios = set(
        (i, j)
        for n in range(
            min_num,
            max_num + 1,
        )
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if (
            i * j <= max_num
            and i * j >= min_num
        )
    )

    target_ratios = sorted(
        target_ratios,
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = (
        find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
            image_size,
        )
    )

    target_width = (
        image_size
        * target_aspect_ratio[0]
    )

    target_height = (
        image_size
        * target_aspect_ratio[1]
    )

    blocks = (
        target_aspect_ratio[0]
        * target_aspect_ratio[1]
    )

    resized_img = image.resize(
        (
            target_width,
            target_height,
        )
    )

    processed_images = []

    for i in range(blocks):
        box = (
            (
                i
                % (
                    target_width
                    // image_size
                )
            )
            * image_size,
            (
                i
                // (
                    target_width
                    // image_size
                )
            )
            * image_size,
            (
                (
                    i
                    % (
                        target_width
                        // image_size
                    )
                )
                + 1
            )
            * image_size,
            (
                (
                    i
                    // (
                        target_width
                        // image_size
                    )
                )
                + 1
            )
            * image_size,
        )

        split_img = resized_img.crop(
            box
        )

        processed_images.append(
            split_img
        )

    assert (
        len(processed_images)
        == blocks
    )

    if (
        use_thumbnail
        and len(processed_images) != 1
    ):
        thumbnail_img = image.resize(
            (
                image_size,
                image_size,
            )
        )

        processed_images.append(
            thumbnail_img
        )

    return processed_images


def load_image(
    image_path: str,
    input_size: int,
    max_num: int,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    transform = build_transform(
        input_size=input_size
    )

    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )

    pixel_values = [
        transform(image)
        for image in images
    ]

    return torch.stack(
        pixel_values
    )


def pick_question_text(
    item: dict[str, Any],
) -> str:
    question = (
        item.get("query")
        or item.get("query_cot")
        or item.get("question")
        or ""
    )

    return str(question)


def pick_image_path(
    root: Path,
    item: dict[str, Any],
) -> Path:
    candidates = []

    raw_image_path = item.get(
        "image_path"
    )

    raw_image = item.get(
        "image"
    )

    if raw_image_path:
        p = Path(
            str(raw_image_path)
        )
        candidates.append(p)

    if raw_image:
        p = Path(
            str(raw_image)
        )

        candidates.append(
            root / p
        )

        candidates.append(
            root / p.name
        )

    if raw_image_path:
        p = Path(
            str(raw_image_path)
        )

        candidates.append(
            root / p
        )

        candidates.append(
            root / p.name
        )

    seen = set()

    for candidate in candidates:
        candidate = (
            candidate.expanduser()
        )

        key = str(candidate)

        if key in seen:
            continue

        seen.add(key)

        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate image for "
        f"id={item.get('id')}. "
        f"Tried: "
        f"{[str(x) for x in candidates]}"
    )


def validate_annotation(
    data: Any,
    expected_prefixes: int,
):
    if not isinstance(data, list):
        raise TypeError(
            "Annotation must be a "
            "top-level JSON list."
        )

    total_prefixes = 0
    seen_prefix_ids = set()

    for question_index, item in enumerate(
        data
    ):
        if not isinstance(item, dict):
            raise TypeError(
                f"Annotation item "
                f"{question_index} "
                f"is not an object."
            )

        prefix_ids = item.get(
            "prefix_ids"
        )

        solutions_splits = item.get(
            "solutions_splits"
        )

        if not isinstance(
            prefix_ids,
            list,
        ):
            raise ValueError(
                f"Item {question_index}: "
                "prefix_ids must be a list."
            )

        if not isinstance(
            solutions_splits,
            list,
        ):
            raise ValueError(
                f"Item {question_index}: "
                "solutions_splits must "
                "be a list."
            )

        if (
            len(prefix_ids)
            != len(solutions_splits)
        ):
            raise ValueError(
                f"Item {question_index}: "
                f"{len(prefix_ids)} "
                "prefix_ids but "
                f"{len(solutions_splits)} "
                "solutions_splits."
            )

        for local_index, (
            prefix_id,
            steps,
        ) in enumerate(
            zip(
                prefix_ids,
                solutions_splits,
            )
        ):
            if not isinstance(
                prefix_id,
                str,
            ):
                raise ValueError(
                    f"Item {question_index}, "
                    f"prefix {local_index}: "
                    "invalid prefix_id."
                )

            if prefix_id in seen_prefix_ids:
                raise ValueError(
                    "Duplicate prefix_id: "
                    f"{prefix_id}"
                )

            seen_prefix_ids.add(
                prefix_id
            )

            if (
                not isinstance(
                    steps,
                    list,
                )
                or not steps
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    "solutions_splits "
                    "must contain a "
                    "non-empty list of steps."
                )

            if not all(
                isinstance(step, str)
                for step in steps
            ):
                raise ValueError(
                    f"{prefix_id}: "
                    "all steps must be strings."
                )

        total_prefixes += len(
            prefix_ids
        )

    if (
        total_prefixes
        != expected_prefixes
    ):
        raise ValueError(
            f"Expected "
            f"{expected_prefixes} "
            f"prefixes, got "
            f"{total_prefixes}."
        )

    return total_prefixes


def get_local_indices(
    total_size: int,
    world_size: int,
    rank: int,
):
    shard_size = (
        total_size // world_size
    )

    remainder = (
        total_size % world_size
    )

    shard_sizes = [
        shard_size
        + int(r < remainder)
        for r in range(world_size)
    ]

    begin = sum(
        shard_sizes[:rank]
    )

    end = begin + shard_sizes[rank]

    return range(
        begin,
        end,
    )


def main():
    args = parse_args()

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    # Each rank independently evaluates a shard on one GPU.
    # Distributed communication is only needed for CPU-side
    # synchronization and object gathering, so use Gloo rather
    # than NCCL.
    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    if world_size > 1:
        dist.init_process_group(
            backend="gloo"
        )

    random.seed(
        args.seed + rank
    )

    torch.manual_seed(
        args.seed + rank
    )

    annotation_path = (
        args.annotation
        .expanduser()
        .resolve()
    )

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        out_dir
        / args.output_name
    )

    if (
        rank == 0
        and output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: "
            f"{output_path}\n"
            "Use --overwrite."
        )

    if world_size > 1:
        dist.barrier()

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        annotation = json.load(f)

    total_prefixes = (
        validate_annotation(
            annotation,
            args.expected_prefixes,
        )
    )

    if rank == 0:
        print(
            "========================================"
        )
        print(
            "Official VisualPRM MathVision calibration"
        )
        print(
            f"Checkpoint : "
            f"{args.checkpoint}"
        )
        print(
            f"Annotation : "
            f"{annotation_path}"
        )
        print(
            f"Image root : "
            f"{root}"
        )
        print(
            f"Questions  : "
            f"{len(annotation)}"
        )
        print(
            f"Prefixes   : "
            f"{total_prefixes}"
        )
        print(
            f"GPUs       : "
            f"{world_size}"
        )
        print(
            f"Output     : "
            f"{output_path}"
        )
        print(
            "========================================"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )
    )

    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    model = (
        model.eval()
        .to(device)
    )

    if not hasattr(
        model,
        "generate_steps_with_soft_score",
    ):
        raise AttributeError(
            "Loaded model does not expose "
            "generate_steps_with_soft_score(). "
            "Check that this is "
            "OpenGVLab/VisualPRM-8B-v1_1."
        )

    image_size = (
        model.config.force_image_size
        or model.config.vision_config.image_size
    )

    local_indices = (
        get_local_indices(
            len(annotation),
            world_size,
            rank,
        )
    )

    local_outputs = []

    progress = tqdm(
        local_indices,
        desc=f"rank {rank}",
        disable=False,
    )

    for question_index in progress:
        original_item = (
            annotation[question_index]
        )

        # Copy because we will add model output.
        item = json.loads(
            json.dumps(
                original_item,
                ensure_ascii=False,
            )
        )

        question = (
            pick_question_text(item)
        )

        if not question:
            raise ValueError(
                f"Item {question_index}: "
                "empty question."
            )

        image_path = (
            pick_image_path(
                root,
                item,
            )
        )

        pixel_values = load_image(
            str(image_path),
            input_size=image_size,
            max_num=args.max_num,
        )

        pixel_values = (
            pixel_values
            .to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
        )

        solutions_splits = (
            item["solutions_splits"]
        )

        prefix_ids = (
            item["prefix_ids"]
        )

        all_step_scores = []

        for local_index, (
            prefix_id,
            steps,
        ) in enumerate(
            zip(
                prefix_ids,
                solutions_splits,
            )
        ):
            try:
                with torch.inference_mode():
                    result = (
                        model
                        .generate_steps_with_soft_score(
                            tokenizer=tokenizer,
                            question=question,
                            response=steps,
                            pixel_values=pixel_values,
                            num_patches_list=[
                                pixel_values.shape[0]
                            ],
                        )
                    )

            except Exception as exc:
                raise RuntimeError(
                    "VisualPRM inference failed for "
                    f"question_index="
                    f"{question_index}, "
                    f"prefix_id={prefix_id}, "
                    f"num_steps={len(steps)}, "
                    f"image={image_path}"
                ) from exc

            if not isinstance(
                result,
                list,
            ):
                raise TypeError(
                    f"{prefix_id}: "
                    "generate_steps_with_soft_score "
                    "did not return a list."
                )

            if len(result) != len(steps):
                raise ValueError(
                    f"{prefix_id}: "
                    f"got {len(result)} scores "
                    f"for {len(steps)} steps."
                )

            scores = []

            for step_index, row in enumerate(
                result
            ):
                if not isinstance(
                    row,
                    dict,
                ):
                    raise TypeError(
                        f"{prefix_id}: "
                        f"step {step_index} "
                        "output is not a dict."
                    )

                score = float(
                    row["score"]
                )

                if not (
                    0.0 <= score <= 1.0
                ):
                    raise ValueError(
                        f"{prefix_id}: "
                        f"invalid score={score}."
                    )

                scores.append(
                    score
                )

            all_step_scores.append(
                scores
            )

        # Use a generic field and also prm_mu for easier
        # compatibility with existing diagnostics.
        item[
            "visualprm_scores"
        ] = all_step_scores

        item[
            "prm_scores"
        ] = all_step_scores

        item[
            "prm_mu"
        ] = all_step_scores

        item[
            "visualprm_model"
        ] = args.checkpoint

        local_outputs.append(
            (
                question_index,
                item,
            )
        )

        del pixel_values

    if world_size > 1:
        gathered = [
            None
            for _ in range(
                world_size
            )
        ]

        dist.all_gather_object(
            gathered,
            local_outputs,
        )

        merged_pairs = list(
            itertools.chain.from_iterable(
                gathered
            )
        )

    else:
        merged_pairs = (
            local_outputs
        )

    if rank == 0:
        merged_pairs.sort(
            key=lambda x: x[0]
        )

        if len(merged_pairs) != len(
            annotation
        ):
            raise ValueError(
                f"Expected "
                f"{len(annotation)} "
                "question outputs, got "
                f"{len(merged_pairs)}."
            )

        merged_outputs = [
            item
            for _, item
            in merged_pairs
        ]

        actual_prefixes = sum(
            len(item["prefix_ids"])
            for item in merged_outputs
        )

        if (
            actual_prefixes
            != args.expected_prefixes
        ):
            raise ValueError(
                f"Expected "
                f"{args.expected_prefixes} "
                "output prefixes, got "
                f"{actual_prefixes}."
            )

        tmp_path = Path(
            str(output_path)
            + ".tmp"
        )

        with tmp_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                merged_outputs,
                f,
                ensure_ascii=False,
                indent=2,
            )

        tmp_path.replace(
            output_path
        )

        print()
        print(
            "=== Official VisualPRM inference ==="
        )
        print(
            f"Questions: "
            f"{len(merged_outputs)}"
        )
        print(
            f"Prefixes : "
            f"{actual_prefixes}"
        )
        print(
            f"Output   : "
            f"{output_path}"
        )
        print()
        print(
            "[PASS] VisualPRM inference completed."
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()