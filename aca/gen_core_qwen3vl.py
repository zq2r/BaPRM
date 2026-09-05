from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# vLLM V1 + Qwen3-VL works reliably with spawn.
# This file is executed from a real .py entrypoint, not stdin.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def select_best_rollouts(
    seg_lists: Sequence[Sequence[str]],
    k: int,
) -> List[List[str]]:
    """
    Same fallback selector as the existing InternVL generator:
    deduplicate first, then prefer valid trajectories with more steps.
    """
    seen = set()
    deduped: List[List[str]] = []

    for segs in seg_lists:
        segs_norm = tuple(
            _normalize_text(x)
            for x in segs
            if _normalize_text(x)
        )
        if not segs_norm:
            continue
        if segs_norm in seen:
            continue

        seen.add(segs_norm)
        deduped.append(list(segs_norm))

    valid = [
        s
        for s in deduped
        if len(s) >= 2 and all(x.strip() for x in s)
    ]
    valid.sort(
        key=lambda s: (len(s), sum(len(x) for x in s)),
        reverse=True,
    )

    result = valid[:k]

    if len(result) < k:
        rest = [
            s
            for s in deduped
            if len(s) >= 1
            and all(x.strip() for x in s)
            and s not in result
        ]
        rest.sort(
            key=lambda s: (len(s), sum(len(x) for x in s)),
            reverse=True,
        )
        result.extend(rest[: k - len(result)])

    return result


def make_cot_prompt(question: str) -> str:
    """
    IMPORTANT:
    Unlike InternVL, do NOT insert a literal <image> token here.
    Qwen3-VL image inputs are represented explicitly in chat messages.
    """
    return f"""You are an AI assistant that must strictly follow the response protocol below.

1) Reasoning phase:
- Analyze the question thoroughly and consider plausible solution strategies.
- Proceed step by step and wrap each reasoning step in <step>...</step>.
- Keep each step concise, factual, and focused on the question and the image (if provided).
- Do NOT include any text outside <step> tags in this phase.

2) Final answer:
- After the reasoning steps, add a single blank line.
- Provide the final answer wrapped in exactly one <answer>...</answer> tag.
- The final answer must be self-contained and must NOT reference the reasoning text.
- Output nothing outside the required tags and nothing after </answer>.

Question: {question}"""


def make_continue_prompt(
    question: str,
    prefix_steps: Sequence[str],
    extra_instruction: Optional[str] = None,
) -> str:
    prev = "".join(
        f"<step>{_normalize_text(s)}</step>"
        for s in prefix_steps
        if _normalize_text(s)
    )

    extra = ""
    if extra_instruction and _normalize_text(extra_instruction):
        extra = _normalize_text(extra_instruction) + "\n"

    return f"""You are given a math problem and previous reasoning steps that have already been completed.

Continue reasoning from the NEXT step.
Do NOT repeat, rewrite, or modify any previous step.

{extra}Output STRICTLY and ONLY using these XML tags:
- Wrap every NEW reasoning step in <step>...</step>.
- End with exactly ONE <answer>...</answer> tag as the LAST tag.
- Do NOT output any text outside these tags.
- Do NOT output anything after </answer>.

Question: {question}

Previous Steps:
{prev}"""


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenResult:
    segments: List[str]
    token_len: int
    raw_text: str = ""


class Qwen3VLGenerator:
    """
    Qwen3-VL generator backed by the vLLM Python API.

    Public behavior intentionally mirrors InternVL25Generator so that
    downstream calibration/TTS code does not need to know which rollout
    policy is being used.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    ):
        try:
            from transformers import AutoProcessor
            from vllm import LLM
        except Exception as e:
            raise RuntimeError(
                "Missing Qwen3-VL generator dependencies. "
                "Expected transformers, vllm, qwen-vl-utils and Pillow."
            ) from e

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as e:
            raise RuntimeError(
                "qwen-vl-utils is required for Qwen3-VL generation."
            ) from e

        self.model_path = model
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.limit_mm_per_prompt = (
            limit_mm_per_prompt or {"image": 8}
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.tokenizer = self.processor.tokenizer
        self._process_vision_info = process_vision_info

        self._step_answer_pattern = re.compile(
            r"<step>(.*?)</step>|<answer>(.*?)</answer>",
            re.DOTALL,
        )

        special_tokens = list(
            getattr(self.tokenizer, "all_special_tokens", []) or []
        )

        self._special_tokens = [
            t
            for t in special_tokens
            if t
            not in [
                "<step>",
                "</step>",
                "<answer>",
                "</answer>",
            ]
        ]

        self._special_re = (
            re.compile("|".join(map(re.escape, self._special_tokens)))
            if self._special_tokens
            else None
        )

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
        )

    @staticmethod
    def _normalize_image_paths(
        image_path: Optional[Union[str, List[str]]],
    ) -> List[str]:
        if not image_path:
            return []

        if isinstance(image_path, list):
            raw_paths = image_path
        else:
            raw_paths = [image_path]

        result: List[str] = []

        for raw in raw_paths:
            if raw is None or not str(raw).strip():
                continue

            path = Path(str(raw)).expanduser().resolve()

            if not path.is_file():
                raise FileNotFoundError(
                    f"Image does not exist: {path}"
                )

            # qwen-vl-utils officially supports local file:// URIs.
            result.append(path.as_uri())

        return result

    def _build_messages(
        self,
        user_content: str,
        image_path: Optional[Union[str, List[str]]],
    ) -> List[Dict[str, Any]]:
        image_uris = self._normalize_image_paths(image_path)

        content: List[Dict[str, Any]] = []

        for uri in image_uris:
            content.append(
                {
                    "type": "image",
                    "image": uri,
                }
            )

        content.append(
            {
                "type": "text",
                "text": str(user_content),
            }
        )

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _prepare_vllm_input(
        self,
        user_content: str,
        image_path: Optional[Union[str, List[str]]],
    ) -> Dict[str, Any]:
        messages = self._build_messages(
            user_content=user_content,
            image_path=image_path,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = (
            self._process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        )

        mm_data: Dict[str, Any] = {}

        if image_inputs is not None:
            mm_data["image"] = image_inputs

        if video_inputs is not None:
            mm_data["video"] = video_inputs

        inp: Dict[str, Any] = {
            "prompt": text,
        }

        if mm_data:
            inp["multi_modal_data"] = mm_data

        # Keep this exactly aligned with the official Qwen3-VL
        # offline-vLLM preprocessing path.
        if video_kwargs is not None:
            inp["mm_processor_kwargs"] = video_kwargs

        return inp

    def _parse_segments(self, text: str) -> List[str]:
        if text is None:
            return []

        resp = str(text)

        if self._special_re is not None:
            resp = re.sub(self._special_re, "", resp)
                    
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            resp,
            re.DOTALL,
        )
        if "<step>" in resp or "<answer>" in resp:
            if (
                answer_match is None
                or not _normalize_text(answer_match.group(1))
            ):
                return []

        matches = re.findall(
            self._step_answer_pattern,
            resp,
        )
            
        matches = re.findall(
            self._step_answer_pattern,
            resp,
        )

        segs = (
            [
                m[0] if m[0] else m[1]
                for m in matches
            ]
            if matches
            else []
        )

        segs = [
            _normalize_text(s)
            for s in segs
            if _normalize_text(s)
        ]

        # Same fallback behavior as the InternVL backend.
        if not segs:
            m = re.search(
                r"Answer:\s*(?:The final answer is\s*)?(.*)",
                resp,
                re.IGNORECASE,
            )
            if m and _normalize_text(m.group(1)):
                return [_normalize_text(m.group(1))]

        if not segs and _normalize_text(resp):
            return [_normalize_text(resp)]

        return segs

    def generate(
        self,
        user_prompt: str,
        image_path: Optional[Union[str, List[str]]],
        n: int,
        sampling: SamplingConfig,
    ) -> List[GenResult]:
        from vllm import SamplingParams

        inp = self._prepare_vllm_input(
            user_content=user_prompt,
            image_path=image_path,
        )

        sp_kwargs: Dict[str, Any] = {
            "temperature": float(sampling.temperature),
            "max_tokens": int(sampling.max_new_tokens),
            "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k),
            "repetition_penalty": float(
                sampling.repetition_penalty
            ),
            "skip_special_tokens": False,
            "n": max(1, int(n)),
        }

        if (
            sampling.seed is not None
            and int(sampling.seed) >= 0
        ):
            sp_kwargs["seed"] = int(sampling.seed)

        sampling_params = SamplingParams(**sp_kwargs)

        outputs = self.llm.generate(
            [inp],
            sampling_params=sampling_params,
        )

        if not outputs:
            return []

        results: List[GenResult] = []

        for out in outputs[0].outputs:
            token_ids = getattr(out, "token_ids", []) or []
            raw_text = str(
                getattr(out, "text", "") or ""
            )

            results.append(
                GenResult(
                    segments=self._parse_segments(raw_text),
                    token_len=int(len(token_ids)),
                    raw_text=raw_text,
                )
            )

        return results


def oversample_and_select(
    raw: Sequence[GenResult],
    k: int,
) -> Tuple[List[GenResult], int]:
    """
    Same semantics as the existing InternVL backend.
    """
    k = int(k)

    if k <= 0:
        return [], 0

    seg_lists = [r.segments for r in raw]
    selected_seg_lists = select_best_rollouts(
        seg_lists,
        k=k,
    )

    selected: List[GenResult] = []
    effective_tokens = 0
    used = set()

    for segs in selected_seg_lists:
        key = tuple(
            _normalize_text(x)
            for x in segs
            if _normalize_text(x)
        )

        picked = None

        for i, r in enumerate(raw):
            if i in used:
                continue

            rkey = tuple(
                _normalize_text(x)
                for x in r.segments
                if _normalize_text(x)
            )

            if rkey == key:
                picked = i
                break

        if picked is None:
            selected.append(
                GenResult(
                    segments=list(segs),
                    token_len=0,
                )
            )
        else:
            used.add(picked)
            selected.append(raw[picked])
            effective_tokens += int(raw[picked].token_len)

    return selected, int(effective_tokens)
