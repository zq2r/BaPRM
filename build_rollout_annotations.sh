#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/home/admin/workspace/aop_lab/app_data/datasets}"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Generator model used to sample reasoning rollouts.
GEN_MODEL=${GEN_MODEL:-"/home/admin/workspace/aop_lab/app_data/model/InternVL3-8B"}

# Judge server.
JUDGE_API_BASE=${JUDGE_API_BASE:-"http://127.0.0.1:8888/v1"}
JUDGE_MODEL=${JUDGE_MODEL:-"Qwen2.5-32B-Instruct"}

# Rollout settings.
NUM_ROLLOUTS=${NUM_ROLLOUTS:-16}
OVERSAMPLE=${OVERSAMPLE:-2.0}
FLUSH_EVERY=${FLUSH_EVERY:-1}

# Use one GPU for rollout generation by default.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"7"}

# =========================
# OlympiadBench: multi-image version
# =========================
python -m eval.build_eval_rollouts_annotation_olympiadbench \
  --input "${DATASET_ROOT}/OlympiadBench/seed_dataset.json" \
  --output "${DATASET_ROOT}/OlympiadBench/OlympiadBench_rollout_annotation_InternVL8B_oversample.json" \
  --image_root "${DATASET_ROOT}/OlympiadBench" \
  --generator_model "${GEN_MODEL}" \
  --num_rollouts "${NUM_ROLLOUTS}" \
  --judge_api_base "${JUDGE_API_BASE}" \
  --judge_model "${JUDGE_MODEL}" \
  --select_by_llm_quality \
  --oversample "${OVERSAMPLE}" \
  --flush_every "${FLUSH_EVERY}"