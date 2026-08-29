#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ============================================================
# BayesianPRM Stage-2 Training
#
# This script ONLY trains the Bayesian belief network.
#
# Required input:
#   A pretrained EnsemblePRM checkpoint.
#
# The frozen ensemble architecture (num_heads, hidden_dim,
# dropout, prior-network settings, etc.) is loaded directly
# from the checkpoint config and must NOT be re-specified here.
# ============================================================


# ============================================================
# Offline
# ============================================================
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}


# ============================================================
# Distributed / device
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}

GPUS=${GPUS:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-${GPUS}}
export MASTER_PORT=${MASTER_PORT:-4320}

model_name=${model_name:-"InternVL3-8B"}


# ============================================================
# Paths
# ============================================================

# REQUIRED.
# Explicitly specify the exact EnsemblePRM checkpoint.
ENSEMBLE_CHECKPOINT=${ENSEMBLE_CHECKPOINT:-""}

if [ -z "${ENSEMBLE_CHECKPOINT}" ]; then
    echo "ERROR: ENSEMBLE_CHECKPOINT must be explicitly specified."
    echo
    echo "Example:"
    echo "  ENSEMBLE_CHECKPOINT=/path/to/checkpoint-xxx \\"
    echo "  bash shell/scripts/visualprm400k_train_bayesian_prm.sh"
    exit 1
fi

if [ ! -d "${ENSEMBLE_CHECKPOINT}" ]; then
    echo "ERROR: Ensemble checkpoint does not exist:"
    echo "  ${ENSEMBLE_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${ENSEMBLE_CHECKPOINT}/config.json" ]; then
    echo "ERROR: config.json not found in ensemble checkpoint:"
    echo "  ${ENSEMBLE_CHECKPOINT}"
    exit 1
fi


BAYESIAN_OUTPUT_DIR=${BAYESIAN_OUTPUT_DIR:-"${REPO_ROOT}/log/bayesian-${model_name}-visualprm400k"}

META_PATH=${META_PATH:-"${REPO_ROOT}/shell/data/meta_visualprm400k_beta_binom.json"}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/configs/zero_stage3_config.json"}

mkdir -p "${BAYESIAN_OUTPUT_DIR}"


# ============================================================
# Inspect EnsemblePRM checkpoint
#
# Ensemble architecture comes ONLY from config.json.
# ============================================================
python - "${ENSEMBLE_CHECKPOINT}" <<'PY'
import json
import os
import sys

ckpt = sys.argv[1]
config_path = os.path.join(ckpt, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

required = [
    "ensemble_prm_num_heads",
    "ensemble_prm_hidden_dim",
    "ensemble_prm_dropout",
    "ensemble_prm_use_prior_network",
    "ensemble_prm_prior_scale",
]

missing = [k for k in required if k not in cfg]

if missing:
    raise RuntimeError(
        "The supplied checkpoint is missing required EnsemblePRM "
        f"configuration fields: {missing}"
    )

print("========== Loaded EnsemblePRM config ==========")
for key in required:
    print(f"{key}: {cfg[key]}")

print(
    "ensemble_prm_bootstrap_prob:",
    cfg.get("ensemble_prm_bootstrap_prob", "<not stored>")
)
print("checkpoint prm_loss_type:", cfg.get("prm_loss_type"))
print("================================================")
PY


# ============================================================
# Bayesian belief network
# ============================================================

# Belief-network architecture.
BELIEF_HIDDEN_DIM=${BELIEF_HIDDEN_DIM:-256}
BELIEF_DROPOUT=${BELIEF_DROPOUT:-0.0}
BELIEF_USE_REWARD_PROBS=${BELIEF_USE_REWARD_PROBS:-True}

# Reliability posterior objective:
#
#   L_rel =
#       - E_{alpha_rel}[log likelihood]
#       + beta_1 KL(alpha_rel || Uniform)
#
BELIEF_BETA_KL=${BELIEF_BETA_KL:-0.05}

# False:
#   use the full Binomial log likelihood
#       K log(mu) + (N-K) log(1-mu)
#
# True:
#   divide the likelihood by N.
BELIEF_LOGLIK_NORMALIZE_BY_N=${BELIEF_LOGLIK_NORMALIZE_BY_N:-False}


# ============================================================
# Conservatism-aware belief calibration
#
# alpha_final_m
#   ∝ alpha_rel_m * exp(-reward_m / beta_2)
#
# NOTE:
# beta_2 does NOT affect belief-network training gradients.
# It can be overridden at evaluation time without retraining.
# ============================================================
BELIEF_USE_CONSERVATISM=${BELIEF_USE_CONSERVATISM:-True}
BELIEF_CONSERVATISM_BETA=${BELIEF_CONSERVATISM_BETA:-0.1}


# ============================================================
# PRM data split
#
# The EnsemblePRM checkpoint must have been trained on the
# complementary "ensemble" subset using exactly the same:
#
#   META_PATH
#   split ratio
#   split seed
#
# BayesianPRM uses the "belief" subset.
# ============================================================
PRM_DATA_SPLIT_ENABLE=${PRM_DATA_SPLIT_ENABLE:-True}
PRM_DATA_SPLIT_RATIO=${PRM_DATA_SPLIT_RATIO:-0.8}
PRM_DATA_SPLIT_SEED=${PRM_DATA_SPLIT_SEED:-42}


# ============================================================
# Optimization
# ============================================================
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}

DENOM=$((PER_DEVICE_BATCH_SIZE * GPUS))

if (( BATCH_SIZE % DENOM != 0 )); then
    echo "ERROR: BATCH_SIZE must be divisible by"
    echo "       PER_DEVICE_BATCH_SIZE * GPUS."
    echo
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    echo "GPUS=${GPUS}"
    exit 1
fi

GRADIENT_ACC=$((BATCH_SIZE / DENOM))

LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}

SAVE_STEPS=${SAVE_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}

# True = much smaller checkpoints, but optimizer/scheduler state
# is not available for a strict training resume.
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-True}


# ============================================================
# Optional short smoke test
# ============================================================
MAX_STEPS=${MAX_STEPS:-""}
MAX_STEPS_ARGS=()

if [ -n "${MAX_STEPS}" ]; then
    MAX_STEPS_ARGS+=(--max_steps "${MAX_STEPS}")
fi


# ============================================================
# Resume Bayesian belief training
# ============================================================
RESUME_BELIEF_TRAINING=${RESUME_BELIEF_TRAINING:-0}

BAYESIAN_MODEL_PATH="${ENSEMBLE_CHECKPOINT}"
BAYESIAN_RESUME_ARGS=()

if [ "${RESUME_BELIEF_TRAINING}" = "1" ]; then

    LATEST_BAYESIAN_CHECKPOINT="$(
        find "${BAYESIAN_OUTPUT_DIR}" \
            -maxdepth 1 \
            -type d \
            -name "checkpoint-*" \
            2>/dev/null \
        | sort -V \
        | tail -n 1
    )"

    if [ -z "${LATEST_BAYESIAN_CHECKPOINT}" ]; then
        echo "ERROR: RESUME_BELIEF_TRAINING=1 but no Bayesian checkpoint found."
        exit 1
    fi

    BAYESIAN_MODEL_PATH="${LATEST_BAYESIAN_CHECKPOINT}"

    BAYESIAN_RESUME_ARGS+=(
        --resume_from_checkpoint "${LATEST_BAYESIAN_CHECKPOINT}"
    )

    echo "Resume BayesianPRM from:"
    echo "  ${LATEST_BAYESIAN_CHECKPOINT}"

else

    echo "Fresh BayesianPRM belief training."
    echo "Removing old Bayesian checkpoints under:"
    echo "  ${BAYESIAN_OUTPUT_DIR}"

    rm -rf "${BAYESIAN_OUTPUT_DIR}"/checkpoint-*
fi


# ============================================================
# Runtime
# ============================================================
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

CUDA_TARGETS_INCLUDE="${CUDA_HOME:-}/targets/x86_64-linux/include"
CUDA_CCCL_INCLUDE="${CUDA_HOME:-}/targets/x86_64-linux/include/cccl"

if [ -d "${CUDA_TARGETS_INCLUDE}" ]; then
    export CPATH="${CUDA_TARGETS_INCLUDE}:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_TARGETS_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi

if [ -d "${CUDA_CCCL_INCLUDE}" ]; then
    export CPATH="${CUDA_CCCL_INCLUDE}:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_CCCL_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/cache/torch_extensions}
mkdir -p "${TORCH_EXTENSIONS_DIR}"


# ============================================================
# WandB
# ============================================================
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-"Beta-PRM"}
export WANDB_NAME=${WANDB_NAME:-"bayesian-${model_name}-visualprm400k"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"bayesian-${model_name}"}
export WANDB_TAGS=${WANDB_TAGS:-"visualprm400k,bayesian"}
export WANDB_DIR=${WANDB_DIR:-"${BAYESIAN_OUTPUT_DIR}/wandb"}

mkdir -p "${WANDB_DIR}"


# ============================================================
# Summary
# ============================================================
echo "================ BayesianPRM training ================"
echo "ENSEMBLE_CHECKPOINT: ${ENSEMBLE_CHECKPOINT}"
echo "BAYESIAN_MODEL_PATH: ${BAYESIAN_MODEL_PATH}"
echo "BAYESIAN_OUTPUT_DIR: ${BAYESIAN_OUTPUT_DIR}"
echo
echo "META_PATH: ${META_PATH}"
echo "PRM_DATA_SPLIT_ENABLE: ${PRM_DATA_SPLIT_ENABLE}"
echo "PRM_DATA_SPLIT_RATIO: ${PRM_DATA_SPLIT_RATIO}"
echo "PRM_DATA_SPLIT_SEED: ${PRM_DATA_SPLIT_SEED}"
echo
echo "BELIEF_HIDDEN_DIM: ${BELIEF_HIDDEN_DIM}"
echo "BELIEF_DROPOUT: ${BELIEF_DROPOUT}"
echo "BELIEF_USE_REWARD_PROBS: ${BELIEF_USE_REWARD_PROBS}"
echo "BELIEF_BETA_KL (beta_1): ${BELIEF_BETA_KL}"
echo "BELIEF_LOGLIK_NORMALIZE_BY_N: ${BELIEF_LOGLIK_NORMALIZE_BY_N}"
echo
echo "BELIEF_USE_CONSERVATISM: ${BELIEF_USE_CONSERVATISM}"
echo "BELIEF_CONSERVATISM_BETA (beta_2): ${BELIEF_CONSERVATISM_BETA}"
echo
echo "GPUS: ${GPUS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "GRADIENT_ACC: ${GRADIENT_ACC}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
echo "========================================================"


# ============================================================
# Train Bayesian belief network
# ============================================================
python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/src/internvl/train/internvl_chat_finetune_beta_binom.py" \
    \
    --model_name_or_path "${BAYESIAN_MODEL_PATH}" \
    --prm_loss_type bayesian_prm \
    \
    --conv_style "internvl2_5" \
    --use_fast_tokenizer False \
    --force_image_size 448 \
    --max_dynamic_patch 6 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.4 \
    --vision_select_layer -1 \
    --max_seq_length 8192 \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version v2 \
    \
    --meta_path "${META_PATH}" \
    --prm_data_split_enable "${PRM_DATA_SPLIT_ENABLE}" \
    --prm_data_split_ratio "${PRM_DATA_SPLIT_RATIO}" \
    --prm_data_split_seed "${PRM_DATA_SPLIT_SEED}" \
    --prm_data_split_part belief \
    \
    --belief_hidden_dim "${BELIEF_HIDDEN_DIM}" \
    --belief_dropout "${BELIEF_DROPOUT}" \
    --belief_beta_kl "${BELIEF_BETA_KL}" \
    --belief_use_reward_probs "${BELIEF_USE_REWARD_PROBS}" \
    --belief_loglik_normalize_by_n "${BELIEF_LOGLIK_NORMALIZE_BY_N}" \
    --belief_use_conservatism "${BELIEF_USE_CONSERVATISM}" \
    --belief_conservatism_beta "${BELIEF_CONSERVATISM_BETA}" \
    \
    --freeze_llm True \
    --freeze_mlp True \
    --freeze_backbone True \
    --grad_checkpoint True \
    \
    --output_dir "${BAYESIAN_OUTPUT_DIR}" \
    --overwrite_output_dir True \
    "${BAYESIAN_RESUME_ARGS[@]}" \
    \
    --do_train True \
    --bf16 True \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    "${MAX_STEPS_ARGS[@]}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACC}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type cosine \
    \
    --dataloader_num_workers 4 \
    --group_by_length True \
    \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    \
    --logging_steps 1 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --report_to wandb \
    --run_name "${WANDB_NAME}" \
    \
    2>&1 | python -u -c '
import sys
from collections import deque
from pathlib import Path

log_file = Path(sys.argv[1])
max_lines = int(sys.argv[2])
flush_interval = int(sys.argv[3])

buf = deque(maxlen=max_lines)
log_file.parent.mkdir(parents=True, exist_ok=True)
log_file.write_text("", encoding="utf-8")

def dump():
    tmp = log_file.with_suffix(log_file.suffix + ".tmp")
    tmp.write_text("".join(buf), encoding="utf-8")
    tmp.replace(log_file)

for i, line in enumerate(sys.stdin, 1):
    sys.stdout.write(line)
    sys.stdout.flush()
    buf.append(line)

    if i % flush_interval == 0:
        dump()

dump()
' "${BAYESIAN_OUTPUT_DIR}/training_log.txt" 1000 100