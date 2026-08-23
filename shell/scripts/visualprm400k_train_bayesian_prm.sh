#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

is_true() {
  local value="${1:-False}"
  [ "${value}" = "True" ] || [ "${value}" = "true" ] || [ "${value}" = "1" ]
}

# =========================
# Offline setting
# =========================
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

# =========================
# Common configs
# =========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
GPUS=${GPUS:-4}
model_name=${model_name:-"InternVL3-2B"}
export MASTER_PORT=${MASTER_PORT:-4320}

ENSEMBLE_PRM_BOOTSTRAP_PROB=${ENSEMBLE_PRM_BOOTSTRAP_PROB:-0.5}
BELIEF_BETA_KL=${BELIEF_BETA_KL:-0.05}

# Prior-network switch must be defined before default paths,
# because output directory names depend on it.
ENSEMBLE_PRM_USE_PRIOR_NETWORK=${ENSEMBLE_PRM_USE_PRIOR_NETWORK:-True}
ENSEMBLE_PRM_PRIOR_SCALE=${ENSEMBLE_PRM_PRIOR_SCALE:-1.0}

# Whether to skip ensemble training.
# 0: train ensemble first, then train belief network.
# 1: directly load ensemble checkpoint and train belief network.
LOAD_ENSEMBLE_CHECKPOINT=${LOAD_ENSEMBLE_CHECKPOINT:-1}

# Stage resume switches.
RESUME_ENSEMBLE_TRAINING=${RESUME_ENSEMBLE_TRAINING:-0}
RESUME_BELIEF_TRAINING=${RESUME_BELIEF_TRAINING:-0}

SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-True}

# =========================
# Paths
# =========================
if is_true "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"; then
  DEFAULT_ENSEMBLE_OUTPUT_DIR="${REPO_ROOT}/log/ensemble-prior-${model_name}-visualprm400k"
  DEFAULT_BAYESIAN_OUTPUT_DIR="${REPO_ROOT}/log/bayesian-prior-${model_name}-visualprm400k"
else
  DEFAULT_ENSEMBLE_OUTPUT_DIR="${REPO_ROOT}/log/ensemble-${model_name}-visualprm400k"
  DEFAULT_BAYESIAN_OUTPUT_DIR="${REPO_ROOT}/log/bayesian-${model_name}-visualprm400k"
fi

ENSEMBLE_OUTPUT_DIR=${ENSEMBLE_OUTPUT_DIR:-"${DEFAULT_ENSEMBLE_OUTPUT_DIR}"}
BAYESIAN_OUTPUT_DIR=${BAYESIAN_OUTPUT_DIR:-"${DEFAULT_BAYESIAN_OUTPUT_DIR}"}

# Optional explicit ensemble checkpoint.
# If empty, the script will search latest checkpoint under ENSEMBLE_OUTPUT_DIR.
ENSEMBLE_CHECKPOINT=${ENSEMBLE_CHECKPOINT:-""}

META_PATH=${META_PATH:-"${REPO_ROOT}/shell/data/meta_visualprm400k_beta_binom.json"}
MODEL_PATH=${MODEL_PATH:-"/home/admin/workspace/aop_lab/app_data/model/${model_name}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/configs/zero_stage3_config.json"}

# =========================
# Optional PRM data split
# =========================
PRM_DATA_SPLIT_ENABLE=${PRM_DATA_SPLIT_ENABLE:-True}
PRM_DATA_SPLIT_RATIO=${PRM_DATA_SPLIT_RATIO:-0.8}
PRM_DATA_SPLIT_SEED=${PRM_DATA_SPLIT_SEED:-42}

mkdir -p "${ENSEMBLE_OUTPUT_DIR}"
mkdir -p "${BAYESIAN_OUTPUT_DIR}"

# =========================
# Ensemble PRM hyperparameters
# =========================
ENSEMBLE_PRM_NUM_HEADS=${ENSEMBLE_PRM_NUM_HEADS:-5}
ENSEMBLE_PRM_HIDDEN_DIM=${ENSEMBLE_PRM_HIDDEN_DIM:-256}
ENSEMBLE_PRM_DROPOUT=${ENSEMBLE_PRM_DROPOUT:-0.0}

# =========================
# Bayesian belief hyperparameters
# =========================
BELIEF_HIDDEN_DIM=${BELIEF_HIDDEN_DIM:-256}
BELIEF_DROPOUT=${BELIEF_DROPOUT:-0.0}
BELIEF_USE_REWARD_PROBS=${BELIEF_USE_REWARD_PROBS:-True}
BELIEF_LOGLIK_NORMALIZE_BY_N=${BELIEF_LOGLIK_NORMALIZE_BY_N:-False}

BELIEF_USE_CONSERVATISM=${BELIEF_USE_CONSERVATISM:-False}
BELIEF_CONSERVATISM_BETA=${BELIEF_CONSERVATISM_BETA:-0.1}
BELIEF_HYBRID_LAMBDA=${BELIEF_HYBRID_LAMBDA:-1.0}

# =========================
# Batch / distributed configs
# =========================
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS}

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# =========================
# CUDA include path fix
# =========================
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

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.cache/torch_extensions}
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# =========================
# Shared Beta-Binom compatibility args
# These are skipped internally when prm_loss_type != beta_binom.
# =========================
BETA_BINOM_KAPPA_MIN=${BETA_BINOM_KAPPA_MIN:-1e-3}
BETA_BINOM_KAPPA_INIT=${BETA_BINOM_KAPPA_INIT:-4.0}
BETA_BINOM_EVI_REG=${BETA_BINOM_EVI_REG:-5e-2}
BETA_BINOM_KAPPA_HEAD_LR_MULT=${BETA_BINOM_KAPPA_HEAD_LR_MULT:-10.0}

# =========================
# Optional smoke-test max steps
# Set MAX_STEPS=2 for quick debug.
# Empty means normal epoch-based training.
# =========================
MAX_STEPS=${MAX_STEPS:-""}
MAX_STEPS_ARGS=()
if [ -n "${MAX_STEPS}" ]; then
  MAX_STEPS_ARGS+=(--max_steps "${MAX_STEPS}")
fi

# =========================
# Default WandB names
# =========================
if is_true "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"; then
  DEFAULT_ENSEMBLE_WANDB_NAME="ensemble-prior-${model_name}-visualprm400k"
  DEFAULT_ENSEMBLE_WANDB_RUN_GROUP="ensemble-prior-${model_name}"
  DEFAULT_ENSEMBLE_WANDB_TAGS="visualprm400k,ensemble,ensemble_prior"

  DEFAULT_BAYESIAN_WANDB_NAME="bayesian-prior-${model_name}-visualprm400k"
  DEFAULT_BAYESIAN_WANDB_RUN_GROUP="bayesian-prior-${model_name}"
  DEFAULT_BAYESIAN_WANDB_TAGS="visualprm400k,bayesian,ensemble_prior"
else
  DEFAULT_ENSEMBLE_WANDB_NAME="ensemble-${model_name}-visualprm400k"
  DEFAULT_ENSEMBLE_WANDB_RUN_GROUP="ensemble-${model_name}"
  DEFAULT_ENSEMBLE_WANDB_TAGS="visualprm400k,ensemble"

  DEFAULT_BAYESIAN_WANDB_NAME="bayesian-${model_name}-visualprm400k"
  DEFAULT_BAYESIAN_WANDB_RUN_GROUP="bayesian-${model_name}"
  DEFAULT_BAYESIAN_WANDB_TAGS="visualprm400k,bayesian"
fi

# =========================
# Stage 1: train or load ensemble PRM
# =========================
if [ "${LOAD_ENSEMBLE_CHECKPOINT}" = "0" ]; then
  echo "========== Stage 1: Train ensemble PRM =========="

  OUTPUT_DIR="${ENSEMBLE_OUTPUT_DIR}" \
  RESUME_TRAINING="${RESUME_ENSEMBLE_TRAINING}" \
  SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  GPUS="${GPUS}" \
  model_name="${model_name}" \
  MASTER_PORT="${MASTER_PORT}" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  WANDB_PROJECT="${WANDB_PROJECT:-Beta-PRM}" \
  WANDB_NAME="${ENSEMBLE_WANDB_NAME:-${DEFAULT_ENSEMBLE_WANDB_NAME}}" \
  WANDB_RUN_GROUP="${ENSEMBLE_WANDB_RUN_GROUP:-${DEFAULT_ENSEMBLE_WANDB_RUN_GROUP}}" \
  WANDB_TAGS="${ENSEMBLE_WANDB_TAGS:-${DEFAULT_ENSEMBLE_WANDB_TAGS}}" \
  META_PATH="${META_PATH}" \
  MODEL_PATH="${MODEL_PATH}" \
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
  MAX_STEPS="${MAX_STEPS}" \
  PRM_DATA_SPLIT_ENABLE="${PRM_DATA_SPLIT_ENABLE}" \
  PRM_DATA_SPLIT_RATIO="${PRM_DATA_SPLIT_RATIO}" \
  PRM_DATA_SPLIT_SEED="${PRM_DATA_SPLIT_SEED}" \
  PRM_DATA_SPLIT_PART="ensemble" \
  ENSEMBLE_PRM_NUM_HEADS="${ENSEMBLE_PRM_NUM_HEADS}" \
  ENSEMBLE_PRM_HIDDEN_DIM="${ENSEMBLE_PRM_HIDDEN_DIM}" \
  ENSEMBLE_PRM_DROPOUT="${ENSEMBLE_PRM_DROPOUT}" \
  ENSEMBLE_PRM_USE_PRIOR_NETWORK="${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" \
  ENSEMBLE_PRM_PRIOR_SCALE="${ENSEMBLE_PRM_PRIOR_SCALE}" \
  ENSEMBLE_PRM_BOOTSTRAP_PROB="${ENSEMBLE_PRM_BOOTSTRAP_PROB}" \
  bash "${REPO_ROOT}/shell/scripts/visualprm400k_train_ensemble_prm.sh"

  echo "========== Stage 1 finished =========="
else
  echo "========== Stage 1 skipped: LOAD_ENSEMBLE_CHECKPOINT=1 =========="
fi

# =========================
# Resolve ensemble checkpoint
# =========================
if [ -z "${ENSEMBLE_CHECKPOINT}" ]; then
  ENSEMBLE_CHECKPOINT="$(
    find "${ENSEMBLE_OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
fi

if [ -z "${ENSEMBLE_CHECKPOINT}" ] || [ ! -d "${ENSEMBLE_CHECKPOINT}" ]; then
  echo "ERROR: No valid ensemble checkpoint found."
  echo "ENSEMBLE_CHECKPOINT='${ENSEMBLE_CHECKPOINT}'"
  echo "ENSEMBLE_OUTPUT_DIR='${ENSEMBLE_OUTPUT_DIR}'"
  exit 1
fi

echo "Using ensemble checkpoint: ${ENSEMBLE_CHECKPOINT}"

# =========================
# Stage 2 resume / fresh-start logic
# =========================
BAYESIAN_RESUME_ARGS=()
BAYESIAN_MODEL_PATH="${ENSEMBLE_CHECKPOINT}"

if [ "${RESUME_BELIEF_TRAINING}" = "1" ]; then
  LATEST_BAYESIAN_CHECKPOINT="$(
    find "${BAYESIAN_OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"

  if [ -n "${LATEST_BAYESIAN_CHECKPOINT}" ] && [ -d "${LATEST_BAYESIAN_CHECKPOINT}" ]; then
    echo "Auto resume Bayesian belief training from: ${LATEST_BAYESIAN_CHECKPOINT}"
    BAYESIAN_MODEL_PATH="${LATEST_BAYESIAN_CHECKPOINT}"
    BAYESIAN_RESUME_ARGS+=(--resume_from_checkpoint "${LATEST_BAYESIAN_CHECKPOINT}")
  else
    echo "RESUME_BELIEF_TRAINING=1 but no Bayesian checkpoint found. Start belief training from ensemble checkpoint."
  fi
else
  echo "Resume disabled for Bayesian belief training. Remove old Bayesian checkpoints."
  rm -rf "${BAYESIAN_OUTPUT_DIR}"/checkpoint-*
fi

# =========================
# WandB for Stage 2
# =========================
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-"Beta-PRM"}
export WANDB_NAME=${BAYESIAN_WANDB_NAME:-"${DEFAULT_BAYESIAN_WANDB_NAME}"}
export WANDB_RUN_GROUP=${BAYESIAN_WANDB_RUN_GROUP:-"${DEFAULT_BAYESIAN_WANDB_RUN_GROUP}"}
export WANDB_TAGS=${BAYESIAN_WANDB_TAGS:-"${DEFAULT_BAYESIAN_WANDB_TAGS}"}
export WANDB_DIR=${WANDB_DIR:-"${BAYESIAN_OUTPUT_DIR}/wandb"}
mkdir -p "${WANDB_DIR}"

# =========================
# Stage 2: train BayesianPRM belief network
# =========================
echo "========== Stage 2: Train BayesianPRM belief network =========="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "BAYESIAN_MODEL_PATH: ${BAYESIAN_MODEL_PATH}"
echo "ENSEMBLE_CHECKPOINT: ${ENSEMBLE_CHECKPOINT}"
echo "ENSEMBLE_OUTPUT_DIR: ${ENSEMBLE_OUTPUT_DIR}"
echo "BAYESIAN_OUTPUT_DIR: ${BAYESIAN_OUTPUT_DIR}"
echo "META_PATH: ${META_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUS: ${GPUS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "GRADIENT_ACC: ${GRADIENT_ACC}"
echo "PRM_LOSS_TYPE: bayesian_prm"
echo "ENSEMBLE_PRM_NUM_HEADS: ${ENSEMBLE_PRM_NUM_HEADS}"
echo "ENSEMBLE_PRM_HIDDEN_DIM: ${ENSEMBLE_PRM_HIDDEN_DIM}"
echo "ENSEMBLE_PRM_DROPOUT: ${ENSEMBLE_PRM_DROPOUT}"
echo "ENSEMBLE_PRM_USE_PRIOR_NETWORK: ${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"
echo "ENSEMBLE_PRM_PRIOR_SCALE: ${ENSEMBLE_PRM_PRIOR_SCALE}"
echo "PRM_DATA_SPLIT_ENABLE: ${PRM_DATA_SPLIT_ENABLE}"
echo "PRM_DATA_SPLIT_RATIO: ${PRM_DATA_SPLIT_RATIO}"
echo "PRM_DATA_SPLIT_SEED: ${PRM_DATA_SPLIT_SEED}"
echo "BELIEF_HIDDEN_DIM: ${BELIEF_HIDDEN_DIM}"
echo "BELIEF_DROPOUT: ${BELIEF_DROPOUT}"
echo "BELIEF_BETA_KL: ${BELIEF_BETA_KL}"
echo "BELIEF_USE_REWARD_PROBS: ${BELIEF_USE_REWARD_PROBS}"
echo "BELIEF_LOGLIK_NORMALIZE_BY_N: ${BELIEF_LOGLIK_NORMALIZE_BY_N}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_NAME: ${WANDB_NAME}"
echo "WANDB_RUN_GROUP: ${WANDB_RUN_GROUP}"
echo "WANDB_TAGS: ${WANDB_TAGS}"
echo "=================================="

python -m torch.distributed.run \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=${MASTER_PORT} \
  "${REPO_ROOT}/src/internvl/train/internvl_chat_finetune_beta_binom.py" \
  --model_name_or_path "${BAYESIAN_MODEL_PATH}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "${BAYESIAN_OUTPUT_DIR}" \
  "${BAYESIAN_RESUME_ARGS[@]}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  "${MAX_STEPS_ARGS[@]}" \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_only_model ${SAVE_ONLY_MODEL} \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --prm_loss_type bayesian_prm \
  --ensemble_prm_num_heads ${ENSEMBLE_PRM_NUM_HEADS} \
  --ensemble_prm_hidden_dim ${ENSEMBLE_PRM_HIDDEN_DIM} \
  --ensemble_prm_dropout ${ENSEMBLE_PRM_DROPOUT} \
  --ensemble_prm_use_prior_network ${ENSEMBLE_PRM_USE_PRIOR_NETWORK} \
  --ensemble_prm_prior_scale ${ENSEMBLE_PRM_PRIOR_SCALE} \
  --belief_hidden_dim ${BELIEF_HIDDEN_DIM} \
  --belief_dropout ${BELIEF_DROPOUT} \
  --belief_beta_kl ${BELIEF_BETA_KL} \
  --belief_use_reward_probs ${BELIEF_USE_REWARD_PROBS} \
  --belief_loglik_normalize_by_n ${BELIEF_LOGLIK_NORMALIZE_BY_N} \
  --belief_use_conservatism ${BELIEF_USE_CONSERVATISM} \
  --belief_conservatism_beta ${BELIEF_CONSERVATISM_BETA} \
  --belief_hybrid_lambda ${BELIEF_HYBRID_LAMBDA} \
  --prm_data_split_enable ${PRM_DATA_SPLIT_ENABLE} \
  --prm_data_split_ratio ${PRM_DATA_SPLIT_RATIO} \
  --prm_data_split_seed ${PRM_DATA_SPLIT_SEED} \
  --prm_data_split_part belief \
  --beta_binom_kappa_min ${BETA_BINOM_KAPPA_MIN} \
  --beta_binom_kappa_init ${BETA_BINOM_KAPPA_INIT} \
  --beta_binom_evi_reg ${BETA_BINOM_EVI_REG} \
  --beta_binom_kappa_head_lr_mult ${BETA_BINOM_KAPPA_HEAD_LR_MULT} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "wandb" \
  --run_name "${WANDB_NAME}" \
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