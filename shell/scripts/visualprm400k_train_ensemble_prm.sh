#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Offline setting
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# =========================
# Basic configs
# =========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
GPUS=${GPUS:-4}
model_name=${model_name:-"InternVL3-8B"}
export MASTER_PORT=${MASTER_PORT:-4311}
ENSEMBLE_PRM_BOOTSTRAP_PROB=${ENSEMBLE_PRM_BOOTSTRAP_PROB:-1.0}
RESUME_TRAINING=${RESUME_TRAINING:-0}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-True}

# =========================
# Ensemble PRM hyperparams
# =========================
ENSEMBLE_PRM_NUM_HEADS=${ENSEMBLE_PRM_NUM_HEADS:-20}
ENSEMBLE_PRM_HIDDEN_DIM=${ENSEMBLE_PRM_HIDDEN_DIM:-256}
ENSEMBLE_PRM_DROPOUT=${ENSEMBLE_PRM_DROPOUT:-0.0}
ENSEMBLE_PRM_USE_PRIOR_NETWORK=${ENSEMBLE_PRM_USE_PRIOR_NETWORK:-True}
ENSEMBLE_PRM_PRIOR_SCALE=${ENSEMBLE_PRM_PRIOR_SCALE:-10}

# Use repo-relative log dir by default.
if [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "True" ] || [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "true" ]; then
  DEFAULT_OUTPUT_DIR="${REPO_ROOT}/log/ensemble-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}-visualprm400k"
else
  DEFAULT_OUTPUT_DIR="${REPO_ROOT}/log/ensemble-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}-visualprm400k"
fi

OUTPUT_DIR=${OUTPUT_DIR:-"${DEFAULT_OUTPUT_DIR}"}


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# =========================
# Resume / fresh-start logic
# =========================
#
# RESUME_TRAINING=1:
#   - Find the latest checkpoint.
#   - Resume only if it contains complete Trainer + DeepSpeed states.
#   - Otherwise remove incomplete checkpoints and start from scratch.
#
# RESUME_TRAINING=0:
#   - Remove old checkpoints and start from scratch.
# =========================

RESUME_ARGS=()

is_complete_checkpoint() {
  local ckpt="$1"
  local ds_step_dir=""

  # Hugging Face Trainer state.
  if [ ! -s "${ckpt}/trainer_state.json" ]; then
    return 1
  fi

  # DeepSpeed training state directory.
  ds_step_dir=$(
    find "${ckpt}" \
      -maxdepth 1 \
      -type d \
      -name "global_step*" \
      -print -quit \
      2>/dev/null
  )

  if [ -z "${ds_step_dir}" ]; then
    return 1
  fi

  # DeepSpeed model state.
  if ! find "${ds_step_dir}" \
      -maxdepth 1 \
      -type f \
      -name "*model_states.pt" \
      -size +0c \
      -print -quit \
      2>/dev/null | grep -q .; then
    return 1
  fi

  # DeepSpeed optimizer state.
  if ! find "${ds_step_dir}" \
      -maxdepth 1 \
      -type f \
      -name "*optim_states.pt" \
      -size +0c \
      -print -quit \
      2>/dev/null | grep -q .; then
    return 1
  fi

  return 0
}

if [ "${RESUME_TRAINING}" = "1" ]; then

  LATEST_CHECKPOINT=$(
    find "${OUTPUT_DIR}" \
      -maxdepth 1 \
      -type d \
      -name "checkpoint-*" \
      2>/dev/null \
    | sort -V \
    | tail -n 1
  )

  if [ -z "${LATEST_CHECKPOINT}" ]; then
    echo "============================================================"
    echo "[INFO] RESUME_TRAINING=1"
    echo "[INFO] No checkpoint found."
    echo "[INFO] Start EnsemblePRM training from scratch."
    echo "============================================================"

  elif is_complete_checkpoint "${LATEST_CHECKPOINT}"; then
    echo "============================================================"
    echo "[INFO] Found complete EnsemblePRM checkpoint:"
    echo "       ${LATEST_CHECKPOINT}"
    echo "[INFO] Resume training from this checkpoint."
    echo "============================================================"

    RESUME_ARGS+=(
      --resume_from_checkpoint "${LATEST_CHECKPOINT}"
    )

  else
    echo "============================================================"
    echo "[WARNING] Latest BetaPRM checkpoint is incomplete:"
    echo "          ${LATEST_CHECKPOINT}"
    echo "[WARNING] Cannot perform true resume."
    echo "[INFO] Removing old checkpoints."
    echo "[INFO] Start EnsemblePRM training from scratch."
    echo "============================================================"

    rm -rf "${OUTPUT_DIR}"/checkpoint-*
  fi

else
  echo "============================================================"
  echo "[INFO] RESUME_TRAINING=0"
  echo "[INFO] Removing old checkpoints."
  echo "[INFO] Start EnsemblePRM training from scratch."
  echo "============================================================"

  rm -rf "${OUTPUT_DIR}"/checkpoint-*
fi

# =========================
# WandB
# =========================
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-"Beta-PRM"}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}/wandb"}
mkdir -p "${WANDB_DIR}"

if [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "True" ] || [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "true" ]; then
  DEFAULT_WANDB_NAME="ensemble-prior-${model_name}-visualprm400k"
  DEFAULT_WANDB_RUN_GROUP="ensemble-prior-${model_name}"
  DEFAULT_WANDB_TAGS="visualprm400k,ensemble_prior"
else
  DEFAULT_WANDB_NAME="ensemble-${model_name}-visualprm400k"
  DEFAULT_WANDB_RUN_GROUP="ensemble-${model_name}"
  DEFAULT_WANDB_TAGS="visualprm400k"
fi

export WANDB_NAME=${WANDB_NAME:-"${DEFAULT_WANDB_NAME}"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"${DEFAULT_WANDB_RUN_GROUP}"}
export WANDB_TAGS=${WANDB_TAGS:-"${DEFAULT_WANDB_TAGS}"}

# =========================
# Batch / data / model paths
# =========================
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

META_PATH=${META_PATH:-"${REPO_ROOT}/shell/data/meta_visualprm400k_beta_binom.json"}
MODEL_PATH=${MODEL_PATH:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model/${model_name}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/configs/zero_stage3_config.json"}

# =========================
# Optional PRM data split
# =========================
PRM_DATA_SPLIT_ENABLE=${PRM_DATA_SPLIT_ENABLE:-True}
PRM_DATA_SPLIT_RATIO=${PRM_DATA_SPLIT_RATIO:-0.8}
PRM_DATA_SPLIT_SEED=${PRM_DATA_SPLIT_SEED:-42}
PRM_DATA_SPLIT_PART=${PRM_DATA_SPLIT_PART:-ensemble}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS}

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
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

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/cache/torch_extensions}
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# =========================
# Beta-Binom args
# Kept for compatibility with the shared training entry.
# They are skipped internally when prm_loss_type=ensemble_prm.
# =========================
BETA_BINOM_KAPPA_MIN=${BETA_BINOM_KAPPA_MIN:-1e-3}
BETA_BINOM_KAPPA_INIT=${BETA_BINOM_KAPPA_INIT:-4.0}
BETA_BINOM_EVI_REG=${BETA_BINOM_EVI_REG:-5e-2}
BETA_BINOM_KAPPA_HEAD_LR_MULT=${BETA_BINOM_KAPPA_HEAD_LR_MULT:-10.0}



echo "========== Train Config =========="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "META_PATH: ${META_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUS: ${GPUS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "GRADIENT_ACC: ${GRADIENT_ACC}"
echo "PRM_LOSS_TYPE: ensemble_prm"
echo "ENSEMBLE_PRM_NUM_HEADS: ${ENSEMBLE_PRM_NUM_HEADS}"
echo "ENSEMBLE_PRM_HIDDEN_DIM: ${ENSEMBLE_PRM_HIDDEN_DIM}"
echo "ENSEMBLE_PRM_DROPOUT: ${ENSEMBLE_PRM_DROPOUT}"
echo "ENSEMBLE_PRM_USE_PRIOR_NETWORK: ${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"
echo "ENSEMBLE_PRM_PRIOR_SCALE: ${ENSEMBLE_PRM_PRIOR_SCALE}"
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
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  "${RESUME_ARGS[@]}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_only_model ${SAVE_ONLY_MODEL} \
  --save_steps 10 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --prm_loss_type ensemble_prm \
  --ensemble_prm_num_heads ${ENSEMBLE_PRM_NUM_HEADS} \
  --ensemble_prm_hidden_dim ${ENSEMBLE_PRM_HIDDEN_DIM} \
  --ensemble_prm_dropout ${ENSEMBLE_PRM_DROPOUT} \
  --ensemble_prm_use_prior_network ${ENSEMBLE_PRM_USE_PRIOR_NETWORK} \
  --ensemble_prm_prior_scale ${ENSEMBLE_PRM_PRIOR_SCALE} \
  --ensemble_prm_bootstrap_prob ${ENSEMBLE_PRM_BOOTSTRAP_PROB} \
  --prm_data_split_enable ${PRM_DATA_SPLIT_ENABLE} \
  --prm_data_split_ratio ${PRM_DATA_SPLIT_RATIO} \
  --prm_data_split_seed ${PRM_DATA_SPLIT_SEED} \
  --prm_data_split_part ${PRM_DATA_SPLIT_PART} \
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
' "${OUTPUT_DIR}/training_log.txt" 1000 100