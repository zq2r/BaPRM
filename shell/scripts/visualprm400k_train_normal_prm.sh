#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Offline setting
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# hyperparameters needed to specify
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
GPUS=${GPUS:-4}
model_name=${model_name:-"InternVL2_5-2B"}
export MASTER_PORT=${MASTER_PORT:-4320}
RESUME_TRAINING=${RESUME_TRAINING:-0}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-True}

OUTPUT_DIR=${OUTPUT_DIR:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/project/Beta-Binomial-PRM/log/normal-${model_name}-visualprm400k"}
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
# =========================
# Resume / fresh-start logic
# =========================
# RESUME_TRAINING=1: auto resume from latest checkpoint if exists.
# RESUME_TRAINING=0: start from scratch and remove old checkpoints.
RESUME_ARGS=()

if [ "${RESUME_TRAINING}" = "1" ]; then
  LATEST_CHECKPOINT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n 1)

  if [ -n "${LATEST_CHECKPOINT}" ]; then
    echo "Auto resume enabled. Found latest checkpoint: ${LATEST_CHECKPOINT}"
    RESUME_ARGS+=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
  else
    echo "Auto resume enabled, but no checkpoint found. Start training from scratch."
  fi
else
  echo "Resume disabled. Remove old checkpoints and start from scratch."
  rm -rf "${OUTPUT_DIR}"/checkpoint-*
fi



export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-"Beta-PRM"}
export WANDB_NAME=${WANDB_NAME:-"normal-${model_name}-visualprm400k"}
# group: hyperparameter
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"normal-${model_name}"}
# tag: dataset
export WANDB_TAGS=${WANDB_TAGS:-"visualprm400k"}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}/wandb"}
mkdir -p "${WANDB_DIR}"


BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

META_PATH=${META_PATH:-"${REPO_ROOT}/shell/data/meta_visualprm400k_beta_binom.json"}
MODEL_PATH=${MODEL_PATH:-"/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/model/${model_name}"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/configs/zero_stage3_config.json"}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS}

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

CUDA_TARGETS_INCLUDE="${CUDA_HOME}/targets/x86_64-linux/include"
CUDA_CCCL_INCLUDE="${CUDA_HOME}/targets/x86_64-linux/include/cccl"

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

# Beta-Binom PRM hyperparams
BETA_BINOM_KAPPA_MIN=${BETA_BINOM_KAPPA_MIN:-1e-3}
BETA_BINOM_KAPPA_INIT=${BETA_BINOM_KAPPA_INIT:-4.0}
BETA_BINOM_EVI_REG=${BETA_BINOM_EVI_REG:-5e-2}
BETA_BINOM_KAPPA_HEAD_LR_MULT=${BETA_BINOM_KAPPA_HEAD_LR_MULT:-10.0}


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
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 100 \
  --max_seq_length 8192 \
  --prm_loss_type normal_prm \
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
' "${OUTPUT_DIR}/training_log.txt" 5000 100