#!/usr/bin/env bash

set -euo pipefail

# HSA-DPO Training Script for LLaVA-v1.5

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Load repo-local overrides if present.
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

# Training configuration
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCH="${EPOCH:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
USE_CHOSEN_SCORE="${USE_CHOSEN_SCORE:-False}"
USE_REJECTED_SCORE="${USE_REJECTED_SCORE:-True}"
DPO_LOSS_TYPE="${DPO_LOSS_TYPE:-hsa_weighted}"
SEVERITY_MARGIN_SCALE="${SEVERITY_MARGIN_SCALE:-0.5}"
SEVERITY_SCORE_NORMALIZER="${SEVERITY_SCORE_NORMALIZER:-3.0}"

# Project-local defaults. Override with env vars if needed.
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/hsa_dpo/data/images}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/llava-v1.5-13b}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/hsa_dpo_llava}"
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/hsa_dpo/models/llava-v1_5/scripts/zero3.json}"

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
        return
    fi
    echo 0
}

AVAILABLE_GPUS="$(detect_gpu_count)"
REQUESTED_GPUS="${NUM_GPUS:-}"
if [ -z "${REQUESTED_GPUS}" ]; then
    if [ "${AVAILABLE_GPUS}" -gt 0 ]; then
        NUM_GPUS="${AVAILABLE_GPUS}"
    else
        NUM_GPUS=2
    fi
else
    NUM_GPUS="${REQUESTED_GPUS}"
fi

if [ "${AVAILABLE_GPUS}" -le 0 ]; then
    echo "No GPUs detected via nvidia-smi; cannot launch DeepSpeed training." >&2
    exit 1
fi

if [ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
    echo "Requested NUM_GPUS=${NUM_GPUS}, but only ${AVAILABLE_GPUS} GPU(s) are available. Using ${AVAILABLE_GPUS} instead." >&2
    NUM_GPUS="${AVAILABLE_GPUS}"
fi

PER_STEP_BATCH=$((BATCH_SIZE * NUM_GPUS))
if [ "${PER_STEP_BATCH}" -le 0 ]; then
    echo "Invalid batch configuration: BATCH_SIZE=${BATCH_SIZE}, NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
fi

if [ -z "${GRADIENT_ACCUMULATION_STEPS}" ]; then
    GRADIENT_ACCUMULATION_STEPS=$(((TOTAL_BATCH_SIZE + PER_STEP_BATCH - 1) / PER_STEP_BATCH))
fi

if [ "${GRADIENT_ACCUMULATION_STEPS}" -le 0 ]; then
    echo "Invalid GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" >&2
    exit 1
fi

EFFECTIVE_TOTAL_BATCH=$((PER_STEP_BATCH * GRADIENT_ACCUMULATION_STEPS))
if [ "${EFFECTIVE_TOTAL_BATCH}" -ne "${TOTAL_BATCH_SIZE}" ]; then
    echo "Warning: effective total batch size ${EFFECTIVE_TOTAL_BATCH} differs from target TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}." >&2
    echo "Set BATCH_SIZE, NUM_GPUS, or GRADIENT_ACCUMULATION_STEPS if you need an exact total batch." >&2
fi

# Training script entry point
ENTRY="${ENTRY:-${REPO_ROOT}/hsa_dpo/models/llava-v1_5/train_dpo.py}"
LLAVA_ROOT="${REPO_ROOT}/hsa_dpo/models/llava-v1_5"

require_path() {
    local path="$1"
    local label="$2"

    if [ ! -e "${path}" ]; then
        echo "Missing ${label}: ${path}" >&2
        exit 1
    fi
}

if ! command -v deepspeed >/dev/null 2>&1; then
    echo "deepspeed is not installed or not on PATH. Install the Linux training extras before running this script." >&2
    exit 1
fi

require_path "${DATA_PATH}" "preference dataset"
require_path "${IMAGE_FOLDER}" "image folder"
require_path "${MODEL_PATH}" "base model directory"
require_path "${ENTRY}" "training entrypoint"
require_path "${DS_CONFIG}" "DeepSpeed config"
require_path "${LLAVA_ROOT}" "LLaVA source tree"

# Keep both the repo package and the bundled LLaVA tree importable.
export PYTHONPATH="${REPO_ROOT}:${LLAVA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting HSA-DPO training..."
echo "Repo root: ${REPO_ROOT}"
echo "Data path: ${DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Model path: ${MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Using ${NUM_GPUS} GPUs"
echo "Per-device train batch size: ${BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective total batch size: ${EFFECTIVE_TOTAL_BATCH}"
echo "Chosen score weighting: ${USE_CHOSEN_SCORE}"
echo "Rejected score weighting: ${USE_REJECTED_SCORE}"
echo "DPO loss type: ${DPO_LOSS_TYPE}"

mkdir -p "${OUTPUT_DIR}"

deepspeed --num_gpus="${NUM_GPUS}" "${ENTRY}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version v1 \
    --desc_data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 0 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCH}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --deepspeed "${DS_CONFIG}" \
    --beta 0.1 \
    --use_chosen_score "${USE_CHOSEN_SCORE}" \
    --use_rejected_score "${USE_REJECTED_SCORE}" \
    --dpo_loss_type "${DPO_LOSS_TYPE}" \
    --severity_margin_scale "${SEVERITY_MARGIN_SCALE}" \
    --severity_score_normalizer "${SEVERITY_SCORE_NORMALIZER}"

echo "Training completed!"
