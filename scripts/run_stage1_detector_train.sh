#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
DATA_PATH="${DATA_PATH:-output/fghd/stage1/detector_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/stage1/detector_llava}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-hsa_dpo/models/llava-v1_5/scripts/zero3.json}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "Missing detector training data: ${DATA_PATH}" >&2
  echo "Run Stage 1 detector dataset prep first: bash scripts/run_stage1_detector_dataset.sh" >&2
  exit 1
fi

CMD=(
  deepspeed hsa_dpo/models/llava-v1_5/llava/train/train_mem.py
  --lora_enable True
  --lora_r "${LORA_R:-128}"
  --lora_alpha "${LORA_ALPHA:-256}"
  --mm_projector_lr "${MM_PROJECTOR_LR:-2e-5}"
  --deepspeed "${DEEPSPEED_CONFIG}"
  --model_name_or_path "${MODEL_PATH}"
  --version "${MODEL_VERSION:-v1}"
  --data_path "${DATA_PATH}"
  --image_folder "${IMAGE_FOLDER}"
  --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
  --mm_projector_type "${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
  --mm_vision_select_layer "${MM_VISION_SELECT_LAYER:--2}"
  --mm_use_im_start_end "${MM_USE_IM_START_END:-False}"
  --mm_use_im_patch_token "${MM_USE_IM_PATCH_TOKEN:-False}"
  --image_aspect_ratio "${IMAGE_ASPECT_RATIO:-pad}"
  --group_by_modality_length True
  --bf16 "${BF16:-True}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
  --evaluation_strategy "no"
  --save_strategy "steps"
  --save_steps "${SAVE_STEPS:-1000}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
  --learning_rate "${LEARNING_RATE:-2e-4}"
  --weight_decay "${WEIGHT_DECAY:-0.0}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --tf32 "${TF32:-True}"
  --model_max_length "${MODEL_MAX_LENGTH:-2048}"
  --gradient_checkpointing True
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
  --lazy_preprocess True
  --report_to "${REPORT_TO:-none}"
)

"${CMD[@]}" "$@"
