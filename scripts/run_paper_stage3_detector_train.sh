#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

DATA_PATH="${DATA_PATH:-output/fghd/paper_stage2/detector_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/paper_stage3/detector_lora}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
LLAVA_REPO="${LLAVA_REPO:-${REPO_ROOT}/hsa_dpo/models/llava-v1_5}"

if [ ! -d "${LLAVA_REPO}/llava" ]; then
  echo "Local LLaVA package not found: ${LLAVA_REPO}/llava" >&2
  exit 1
fi

export PYTHONPATH="${LLAVA_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "Paper Stage 3 detector data not found: ${DATA_PATH}" >&2
  echo "Run: bash scripts/run_paper_stage2_detector_dataset.sh" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH}" \
DATA_PATH="${DATA_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}" \
LORA_R="${LORA_R:-128}" \
LORA_ALPHA="${LORA_ALPHA:-256}" \
LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}" \
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}" \
bash scripts/run_stage1_detector_train.sh "$@"
