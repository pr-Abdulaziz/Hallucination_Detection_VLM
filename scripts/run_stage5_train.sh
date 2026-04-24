#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VAST_LOCAL_ENV="${REPO_ROOT}/scripts/vastai/defaults.local.env"
if [ -f "${VAST_LOCAL_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${VAST_LOCAL_ENV}"
  set +a
fi

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/output/fghd/stage4/final_preference_pairs.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/fghd/stage5_llava_margin}"
DPO_LOSS_TYPE="${DPO_LOSS_TYPE:-severity_margin}"
SEVERITY_MARGIN_SCALE="${SEVERITY_MARGIN_SCALE:-0.5}"
SEVERITY_SCORE_NORMALIZER="${SEVERITY_SCORE_NORMALIZER:-3.0}"
USE_CHOSEN_SCORE="${USE_CHOSEN_SCORE:-False}"
USE_REJECTED_SCORE="${USE_REJECTED_SCORE:-False}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCH="${EPOCH:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
MAX_STEPS="${MAX_STEPS:--1}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "Stage 5 input not found: ${DATA_PATH}" >&2
  echo "Run Stage 4 first:  bash scripts/run_stage4_rewrite.sh" >&2
  exit 1
fi

echo "Stage 5 uses final Stage 4 preference pairs."
echo "DPO loss type: ${DPO_LOSS_TYPE}"
echo "Paper-fair target total batch size: ${TOTAL_BATCH_SIZE}"

DATA_PATH="${DATA_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DPO_LOSS_TYPE="${DPO_LOSS_TYPE}" \
SEVERITY_MARGIN_SCALE="${SEVERITY_MARGIN_SCALE}" \
SEVERITY_SCORE_NORMALIZER="${SEVERITY_SCORE_NORMALIZER}" \
USE_CHOSEN_SCORE="${USE_CHOSEN_SCORE}" \
USE_REJECTED_SCORE="${USE_REJECTED_SCORE}" \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCH="${EPOCH}" \
LEARNING_RATE="${LEARNING_RATE}" \
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
MAX_STEPS="${MAX_STEPS}" \
bash "${REPO_ROOT}/hsa_dpo_train.sh" "$@"
