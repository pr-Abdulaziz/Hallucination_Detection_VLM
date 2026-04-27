#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/output/fghd/paper_stage4/preference_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/fghd/paper_stage5_dpo}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "Paper Stage 5 DPO input not found: ${DATA_PATH}" >&2
  echo "Run: bash scripts/run_paper_stage4_rewrite.sh" >&2
  exit 1
fi

DATA_PATH="${DATA_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
DPO_LOSS_TYPE="${DPO_LOSS_TYPE:-standard}" \
USE_CHOSEN_SCORE="${USE_CHOSEN_SCORE:-False}" \
USE_REJECTED_SCORE="${USE_REJECTED_SCORE:-False}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
EPOCH="${EPOCH:-3}" \
LEARNING_RATE="${LEARNING_RATE:-2e-6}" \
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}" \
bash "${REPO_ROOT}/hsa_dpo_train.sh" "$@"
