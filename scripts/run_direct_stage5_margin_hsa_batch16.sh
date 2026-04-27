#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl}" \
IMAGE_FOLDER="${IMAGE_FOLDER:-hsa_dpo/data/images}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/exp_direct_margin_hsa}" \
EPOCH="${EPOCH:-2}" \
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
DPO_LOSS_TYPE=severity_margin \
SEVERITY_MARGIN_SCALE="${SEVERITY_MARGIN_SCALE:-0.5}" \
SEVERITY_SCORE_NORMALIZER="${SEVERITY_SCORE_NORMALIZER:-3.0}" \
USE_REJECTED_SCORE=False \
USE_CHOSEN_SCORE=False \
bash scripts/run_paper_stage5_train_hsa.sh
