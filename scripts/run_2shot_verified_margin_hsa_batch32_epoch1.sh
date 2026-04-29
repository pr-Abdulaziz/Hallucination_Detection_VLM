#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-output/fghd/released_pref_stage5_openai_verify_2shot_experiment/final_verified_preference_pairs_train_ready.jsonl}" \
IMAGE_FOLDER="${IMAGE_FOLDER:-hsa_dpo/data/images}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/exp_2shot_verified_margin_hsa_b32_e1}" \
EPOCH="${EPOCH:-1}" \
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
LEARNING_RATE="${LEARNING_RATE:-2e-6}" \
MAX_STEPS="${MAX_STEPS:--1}" \
DPO_LOSS_TYPE=severity_margin \
SEVERITY_MARGIN_SCALE="${SEVERITY_MARGIN_SCALE:-0.5}" \
SEVERITY_SCORE_NORMALIZER="${SEVERITY_SCORE_NORMALIZER:-3.0}" \
USE_REJECTED_SCORE=False \
USE_CHOSEN_SCORE=False \
bash hsa_dpo_train.sh
