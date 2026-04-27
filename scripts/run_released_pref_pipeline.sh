#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/run_released_pref_stage3_validate.sh
bash scripts/run_released_pref_stage4_repair.sh

DATA_PATH="${DATA_PATH:-output/fghd/released_pref_stage4/final_preference_pairs.jsonl}" \
IMAGE_FOLDER="${IMAGE_FOLDER:-hsa_dpo/data/images}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/released_pref_stage5_hsa_dpo}" \
bash scripts/run_paper_stage5_train_hsa.sh
