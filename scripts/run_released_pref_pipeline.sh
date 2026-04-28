#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SHOT_MODE="${SHOT_MODE:-zero_shot}"
case "${SHOT_MODE}" in
  zero|zero_shot)
    SHOT_MODE="zero_shot"
    DEFAULT_STAGE3_DIR="output/fghd/released_pref_stage3"
    DEFAULT_STAGE4_DIR="output/fghd/released_pref_stage4"
    DEFAULT_STAGE5_DIR="output/fghd/released_pref_stage5_hsa_dpo"
    ;;
  two|2shot|two_shot)
    SHOT_MODE="two_shot"
    DEFAULT_STAGE3_DIR="output/fghd/released_pref_stage3_2shot_experiment"
    DEFAULT_STAGE4_DIR="output/fghd/released_pref_stage4_2shot_experiment"
    DEFAULT_STAGE5_DIR="output/fghd/released_pref_stage5_2shot_experiment"
    ;;
  *)
    echo "Unsupported SHOT_MODE: ${SHOT_MODE}. Use zero_shot or two_shot." >&2
    exit 2
    ;;
esac

STAGE3_DIR="${STAGE3_DIR:-${DEFAULT_STAGE3_DIR}}"
STAGE4_DIR="${STAGE4_DIR:-${DEFAULT_STAGE4_DIR}}"
export SHOT_MODE PROMPT_MODE="${PROMPT_MODE:-${SHOT_MODE}}" EXPERIMENT_MODE="${EXPERIMENT_MODE:-${SHOT_MODE}}" STAGE3_DIR STAGE4_DIR

printf 'Released preference pipeline mode: %s\n' "${SHOT_MODE}"
printf 'Stage 3 outputs: %s\n' "${STAGE3_DIR}"
printf 'Stage 4 preference outputs: %s\n' "${STAGE4_DIR}"
printf 'Stage 5 training output: %s\n' "${OUTPUT_DIR:-${DEFAULT_STAGE5_DIR}}"

bash scripts/run_released_pref_stage3_validate.sh
bash scripts/run_released_pref_stage4_repair.sh

DATA_PATH="${DATA_PATH:-${STAGE4_DIR}/final_preference_pairs.jsonl}" \
IMAGE_FOLDER="${IMAGE_FOLDER:-hsa_dpo/data/images}" \
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_STAGE5_DIR}}" \
DPO_LOSS_TYPE="${DPO_LOSS_TYPE:-severity_margin}" \
SEVERITY_MARGIN_SCALE="${SEVERITY_MARGIN_SCALE:-0.5}" \
bash scripts/run_paper_stage5_train_hsa.sh
