#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SHOT_MODE="${SHOT_MODE:-${EXPERIMENT_MODE:-${PROMPT_MODE:-zero_shot}}}"
case "${SHOT_MODE}" in
  zero|zero_shot)
    EXPERIMENT_MODE="zero_shot"
    DEFAULT_STAGE3_DIR="output/fghd/released_pref_stage3"
    DEFAULT_STAGE4_DIR="output/fghd/released_pref_stage4"
    ;;
  two|2shot|two_shot)
    EXPERIMENT_MODE="two_shot"
    DEFAULT_STAGE3_DIR="output/fghd/released_pref_stage3_2shot_experiment"
    DEFAULT_STAGE4_DIR="output/fghd/released_pref_stage4_2shot_experiment"
    ;;
  *)
    echo "Unsupported SHOT_MODE/EXPERIMENT_MODE: ${SHOT_MODE}. Use zero_shot or two_shot." >&2
    exit 2
    ;;
esac
STAGE3_DIR="${STAGE3_DIR:-${DEFAULT_STAGE3_DIR}}"
STAGE4_DIR="${STAGE4_DIR:-${DEFAULT_STAGE4_DIR}}"
REJECTED_INPUT="${REJECTED_INPUT:-${STAGE3_DIR}/rejected_for_repair.jsonl}"
ACCEPTED_INPUT="${ACCEPTED_INPUT:-${STAGE3_DIR}/validated_preferences.jsonl}"
REPAIR_OUT="${REPAIR_OUT:-${STAGE4_DIR}/repair_records.jsonl}"
REPAIRED_PREFERENCES_OUT="${REPAIRED_PREFERENCES_OUT:-${STAGE4_DIR}/repaired_preferences.jsonl}"
FINAL_PREFERENCES_OUT="${FINAL_PREFERENCES_OUT:-${STAGE4_DIR}/final_preference_pairs.jsonl}"
STATS_OUT="${STATS_OUT:-${STAGE4_DIR}/stats.json}"
BACKEND="${BACKEND:-llava}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-hsa_dpo/data/images}"

printf 'Stage 4 repair mode: %s\n' "${EXPERIMENT_MODE}"
printf 'Stage 4 repair input: %s\n' "${REJECTED_INPUT}"
printf 'Stage 4 final preferences: %s\n' "${FINAL_PREFERENCES_OUT}"

CMD=(
  python -m fg_pipeline.paper.run_released_pref_stage4_repair
  --rejected-input "${REJECTED_INPUT}"
  --accepted-input "${ACCEPTED_INPUT}"
  --repair-out "${REPAIR_OUT}"
  --repaired-preferences-out "${REPAIRED_PREFERENCES_OUT}"
  --final-preferences-out "${FINAL_PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
  --experiment-mode "${EXPERIMENT_MODE}"
  --backend "${BACKEND}"
  --image-root "${IMAGE_ROOT}"
)

if [ "${BACKEND}" = "llava" ]; then
  CMD+=(--model-path "${MODEL_PATH}" --conv-mode "${CONV_MODE:-vicuna_v1}" --temperature "${TEMPERATURE:-0.0}")
  if [ -n "${MODEL_BASE:-}" ]; then
    CMD+=(--model-base "${MODEL_BASE}")
  fi
  if [ -n "${MAX_NEW_TOKENS:-}" ]; then
    CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  fi
fi
if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi
if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}" "$@"
