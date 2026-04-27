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

REJECTED_INPUT="${REJECTED_INPUT:-output/fghd/released_pref_stage3/rejected_for_repair.jsonl}"
ACCEPTED_INPUT="${ACCEPTED_INPUT:-output/fghd/released_pref_stage3/validated_preferences.jsonl}"
REPAIR_OUT="${REPAIR_OUT:-output/fghd/released_pref_stage4/repair_records.jsonl}"
REPAIRED_PREFERENCES_OUT="${REPAIRED_PREFERENCES_OUT:-output/fghd/released_pref_stage4/repaired_preferences.jsonl}"
FINAL_PREFERENCES_OUT="${FINAL_PREFERENCES_OUT:-output/fghd/released_pref_stage4/final_preference_pairs.jsonl}"
STATS_OUT="${STATS_OUT:-output/fghd/released_pref_stage4/stats.json}"
BACKEND="${BACKEND:-llava}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-hsa_dpo/data/images}"

CMD=(
  python -m fg_pipeline.paper.run_released_pref_stage4_repair
  --rejected-input "${REJECTED_INPUT}"
  --accepted-input "${ACCEPTED_INPUT}"
  --repair-out "${REPAIR_OUT}"
  --repaired-preferences-out "${REPAIRED_PREFERENCES_OUT}"
  --final-preferences-out "${FINAL_PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
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
