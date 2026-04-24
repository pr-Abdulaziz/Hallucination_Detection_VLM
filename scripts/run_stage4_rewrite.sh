#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VAST_LOCAL_ENV="${REPO_ROOT}/scripts/vastai/defaults.local.env"
if [ -f "${VAST_LOCAL_ENV}" ]; then
  _VAST_OVERRIDE_KEYS=(
    STAGE4_INPUT STAGE3_PREFERENCES STAGE4_OUTPUT_DIR STAGE4_OUTPUT
    REPAIR_PREFERENCES_OUT FINAL_PREFERENCES_OUT STAGE4_STATS_OUT
    STAGE4_BACKEND MODEL_PATH MODEL_BASE CONV_MODE IMAGE_ROOT
    STAGE4_MAX_NEW_TOKENS STAGE4_TEMPERATURE RESUME LIMIT STRICT
  )
  _VAST_OVERRIDES=()
  for _key in "${_VAST_OVERRIDE_KEYS[@]}"; do
    if [ "${!_key+x}" = "x" ]; then
      _VAST_OVERRIDES+=("${_key}=${!_key}")
    fi
  done
  set -a
  # shellcheck disable=SC1090
  source "${VAST_LOCAL_ENV}"
  set +a
  for _assignment in "${_VAST_OVERRIDES[@]}"; do
    export "${_assignment}"
  done
  unset _VAST_OVERRIDE_KEYS _VAST_OVERRIDES _key _assignment
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

INPUT="${STAGE4_INPUT:-output/fghd/stage3/vote_records.jsonl}"
STAGE3_PREFERENCES="${STAGE3_PREFERENCES:-output/fghd/stage3/preference_pairs.jsonl}"
OUTPUT_DIR="${STAGE4_OUTPUT_DIR:-output/fghd/stage4}"
OUTPUT="${STAGE4_OUTPUT:-${OUTPUT_DIR}/repair_records.jsonl}"
REPAIR_PREFERENCES_OUT="${REPAIR_PREFERENCES_OUT:-${OUTPUT_DIR}/repair_preferences.jsonl}"
FINAL_PREFERENCES_OUT="${FINAL_PREFERENCES_OUT:-${OUTPUT_DIR}/final_preference_pairs.jsonl}"
STATS_OUT="${STAGE4_STATS_OUT:-${OUTPUT_DIR}/stats.json}"
BACKEND="${STAGE4_BACKEND:-llava}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-13b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"
MAX_NEW_TOKENS="${STAGE4_MAX_NEW_TOKENS:-128}"
TEMPERATURE="${STAGE4_TEMPERATURE:-0.0}"
RESUME="${RESUME:-0}"

if [ ! -f "${INPUT}" ]; then
  echo "Stage 4 input not found: ${INPUT}" >&2
  echo "Run Stage 3 first:  bash scripts/run_stage3_validate.sh" >&2
  exit 1
fi

CMD=(
  python -m fg_pipeline.stage4.run_stage4_repair
  --input "${INPUT}"
  --stage3-preferences "${STAGE3_PREFERENCES}"
  --output "${OUTPUT}"
  --repair-preferences-out "${REPAIR_PREFERENCES_OUT}"
  --final-preferences-out "${FINAL_PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

if [ "${BACKEND}" = "llava" ]; then
  CMD+=(
    --model-path "${MODEL_PATH}"
    --conv-mode "${CONV_MODE}"
    --image-root "${IMAGE_ROOT}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
  )
  if [ -n "${MODEL_BASE}" ]; then
    CMD+=(--model-base "${MODEL_BASE}")
  fi
fi

"${CMD[@]}" "$@"
