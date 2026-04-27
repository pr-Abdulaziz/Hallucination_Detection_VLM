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

INPUT="${INPUT:-output/fghd/paper_stage3/detections.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/paper_stage4/rewrite_records.jsonl}"
PREFERENCES_OUT="${PREFERENCES_OUT:-output/fghd/paper_stage4/preference_pairs.jsonl}"
STATS_OUT="${STATS_OUT:-output/fghd/paper_stage4/stats.json}"
BACKEND="${BACKEND:-llava_api_critic}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"
API_CRITIC="${API_CRITIC:-gemini_openai}"

CMD=(
  python -m fg_pipeline.paper.run_stage4_rewrite
  --input "${INPUT}"
  --output "${OUTPUT}"
  --preferences-out "${PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
  --image-root "${IMAGE_ROOT}"
)

if [ "${BACKEND}" = "llava" ] || [ "${BACKEND}" = "llava_api_critic" ]; then
  CMD+=(--model-path "${MODEL_PATH}" --conv-mode "${CONV_MODE:-vicuna_v1}" --temperature "${TEMPERATURE:-0.0}")
  if [ -n "${MAX_NEW_TOKENS:-}" ]; then
    CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  fi
  if [ -n "${MODEL_BASE:-}" ]; then
    CMD+=(--model-base "${MODEL_BASE}")
  fi
fi
if [ "${BACKEND}" = "llava_api_critic" ]; then
  CMD+=(
    --api-critic "${API_CRITIC}"
    --gemini-model "${GEMINI_MODEL:-gemini-2.5-flash-lite}"
    --openai-model "${OPENAI_MODEL:-gpt-4o-mini}"
    --critic-timeout-seconds "${CRITIC_TIMEOUT_SECONDS:-60}"
    --critic-retries "${CRITIC_RETRIES:-3}"
  )
  if [ -n "${CRITIC_MAX_OUTPUT_TOKENS:-}" ]; then
    CMD+=(--critic-max-output-tokens "${CRITIC_MAX_OUTPUT_TOKENS}")
  fi
fi
if [ "${ALLOW_MISSING_IMAGES:-0}" = "1" ]; then
  CMD+=(--allow-missing-images)
fi
if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi
if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}" "$@"
