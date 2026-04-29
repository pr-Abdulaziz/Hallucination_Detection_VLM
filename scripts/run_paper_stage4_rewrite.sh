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

INPUT="${INPUT:-output/fghd/paper_stage1/d_faif.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/paper_stage4/rewrite_records.jsonl}"
PREFERENCES_OUT="${PREFERENCES_OUT:-output/fghd/paper_stage4/preference_pairs.jsonl}"
STATS_OUT="${STATS_OUT:-output/fghd/paper_stage4/stats.json}"
BACKEND="${BACKEND:-llava}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

CMD=(
  python -m fg_pipeline.paper.run_stage4_rewrite
  --input "${INPUT}"
  --output "${OUTPUT}"
  --preferences-out "${PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
  --image-root "${IMAGE_ROOT}"
  --temperature "${TEMPERATURE}"
)

if [ "${BACKEND}" = "llava" ] || [ "${BACKEND}" = "llava_api_critic" ]; then
  CMD+=(--model-path "${MODEL_PATH}" --conv-mode "${CONV_MODE}")
  if [ -n "${MODEL_BASE:-}" ]; then
    CMD+=(--model-base "${MODEL_BASE}")
  fi
  if [ -n "${MAX_NEW_TOKENS:-}" ]; then
    CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
  fi
fi
if [ "${BACKEND}" = "llava_api_critic" ]; then
  CMD+=(--api-critic "${API_CRITIC:-gemini_openai}")
  CMD+=(--gemini-model "${GEMINI_MODEL:-gemini-2.5-flash-lite}")
  CMD+=(--openai-model "${OPENAI_MODEL:-gpt-4o-mini}")
  CMD+=(--critic-timeout-seconds "${CRITIC_TIMEOUT_SECONDS:-60}")
  CMD+=(--critic-retries "${CRITIC_RETRIES:-3}")
  if [ -n "${CRITIC_MAX_OUTPUT_TOKENS:-}" ]; then
    CMD+=(--critic-max-output-tokens "${CRITIC_MAX_OUTPUT_TOKENS}")
  fi
fi
if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi
if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi
if [ "${ALLOW_MISSING_IMAGES:-0}" = "1" ]; then
  CMD+=(--allow-missing-images)
fi

"${CMD[@]}" "$@"
