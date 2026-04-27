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

INPUT="${INPUT:-fg_pipeline/data/hsa_dpo_detection.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/paper_stage3/detections.jsonl}"
STATS_OUT="${STATS_OUT:-output/fghd/paper_stage3/detection_stats.json}"
BACKEND="${BACKEND:-llava_detector}"
if [ "${BACKEND}" = "api_judge" ]; then
  MODEL_PATH="${MODEL_PATH:-}"
  MODEL_BASE="${MODEL_BASE:-}"
else
  MODEL_PATH="${MODEL_PATH:-output/fghd/paper_stage3/detector_lora}"
  MODEL_BASE="${MODEL_BASE:-models/llava-v1.5-7b}"
fi
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"

CMD=(
  python -m fg_pipeline.paper.run_stage3_detect
  --input "${INPUT}"
  --output "${OUTPUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
  --image-root "${IMAGE_ROOT}"
  --max-new-tokens "${MAX_NEW_TOKENS:-384}"
  --torch-dtype "${TORCH_DTYPE:-bfloat16}"
  --min-pixels "${MIN_PIXELS:-200704}"
  --max-pixels "${MAX_PIXELS:-401408}"
  --api-judge "${API_JUDGE:-gemini_openai}"
  --gemini-model "${GEMINI_MODEL:-gemini-2.5-flash-lite}"
  --openai-model "${OPENAI_MODEL:-gpt-4o-mini}"
  --judge-timeout-seconds "${JUDGE_TIMEOUT_SECONDS:-60}"
  --judge-retries "${JUDGE_RETRIES:-3}"
  --api-decision-rule "${API_DECISION_RULE:-either}"
)

if [ "${BACKEND}" = "llava_detector" ]; then
  CMD+=(--model-path "${MODEL_PATH}" --model-base "${MODEL_BASE}" --conv-mode "${CONV_MODE:-vicuna_v1}")
fi
if [ -n "${JUDGE_MAX_OUTPUT_TOKENS:-}" ]; then
  CMD+=(--judge-max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}")
fi
if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi
if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}" "$@"
