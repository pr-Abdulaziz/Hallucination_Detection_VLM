#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VAST_LOCAL_ENV="${REPO_ROOT}/scripts/vastai/defaults.local.env"
if [ -f "${VAST_LOCAL_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${VAST_LOCAL_ENV}"
  set +a
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

INPUT="${INPUT:-output/fghd/stage2/rewrites.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/stage3}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/vote_records.jsonl}"
PREFERENCES_OUT="${PREFERENCES_OUT:-${OUTPUT_DIR}/preference_pairs.jsonl}"
STATS_OUT="${STATS_OUT:-${OUTPUT_DIR}/stats.json}"
BACKEND="${BACKEND:-heuristic}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-}"
LLAVA_MODEL_BASE="${LLAVA_MODEL_BASE:-}"
LLAVA_CONV_MODE="${LLAVA_CONV_MODE:-vicuna_v1}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-256}"
LLAVA_MAX_NEW_TOKENS="${LLAVA_MAX_NEW_TOKENS:-256}"

if [ "${BACKEND}" = "heuristic" ] && [ -n "${QWEN_MODEL_PATH}" ] && [ -n "${LLAVA_MODEL_PATH}" ]; then
  BACKEND="qwen_llava_ensemble"
fi

if [ ! -f "${INPUT}" ]; then
  echo "Stage 3 input not found: ${INPUT}" >&2
  echo "Run Stage 2 first:  bash scripts/run_stage2_rewrites.sh" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m fg_pipeline.stage3.run_stage3
  --input "${INPUT}"
  --output "${OUTPUT}"
  --preferences-out "${PREFERENCES_OUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

if [ "${BACKEND}" = "qwen_llava_ensemble" ]; then
  if [ -z "${QWEN_MODEL_PATH}" ] || [ -z "${LLAVA_MODEL_PATH}" ]; then
    echo "qwen_llava_ensemble requires QWEN_MODEL_PATH and LLAVA_MODEL_PATH" >&2
    exit 1
  fi
  CMD+=(
    --qwen-model-path "${QWEN_MODEL_PATH}"
    --llava-model-path "${LLAVA_MODEL_PATH}"
    --image-root "${IMAGE_ROOT}"
    --llava-conv-mode "${LLAVA_CONV_MODE}"
    --qwen-max-new-tokens "${QWEN_MAX_NEW_TOKENS}"
    --llava-max-new-tokens "${LLAVA_MAX_NEW_TOKENS}"
  )
  if [ -n "${LLAVA_MODEL_BASE}" ]; then
    CMD+=(--llava-model-base "${LLAVA_MODEL_BASE}")
  fi
fi

"${CMD[@]}" "$@"
