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
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-}"
LLAVA_MODEL_BASE="${LLAVA_MODEL_BASE:-}"
LLAVA_CONV_MODE="${LLAVA_CONV_MODE:-vicuna_v1}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"
LLAVA_MAX_NEW_TOKENS="${LLAVA_MAX_NEW_TOKENS:-64}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"
GEMINI_MAX_OUTPUT_TOKENS="${GEMINI_MAX_OUTPUT_TOKENS:-64}"
ROW_WORKERS="${ROW_WORKERS:-1}"
LLAVA_DEVICE="${LLAVA_DEVICE:-}"
RESUME="${RESUME:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"

if [ "${BACKEND}" = "heuristic" ] && [ -n "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]; then
  BACKEND="gemini_two_vote"
fi

if [ "${BACKEND}" = "gemini_two_vote" ] && [ "${ROW_WORKERS}" = "1" ]; then
  ROW_WORKERS=4
fi

if [ "${BACKEND}" = "gemini_llava_two_vote" ] && [ -z "${LLAVA_DEVICE}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    if [ "${GPU_COUNT}" -ge 1 ]; then
      LLAVA_DEVICE="cuda:0"
    fi
  fi
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
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --row-workers "${ROW_WORKERS}"
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

if [ "${BACKEND}" = "gemini_llava_two_vote" ]; then
  if [ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ] || [ -z "${LLAVA_MODEL_PATH}" ]; then
    echo "gemini_llava_two_vote requires GEMINI_API_KEY or GOOGLE_API_KEY, plus LLAVA_MODEL_PATH" >&2
    exit 1
  fi
  CMD+=(
    --llava-model-path "${LLAVA_MODEL_PATH}"
    --image-root "${IMAGE_ROOT}"
    --llava-conv-mode "${LLAVA_CONV_MODE}"
    --llava-max-new-tokens "${LLAVA_MAX_NEW_TOKENS}"
    --gemini-model "${GEMINI_MODEL}"
    --gemini-max-output-tokens "${GEMINI_MAX_OUTPUT_TOKENS}"
  )
  if [ -n "${LLAVA_DEVICE}" ]; then
    CMD+=(--llava-device "${LLAVA_DEVICE}")
  fi
  if [ -n "${LLAVA_MODEL_BASE}" ]; then
    CMD+=(--llava-model-base "${LLAVA_MODEL_BASE}")
  fi
fi

if [ "${BACKEND}" = "gemini_two_vote" ]; then
  if [ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]; then
    echo "gemini_two_vote requires GEMINI_API_KEY or GOOGLE_API_KEY" >&2
    exit 1
  fi
  CMD+=(
    --image-root "${IMAGE_ROOT}"
    --gemini-model "${GEMINI_MODEL}"
    --gemini-max-output-tokens "${GEMINI_MAX_OUTPUT_TOKENS}"
  )
fi

"${CMD[@]}" "$@"
