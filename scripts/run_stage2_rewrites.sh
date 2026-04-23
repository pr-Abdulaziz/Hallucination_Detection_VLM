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

INPUT="${INPUT:-output/fghd/stage1/detection_critiques.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/stage2}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/rewrites.jsonl}"
STATS_OUT="${STATS_OUT:-${OUTPUT_DIR}/stats.json}"
BACKEND="${BACKEND:-template}"

if [ ! -f "${INPUT}" ]; then
  echo "Stage 2 input not found: ${INPUT}" >&2
  echo "Run Stage 1 first:  bash scripts/run_stage1_critiques.sh" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m fg_pipeline.stage2.run_stage2
  --input "${INPUT}"
  --output "${OUTPUT}"
  --stats-out "${STATS_OUT}"
  --backend "${BACKEND}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

if [ -n "${MODEL_PATH:-}" ]; then
  CMD+=(--model-path "${MODEL_PATH}")
fi

if [ -n "${MODEL_BASE:-}" ]; then
  CMD+=(--model-base "${MODEL_BASE}")
fi

if [ -n "${CONV_MODE:-}" ]; then
  CMD+=(--conv-mode "${CONV_MODE}")
fi

if [ -n "${IMAGE_ROOT:-}" ]; then
  CMD+=(--image-root "${IMAGE_ROOT}")
fi

"${CMD[@]}" "$@"
