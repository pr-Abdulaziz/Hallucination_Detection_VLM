#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

INPUT="${INPUT:-fg_pipeline/data/hsa_dpo_detection.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/paper_stage1/d_faif.jsonl}"
STATS_OUT="${STATS_OUT:-output/fghd/paper_stage1/stats.json}"

CMD=(
  python -m fg_pipeline.paper.run_stage1_faif
  --input "${INPUT}"
  --output "${OUTPUT}"
  --stats-out "${STATS_OUT}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}" "$@"
