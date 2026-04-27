#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

INPUT="${INPUT:-output/fghd/paper_stage1/d_faif.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/paper_stage2/detector_train.json}"
STATS_OUT="${STATS_OUT:-output/fghd/paper_stage2/detector_split_stats.json}"
SEED="${SEED:-42}"
NON_HALLUCINATED_RATIO="${NON_HALLUCINATED_RATIO:-1.2}"

CMD=(
  python -m fg_pipeline.paper.run_stage2_detector_dataset
  --input "${INPUT}"
  --output "${OUTPUT}"
  --stats-out "${STATS_OUT}"
  --seed "${SEED}"
  --non-hallucinated-ratio "${NON_HALLUCINATED_RATIO}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}" "$@"
