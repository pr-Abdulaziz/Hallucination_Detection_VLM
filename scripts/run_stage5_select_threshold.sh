#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT="${1:-output/fghd/D_rewrite.jsonl}"
OUTPUT_REPORT="${2:-output/fghd/D_tau_c_report.json}"
BACKEND="${3:-heuristic}"
shift || true
shift || true
shift || true

python -m fg_pipeline.verification.run_select_threshold \
  --input "${INPUT}" \
  --output-report "${OUTPUT_REPORT}" \
  --backend "${BACKEND}" \
  "$@"
