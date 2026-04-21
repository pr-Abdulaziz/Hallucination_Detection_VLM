#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Usage:
#   scripts/run_stage5_verify.sh [INPUT] [OUTPUT] [BACKEND] [EXTRA_ARGS...]
#
# Examples:
#   scripts/run_stage5_verify.sh \
#     output/fghd/D_rewrite.jsonl output/fghd/D_pref_clean.jsonl heuristic \
#     --threshold-report output/fghd/D_tau_c_report.json --limit 1000

INPUT="${1:-output/fghd/D_rewrite.jsonl}"
OUTPUT="${2:-output/fghd/D_pref_clean.jsonl}"
BACKEND="${3:-heuristic}"
shift || true
shift || true
shift || true

python -m fg_pipeline.verification.run_verify \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --backend "${BACKEND}" \
  "$@"
