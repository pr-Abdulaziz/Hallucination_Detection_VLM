#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m fg_pipeline.verification.run_verify \
  --input "${1:-output/fghd/D_rewrite.jsonl}" \
  --output "${2:-output/fghd/D_pref_clean.jsonl}"
