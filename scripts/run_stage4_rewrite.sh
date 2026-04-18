#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m fg_pipeline.rewrite.run_rewrite \
  --input "${1:-output/fghd/D_det.jsonl}" \
  --output "${2:-output/fghd/D_rewrite.jsonl}"
