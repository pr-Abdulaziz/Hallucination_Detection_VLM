#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m fg_pipeline.confidence.run_detect \
  --input "${1:-hsa_dpo/data/hsa_dpo_detection.jsonl}" \
  --output "${2:-output/fghd/D_det.jsonl}"
