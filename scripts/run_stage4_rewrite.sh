#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Usage:
#   scripts/run_stage4_rewrite.sh [INPUT] [OUTPUT] [BACKEND] [EXTRA_ARGS...]
#
# Examples:
#   # Smoke-only template backend (CPU, offline)
#   scripts/run_stage4_rewrite.sh \
#     output/fghd/D_det.jsonl output/fghd/D_rewrite.jsonl template \
#     --confidence-threshold 0.5 --limit 20
#
#   # Real Stage 4 rewrite with LLaVA-v1.5 on GPU
#   scripts/run_stage4_rewrite.sh \
#     output/fghd/D_det.jsonl output/fghd/D_rewrite.jsonl llava \
#     --model-path /models/llava-v1.5-13b \
#     --image-root /data/vg \
#     --confidence-threshold 0.5

INPUT="${1:-output/fghd/D_det.jsonl}"
OUTPUT="${2:-output/fghd/D_rewrite.jsonl}"
BACKEND="${3:-template}"
shift || true
shift || true
shift || true

python -m fg_pipeline.rewrite.run_rewrite \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --backend "${BACKEND}" \
  "$@"
