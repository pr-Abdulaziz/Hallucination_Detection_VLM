#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT="${INPUT:-output/fghd/D_det.jsonl}"
REPORT="${REPORT:-output/fghd/D_det_calibration.json}"
OUTPUT_CALIBRATED="${OUTPUT_CALIBRATED:-output/fghd/D_det_calibrated.jsonl}"

python -m fg_pipeline.confidence.run_calibrate \
  --input "${INPUT}" \
  --report "${REPORT}" \
  --output-calibrated "${OUTPUT_CALIBRATED}" \
  "$@"
