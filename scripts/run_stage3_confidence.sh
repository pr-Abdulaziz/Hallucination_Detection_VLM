#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT="${INPUT:-fg_pipeline/data/hsa_dpo_detection.jsonl}"
OUTPUT="${OUTPUT:-output/fghd/D_det.jsonl}"
SCORER="${SCORER:-bootstrap}"

EXTRA_ARGS=()

if [[ "${SCORER}" == "log_prob" ]]; then
  MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-13b}"
  DEVICE="${DEVICE:-auto}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"
  EXTRA_ARGS+=(
    --model-path "${MODEL_PATH}"
    --device "${DEVICE}"
    --temperature "${TEMPERATURE}"
    --image-root "${IMAGE_ROOT}"
  )
fi

python -m fg_pipeline.confidence.run_detect \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --scorer "${SCORER}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
