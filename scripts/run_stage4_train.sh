#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VAST_LOCAL_ENV="${REPO_ROOT}/scripts/vastai/defaults.local.env"
if [ -f "${VAST_LOCAL_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${VAST_LOCAL_ENV}"
  set +a
fi

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/output/fghd/stage3/preference_pairs.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/fghd/stage4_llava}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "Stage 4 input not found: ${DATA_PATH}" >&2
  echo "Run Stage 3 first:  bash scripts/run_stage3_validate.sh" >&2
  exit 1
fi

echo "Stage 4 wrapper uses Stage 3 preference pairs."
echo "Current limitation: the released detection data does not expose the original"
echo "user prompt separately, so the Stage 3 'question' field may mirror the"
echo "assessed candidate sentence from Stage 1."

DATA_PATH="${DATA_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${REPO_ROOT}/hsa_dpo_train.sh" "$@"
