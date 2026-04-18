#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/output/fghd/D_pref_clean.jsonl}"
export IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/hsa_dpo/data/images}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/fghd/adaptive_dpo}"

echo "Reusing the original HSA-DPO training stack with:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

bash "${REPO_ROOT}/hsa_dpo_train.sh"
