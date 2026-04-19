#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

export NUM_GPUS="${NUM_GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EPOCH="${EPOCH:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/hsa_dpo_pilot}"

echo "Starting pilot HSA-DPO run with:"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  EPOCH=${EPOCH}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

bash "${REPO_ROOT}/hsa_dpo_train.sh"
