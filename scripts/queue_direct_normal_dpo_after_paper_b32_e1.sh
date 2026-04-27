#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PAPER_OUTPUT="${PAPER_OUTPUT:-output/fghd/exp_direct_paper_hsa_b32_e1}"
PAPER_LOG="${PAPER_LOG:-output/fghd/stage5_paper_hsa_b32_e1.log}"
NORMAL_LOG="${NORMAL_LOG:-output/fghd/stage5_normal_dpo_b32_e1.log}"
CHECK_EVERY="${CHECK_EVERY:-60}"

mkdir -p "$(dirname "${NORMAL_LOG}")"

echo "Waiting for paper HSA-DPO batch32 epoch1 run to finish: ${PAPER_OUTPUT}"
while pgrep -af "train_dpo.py|deepspeed" | grep -F "${PAPER_OUTPUT}" >/dev/null 2>&1; do
  sleep "${CHECK_EVERY}"
done

if [ ! -f "${PAPER_OUTPUT}/trainer_state.json" ] && ! grep -q "Training completed" "${PAPER_LOG}" 2>/dev/null; then
  echo "Paper HSA-DPO batch32 epoch1 does not look complete; normal DPO will not start." >&2
  echo "Expected ${PAPER_OUTPUT}/trainer_state.json or Training completed in ${PAPER_LOG}." >&2
  exit 1
fi

if pgrep -af "train_dpo.py|deepspeed" | grep -E "exp_direct_normal_dpo_b32_e1|exp_direct_margin_hsa" >/dev/null 2>&1; then
  echo "Another direct Stage 5 training run is already active; normal DPO will not start." >&2
  exit 1
fi

echo "Starting normal DPO batch32 epoch1 baseline. Log: ${NORMAL_LOG}"
bash scripts/run_direct_stage5_normal_dpo_batch32_epoch1.sh > "${NORMAL_LOG}" 2>&1
