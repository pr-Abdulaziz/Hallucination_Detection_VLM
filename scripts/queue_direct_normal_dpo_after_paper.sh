#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PAPER_OUTPUT="${PAPER_OUTPUT:-output/fghd/exp_direct_paper_hsa}"
PAPER_LOG="${PAPER_LOG:-output/fghd/stage5_paper_hsa_batch16.log}"
NORMAL_LOG="${NORMAL_LOG:-output/fghd/stage5_normal_dpo_batch16.log}"
CHECK_EVERY="${CHECK_EVERY:-60}"

mkdir -p "$(dirname "${NORMAL_LOG}")"

echo "Waiting for paper HSA-DPO run to finish: ${PAPER_OUTPUT}"
while pgrep -af "train_dpo.py|deepspeed" | grep -F "${PAPER_OUTPUT}" >/dev/null 2>&1; do
  sleep "${CHECK_EVERY}"
done

if [ ! -f "${PAPER_OUTPUT}/trainer_state.json" ] && ! grep -q "Training completed" "${PAPER_LOG}" 2>/dev/null; then
  echo "Paper HSA-DPO does not look complete; normal DPO will not start." >&2
  echo "Expected ${PAPER_OUTPUT}/trainer_state.json or Training completed in ${PAPER_LOG}." >&2
  exit 1
fi

if pgrep -af "train_dpo.py|deepspeed" | grep -E "exp_direct_normal_dpo|exp_direct_margin_hsa" >/dev/null 2>&1; then
  echo "Another direct Stage 5 training run is already active; normal DPO will not start." >&2
  exit 1
fi

echo "Starting normal DPO baseline. Log: ${NORMAL_LOG}"
bash scripts/run_direct_stage5_normal_dpo_batch16.sh > "${NORMAL_LOG}" 2>&1
