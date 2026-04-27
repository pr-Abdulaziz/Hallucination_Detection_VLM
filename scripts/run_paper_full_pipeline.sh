#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "Paper pipeline: Stage 1 FAIF"
bash scripts/run_paper_stage1_faif.sh

echo "Paper pipeline: Stage 2 detector dataset"
bash scripts/run_paper_stage2_detector_dataset.sh

if [ "${RUN_DETECTOR_TRAIN:-1}" = "1" ]; then
  echo "Paper pipeline: Stage 3 detector training"
  bash scripts/run_paper_stage3_detector_train.sh
else
  echo "RUN_DETECTOR_TRAIN=0; skipping detector training"
fi

if [ "${RUN_DETECTOR_INFER:-1}" = "1" ]; then
  echo "Paper pipeline: Stage 3 detector inference"
  bash scripts/run_paper_stage3_detect.sh
else
  echo "RUN_DETECTOR_INFER=0; skipping detector inference"
fi

if [ "${RUN_REWRITE:-1}" = "1" ]; then
  echo "Paper pipeline: Stage 4 detect-then-rewrite"
  bash scripts/run_paper_stage4_rewrite.sh
else
  echo "RUN_REWRITE=0; skipping Stage 4 rewrite"
fi

if [ "${RUN_HSA:-1}" = "1" ]; then
  echo "Paper pipeline: Stage 5 HSA-DPO"
  bash scripts/run_paper_stage5_train_hsa.sh
else
  echo "RUN_HSA=0; skipping HSA-DPO training"
fi
