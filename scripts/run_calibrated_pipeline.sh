#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

D_DET_INPUT="${D_DET_INPUT:-output/fghd/D_det.jsonl}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:-output/fghd/D_det_calibration.json}"
CALIBRATED_DET="${CALIBRATED_DET:-output/fghd/D_det_calibrated.jsonl}"
REWRITE_OUTPUT="${REWRITE_OUTPUT:-output/fghd/D_rewrite_grouped.jsonl}"
TAU_C_REPORT="${TAU_C_REPORT:-output/fghd/D_tau_c_report_grouped.json}"
PREF_OUTPUT="${PREF_OUTPUT:-output/fghd/D_pref_clean_grouped.jsonl}"
REWRITE_BACKEND="${REWRITE_BACKEND:-llava}"
VERIFY_BACKEND="${VERIFY_BACKEND:-heuristic}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/llava-v1.5-13b}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}}"
CRC_METHOD="${CRC_METHOD:-cv_crc}"
CRC_ALPHA="${CRC_ALPHA:-0.10}"
CRC_FOLDS="${CRC_FOLDS:-5}"
MIN_ACCEPTED="${MIN_ACCEPTED:-100}"
RUN_STAGE6="${RUN_STAGE6:-auto}"          # auto | true | false
MIN_STAGE6_GPUS="${MIN_STAGE6_GPUS:-2}"   # paper-like default

detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
    return
  fi
  echo 0
}

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "${path}" ]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

AVAILABLE_GPUS="$(detect_gpu_count)"
SHOULD_RUN_STAGE6="false"
case "${RUN_STAGE6}" in
  true)
    if [ "${AVAILABLE_GPUS}" -lt "${MIN_STAGE6_GPUS}" ]; then
      echo "RUN_STAGE6=true but only ${AVAILABLE_GPUS} GPU(s) are available; need at least ${MIN_STAGE6_GPUS} for the current Stage 6 setup." >&2
      exit 1
    fi
    SHOULD_RUN_STAGE6="true"
    ;;
  false)
    SHOULD_RUN_STAGE6="false"
    ;;
  auto)
    if [ "${AVAILABLE_GPUS}" -ge "${MIN_STAGE6_GPUS}" ]; then
      SHOULD_RUN_STAGE6="true"
    fi
    ;;
  *)
    echo "RUN_STAGE6 must be one of: auto, true, false" >&2
    exit 1
    ;;
esac

require_path "${D_DET_INPUT}" "Stage 3 D_det input"
require_path "${MODEL_PATH}" "LLaVA model directory"
require_path "${IMAGE_ROOT}" "image root"

echo "Running calibrated grouped pipeline with:"
echo "  D_DET_INPUT=${D_DET_INPUT}"
echo "  CALIBRATION_REPORT=${CALIBRATION_REPORT}"
echo "  CALIBRATED_DET=${CALIBRATED_DET}"
echo "  REWRITE_OUTPUT=${REWRITE_OUTPUT}"
echo "  TAU_C_REPORT=${TAU_C_REPORT}"
echo "  PREF_OUTPUT=${PREF_OUTPUT}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  IMAGE_ROOT=${IMAGE_ROOT}"
echo "  CRC_METHOD=${CRC_METHOD}"
echo "  CRC_ALPHA=${CRC_ALPHA}"
echo "  CRC_FOLDS=${CRC_FOLDS}"
echo "  MIN_ACCEPTED=${MIN_ACCEPTED}"
echo "  AVAILABLE_GPUS=${AVAILABLE_GPUS}"
echo "  RUN_STAGE6=${RUN_STAGE6}"

OUTPUT_CALIBRATED="${CALIBRATED_DET}" \
REPORT="${CALIBRATION_REPORT}" \
INPUT="${D_DET_INPUT}" \
bash scripts/run_stage3_calibrate.sh

bash scripts/run_stage4_rewrite.sh \
  "${CALIBRATED_DET}" \
  "${REWRITE_OUTPUT}" \
  "${REWRITE_BACKEND}" \
  --model-path "${MODEL_PATH}" \
  --image-root "${IMAGE_ROOT}" \
  --threshold-report "${CALIBRATION_REPORT}"

bash scripts/run_stage5_select_threshold.sh \
  "${REWRITE_OUTPUT}" \
  "${TAU_C_REPORT}" \
  "${VERIFY_BACKEND}" \
  --method "${CRC_METHOD}" \
  --alpha "${CRC_ALPHA}" \
  --folds "${CRC_FOLDS}" \
  --min-accepted "${MIN_ACCEPTED}"

bash scripts/run_stage5_verify.sh \
  "${REWRITE_OUTPUT}" \
  "${PREF_OUTPUT}" \
  "${VERIFY_BACKEND}" \
  --threshold-report "${TAU_C_REPORT}"

if [ "${SHOULD_RUN_STAGE6}" = "true" ]; then
  echo "Stage 6 preflight passed; starting training."
  DATA_PATH="${PREF_OUTPUT}" \
  IMAGE_FOLDER="${IMAGE_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  bash scripts/run_stage6_train.sh
else
  echo "Stage 6 skipped."
  if [ "${RUN_STAGE6}" = "auto" ]; then
    echo "  Reason: only ${AVAILABLE_GPUS} GPU(s) available, below MIN_STAGE6_GPUS=${MIN_STAGE6_GPUS}."
  fi
fi

echo "Pipeline completed."
