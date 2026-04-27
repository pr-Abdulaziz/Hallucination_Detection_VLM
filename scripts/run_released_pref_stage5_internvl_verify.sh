#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

REPAIRED_INPUT="${REPAIRED_INPUT:-output/fghd/released_pref_stage4_and_gate/repaired_preferences.jsonl}"
ACCEPTED_INPUT="${ACCEPTED_INPUT:-output/fghd/released_pref_stage3_and_gate/passed_by_either.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fghd/released_pref_stage5_internvl_verify}"
# Do not reuse the project-wide MODEL_PATH from .env here; that usually points
# to LLaVA. This verifier must load the local InternVL model unless explicitly
# overridden with INTERNVL_MODEL_PATH.
INTERNVL_MODEL_PATH="${INTERNVL_MODEL_PATH:-/root/models/InternVL-Chat-V1-2-Plus}"
IMAGE_ROOT="${IMAGE_ROOT:-hsa_dpo/data/images}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"

CMD=(
  python -m fg_pipeline.paper.run_released_pref_stage5_internvl_verify
  --repaired-input "${REPAIRED_INPUT}"
  --accepted-input "${ACCEPTED_INPUT}"
  --verification-out "${OUTPUT_DIR}/verification_records.jsonl"
  --approved-out "${OUTPUT_DIR}/approved_repaired_preferences.jsonl"
  --failed-out "${OUTPUT_DIR}/failed_repaired_preferences.jsonl"
  --final-preferences-out "${OUTPUT_DIR}/final_verified_preference_pairs.jsonl"
  --stats-out "${OUTPUT_DIR}/stats.json"
  --model-path "${INTERNVL_MODEL_PATH}"
  --image-root "${IMAGE_ROOT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
)

if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi
if [ "${RESUME:-0}" = "1" ]; then
  CMD+=(--resume)
fi
if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi
if [ "${LOAD_IN_8BIT:-1}" = "0" ]; then
  CMD+=(--no-load-in-8bit)
fi

"${CMD[@]}" "$@"
