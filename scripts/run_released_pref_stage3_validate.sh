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

INPUT="${INPUT:-hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-hsa_dpo/data/images}"
SHOT_MODE="${SHOT_MODE:-${PROMPT_MODE:-zero_shot}}"
case "${SHOT_MODE}" in
  zero|zero_shot)
    PROMPT_MODE="zero_shot"
    DEFAULT_STAGE3_DIR="output/fghd/released_pref_stage3"
    ;;
  two|2shot|two_shot)
    PROMPT_MODE="two_shot"
    DEFAULT_STAGE3_DIR="output/fghd/released_pref_stage3_2shot_experiment"
    ;;
  *)
    echo "Unsupported SHOT_MODE/PROMPT_MODE: ${SHOT_MODE}. Use zero_shot or two_shot." >&2
    exit 2
    ;;
esac
STAGE3_DIR="${STAGE3_DIR:-${DEFAULT_STAGE3_DIR}}"
ACCEPTED_OUT="${ACCEPTED_OUT:-${STAGE3_DIR}/validated_preferences.jsonl}"
REJECTED_OUT="${REJECTED_OUT:-${STAGE3_DIR}/rejected_for_repair.jsonl}"
AUDIT_OUT="${AUDIT_OUT:-${STAGE3_DIR}/judgement_records.jsonl}"
STATS_OUT="${STATS_OUT:-${STAGE3_DIR}/stats.json}"

printf 'Stage 3 validation mode: %s\n' "${PROMPT_MODE}"
printf 'Stage 3 validation outputs: %s\n' "${STAGE3_DIR}"

CMD=(
  python -m fg_pipeline.paper.run_released_pref_stage3_validate
  --input "${INPUT}"
  --image-root "${IMAGE_ROOT}"
  --accepted-out "${ACCEPTED_OUT}"
  --rejected-out "${REJECTED_OUT}"
  --audit-out "${AUDIT_OUT}"
  --stats-out "${STATS_OUT}"
  --api-judge "${API_JUDGE:-gemini_openai}"
  --gemini-model "${GEMINI_MODEL:-gemini-2.5-flash-lite}"
  --openai-model "${OPENAI_MODEL:-gpt-4o-mini}"
  --decision-rule "${DECISION_RULE:-either}"
  --prompt-mode "${PROMPT_MODE}"
  --timeout-seconds "${JUDGE_TIMEOUT_SECONDS:-60}"
  --retries "${JUDGE_RETRIES:-3}"
  --workers "${WORKERS:-1}"
)

if [ -n "${JUDGE_MAX_OUTPUT_TOKENS:-}" ]; then
  CMD+=(--max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}")
fi
if [ -n "${LIMIT:-}" ]; then
  CMD+=(--limit "${LIMIT}")
fi
if [ "${STRICT:-0}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}" "$@"
