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
ACCEPTED_OUT="${ACCEPTED_OUT:-output/fghd/released_pref_stage3/validated_preferences.jsonl}"
REJECTED_OUT="${REJECTED_OUT:-output/fghd/released_pref_stage3/rejected_for_repair.jsonl}"
AUDIT_OUT="${AUDIT_OUT:-output/fghd/released_pref_stage3/judgement_records.jsonl}"
STATS_OUT="${STATS_OUT:-output/fghd/released_pref_stage3/stats.json}"

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
  --timeout-seconds "${JUDGE_TIMEOUT_SECONDS:-60}"
  --retries "${JUDGE_RETRIES:-3}"
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
