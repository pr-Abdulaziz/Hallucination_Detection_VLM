#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

USER_OPENAI_MODEL="${OPENAI_MODEL:-}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
USER_REPAIRED_INPUT="${REPAIRED_INPUT:-}"
USER_BASE_ACCEPTED_INPUT="${BASE_ACCEPTED_INPUT:-}"
USER_IMAGE_ROOT="${IMAGE_ROOT:-}"
USER_JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-}"
USER_JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-}"
USER_JUDGE_RETRIES="${JUDGE_RETRIES:-}"
USER_SHOT_MODE="${SHOT_MODE:-${EXPERIMENT_MODE:-${PROMPT_MODE:-}}}"
USER_PROMPT_MODE="${PROMPT_MODE:-}"

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

SHOT_MODE="${USER_SHOT_MODE:-${SHOT_MODE:-${EXPERIMENT_MODE:-${PROMPT_MODE:-zero_shot}}}}"
case "${SHOT_MODE}" in
  zero|zero_shot)
    SHOT_MODE="zero_shot"
    DEFAULT_REPAIRED_INPUT="output/fghd/released_pref_stage4_and_gate/repaired_preferences.jsonl"
    DEFAULT_BASE_ACCEPTED_INPUT="output/fghd/released_pref_stage3_and_gate/passed_by_either.jsonl"
    DEFAULT_OUTPUT_DIR="output/fghd/released_pref_stage5_openai_verify"
    ;;
  two|2shot|two_shot)
    SHOT_MODE="two_shot"
    DEFAULT_REPAIRED_INPUT="output/fghd/released_pref_stage4_2shot_experiment/repaired_preferences.jsonl"
    DEFAULT_BASE_ACCEPTED_INPUT="output/fghd/released_pref_stage3_2shot_experiment/validated_preferences.jsonl"
    DEFAULT_OUTPUT_DIR="output/fghd/released_pref_stage5_openai_verify_2shot_experiment"
    ;;
  *)
    echo "Unsupported SHOT_MODE/EXPERIMENT_MODE/PROMPT_MODE: ${SHOT_MODE}. Use zero_shot or two_shot." >&2
    exit 2
    ;;
esac

PROMPT_MODE="${USER_PROMPT_MODE:-${SHOT_MODE}}"
if [ "${PROMPT_MODE}" = "zero" ]; then
  PROMPT_MODE="zero_shot"
elif [ "${PROMPT_MODE}" = "two" ] || [ "${PROMPT_MODE}" = "2shot" ]; then
  PROMPT_MODE="two_shot"
fi
if [ "${PROMPT_MODE}" != "zero_shot" ] && [ "${PROMPT_MODE}" != "two_shot" ]; then
  echo "Unsupported PROMPT_MODE: ${PROMPT_MODE}. Use zero_shot or two_shot." >&2
  exit 2
fi

REPAIRED_INPUT="${USER_REPAIRED_INPUT:-${REPAIRED_INPUT:-${DEFAULT_REPAIRED_INPUT}}}"
BASE_ACCEPTED_INPUT="${USER_BASE_ACCEPTED_INPUT:-${BASE_ACCEPTED_INPUT:-${DEFAULT_BASE_ACCEPTED_INPUT}}}"
IMAGE_ROOT="${USER_IMAGE_ROOT:-${IMAGE_ROOT:-hsa_dpo/data/images}}"
OUTPUT_DIR="${USER_OUTPUT_DIR:-${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}}"
OPENAI_MODEL="${USER_OPENAI_MODEL:-${OPENAI_MODEL:-gpt-4.1-mini}}"
JUDGE_TIMEOUT_SECONDS="${USER_JUDGE_TIMEOUT_SECONDS:-${JUDGE_TIMEOUT_SECONDS:-90}}"
JUDGE_RETRIES="${USER_JUDGE_RETRIES:-${JUDGE_RETRIES:-4}}"
JUDGE_MAX_OUTPUT_TOKENS="${USER_JUDGE_MAX_OUTPUT_TOKENS:-${JUDGE_MAX_OUTPUT_TOKENS:-512}}"

mkdir -p "${OUTPUT_DIR}"

printf 'OpenAI repair verification mode: %s\n' "${SHOT_MODE}"
printf 'OpenAI repair verification prompt mode: %s\n' "${PROMPT_MODE}"
printf 'OpenAI verifier model: %s\n' "${OPENAI_MODEL}"
printf 'Repaired input: %s\n' "${REPAIRED_INPUT}"
printf 'Base accepted input: %s\n' "${BASE_ACCEPTED_INPUT}"
printf 'Verification output: %s\n' "${OUTPUT_DIR}"

VALIDATE_CMD=(
  python -m fg_pipeline.paper.run_released_pref_stage3_validate
  --input "${REPAIRED_INPUT}"
  --image-root "${IMAGE_ROOT}"
  --accepted-out "${OUTPUT_DIR}/approved_repaired_preferences.jsonl"
  --rejected-out "${OUTPUT_DIR}/failed_repaired_preferences.jsonl"
  --audit-out "${OUTPUT_DIR}/verification_records.jsonl"
  --stats-out "${OUTPUT_DIR}/verification_stats.json"
  --api-judge openai
  --openai-model "${OPENAI_MODEL}"
  --decision-rule either
  --prompt-mode "${PROMPT_MODE}"
  --timeout-seconds "${JUDGE_TIMEOUT_SECONDS}"
  --retries "${JUDGE_RETRIES}"
  --max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}"
)

if [ -n "${LIMIT:-}" ]; then
  VALIDATE_CMD+=(--limit "${LIMIT}")
fi
if [ "${STRICT:-0}" = "1" ]; then
  VALIDATE_CMD+=(--strict)
fi

"${VALIDATE_CMD[@]}"

python - "${BASE_ACCEPTED_INPUT}" \
  "${OUTPUT_DIR}/approved_repaired_preferences.jsonl" \
  "${OUTPUT_DIR}/failed_repaired_preferences.jsonl" \
  "${OUTPUT_DIR}/verification_stats.json" \
  "${OUTPUT_DIR}/final_verified_preference_pairs.jsonl" \
  "${OUTPUT_DIR}/stats.json" \
  "${OPENAI_MODEL}" \
  "${PROMPT_MODE}" \
  "${SHOT_MODE}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
approved_path = Path(sys.argv[2])
failed_path = Path(sys.argv[3])
verification_stats_path = Path(sys.argv[4])
final_path = Path(sys.argv[5])
stats_path = Path(sys.argv[6])
model = sys.argv[7]
prompt_mode = sys.argv[8]
shot_mode = sys.argv[9]

def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

base_rows = read_jsonl(base_path)
approved_rows = read_jsonl(approved_path)
failed_rows = read_jsonl(failed_path)
final_rows = base_rows + approved_rows
write_jsonl(final_path, final_rows)

verification_stats = {}
if verification_stats_path.exists():
    verification_stats = json.loads(verification_stats_path.read_text(encoding="utf-8"))

payload = {
    "stage": "released_pref_stage5_openai_verify",
    "verifier_model": model,
    "prompt_mode": prompt_mode,
    "experiment_mode": shot_mode,
    "base_accepted_input": str(base_path),
    "approved_repaired_input": str(approved_path),
    "failed_repaired_input": str(failed_path),
    "base_accepted_rows": len(base_rows),
    "approved_repaired_rows": len(approved_rows),
    "failed_repaired_rows": len(failed_rows),
    "checked_repaired_rows": len(approved_rows) + len(failed_rows),
    "final_preference_rows": len(final_rows),
    "final_preferences_out": str(final_path),
    "verification_stats": verification_stats,
}
stats_path.parent.mkdir(parents=True, exist_ok=True)
stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    f"OpenAI Stage 5 verified {payload['checked_repaired_rows']} repaired row(s): "
    f"{payload['approved_repaired_rows']} approved, {payload['failed_repaired_rows']} failed"
)
print(f"Final verified preferences -> {final_path}")
print(f"Stats -> {stats_path}")
PY
