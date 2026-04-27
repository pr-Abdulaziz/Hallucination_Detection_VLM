#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/eval}"
OLD_OUTPUT_ROOT="${OLD_OUTPUT_ROOT:-output/old_output_results}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BACKUP_DIR="${BACKUP_DIR:-${OLD_OUTPUT_ROOT}/eval_before_stage5_current_parts_${TIMESTAMP}}"
MODELS_JSON="${MODELS_JSON:-output/fghd/current_stage5_parts_models.eval.json}"
BENCHMARKS="${BENCHMARKS:-pope_adv,object_halbench,amber}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-stage5_current_parts_eval_smoke_${TIMESTAMP}}"
FULL_RUN_NAME="${FULL_RUN_NAME:-stage5_current_parts_eval_full_${TIMESTAMP}}"
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
POLL_SECONDS="${POLL_SECONDS:-120}"

BASE_MODEL="${BASE_MODEL:-models/llava-v1.5-7b}"

DIRECT_PAPER_HSA_DIR="${DIRECT_PAPER_HSA_DIR:-output/fghd/exp_direct_paper_hsa_b32_e1}"
NORMAL_DPO_DIR="${NORMAL_DPO_DIR:-output/fghd/exp_direct_normal_dpo_b32_e1}"
API_VERIFIED_MARGIN_DIR="${API_VERIFIED_MARGIN_DIR:-output/fghd/exp_api_verified_margin_hsa_b32_e1}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

backup_old_eval_outputs() {
  mkdir -p "${OLD_OUTPUT_ROOT}"
  if [ -d "${OUTPUT_ROOT}" ] && [ -z "${SKIP_EVAL_BACKUP:-}" ]; then
    if [ ! -e "${BACKUP_DIR}" ]; then
      log "Backing up existing evaluation outputs to ${BACKUP_DIR}"
      mkdir -p "${BACKUP_DIR}"
      rsync -a "${OUTPUT_ROOT}/" "${BACKUP_DIR}/"
    else
      log "Evaluation backup already exists: ${BACKUP_DIR}"
    fi
  else
    log "No existing evaluation output backup needed."
  fi
}

require_completed_lora() {
  local dir="$1"
  local label="$2"
  test -s "${dir}/adapter_model.safetensors" && test -s "${dir}/trainer_state.json" && return 0
  log "Waiting for ${label}: missing ${dir}/adapter_model.safetensors or trainer_state.json"
  return 1
}

wait_for_training_outputs() {
  log "Waiting for required training artifacts."
  while true; do
    local ready=1
    require_completed_lora "${DIRECT_PAPER_HSA_DIR}" "direct paper HSA-DPO" || ready=0
    require_completed_lora "${NORMAL_DPO_DIR}" "Part A normal DPO" || ready=0
    require_completed_lora "${API_VERIFIED_MARGIN_DIR}" "Part B API-verified margin HSA-DPO" || ready=0
    if [ "${ready}" = "1" ]; then
      log "All required training artifacts are ready."
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

write_model_manifest() {
  mkdir -p "$(dirname "${MODELS_JSON}")"
  cat >"${MODELS_JSON}" <<JSON
[
  {
    "model_id": "llava15_base_7b",
    "model_path": "${BASE_MODEL}",
    "model_base": null,
    "kind": "base",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": ${MAX_NEW_TOKENS}
  },
  {
    "model_id": "direct_paper_hsa_b32_e1",
    "model_path": "${DIRECT_PAPER_HSA_DIR}",
    "model_base": "${BASE_MODEL}",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": ${MAX_NEW_TOKENS}
  },
  {
    "model_id": "direct_normal_dpo_b32_e1",
    "model_path": "${NORMAL_DPO_DIR}",
    "model_base": "${BASE_MODEL}",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": ${MAX_NEW_TOKENS}
  },
  {
    "model_id": "api_verified_margin_hsa_b32_e1",
    "model_path": "${API_VERIFIED_MARGIN_DIR}",
    "model_base": "${BASE_MODEL}",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": ${MAX_NEW_TOKENS}
  }
]
JSON
  python -m json.tool "${MODELS_JSON}" >/dev/null
  log "Wrote model manifest: ${MODELS_JSON}"
}

preflight() {
  log "Running evaluation preflight."
  python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

repo = Path(".")
models = json.loads(Path(os.environ["MODELS_JSON"]).read_text(encoding="utf-8"))
missing = []
for model in models:
    model_path = repo / model["model_path"]
    if not model_path.exists():
        missing.append(f"missing model path: {model_path}")
    if model.get("kind") == "lora":
        for name in ("adapter_model.safetensors", "trainer_state.json"):
            if not (model_path / name).exists():
                missing.append(f"missing {name}: {model_path / name}")
    if model.get("model_base") and not (repo / model["model_base"]).exists():
        missing.append(f"missing model base: {repo / model['model_base']}")

benchmarks = [b.strip() for b in os.environ["BENCHMARKS"].split(",") if b.strip()]
asset_checks = {
    "pope_adv": [
        repo / "playground/data/eval/pope/llava_pope_test.jsonl",
        repo / "playground/data/eval/pope/val2014",
    ],
    "object_halbench": [
        repo / "playground/data/eval/object-halbench/questions.jsonl",
        repo / "playground/data/eval/object-halbench/annotations.jsonl",
        repo / "playground/data/eval/object-halbench/images",
    ],
    "amber": [
        repo / "playground/data/eval/amber/query_generative.jsonl",
        repo / "playground/data/eval/amber/annotations.jsonl",
        repo / "playground/data/eval/amber/images",
    ],
}
for benchmark in benchmarks:
    for path in asset_checks.get(benchmark, []):
        if not path.exists():
            missing.append(f"missing benchmark asset for {benchmark}: {path}")

if missing:
    raise SystemExit("Preflight failed:\n- " + "\n- ".join(missing))
print("Preflight OK.")
PY
  python - <<'PY'
from __future__ import annotations

missing = []
try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.word_tokenize("A dog sits near a car.")
    nltk.pos_tag(["dog", "car"])
    if not wn.synsets("dog"):
        missing.append("wordnet returned no synsets")
except Exception as exc:  # noqa: BLE001 - preflight should report dependency failures clearly.
    missing.append(f"NLTK runtime resources are incomplete: {exc}")

try:
    import spacy
    spacy.load("en_core_web_lg")
except Exception as exc:  # noqa: BLE001
    missing.append(f"spaCy en_core_web_lg is unavailable: {exc}")

if missing:
    raise SystemExit("Runtime metric dependency preflight failed:\n- " + "\n- ".join(missing))

print("Runtime metric dependency preflight OK.")
PY
}

run_eval() {
  local run_name="$1"
  local limit="${2:-}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  if [ -e "${run_dir}" ]; then
    log "Evaluation run already exists, refusing to overwrite: ${run_dir}"
    exit 2
  fi

  local cmd=(
    python -m fg_pipeline.eval.run_eval
    --run-name "${run_name}"
    --models-json "${MODELS_JSON}"
    --benchmarks "${BENCHMARKS}"
    --supplemental
    --general
    --output-root "${OUTPUT_ROOT}"
  )
  if [ -n "${limit}" ]; then
    cmd+=(--limit "${limit}")
  fi

  log "Starting evaluation run: ${run_name}"
  "${cmd[@]}"
  test -s "${run_dir}/comparison/summary.csv"
  test -s "${run_dir}/comparison/supplemental_eval.md"
  {
    printf '\n## Run Note\n\n'
    printf 'This evaluation compares LLaVA-1.5-7B base, direct paper-style HSA-DPO, Part A normal DPO, and Part B API-verified severity-margin HSA-DPO.\n'
    printf 'Existing older evaluation outputs were backed up to `%s` before this run.\n' "${BACKUP_DIR}"
  } >>"${run_dir}/comparison/supplemental_eval.md"
  log "Finished evaluation run: ${run_name}"
}

main() {
  export MODELS_JSON BENCHMARKS
  backup_old_eval_outputs
  wait_for_training_outputs
  write_model_manifest
  preflight
  run_eval "${SMOKE_RUN_NAME}" "${SMOKE_LIMIT}"
  run_eval "${FULL_RUN_NAME}" ""
  log "All monitoring and evaluation work completed."
  log "Smoke report: ${OUTPUT_ROOT}/${SMOKE_RUN_NAME}/comparison"
  log "Full report: ${OUTPUT_ROOT}/${FULL_RUN_NAME}/comparison"
}

main "$@"
