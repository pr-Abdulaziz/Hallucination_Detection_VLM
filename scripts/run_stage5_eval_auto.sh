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
  set +e
  source "${REPO_ROOT}/.env"
  env_rc=$?
  set -e
  set +a
  if [ "${env_rc}" -ne 0 ]; then
    echo "Warning: .env could not be fully sourced; continuing because this automatic evaluation does not require API keys." >&2
  fi
fi

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/eval}"
MODEL_MANIFEST="${MODEL_MANIFEST:-models.eval.json}"
BENCHMARKS="${BENCHMARKS:-pope_adv,object_halbench,amber}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-stage5_auto_eval_smoke}"
FULL_RUN_NAME="${FULL_RUN_NAME:-stage5_auto_eval_full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
CLEAN_EVAL="${CLEAN_EVAL:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-${LIMIT:-5}}"
FULL_LIMIT="${FULL_LIMIT:-}"
EXPECTED_GLOBAL_STEP="${EXPECTED_GLOBAL_STEP:-}"

if [ -z "${RUN_FULL+x}" ]; then
  if [ -n "${LIMIT:-}" ]; then
    RUN_FULL=0
  else
    RUN_FULL=1
  fi
fi

NOTE="This evaluation uses automatic metrics only and does not use OpenAI API keys or external judge APIs."
echo "${NOTE}"

cat > "${MODEL_MANIFEST}" <<EOF
[
  {
    "model_id": "llava15_base_7b",
    "model_path": "models/llava-v1.5-7b",
    "model_base": null,
    "kind": "base",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": ${MAX_NEW_TOKENS}
  },
  {
    "model_id": "ours_stage5_lora",
    "model_path": "output/fghd/paper_stage5_hsa_dpo",
    "model_base": "models/llava-v1.5-7b",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": ${MAX_NEW_TOKENS}
  }
]
EOF

python -m json.tool "${MODEL_MANIFEST}" >/dev/null

AVAILABLE_BENCHMARKS="$(
  BENCHMARKS="${BENCHMARKS}" \
  ALLOW_PARTIAL="${ALLOW_PARTIAL}" \
  EXPECTED_GLOBAL_STEP="${EXPECTED_GLOBAL_STEP}" \
  python - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


repo = Path(".").resolve()
benchmarks = [item.strip() for item in os.environ["BENCHMARKS"].split(",") if item.strip()]
allow_partial = os.environ.get("ALLOW_PARTIAL") == "1"
expected_global_step_raw = os.environ.get("EXPECTED_GLOBAL_STEP", "").strip()
allowed = {"pope_adv", "object_halbench", "amber"}
unsupported = [name for name in benchmarks if name not in allowed]
if unsupported:
    raise SystemExit(
        "Unsupported automatic benchmark(s): "
        + ", ".join(unsupported)
        + ". Allowed: amber, object_halbench, pope_adv. Judge-based benchmarks are intentionally excluded."
    )

required_paths = [
    (repo / "models/llava-v1.5-7b", "LLaVA-1.5 base model directory"),
    (repo / "output/fghd/paper_stage5_hsa_dpo/adapter_model.safetensors", "Stage 5 LoRA adapter"),
    (repo / "output/fghd/paper_stage5_hsa_dpo/trainer_state.json", "Stage 5 trainer state"),
]
missing_required = [f"{label}: {path}" for path, label in required_paths if not path.exists()]
if missing_required:
    raise SystemExit("Missing required model/training files:\n- " + "\n- ".join(missing_required))

state_path = repo / "output/fghd/paper_stage5_hsa_dpo/trainer_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
actual_step = int(state.get("global_step", -1))
if expected_global_step_raw:
    expected_global_step = int(expected_global_step_raw)
    if actual_step != expected_global_step:
        raise SystemExit(
            f"Unexpected Stage 5 global_step={actual_step}; expected {expected_global_step}. "
            "Set EXPECTED_GLOBAL_STEP to override intentionally."
        )
eprint(f"Stage 5 trainer_state global_step={actual_step}.")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(item)
    return rows


def check_images(rows: list[dict], image_root: Path, benchmark: str) -> list[str]:
    missing: list[str] = []
    for idx, row in enumerate(rows):
        image = row.get("image")
        if not image:
            missing.append(f"row {idx}: missing image field")
            continue
        path = image_root / str(image)
        if not path.exists():
            missing.append(str(path))
    return missing


def check_import(module_name: str) -> str | None:
    try:
        __import__(module_name)
    except ImportError as exc:
        return str(exc)
    return None


def validate_pope() -> tuple[bool, list[str]]:
    root = repo / "playground/data/eval/pope"
    question_file = root / "llava_pope_test.jsonl"
    image_root = root / "val2014"
    reasons: list[str] = []
    if not question_file.exists():
        reasons.append(f"missing {question_file}")
    if not image_root.exists() or not any(image_root.iterdir()):
        reasons.append(f"missing or empty {image_root}")
    if reasons:
        return False, reasons
    rows = read_jsonl(question_file)
    if not rows:
        reasons.append(f"empty {question_file}")
    bad_labels = [
        idx
        for idx, row in enumerate(rows)
        if str(row.get("label") or row.get("answer") or row.get("gt_answer") or "").strip().lower()
        not in {"yes", "no"}
    ]
    if bad_labels:
        reasons.append(f"missing yes/no labels for {len(bad_labels)} row(s), first row index: {bad_labels[0]}")
    missing_images = check_images(rows, image_root, "pope_adv")
    if missing_images:
        reasons.append(f"missing {len(missing_images)} referenced image(s), first: {missing_images[0]}")
    return not reasons, reasons


def validate_object_halbench() -> tuple[bool, list[str]]:
    root = repo / "playground/data/eval/object-halbench"
    question_file = root / "questions.jsonl"
    annotation_file = root / "annotations.jsonl"
    synonyms_file = root / "synonyms_refine.txt"
    image_root = root / "images"
    reasons: list[str] = []
    nltk_error = check_import("nltk")
    if nltk_error:
        reasons.append(f"missing Python dependency nltk: {nltk_error}")
    for path in (question_file, annotation_file, synonyms_file):
        if not path.exists():
            reasons.append(f"missing {path}")
    if not image_root.exists() or not any(image_root.iterdir()):
        reasons.append(f"missing or empty {image_root}")
    if reasons:
        return False, reasons
    questions = read_jsonl(question_file)
    annotations = read_jsonl(annotation_file)
    if len(questions) != len(annotations):
        reasons.append(f"question/annotation count mismatch: {len(questions)} vs {len(annotations)}")
    question_ids = {str(row.get("id")) for row in questions}
    annotation_ids = {str(row.get("id")) for row in annotations}
    if question_ids != annotation_ids:
        reasons.append("question/annotation id sets do not match")
    missing_gt = [
        idx
        for idx, row in enumerate(annotations)
        if not isinstance(row.get("gt_objects"), list)
    ]
    if missing_gt:
        reasons.append(
            f"annotations missing precomputed gt_objects for {len(missing_gt)} row(s), first row index: {missing_gt[0]}"
        )
    missing_images = check_images(questions, image_root, "object_halbench")
    if missing_images:
        reasons.append(f"missing {len(missing_images)} referenced image(s), first: {missing_images[0]}")
    return not reasons, reasons


def validate_amber() -> tuple[bool, list[str]]:
    root = repo / "playground/data/eval/amber"
    question_file = root / "query_generative.jsonl"
    annotation_file = root / "annotations.jsonl"
    relation_file = root / "relation.json"
    safe_words_file = root / "safe_words.txt"
    image_root = root / "images"
    reasons: list[str] = []
    nltk_error = check_import("nltk")
    if nltk_error:
        reasons.append(f"missing Python dependency nltk: {nltk_error}")
    spacy_error = check_import("spacy")
    if spacy_error:
        reasons.append(f"missing Python dependency spacy: {spacy_error}")
    else:
        try:
            import spacy

            spacy.load("en_core_web_lg")
        except Exception as exc:
            reasons.append(f"missing spaCy model en_core_web_lg: {exc}")
    for path in (question_file, annotation_file, relation_file, safe_words_file):
        if not path.exists():
            reasons.append(f"missing {path}")
    if not image_root.exists() or not any(image_root.iterdir()):
        reasons.append(f"missing or empty {image_root}")
    if reasons:
        return False, reasons
    questions = read_jsonl(question_file)
    annotations = read_jsonl(annotation_file)
    if len(questions) != len(annotations):
        reasons.append(f"question/annotation count mismatch: {len(questions)} vs {len(annotations)}")
    question_ids = {str(row.get("id")) for row in questions}
    annotation_ids = {str(row.get("id")) for row in annotations}
    if question_ids != annotation_ids:
        reasons.append("question/annotation id sets do not match")
    malformed_annotations = [
        idx
        for idx, row in enumerate(annotations)
        if not isinstance(row.get("truth"), list) or not isinstance(row.get("hallu"), list)
    ]
    if malformed_annotations:
        reasons.append(
            "annotations must contain AMBER truth/hallu lists; "
            f"bad row count={len(malformed_annotations)}, first row index={malformed_annotations[0]}"
        )
    missing_images = check_images(questions, image_root, "amber")
    if missing_images:
        reasons.append(f"missing {len(missing_images)} referenced image(s), first: {missing_images[0]}")
    return not reasons, reasons


validators = {
    "pope_adv": validate_pope,
    "object_halbench": validate_object_halbench,
    "amber": validate_amber,
}
available: list[str] = []
missing: dict[str, list[str]] = {}
for name in benchmarks:
    ok, reasons = validators[name]()
    if ok:
        available.append(name)
        eprint(f"Benchmark available: {name}")
    else:
        missing[name] = reasons
        eprint(f"Benchmark missing: {name}")
        for reason in reasons:
            eprint(f"  - {reason}")

if not available:
    raise SystemExit(
        "No automatic benchmark assets are available. Install POPE/Object HalBench/AMBER assets first."
    )
if missing and not allow_partial:
    raise SystemExit(
        "Some requested automatic benchmark assets are missing. "
        "Install the missing assets or set ALLOW_PARTIAL=1 to run only available benchmarks."
    )

print(",".join(available))
PY
)"

echo "Available automatic benchmark(s): ${AVAILABLE_BENCHMARKS}"

if [ "${PREFLIGHT_ONLY}" = "1" ]; then
  echo "Preflight complete. No generation was run."
  exit 0
fi

prepare_run_dir() {
  local run_name="$1"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  if [ -e "${run_dir}" ]; then
    if [ "${CLEAN_EVAL}" = "1" ]; then
      local resolved
      resolved="$(realpath "${run_dir}")"
      local expected_prefix
      expected_prefix="$(realpath "${OUTPUT_ROOT}")"
      case "${resolved}" in
        "${expected_prefix}"/*)
          rm -rf -- "${resolved}"
          ;;
        *)
          echo "Refusing to clean unexpected evaluation path: ${resolved}" >&2
          exit 2
          ;;
      esac
    else
      echo "Evaluation output already exists: ${run_dir}" >&2
      echo "Set CLEAN_EVAL=1 to remove and recreate it, or change the run name." >&2
      exit 2
    fi
  fi
}

append_note() {
  local run_name="$1"
  local report="${OUTPUT_ROOT}/${run_name}/comparison/supplemental_eval.md"
  if [ -f "${report}" ] && ! grep -Fq "${NOTE}" "${report}"; then
    {
      printf '\n## Evaluation Note\n\n'
      printf '%s\n' "${NOTE}"
    } >>"${report}"
  fi
}

verify_run() {
  local run_name="$1"
  local comparison_dir="${OUTPUT_ROOT}/${run_name}/comparison"
  test -s "${comparison_dir}/summary.csv"
  test -s "${comparison_dir}/supplemental_eval.md"
  IFS=',' read -r -a bench_array <<<"${AVAILABLE_BENCHMARKS}"
  for model_id in llava15_base_7b ours_stage5_lora; do
    for benchmark in "${bench_array[@]}"; do
      test -s "${OUTPUT_ROOT}/${run_name}/models/${model_id}/predictions/${benchmark}.jsonl"
      test -s "${OUTPUT_ROOT}/${run_name}/models/${model_id}/metrics/${benchmark}.json"
    done
  done
  append_note "${run_name}"
}

run_eval() {
  local run_name="$1"
  local limit="${2:-}"
  prepare_run_dir "${run_name}"
  local cmd=(
    python -m fg_pipeline.eval.run_eval
    --run-name "${run_name}"
    --models-json "${MODEL_MANIFEST}"
    --benchmarks "${AVAILABLE_BENCHMARKS}"
    --supplemental
    --general
    --output-root "${OUTPUT_ROOT}"
  )
  if [ -n "${limit}" ]; then
    cmd+=(--limit "${limit}")
  fi
  echo "Running ${run_name} on benchmark(s): ${AVAILABLE_BENCHMARKS}"
  "${cmd[@]}"
  verify_run "${run_name}"
  echo "Verified ${run_name}: ${OUTPUT_ROOT}/${run_name}/comparison"
}

if [ "${RUN_SMOKE}" = "1" ]; then
  run_eval "${SMOKE_RUN_NAME}" "${SMOKE_LIMIT}"
else
  echo "RUN_SMOKE=${RUN_SMOKE}; smoke evaluation skipped."
fi

if [ "${RUN_FULL}" = "1" ]; then
  run_eval "${FULL_RUN_NAME}" "${FULL_LIMIT}"
else
  echo "RUN_FULL=${RUN_FULL}; full evaluation skipped after smoke run."
fi
