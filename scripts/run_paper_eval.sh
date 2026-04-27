#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

RUN_NAME="${RUN_NAME:-paper_stage5_eval}"
MODELS_JSON="${MODELS_JSON:-output/fghd/paper_models.eval.json}"
BENCHMARKS="${BENCHMARKS:-pope_adv,object_halbench,amber}"

mkdir -p "$(dirname "${MODELS_JSON}")"
cat > "${MODELS_JSON}" <<JSON
[
  {
    "model_id": "llava15_base_7b",
    "model_path": "models/llava-v1.5-7b",
    "model_base": null,
    "kind": "base",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": 512
  },
  {
    "model_id": "paper_stage5_hsa_dpo",
    "model_path": "output/fghd/paper_stage5_hsa_dpo",
    "model_base": "models/llava-v1.5-7b",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": 512
  }
]
JSON

python -m json.tool "${MODELS_JSON}" >/dev/null

python -m fg_pipeline.eval.run_eval \
  --run-name "${RUN_NAME}" \
  --models-json "${MODELS_JSON}" \
  --benchmarks "${BENCHMARKS}" \
  --supplemental \
  --general \
  --output-root "${OUTPUT_ROOT:-output/eval}" "$@"
