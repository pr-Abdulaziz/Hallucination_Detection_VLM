#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

python -m pip install -e ".[eval]"

EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/playground/data/eval}"
mkdir -p \
  "${EVAL_ROOT}/mhalubench" \
  "${EVAL_ROOT}/mfhallubench" \
  "${EVAL_ROOT}/pope/val2014" \
  "${EVAL_ROOT}/object-halbench/images" \
  "${EVAL_ROOT}/amber/images" \
  "${EVAL_ROOT}/mmhal-bench/images" \
  "${EVAL_ROOT}/llava-bench-in-the-wild/images"

cat > "${EVAL_ROOT}/README.md" <<'EOF'
# Evaluation Benchmark Layout

Benchmark assets are intentionally not vendored in this repository. Put the
paper-matching assets in these folders before running the full evaluation.

- `mhalubench/`: `predictions.jsonl`, `annotations.jsonl`
- `mfhallubench/`: `predictions.jsonl`, `annotations.jsonl`
- `pope/`: `llava_pope_test.jsonl`, `val2014/`
- `object-halbench/`: `questions.jsonl`, `annotations.jsonl`, `images/`
- `amber/`: `query_generative.jsonl`, `annotations.jsonl`, `images/`
- `mmhal-bench/`: `questions.jsonl`, `images/`
- `llava-bench-in-the-wild/`: `questions.jsonl`, `context.jsonl`, `images/`, optional `answers_gpt4.jsonl`

For strict paper comparison, keep the model manifest decode settings identical
across base and trained models: `temperature=0.0`, `num_beams=1`,
`conv_mode=vicuna_v1`, and one shared `max_new_tokens` value.
EOF

if [ ! -f "${REPO_ROOT}/models.eval.example.json" ]; then
  cat > "${REPO_ROOT}/models.eval.example.json" <<'EOF'
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
    "model_id": "ours_stage5_lora",
    "model_path": "output/fghd/paper_stage5_hsa_dpo",
    "model_base": "models/llava-v1.5-7b",
    "kind": "lora",
    "conv_mode": "vicuna_v1",
    "temperature": 0.0,
    "num_beams": 1,
    "max_new_tokens": 512
  }
]
EOF
fi

echo "Evaluation dependencies installed."
echo "Benchmark folders prepared under: ${EVAL_ROOT}"
echo "Example model manifest: ${REPO_ROOT}/models.eval.example.json"
