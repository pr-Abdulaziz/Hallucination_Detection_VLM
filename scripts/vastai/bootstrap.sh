#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "No Python interpreter found. Install python3 on the Vast instance first."
  exit 1
fi

echo "Using Python: $(command -v "${PYTHON_BIN}")"

if [ ! -d ".venv" ]; then
  "${PYTHON_BIN}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[linux-train]"
python -m pip install -U "huggingface_hub>=0.23.2,<1.0" modelscope

echo
echo "Vast AI bootstrap completed."
echo
echo "Next steps:"
echo "1. source .venv/bin/activate"
echo "2. huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b"
echo "3. bash hsa_dpo_train.sh"
