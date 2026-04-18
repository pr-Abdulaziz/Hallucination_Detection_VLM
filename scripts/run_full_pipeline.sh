#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/run_stage3_confidence.sh
bash scripts/run_stage4_rewrite.sh
bash scripts/run_stage5_verify.sh
bash scripts/run_stage6_train.sh
