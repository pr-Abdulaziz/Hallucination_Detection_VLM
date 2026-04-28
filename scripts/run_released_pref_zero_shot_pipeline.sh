#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SHOT_MODE=zero_shot bash scripts/run_released_pref_pipeline.sh "$@"
