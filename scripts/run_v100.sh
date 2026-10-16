#!/usr/bin/env bash
set -euo pipefail
# After user acceptance/authorization only. One seed, one condition, no sweep.
if [[ "${1:-}" != "--approved-formal-run" ]]; then
  echo "After Phase-0 acceptance and explicit authorization, pass --approved-formal-run." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
python scripts/check_v100.py
python scripts/prepare_fnbench.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/p1_cli.py all --config configs/phase1b_v100.json
