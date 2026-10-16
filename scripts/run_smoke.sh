#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python scripts/p1_cli.py all --config configs/phase0_smoke.json
printf 'Done. Inspect outputs/smoke/regret_summary.csv and outputs/smoke/validation_accuracy.png\n'
