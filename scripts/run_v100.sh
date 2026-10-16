#!/usr/bin/env bash
set -euo pipefail
# EXPLICIT APPROVAL REQUIRED: This script can train for 120 rounds; Codex should not auto-run it.
python scripts/p1_cli.py all --config configs/phase0_v100.json
