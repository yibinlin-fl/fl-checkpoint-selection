#!/usr/bin/env bash
set -euo pipefail
# Phase-1A only: two real CIFAR rounds, one seed, followed by offline noise replay.
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/check_v100.py
python scripts/prepare_fnbench.py
python scripts/run_phase1a.py --config configs/phase1a_v100_smoke.json "$@"
