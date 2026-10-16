#!/usr/bin/env python3
"""Prepare full official CIFAR data manifests; no model/GPU inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1.config import load_config
from p1.dataset_audit import prepare_real_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1a_v100_smoke.json")
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepare_real_data(cfg, args.out or cfg["prepared_data_dir"])


if __name__ == "__main__":
    main()
