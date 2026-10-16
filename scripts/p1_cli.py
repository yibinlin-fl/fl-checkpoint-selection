#!/usr/bin/env python3
"""Single Phase-0 CLI; command `all` invokes train/evaluate/select/audit in order."""
import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p1.training import run_training
from p1.evaluation import run_evaluation
from p1.selection import run_selection
from p1.audit import run_audit
from p1.plots import run_plot
from p1.config import load_config


def main():
    parser = argparse.ArgumentParser(description="P1 FL checkpoint-selection Phase-0")
    parser.add_argument("command", choices=["train", "evaluate", "select", "audit", "plot", "all"])
    parser.add_argument("--config", default="configs/phase0_smoke.json")
    parser.add_argument("--out", help="Output directory of an existing training run; overrides config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = args.out or cfg["output_dir"]
    if args.command in ("train", "all"):
        run_training(args.config)
    if args.command in ("evaluate", "all"):
        run_evaluation(out)
    if args.command in ("select", "all"):
        run_selection(out)
    if args.command in ("audit", "all"):
        run_audit(out)
    if args.command in ("plot", "all"):
        run_plot(out)


if __name__ == "__main__":
    main()
