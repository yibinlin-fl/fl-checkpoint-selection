#!/usr/bin/env python3
"""Authorized V100 2..5-round smoke. No fallback, long training, or multi-seed sweep."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1.artifacts import verify_frozen
from p1.config import load_config
from p1.phase1a import require_v100, summarize_cost, validate_phase1a_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1a_v100_smoke.json")
    parser.add_argument("--out")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = load_config(args.config)
    validate_phase1a_config(cfg)
    gpu = require_v100()  # Fail before download, expensive training or output creation.
    out = Path(args.out or cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    if not (Path(cfg["prepared_data_dir"]) / "prepared_artifacts.json").exists():
        raise RuntimeError("Run prepare_phase1a_data.py and inspect real-data audit first")
    start = time.perf_counter()
    for command in ("train", "evaluate", "select", "audit", "plot"):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/p1_cli.py"), command,
                                 "--config", args.config, "--out", str(out)], capture_output=True, text=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{command}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            failure = {"stage": command, "exit_code": result.returncode,
                       "stderr": result.stderr, "gpu": gpu, "success": False}
            (out / "phase1a_status.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
    verify_frozen(out)
    costs = summarize_cost(out)
    status = {"success": True, "gpu": gpu, "pipeline_seconds": time.perf_counter() - start,
              "measured_costs": costs, "formal_training_started": False,
              "noise_replay_pending": True}
    (out / "phase1a_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/replay_phase1a_validation.py"),
                    "--source", str(out), "--out", str(out) + "_noise_replay"], check=True)
    print((out / "phase1a_status.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
