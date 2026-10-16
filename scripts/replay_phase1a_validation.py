#!/usr/bin/env python3
"""Replay only validation labels on the SAME saved model sequence; never retrain."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1.artifacts import record_training, sha256, verify_frozen
from p1.audit import run_audit
from p1.evaluation import run_evaluation
from p1.phase1a import validate_phase1a_config
from p1.selection import run_selection
from p1.plots import run_plot


def replay(source, destination):
    source, destination = Path(source), Path(destination)
    verify_frozen(source)
    cfg = json.loads((source / "config_used.json").read_text(encoding="utf-8"))
    validate_phase1a_config(cfg)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Replay needs a fresh directory")
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(destination)
    cfg["validation_noise_seed"] += 1  # Frozen replay rule; do not search seeds or change rates.
    (destination / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for name in ("splits_manifest.json", "train_log.json"):
        shutil.copyfile(source / name, destination / name)
    checkpoints = sorted((source / "checkpoints").glob("round_*.pt"))
    for checkpoint in checkpoints:
        shutil.copyfile(checkpoint, destination / "checkpoints" / checkpoint.name)
    (destination / "run_metadata.json").write_text(json.dumps({
        "replay_of": str(source), "same_checkpoint_sequence": True,
        "new_training_performed": False, "only_validation_noise_seed_changed": True,
        "source_metadata": json.loads((source / "run_metadata.json").read_text(encoding="utf-8")),
    }, indent=2), encoding="utf-8")
    record_training(destination)
    run_evaluation(destination)
    run_selection(destination, cfg["ema_alpha"])
    run_audit(destination)
    run_plot(destination)
    verify_frozen(source)
    verify_frozen(destination)
    assert sha256(source / "splits_manifest.json") == sha256(destination / "splits_manifest.json")
    for checkpoint in checkpoints:
        assert sha256(checkpoint) == sha256(destination / "checkpoints" / checkpoint.name)
    with (source / "private_audit/oracle_test.csv").open(encoding="utf-8", newline="") as f:
        before = list(csv.DictReader(f))
    with (destination / "private_audit/oracle_test.csv").open(encoding="utf-8", newline="") as f:
        after = list(csv.DictReader(f))
    assert before == after, "Same checkpoint test predictions/scores changed"
    result = {"source": str(source), "replay": str(destination),
              "identical_splits": True, "identical_checkpoint_sha256": True,
              "identical_test_predictions": True, "new_training_performed": False,
              "scope": "Offline reuse proof; does not claim independently retrained GPU bitwise reproducibility"}
    (destination / "replay_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    status_path = source / "phase1a_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["noise_replay_pending"] = False
        status["noise_replay_report"] = str(destination / "replay_verification.json")
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/phase1a_v100_seed42")
    parser.add_argument("--out", default="outputs/phase1a_v100_seed42_noise_replay")
    args = parser.parse_args()
    replay(args.source, args.out)
