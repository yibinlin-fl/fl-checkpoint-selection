#!/usr/bin/env python3
"""Read-only audit of actual smoke artifacts; optionally compare a noise-only replay."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1.artifacts import sha256, verify_frozen
from p1.splits import validate_manifest


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--compare")
    args = parser.parse_args()
    out = Path(args.output)
    verify_frozen(out)
    cfg = json.loads((out / "config_used.json").read_text())
    assert cfg["dataset"] == "synthetic-debug", "This verification is Phase-0 only"
    assert cfg["rounds"] == 2
    manifest = json.loads((out / "splits_manifest.json").read_text())
    validate_manifest(manifest, cfg["synthetic_train_size"], cfg["synthetic_test_size"])
    scores = read_csv(out / "validation_scores.csv")
    choices = read_csv(out / "selected_models.csv")
    oracle = read_csv(out / "private_audit/oracle_test.csv")
    assert len(scores) == 2 * 2 * cfg["num_clients"]
    assert len(choices) == 10 and len(oracle) == 2
    assert {r["scenario"] for r in scores} == {"uniform", "heterogeneous"}
    checkpoints = {p.name: sha256(p) for p in sorted((out / "checkpoints").glob("*.pt"))}
    report = {"output": str(out), "dataset": "SYNTHETIC DEBUG ONLY", "rounds": 2,
              "train_n": manifest["n_train"], "validation_n": manifest["n_validation"],
              "independent_test_n": manifest["n_test_independent"],
              "validation_rows": len(scores), "selections": len(choices),
              "regret_rows": len(read_csv(out / "regret_summary.csv")),
              "checkpoint_sha256": checkpoints, "test_predictions": oracle}
    if args.compare:
        other = Path(args.compare)
        verify_frozen(other)
        other_cfg = json.loads((other / "config_used.json").read_text())
        allowed_changes = {"heterogeneous_rates", "validation_noise_seed", "output_dir"}
        assert {k: v for k, v in cfg.items() if k not in allowed_changes} == {
            k: v for k, v in other_cfg.items() if k not in allowed_changes}
        assert cfg["validation_noise_seed"] != other_cfg["validation_noise_seed"]
        assert sha256(out / "splits_manifest.json") == sha256(other / "splits_manifest.json")
        assert checkpoints == {p.name: sha256(p) for p in sorted((other / "checkpoints").glob("*.pt"))}
        assert oracle == read_csv(other / "private_audit/oracle_test.csv")
        report["noise_only_replay"] = {"other": str(other), "identical_splits_checkpoints_test_predictions": True}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
