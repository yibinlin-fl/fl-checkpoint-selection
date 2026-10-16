"""Run provenance and frozen selection integrity (not an OS security sandbox)."""
import csv
import hashlib
import json
from pathlib import Path


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record_training(out):
    out = Path(out)
    files = [out / "config_used.json", out / "splits_manifest.json"]
    files += sorted((out / "checkpoints").glob("round_*.pt"))
    (out / "training_artifacts.json").write_text(json.dumps({
        str(p.relative_to(out)).replace("\\", "/"): sha256(p) for p in files
    }, indent=2), encoding="utf-8")


def verify_training(out):
    out = Path(out)
    expected = json.loads((out / "training_artifacts.json").read_text(encoding="utf-8"))
    actual_ckpts = {p.name for p in (out / "checkpoints").glob("round_*.pt")}
    assert actual_ckpts == {Path(k).name for k in expected if k.startswith("checkpoints/")}, "Checkpoint set changed"
    for name, digest in expected.items():
        assert sha256(out / name) == digest, f"Training artifact changed: {name}"


def verify_frozen(out):
    out = Path(out)
    frozen = json.loads((out / "selection_frozen.json").read_text(encoding="utf-8"))
    for name in ("validation_scores.csv", "selected_models.csv"):
        assert sha256(out / name) == frozen[name], f"Frozen selection changed: {name}"
    verify_training(out)
