"""Offline evaluation: write selection-only scores and separate sealed oracle audit files."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import PixelDataset, load_partition, test_size
from .artifacts import write_csv, verify_training, verify_frozen
from .splits import validate_manifest
from .models import build_model
from .noise import validation_label_views


@torch.no_grad()
def eval_model(model, images, labels, batch_size, device, normalize=False):
    model.eval()
    dataset = PixelDataset(images, labels, augment=False, normalize=normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_hash = hashlib.sha256()
    correct, n, sum_ce = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model output")
        sum_ce += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        predictions = logits.argmax(1)
        prediction_hash.update(predictions.cpu().numpy().astype("<i8").tobytes())
        correct += int((predictions == y).sum().cpu())
        n += len(y)
    if n == 0 or not np.isfinite(sum_ce):
        raise ValueError("Empty dataset or non-finite CE")
    return {"accuracy": correct / n, "cross_entropy": sum_ce / n, "n": n, "predictions_sha256": prediction_hash.hexdigest()}


def run_evaluation(out_dir, *, diagnostic=False):
    out = Path(out_dir)
    verify_training(out)
    if diagnostic:
        verify_frozen(out)
    elif (out / "selection_frozen.json").exists():
        raise FileExistsError("Selection already frozen; use a fresh run")
    cfg = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "splits_manifest.json").read_text(encoding="utf-8"))
    tr_x, tr_y = load_partition(cfg, train=True)
    _, val_ids = validate_manifest(manifest, len(tr_y), test_size(cfg))
    if diagnostic:
        te_x, te_y = load_partition(cfg, train=False)
        assert len(te_y) == manifest["n_test_independent"]
    views, private_meta = validation_label_views(tr_y, val_ids,
                    cfg["heterogeneous_rates"], cfg["validation_noise_seed"])
    audit_path = out / "private_audit"
    audit_path.mkdir(exist_ok=True)
    (audit_path / "noise_manifest.json").write_text(json.dumps(private_meta, indent=2), encoding="utf-8")
    frozen_views = json.dumps({scenario: {str(cid): ys.tolist() for cid, ys in clients.items()}
                              for scenario, clients in views.items()}, sort_keys=True)
    view_path = audit_path / "validation_labels.json"
    if view_path.exists():
        assert view_path.read_text(encoding="utf-8") == frozen_views, "Validation label mapping changed"
    else:
        view_path.write_text(frozen_views, encoding="utf-8")

    ckpts = sorted((out / "checkpoints").glob("round_*.pt"))
    if not ckpts:
        raise RuntimeError("No global checkpoints exist; run train first")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    if cfg.get("backend") == "fnbench":
        from .fnbench import build_official_model
        model = build_official_model().to(device)
    else:
        model = build_model(cfg.get("model", "small_cnn")).to(device)
    normalize = cfg.get("backend") == "fnbench"
    validation_rows = []
    oracle_test, oracle_clean_val = [], []
    for ckpt in ckpts:
        rnd = int(ckpt.stem.split("_")[-1])
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        for cid in sorted(val_ids):
            ids = val_ids[cid]
            im = tr_x[ids]
            clean_labels = tr_y[ids]
            if diagnostic:
                clean = eval_model(model, im, clean_labels, cfg["batch_size"], device, normalize)
                oracle_clean_val.append({"round": rnd, "client_id": cid,
                                        "n": len(ids), "clean_acc": clean["accuracy"]})
            else:
                for scenario in ("uniform", "heterogeneous"):
                    result = eval_model(model, im, views[scenario][cid], cfg["batch_size"], device, normalize)
                    validation_rows.append({"round": rnd, "client_id": cid,
                        "scenario": scenario, "n": len(ids),
                        "noisy_acc": result["accuracy"], "noisy_ce": result["cross_entropy"]})
        if diagnostic:
            res = eval_model(model, te_x, te_y, cfg["batch_size"], device, normalize)
            oracle_test.append({"round": rnd, "test_accuracy": res["accuracy"],
                                "n": res["n"], "predictions_sha256": res["predictions_sha256"]})
        print(f"[offline-evaluate] round={rnd} completed", flush=True)

    if diagnostic:
        write_csv(audit_path / "oracle_test.csv", oracle_test, ["round", "test_accuracy", "n", "predictions_sha256"])
        write_csv(audit_path / "clean_validation.csv", oracle_clean_val,
                  ["round", "client_id", "n", "clean_acc"])
    else:
        write_csv(out / "validation_scores.csv", validation_rows,
                  ["round", "client_id", "scenario", "n", "noisy_acc", "noisy_ce"])
    verify_training(out)
    print("Offline diagnostic audit complete." if diagnostic else "Noisy validation only; official test not loaded.")
