"""Offline evaluation: write selection-only scores and separate sealed oracle audit files."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import PixelDataset, load_images
from .models import build_model
from .noise import validation_label_views


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def eval_model(model, images, labels, batch_size, device):
    model.eval()
    dataset = PixelDataset(images, labels, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, n, sum_ce = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        sum_ce += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        correct += int((logits.argmax(1) == y).sum().cpu())
        n += len(y)
    return {"accuracy": correct / n, "cross_entropy": sum_ce / n, "n": n}


def run_evaluation(out_dir):
    out = Path(out_dir)
    cfg = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "splits_manifest.json").read_text(encoding="utf-8"))
    tr_x, tr_y, te_x, te_y = load_images(cfg)
    val_ids = {int(k): list(v["validation_indices"]) for k, v in manifest["clients"].items()}
    views, private_meta = validation_label_views(tr_y, val_ids,
                    cfg["heterogeneous_rates"], cfg["validation_noise_seed"])
    audit_path = out / "private_audit"
    audit_path.mkdir(exist_ok=True)
    (audit_path / "noise_manifest.json").write_text(json.dumps(private_meta, indent=2), encoding="utf-8")

    ckpts = sorted((out / "checkpoints").glob("round_*.pt"))
    if not ckpts:
        raise RuntimeError("No global checkpoints exist; run train first")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    model = build_model(cfg.get("model", "small_cnn")).to(device)
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
            # OFFLINE ONLY: clean validation truth goes to separate oracle audit file.
            clean = eval_model(model, im, clean_labels, cfg["batch_size"], device)
            oracle_clean_val.append({"round": rnd, "client_id": cid,
                            "n": len(ids), "clean_acc": clean["accuracy"]})
            for scenario in ("uniform", "heterogeneous"):
                result = eval_model(model, im, views[scenario][cid], cfg["batch_size"], device)
                validation_rows.append({"round": rnd, "client_id": cid,
                    "scenario": scenario, "n": len(ids),
                    "noisy_acc": result["accuracy"], "noisy_ce": result["cross_entropy"]})
        # Private oracle: eval after training, never loaded by the selection code.
        res = eval_model(model, te_x, te_y, cfg["batch_size"], device)
        oracle_test.append({"round": rnd, "test_accuracy": res["accuracy"]})
        print(f"[offline-evaluate] round={rnd} completed", flush=True)

    write_csv(out / "validation_scores.csv", validation_rows,
              ["round", "client_id", "scenario", "n", "noisy_acc", "noisy_ce"])
    write_csv(audit_path / "oracle_test.csv", oracle_test, ["round", "test_accuracy"])
    write_csv(audit_path / "clean_validation.csv", oracle_clean_val,
              ["round", "client_id", "n", "clean_acc"])
    print("Offline scoring saved: public selection signals and separate private audit.")
