"""Minimal standalone FedAvg instrumented for checkpoint-selection research.

No validation or test scores are computed in this training process.
This is a research starter, not FNBench's official implementation.
"""
import copy
import json
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import load_config
from .data import PixelDataset, load_images
from .models import build_model
from .noise import flip_labels
from .splits import make_splits, write_manifest


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-checkout"


def fedavg(states, sizes):
    assert len(states) == len(sizes) > 0
    out = {}
    weights = np.asarray(sizes, dtype=float) / sum(sizes)
    for name in states[0]:
        first = states[0][name]
        if torch.is_floating_point(first):
            accumulator = torch.zeros_like(first)
            for state, weight in zip(states, weights):
                accumulator += state[name] * float(weight)
            out[name] = accumulator
        else:
            # BN integer counters: not fractional-weighted tensors.
            out[name] = first.clone()
    return out


def local_update(global_model, images, targets, indices, cfg, device):
    model = copy.deepcopy(global_model).to(device)
    model.train()
    ds = PixelDataset(images[np.asarray(indices)], targets[np.asarray(indices)], augment=True)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg.get("num_workers", 0))
    optim = torch.optim.SGD(model.parameters(), lr=cfg.get("learning_rate", 0.01),
                            momentum=cfg.get("momentum", 0.0))
    criterion = nn.CrossEntropyLoss()
    total_loss, batches = 0.0, 0
    for _ in range(cfg.get("local_epochs", 1)):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(inputs)
            loss = criterion(pred, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; aborting instead of silently continuing")
            loss.backward()
            optim.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, total_loss / max(1, batches)


def run_training(config_path):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA unavailable. Use smoke config on CPU.")
    out = Path(cfg["output_dir"])
    ckpts = out / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)
    # Freeze a complete copy; never edit a running run's config halfway through.
    (out / "config_used.json").write_text(json.dumps({k: v for k, v in cfg.items()
         if not k.startswith("_")}, indent=2), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({
        "git_revision": git_revision(), "torch": torch.__version__,
        "dataset": cfg["dataset"], "is_real_data": cfg["dataset"] == "cifar10",
        "scope": "standalone starter baseline; not official FNBench",
        "train_accesses_validation": False, "train_accesses_official_test": False,
    }, indent=2), encoding="utf-8")

    images, labels, _heldout_images, _heldout_labels = load_images(cfg)
    # DO NOT evaluate _heldout_images/_heldout_labels in this function.
    train_ids, val_ids = make_splits(labels, cfg)
    write_manifest(out, train_ids, val_ids, labels, cfg, len(_heldout_labels))
    # Optional fixed train label noise, disjoint from validation noise.
    train_targets = labels.copy()
    train_noise_rate = cfg.get("training_label_noise_rate", 0.0)
    if train_noise_rate:
        for cid, indices in train_ids.items():
            rng = np.random.default_rng(cfg["seed"] + 8000 + cid)
            new, _ = flip_labels(labels[indices], train_noise_rate, 10, rng)
            train_targets[np.asarray(indices)] = new
    model = build_model(cfg.get("model", "small_cnn")).to("cpu")
    rng = np.random.default_rng(cfg["seed"] + 991)
    log = []
    for rnd in range(1, cfg["rounds"] + 1):
        selected = sorted(rng.choice(cfg["num_clients"],
                       size=cfg.get("clients_per_round", cfg["num_clients"]),
                       replace=False).tolist())
        states, sizes, losses = [], [], []
        for cid in selected:
            state, loss = local_update(model, images, train_targets, train_ids[cid], cfg, device)
            states.append(state)
            sizes.append(len(train_ids[cid]))
            losses.append(loss)
        model.load_state_dict(fedavg(states, sizes))
        print(f"[train] round={rnd}/{cfg['rounds']} train_loss={np.mean(losses):.4f}", flush=True)
        log.append({"round": rnd, "mean_client_train_loss": float(np.mean(losses))})
        if rnd % cfg["checkpoint_every"] == 0 or rnd == cfg["rounds"]:
            # Weight states only. No clean val/test info in model files.
            torch.save(model.state_dict(), ckpts / f"round_{rnd:04d}.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("Training complete; TEST SET NOT EVALUATED during training.")
    return out
