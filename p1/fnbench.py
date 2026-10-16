"""Thin FedAvg instrumentation of pinned FNBench; no replacement training algorithm."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .artifacts import record_training, sha256
from .data import PixelDataset, load_partition, test_size
from .splits import holdout_clients, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def upstream():
    vendor = ROOT / "vendor" / "FNBench"
    pin = json.loads((ROOT / "configs/fnbench_upstream.json").read_text())
    actual = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin["commit"]:
        raise RuntimeError(f"Unexpected FNBench SHA: {actual}")
    expected_patch = (ROOT / pin["patch"]).read_text().replace("\r\n", "\n")
    actual_patch = subprocess.check_output(["git", "-C", str(vendor), "diff", "--", "data/loader.py"], text=True)
    if actual_patch.replace("\r\n", "\n") != expected_patch:
        raise RuntimeError("FNBench loader patch mismatch; run scripts/prepare_fnbench.py")
    changes = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain", "--untracked-files=all"], text=True)
    source_changes = [line.strip() for line in changes.splitlines()
                      if not (line.startswith("?? ") and "/__pycache__/" in line and line.endswith(".pyc"))]
    if source_changes != ["M data/loader.py"]:
        raise RuntimeError(f"Unexpected vendor changes: {changes!r}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return pin


def build_official_model():
    upstream()
    from model_arch.build_model import build_model
    return build_model(SimpleNamespace(model="resnet18", num_classes=10))


def official_args(cfg):
    # Arguments used by the actual FedAvg update and aggregation code.
    return SimpleNamespace(
        method="fedavg", model="resnet18", num_classes=10,
        num_users=cfg["num_clients"], frac=cfg.get("clients_per_round", cfg["num_clients"]) / cfg["num_clients"],
        local_ep=cfg.get("local_epochs", 1), local_bs=cfg["batch_size"],
        lr=cfg.get("learning_rate", .01), momentum=cfg.get("momentum", .9),
        weight_decay=cfg.get("weight_decay", .0005), num_workers=cfg.get("num_workers", 0),
        collate_fn=None, is_svd_loss=False, decorr_coef=.1, verbose=False,
        device=torch.device(cfg.get("device", "cpu")), g_epoch=0,
    )


def train_round(model, locals_, weights, args, selected):
    """Same order/calls as main_fed_LNL.py FedAvg branch."""
    losses = []
    for cid in selected:
        local = locals_[cid]
        local.args = args
        state, loss = local.train(copy.deepcopy(model).to(args.device))
        if not np.isfinite(loss):
            raise ValueError("Non-finite upstream train loss")
        weights.update(cid, state)
        losses.append(float(loss))
    model.load_state_dict(weights.average())
    model.to(args.device)
    weights.global_w_init = copy.deepcopy(model.state_dict())
    weights.init()
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Non-finite upstream state")
    return losses


def run_fnbench_training(cfg):
    from .training import set_seeds, git_revision
    pin = upstream()
    from data.partition import sample_iid, sample_dirichlet
    from flcore.update import get_local_update_objects
    from flcore.federation import LocalModelWeights
    from utils.utils import noisify_label
    import random

    set_seeds(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = official_args(cfg)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2))
    images, labels = load_partition(cfg, train=True)
    # Preserve official sampler behavior, including its class-seeded Dirichlet.
    if cfg["partition"] == "iid":
        if len(labels) % cfg["num_clients"]:
            raise ValueError("Official IID sampler would discard remainder")
        assigned = sample_iid(labels, cfg["num_clients"])
    else:
        if len(labels) < 100 * cfg["num_clients"]:
            raise ValueError("Official Dirichlet requires >=100 samples per client")
        assigned = sample_dirichlet(labels, cfg["num_clients"], cfg["dirichlet_alpha"], 10)
    train, val = holdout_clients(labels, [assigned[i] for i in range(cfg["num_clients"])], cfg)
    if any(len(ids) < args.local_bs for ids in train.values()):
        raise ValueError("Official drop_last=True requires >=batch_size train samples per client")
    write_manifest(out, train, val, labels, cfg, test_size(cfg))
    targets = labels.copy()
    # Reuse original FNBench fixed-rate training corruption; val labels unchanged.
    rate = cfg.get("training_label_noise_rate", 0.)
    for cid in train:
        ids = copy.deepcopy(train[cid])
        random.seed(cfg["seed"])
        random.shuffle(ids)
        for idx in ids[:int(len(ids) * rate)]:
            targets[idx] = noisify_label(targets[idx], num_classes=10, noise_type="symmetric")
    ds = PixelDataset(images, targets, augment=True, normalize=True)
    model = build_official_model().to(args.device)
    weights = LocalModelWeights(False, model, args.num_users, "fedavg", train, args)
    locals_ = get_local_update_objects(args, ds, train, noise_rates=[.2] * args.num_users)
    for local in locals_:
        local.weight = copy.deepcopy(model.state_dict())
    logs = []
    for rnd in range(1, cfg["rounds"] + 1):
        if rnd in cfg.get("schedule", []):
            args.lr *= cfg.get("lr_decay", .1)
        args.g_epoch = rnd - 1
        selected = np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False).tolist()
        losses = train_round(model, locals_, weights, args, selected)
        logs.append({"round": rnd, "clients": selected, "train_sample_counts": [len(train[c]) for c in selected],
                     "mean_client_train_loss": float(np.mean(losses))})
        if rnd % cfg["checkpoint_every"] == 0 or rnd == cfg["rounds"]:
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       out / "checkpoints" / f"round_{rnd:04d}.pt")
        print(f"[FNBench FedAvg / {cfg['dataset']}] round={rnd} loss={np.mean(losses):.6f}", flush=True)
    (out / "train_log.json").write_text(json.dumps(logs, indent=2))
    (out / "run_metadata.json").write_text(json.dumps({
        "git_revision": git_revision(), "fnbench_upstream_sha": pin["commit"],
        "fnbench_patch_sha256": sha256(ROOT / pin["patch"]), "torch": torch.__version__,
        "backend": "FNBench original FedAvg components; external isolation adapter",
        "dataset": cfg["dataset"], "is_real_data": cfg["dataset"] == "cifar10",
        "train_accesses_official_test": False, "seed_control": "Python + NumPy + Torch; upstream main seeds NumPy only",
    }, indent=2))
    record_training(out)
    return out
