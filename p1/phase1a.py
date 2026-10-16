"""Phase-1A authorization bounds, physical GPU checks, and measured cost summaries."""
import json
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_KEYS = (
    "dataset", "seed", "num_clients", "partition", "dirichlet_alpha", "validation_fraction",
    "clients_per_round", "local_epochs", "batch_size", "learning_rate", "momentum", "model",
    "num_workers", "heterogeneous_rates", "validation_noise_seed", "ema_alpha",
    "training_label_noise_rate", "backend", "weight_decay", "download_data", "schedule", "lr_decay",
)


def validate_phase1a_config(cfg):
    baseline = json.loads((ROOT / "configs/phase0_v100.json").read_text(encoding="utf-8"))
    for key in FROZEN_KEYS:
        if cfg.get(key) != baseline.get(key):
            raise ValueError(f"Phase-1A pre-registered setting changed: {key}")
    if cfg.get("phase") != "phase1a" or cfg.get("device") != "cuda":
        raise ValueError("Phase-1A requires its explicit CUDA configuration")
    if not 2 <= cfg["rounds"] <= 5 or cfg["checkpoint_every"] != 1:
        raise ValueError("Phase-1A allows only 2..5 rounds, checkpoint each round")
    if cfg.get("strict_determinism") is not True:
        raise ValueError("Phase-1A requires explicit deterministic execution")


def require_v100():
    if not torch.cuda.is_available():
        raise RuntimeError("V100 unavailable: installed PyTorch has no usable CUDA GPU")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "V100" not in name.upper() or capability != (7, 0):
        raise RuntimeError(f"Phase-1A requires actual V100 (sm_70); detected {name}, {capability}")
    if "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("Installed torch wheel lacks sm_70")
    free, total = torch.cuda.mem_get_info(0)
    x = torch.ones(2, device="cuda")
    if (x + x).sum().item() != 4:
        raise RuntimeError("CUDA arithmetic preflight failed")
    return {"name": name, "capability": list(capability), "free_bytes": free,
            "total_bytes": total, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "wheel_architectures": torch.cuda.get_arch_list()}


def summarize_cost(out_dir):
    out = Path(out_dir)
    cfg = json.loads((out / "config_used.json").read_text(encoding="utf-8"))
    validate_phase1a_config(cfg)
    rows = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    times = [r["round_seconds"] for r in rows]
    if len(times) != cfg["rounds"] or any(t <= 0 for t in times):
        raise ValueError("Incomplete or invalid measured round timings")
    steady = times[1:]  # Report warm-up separately; do not hide first-round cost.
    estimate = times[0] + 119 * statistics.mean(steady)
    report = {"gpu_training_measured": True, "round_seconds": times,
              "estimated_120_training_seconds": estimate,
              "estimate_scope": "training with per-round checkpoint I/O; excludes offline validation/test, install and download",
              "steady_state_samples": len(steady), "short_smoke_estimate_only": True,
              "peak_gpu_allocated_bytes": max(r["peak_gpu_allocated_bytes"] for r in rows),
              "peak_gpu_reserved_bytes": max(r["peak_gpu_reserved_bytes"] for r in rows),
              "checkpoint_bytes": [r["checkpoint_bytes"] for r in rows],
              "estimated_24_checkpoint_bytes": 24 * max(r["checkpoint_bytes"] for r in rows)}
    (out / "cost_estimate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@torch.no_grad()
def verify_checkpoint_roundtrip(model, cfg, images, targets, train_indices, checkpoint, out):
    import hashlib
    from .data import PixelDataset
    from .fnbench import build_official_model
    ids = train_indices[:4]
    dataset = PixelDataset(images[ids], targets[ids], normalize=True)
    x = torch.stack([dataset[i][0] for i in range(len(ids))]).to(cfg["device"])
    model.eval()
    expected = model(x)[0]
    restored = build_official_model().to(cfg["device"])
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored.eval()
    actual = restored(x)[0]
    if not torch.equal(expected, actual):
        raise RuntimeError("Checkpoint reload changed GPU logits")
    result = {"source": "training-only deterministic probe; not validation/test", "n": len(ids),
              "logits_exactly_equal": True,
              "predictions_sha256": hashlib.sha256(actual.argmax(1).cpu().numpy().astype("<i8").tobytes()).hexdigest()}
    (Path(out) / "checkpoint_roundtrip.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
