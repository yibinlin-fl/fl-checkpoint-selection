"""Configuration validation for the Phase-0 experiment."""
import json
from pathlib import Path


def load_config(path):
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    required = ["dataset", "seed", "num_clients", "rounds", "checkpoint_every", "batch_size",
                "validation_fraction", "heterogeneous_rates", "output_dir"]
    missing = [k for k in required if k not in c]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    for key in ("num_clients", "rounds", "checkpoint_every", "batch_size", "local_epochs", "clients_per_round"):
        value = c.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["dataset"] not in ("synthetic-debug", "cifar10"):
        raise ValueError("Unsupported dataset")
    if c.get("backend", "standalone") not in ("standalone", "fnbench"):
        raise ValueError("Unsupported backend")
    if c.get("partition", "iid") not in ("iid", "dirichlet"):
        raise ValueError("Unsupported partition")
    if not 0 < c.get("dirichlet_alpha", .5):
        raise ValueError("dirichlet_alpha must be positive")
    if not 0 <= c.get("training_label_noise_rate", 0.) <= 1:
        raise ValueError("training_label_noise_rate must be in [0, 1]")
    if not 0 < c.get("ema_alpha", .6) <= 1:
        raise ValueError("ema_alpha must be in (0, 1]")
    if c.get("backend") == "fnbench" and c.get("model") != "fnbench_resnet18":
        raise ValueError("FNBench adapter requires official ResNet18")
    if not 0 < c["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(c["heterogeneous_rates"]) != c["num_clients"]:
        raise ValueError("Expected one noise rate per client")
    if not all(0 <= x <= 1 for x in c["heterogeneous_rates"]):
        raise ValueError("All noise rates must lie in [0, 1]")
    if c["checkpoint_every"] < 1 or c["rounds"] < 1:
        raise ValueError("rounds / checkpoint_every must be positive")
    if c.get("clients_per_round", c["num_clients"]) > c["num_clients"]:
        raise ValueError("clients_per_round cannot exceed num_clients")
    c["_config_path"] = str(Path(path).resolve())
    return c
