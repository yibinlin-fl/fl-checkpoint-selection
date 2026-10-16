"""Configuration validation for the Phase-0 experiment."""
import json
from pathlib import Path


def load_config(path):
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    required = ["seed", "num_clients", "rounds", "checkpoint_every", "batch_size",
                "validation_fraction", "heterogeneous_rates", "output_dir"]
    missing = [k for k in required if k not in c]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
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
