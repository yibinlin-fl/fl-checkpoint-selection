"""Deterministic disjoint client indices, train and validation holdouts."""
import json
from pathlib import Path
import numpy as np


def assign_clients(labels, num_clients, partition, alpha, seed):
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_clients)]
    if partition == "iid":
        order = rng.permutation(len(labels))
        return [np.asarray(x, dtype=int) for x in np.array_split(order, num_clients)]
    if partition != "dirichlet":
        raise ValueError("partition must be iid or dirichlet")
    for cls in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == cls))
        props = rng.dirichlet(np.full(num_clients, alpha))
        boundaries = np.rint(np.cumsum(props) * len(indices)).astype(int)
        boundaries = np.maximum.accumulate(np.clip(boundaries, 0, len(indices)))
        split = np.split(indices, boundaries[:-1])
        for cid, chunk in enumerate(split):
            parts[cid].extend(chunk.tolist())
    return [rng.permutation(x).astype(int) for x in parts]


def make_splits(labels, config):
    n = len(labels)
    num_clients = config["num_clients"]
    # A single fixed client assignment, then client-internal train/val split.
    assigned = assign_clients(labels, num_clients,
                              config.get("partition", "iid"),
                              config.get("dirichlet_alpha", 0.5),
                              config["seed"])
    return holdout_clients(labels, assigned, config)


def holdout_clients(labels, assigned, config):
    n = len(labels)
    rng = np.random.default_rng(config["seed"] + 137)
    train, val = {}, {}
    for cid, assigned_c in enumerate(assigned):
        if len(assigned_c) < 4:
            raise ValueError(f"Client {cid} has {len(assigned_c)} samples. Retry seed/alpha; no silent rescue.")
        shuffled = rng.permutation(assigned_c)
        n_val = max(1, int(round(len(shuffled) * config["validation_fraction"])))
        train[cid] = shuffled[n_val:].tolist()
        val[cid] = shuffled[:n_val].tolist()
    validate_splits(train, val, n)
    return train, val


def validate_splits(train, val, n_samples):
    assert set(train) == set(val), "Client keys differ"
    tr = [int(i) for group in train.values() for i in group]
    va = [int(i) for group in val.values() for i in group]
    assert set(tr).isdisjoint(va), "DATA LEAK: train/val intersection"
    assert len(tr) + len(va) == n_samples
    assert len(set(tr + va)) == n_samples
    assert all(0 <= x < n_samples for x in tr + va)
    assert all(len(v) > 0 for v in train.values())
    assert all(len(v) > 0 for v in val.values())


def validate_manifest(manifest, n_samples, n_test):
    train = {int(k): v["train_indices"] for k, v in manifest["clients"].items()}
    val = {int(k): v["validation_indices"] for k, v in manifest["clients"].items()}
    validate_splits(train, val, n_samples)
    # IDs are (official source split, source index), never bare local integers.
    tr = {("official_train", i) for v in train.values() for i in v}
    va = {("official_train", i) for v in val.values() for i in v}
    te = {("official_test", i) for i in range(n_test)}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert manifest["source_namespaces"] == ["official_train", "official_test"]
    assert manifest["n_test_independent"] == n_test
    return train, val


def write_manifest(output, train, val, labels, config, test_size):
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": config["dataset"], "seed": config["seed"],
        "n_train": sum(map(len, train.values())), "n_validation": sum(map(len, val.values())),
        "n_test_independent": int(test_size),
        "source_namespaces": ["official_train", "official_test"],
        "target_distribution": "official CIFAR-10 iid test" if config["dataset"] == "cifar10" else "SYNTHETIC DEBUG ONLY",
        "clients": {
            str(k): {
                "train_indices": list(map(int, train[k])),
                "validation_indices": list(map(int, val[k])),
                "train_class_hist": np.bincount(np.asarray(labels)[train[k]], minlength=10).tolist(),
                "validation_class_hist": np.bincount(np.asarray(labels)[val[k]], minlength=10).tolist(),
            } for k in sorted(train)
        },
        "warning": "Contains sample indices / clean class hist for OFFLINE AUDIT; never pass to selectors"
    }
    validate_manifest(manifest, len(labels), test_size)
    (dest / "splits_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
