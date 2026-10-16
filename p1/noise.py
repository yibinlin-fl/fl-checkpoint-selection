"""Fixed, per-client symmetric validation label flips; deterministic by scenario/seed."""
import numpy as np


def flip_labels(labels, rate, classes, rng):
    """Each selected label is changed to one uniformly sampled *different* class."""
    labels = np.asarray(labels, dtype=np.int64)
    if not 0 <= rate <= 1 or classes < 2 or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Invalid noise rate/classes/labels")
    new = labels.copy()
    mask = rng.random(len(labels)) < rate
    offsets = rng.integers(1, classes, size=int(mask.sum()))
    new[mask] = (new[mask] + offsets) % classes
    return new, mask


def validation_label_views(labels, val_by_client, heterogeneous_rates, seed, classes=10):
    """Returns 3 scenario maps with per-client noisy labels only (not train label changes)."""
    total_n = sum(len(v) for v in val_by_client.values())
    weighted = sum(len(val_by_client[k]) * heterogeneous_rates[k]
                   for k in val_by_client) / total_n
    rates_by_scenario = {
        "clean": [0.0] * len(val_by_client),
        "uniform": [float(weighted)] * len(val_by_client),
        "heterogeneous": list(heterogeneous_rates),
    }
    views = {}
    private_metadata = {}
    for scenario_i, (scenario, rates) in enumerate(rates_by_scenario.items()):
        views[scenario] = {}
        private_metadata[scenario] = {}
        for cid in sorted(val_by_client):
            indices = val_by_client[cid]
            # Stateless scenario x client seed, no changing original arrays.
            rng = np.random.default_rng(seed + 10000 * scenario_i + cid)
            flipped, mask = flip_labels(np.asarray(labels)[indices], rates[cid], classes, rng)
            views[scenario][cid] = flipped
            private_metadata[scenario][str(cid)] = {
                "planned_rate": float(rates[cid]),
                "realized_rate": float(mask.mean()),
                "n_validation": len(indices)
            }
    return views, private_metadata
