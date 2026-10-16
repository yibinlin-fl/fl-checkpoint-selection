"""Actual pinned upstream component parity; skip only if upstream was not fetched."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from p1.data import PixelDataset, make_synthetic
from p1.fnbench import ROOT, upstream, official_args, build_official_model, train_round

pytestmark = pytest.mark.skipif(not (ROOT / "vendor/FNBench/.git").exists(), reason="Fetch pinned FNBench first")


def test_official_components_instrumentation_parity(tmp_path):
    torch.set_num_threads(2)
    upstream()
    from flcore.update import get_local_update_objects
    from flcore.federation import LocalModelWeights
    from p1.training import set_seeds

    set_seeds(27)
    cfg = json.loads((ROOT / "configs/phase0_fnbench_smoke.json").read_text())
    cfg.update(num_clients=2, clients_per_round=2, batch_size=4)
    args = official_args(cfg)
    images, labels = make_synthetic(16, 27)
    ds = PixelDataset(images, labels, augment=True, normalize=True)
    ids = {0: list(range(8)), 1: list(range(8, 16))}
    model = build_official_model()
    locals_ = get_local_update_objects(args, ds, ids, noise_rates=[.2, .2])
    weights = LocalModelWeights(False, model, 2, "fedavg", ids, args)
    instrumented_model = copy.deepcopy(model)
    instrumented_locals = copy.deepcopy(locals_)
    instrumented_weights = copy.deepcopy(weights)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    selected = [1, 0]  # Do not sort official participation order.
    losses = []
    for cid in selected:
        state, loss = locals_[cid].train(copy.deepcopy(model).to(args.device))
        weights.update(cid, state)
        losses.append(loss)
    model.load_state_dict(weights.average())
    model.to(args.device)
    weights.global_w_init = copy.deepcopy(model.state_dict())
    weights.init()
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    got = train_round(instrumented_model, instrumented_locals, instrumented_weights, args, selected)
    assert got == losses
    for name, value in model.state_dict().items():
        assert torch.equal(value, instrumented_model.state_dict()[name]), name
    # Checkpoint serialization doesn't consume training RNG and preserves predictions/BN.
    before = torch.get_rng_state().clone()
    path = tmp_path / "official.pt"
    torch.save(instrumented_model.state_dict(), path)
    assert torch.equal(before, torch.get_rng_state())
    restored = build_official_model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(2, 3, 32, 32)
    instrumented_model.eval(); restored.eval()
    with torch.no_grad():
        assert torch.equal(instrumented_model(x)[0], restored(x)[0])


def test_official_fedavg_bn_integer_semantics_preserved():
    upstream()
    from flcore.federation import FedAvg
    states = [{"w": torch.tensor([1.]), "num_batches_tracked": torch.tensor(2)},
              {"w": torch.tensor([5.]), "num_batches_tracked": torch.tensor(6)}]
    got = FedAvg(states, [1, 3])
    assert got["w"].item() == 4.
    assert got["num_batches_tracked"].item() == 5.
