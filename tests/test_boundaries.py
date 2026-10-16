import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from p1 import data, training, evaluation
from p1.artifacts import sha256, verify_training
from p1.audit import run_audit
from p1.config import load_config
from p1.noise import validation_label_views
from p1.selection import compute_selection, run_selection
from p1.splits import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def make_run(tmp_path):
    cfg = json.loads((ROOT / "configs/phase0_smoke.json").read_text())
    cfg["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, Path(cfg["output_dir"])


def test_pipeline_access_order_frozen_and_invariant(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    config, out = make_run(tmp_path)
    accesses = []
    original = data.load_partition

    def guarded(cfg, *, train):
        if not train:
            assert (out / "selection_frozen.json").exists(), "Test accessed before frozen choice"
        accesses.append(train)
        return original(cfg, train=train)

    monkeypatch.setattr(training, "load_partition", guarded)
    monkeypatch.setattr(evaluation, "load_partition", guarded)
    training.run_training(config)
    assert accesses == [True]
    cfg = json.loads((out / "config_used.json").read_text())
    before = {p.name: sha256(p) for p in (out / "checkpoints").glob("*.pt")}
    evaluation.run_evaluation(out)
    assert all(accesses)
    assert not (out / "private_audit/oracle_test.csv").exists()
    assert not (out / "private_audit/clean_validation.csv").exists()
    with pytest.raises(FileNotFoundError):
        run_audit(out)
    # Selector requires neither config, manifests, checkpoints nor private audit.
    isolated = tmp_path / "selector_only"
    isolated.mkdir()
    (isolated / "validation_scores.csv").write_bytes((out / "validation_scores.csv").read_bytes())
    assert run_selection(isolated) == run_selection(out)
    freeze = (out / "selection_frozen.json").read_bytes()
    run_audit(out)
    assert accesses[-1] is False
    assert (out / "selection_frozen.json").read_bytes() == freeze
    assert {p.name: sha256(p) for p in (out / "checkpoints").glob("*.pt")} == before
    verify_training(out)
    # Generate a different validation view and retrain: training states and predictions identical.
    cfg["heterogeneous_rates"] = [1., .9, .8, .7]
    cfg["validation_noise_seed"] += 999
    cfg["output_dir"] = str(tmp_path / "other")
    config.write_text(json.dumps(cfg))
    training.run_training(config)
    for p in (out / "checkpoints").glob("*.pt"):
        q = Path(cfg["output_dir"]) / "checkpoints" / p.name
        assert sha256(p) == sha256(q)
        a, b = training.build_model("small_cnn"), training.build_model("small_cnn")
        a.load_state_dict(torch.load(p, weights_only=True))
        b.load_state_dict(torch.load(q, weights_only=True))
        x = torch.randn(2, 3, 32, 32)
        a.eval(); b.eval()
        assert torch.equal(a(x), b(x))
    with pytest.raises(FileExistsError):
        training.run_training(config)
    with (out / "selected_models.csv").open("a") as f:
        f.write("tampered")
    with pytest.raises(AssertionError, match="Frozen"):
        run_audit(out)


def test_official_cifar_split_loader_isolation(monkeypatch):
    import torchvision.datasets
    calls = []

    class MockCIFAR:
        def __init__(self, *, root, train, download):
            calls.append((train, download))
            self.data = np.zeros((3, 32, 32, 3), dtype=np.uint8)
            self.targets = [0, 1, 2]

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", MockCIFAR)
    data.load_partition({"dataset": "cifar10"}, train=True)
    assert calls == [(True, False)]
    data.load_partition({"dataset": "cifar10"}, train=False)
    assert calls[-1] == (False, False)


def test_unequal_client_noise_matching_and_rng_isolation():
    labels = np.arange(120) % 10
    original = labels.copy()
    np.random.seed(13)
    state = np.random.get_state()
    views, meta = validation_label_views(labels, {0: list(range(20)), 1: list(range(20, 120))}, [0, .6], 77)
    assert meta["uniform"]["0"]["planned_rate"] == pytest.approx(.5)
    assert np.array_equal(labels, original)
    assert np.array_equal(state[1], np.random.get_state()[1])
    views2, meta2 = validation_label_views(labels, {0: list(range(20)), 1: list(range(20, 120))}, [0, .6], 77)
    assert meta == meta2
    assert all(np.array_equal(views[s][c], views2[s][c]) for s in views for c in views[s])


@pytest.mark.parametrize("change", [dict(noisy_acc=float("nan")), dict(noisy_ce=float("inf")), dict(n=0), dict(scenario="clean"), dict(test_accuracy=.9)])
def test_selector_rejects_bad_direct_api(change):
    row = dict(scenario="uniform", round=1, client_id=0, n=3, noisy_acc=.5, noisy_ce=1.)
    row.update(change)
    with pytest.raises(ValueError):
        compute_selection([row])


def test_selector_rejects_missing_duplicate_grid():
    rows = [dict(scenario="uniform", round=r, client_id=c, n=3, noisy_acc=.5, noisy_ce=1.) for r in (1, 2) for c in (0, 1)]
    with pytest.raises(ValueError, match="Duplicate"):
        compute_selection(rows + [rows[0]])
    with pytest.raises(ValueError, match="grid"):
        compute_selection(rows[:-1])


@pytest.mark.parametrize("key,value", [("clients_per_round", 0), ("local_epochs", 0), ("training_label_noise_rate", -1), ("ema_alpha", 2), ("batch_size", 0)])
def test_config_rejects_invalid(tmp_path, key, value):
    cfg = json.loads((ROOT / "configs/phase0_smoke.json").read_text())
    cfg[key] = value
    p = tmp_path / "invalid.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(p)
