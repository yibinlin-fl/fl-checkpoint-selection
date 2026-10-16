import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from p1.dataset_audit import array_hash, check_content_duplicates, verify_prepared_data
from p1.phase1a import ROOT, validate_phase1a_config, require_v100, summarize_cost


def config():
    return json.loads((ROOT / "configs/phase1a_v100_smoke.json").read_text())


@pytest.mark.parametrize("key,value", [("rounds",120),("rounds",1),("rounds",6),("seed",43),
    ("partition","iid"),("dirichlet_alpha",1.),("training_label_noise_rate",.2),
    ("validation_noise_seed",3031),("device","cpu"),("checkpoint_every",5),("strict_determinism",False)])
def test_phase1a_rejects_unapproved_changes(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_phase1a_config(cfg)


def test_phase1a_allows_only_short_round_bounds():
    cfg = config()
    for rounds in (2, 3, 4, 5):
        cfg["rounds"] = rounds
        validate_phase1a_config(cfg)


def test_gpu_preflight_rejects_cpu_and_non_v100(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="V100 unavailable"):
        require_v100()
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _:"NVIDIA RTX 3050")
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda _:(8,6))
    with pytest.raises(RuntimeError,match="requires actual V100"):
        require_v100()


def test_cost_estimate_uses_actual_rounds_and_marks_limitations(tmp_path):
    (tmp_path/"config_used.json").write_text(json.dumps(config()))
    rows = [{"round_seconds":n,"peak_gpu_allocated_bytes":1000,"peak_gpu_reserved_bytes":2000,
             "checkpoint_bytes":500} for n in (10., 8.)]
    (tmp_path/"train_log.json").write_text(json.dumps(rows))
    result = summarize_cost(tmp_path)
    assert result["estimated_120_training_seconds"] == 10 + 119*8
    assert result["estimated_24_checkpoint_bytes"] == 12000
    assert result["steady_state_samples"] == 1
    assert result["short_smoke_estimate_only"]


def test_exact_content_duplicates_are_reported_without_mutating():
    x = np.zeros((4,2,2,3),dtype=np.uint8)
    x[1]=1; x[2]=2; x[3]=0
    test = np.stack([x[1],np.full_like(x[0],3)])
    before = array_hash(x)
    result = check_content_duplicates(x,test,{0:[0,1]},{0:[2,3]})
    assert result["train_vs_validation_exact_duplicate_pairs"] == [[0,3]]
    assert result["official_train_vs_test_exact_duplicates"][0]["official_train_index"] == 1
    assert result["official_train_vs_test_exact_duplicates"][0]["assigned_split"] == "train"
    assert before == array_hash(x)


def test_data_preparation_hash_tamper_rejected(tmp_path):
    cfg = config(); cfg["prepared_data_dir"] = str(tmp_path)
    f = tmp_path/"registered_settings.json"; f.write_text('{}')
    (tmp_path/"prepared_artifacts.json").write_text(json.dumps({f.name:"wrong_digest"}))
    with pytest.raises(AssertionError,match="Prepared artifact changed"):
        verify_prepared_data(cfg,np.arange(10),{}, {})


def test_real_cifar_frozen_indices_and_noise_mapping():
    prepared = ROOT / "outputs/phase1a_cifar10_prepared_seed42"
    if not (prepared / "prepared_artifacts.json").exists():
        pytest.skip("Real CIFAR preparation not available")
    from p1.artifacts import sha256
    from p1.data import load_partition
    cfg = config()
    for filename, digest in json.loads((prepared / "prepared_artifacts.json").read_text()).items():
        assert sha256(prepared / filename) == digest
    with np.load(prepared / "source_indices.npz", allow_pickle=False) as indices:
        tr = np.concatenate([indices[f"client_{c}_train"] for c in range(10)])
        va = np.concatenate([indices[f"client_{c}_validation"] for c in range(10)])
        assert len(tr) == 40000 and len(va) == 10000
        assert len(np.unique(tr)) == 40000 and len(np.unique(va)) == 10000
        assert not np.intersect1d(tr,va).size
        assert np.array_equal(np.sort(np.concatenate([tr,va])),np.arange(50000))
        assert np.array_equal(indices["official_test_indices"],np.arange(10000))
    _, labels = load_partition(cfg, train=True)
    mapping = json.loads((prepared/"private_audit/validation_labels.json").read_text())
    manifest = json.loads((prepared/"splits_manifest.json").read_text())
    for cid, entry in manifest["clients"].items():
        assert np.array_equal(mapping["clean"][cid],labels[entry["validation_indices"]])
    stats = json.loads((prepared/"dataset_statistics.json").read_text())
    assert stats["model_evaluation_performed"] is False
    assert stats["training_label_noise_rate"] == 0
    assert stats["uniform_expected_validation_noise_rate"] == pytest.approx(.27448)
