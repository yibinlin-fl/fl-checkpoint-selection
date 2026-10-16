"""Audit official CIFAR bytes and freeze pre-registered client indices, without model inference."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity

from .artifacts import sha256
from .data import load_partition
from .fnbench import partition_and_holdout, upstream
from .noise import validation_label_views
from .phase1a import FROZEN_KEYS, validate_phase1a_config
from .splits import write_manifest, validate_manifest


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_official_files(root):
    root = Path(root)
    records = {}
    expected = CIFAR10.train_list + CIFAR10.test_list + [(CIFAR10.meta["filename"], CIFAR10.meta["md5"])]
    for filename, md5 in expected:
        path = root / CIFAR10.base_folder / filename
        if not check_integrity(path, md5):
            raise ValueError(f"Official CIFAR integrity mismatch: {filename}")
        records[filename] = {"official_md5": md5, "sha256": sha256(path), "bytes": path.stat().st_size}
    archive = root / CIFAR10.filename
    if archive.exists():
        if not check_integrity(archive, CIFAR10.tgz_md5):
            raise ValueError("Official archive MD5 mismatch")
        records[CIFAR10.filename] = {"official_md5": CIFAR10.tgz_md5,
                                     "sha256": sha256(archive), "bytes": archive.stat().st_size}
    return records


def check_content_duplicates(images, test_images, train, val):
    """Report inherent identical pixels; never silently deduplicate/reassign official data."""
    train_owner = {i: ("train", cid) for cid, ids in train.items() for i in ids}
    train_owner.update({i: ("validation", cid) for cid, ids in val.items() for i in ids})
    seen, internal = {}, []
    for i, image in enumerate(images):
        digest = array_hash(image)
        if digest in seen:
            internal.append([seen[digest], i])
        else:
            seen[digest] = i
    cross = []
    for i, image in enumerate(test_images):
        match = seen.get(array_hash(image))
        if match is not None:
            split, cid = train_owner[match]
            cross.append({"official_train_index": match, "official_test_index": i,
                          "assigned_split": split, "client": cid})
    cross_holdout = [pair for pair in internal if train_owner[pair[0]][0] != train_owner[pair[1]][0]]
    return {"comparison": "SHA256 exact uint8 pixels (not a semantic near-duplicate detector)",
            "official_train_internal_duplicate_pairs": internal,
            "train_vs_validation_exact_duplicate_pairs": cross_holdout,
            "official_train_vs_test_exact_duplicates": cross,
            "policy": "Report benchmark duplicates; no data deletion, reassignment or threshold tuning"}


def prepare_real_data(cfg, out_dir):
    from .training import git_revision
    validate_phase1a_config(cfg)
    pin = upstream()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Prepared data is immutable; use a fresh path: {out}")
    out.mkdir(parents=True, exist_ok=True)
    files = verify_official_files(cfg.get("data_root", "data_cache"))
    images, labels = load_partition(cfg, train=True)
    if images.shape != (50000, 32, 32, 3) or len(labels) != 50000:
        raise ValueError("Expected complete official CIFAR training source")
    np.random.seed(cfg["seed"])
    train, val = partition_and_holdout(labels, cfg)
    write_manifest(out, train, val, labels, cfg, 10000)
    # Independent repeat, never a seed search or alpha fallback.
    np.random.seed(cfg["seed"])
    train2, val2 = partition_and_holdout(labels, cfg)
    assert train == train2 and val == val2, "Official split replay differs"
    n_train = sum(map(len, train.values()))
    n_val = sum(map(len, val.values()))
    before_labels = array_hash(labels)
    before_train_images = array_hash(images)
    views, noise_meta = validation_label_views(labels, val, cfg["heterogeneous_rates"], cfg["validation_noise_seed"])
    replay_views, replay_noise_meta = validation_label_views(labels, val, cfg["heterogeneous_rates"], cfg["validation_noise_seed"] + 1)
    assert before_labels == array_hash(labels) and before_train_images == array_hash(images)
    assert all(np.array_equal(views["clean"][cid], labels[val[cid]]) for cid in val)
    indices = {f"client_{cid}_{part}": np.asarray(ids, dtype=np.int64)
               for part, clients in (("train", train), ("validation", val)) for cid, ids in clients.items()}
    indices["official_test_indices"] = np.arange(10000, dtype=np.int64)
    np.savez(out / "source_indices.npz", **indices)
    private = out / "private_audit"
    private.mkdir()
    for name, mapping in (("validation_labels", views), ("replay_validation_labels", replay_views)):
        (private / f"{name}.json").write_text(json.dumps({s: {str(c): y.tolist() for c, y in clients.items()}
                                                       for s, clients in mapping.items()}, sort_keys=True), encoding="utf-8")
    (private / "noise_manifest.json").write_text(json.dumps(noise_meta, indent=2), encoding="utf-8")
    (private / "replay_noise_manifest.json").write_text(json.dumps(replay_noise_meta, indent=2), encoding="utf-8")
    # Data integrity only, never compute accuracy/logits or choose any model here.
    test_path = Path(cfg.get("data_root", "data_cache")) / CIFAR10.base_folder / CIFAR10.test_list[0][0]
    with test_path.open("rb") as f:
        test_raw = pickle.load(f, encoding="latin1")
    test_images = test_raw["data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    duplicates = check_content_duplicates(images, test_images, train, val)
    (private / "content_duplicates.json").write_text(json.dumps(duplicates, indent=2), encoding="utf-8")
    registered = {k: cfg[k] for k in FROZEN_KEYS}
    (out / "registered_settings.json").write_text(json.dumps(registered, indent=2), encoding="utf-8")
    train_counts = [len(train[c]) for c in sorted(train)]
    val_counts = [len(val[c]) for c in sorted(val)]
    stats = {"dataset": "official CIFAR-10", "n_official_train": 50000, "n_train": n_train,
             "n_validation": n_val, "n_official_test": 10000, "train_n_by_client": train_counts,
             "validation_n_by_client": val_counts, "official_batch_files": files,
             "clean_train_labels_sha256": before_labels, "train_images_sha256": before_train_images,
             "training_label_noise_rate": cfg["training_label_noise_rate"],
             "uniform_expected_validation_noise_rate": noise_meta["uniform"]["0"]["planned_rate"],
             "split_replay_identical": True, "noise_generation_kept_source_arrays_identical": True,
             "replay_only_noise_seed": cfg["validation_noise_seed"] + 1,
             "sgd_batches_per_round": sum(n // cfg["batch_size"] for n in train_counts) * cfg["local_epochs"],
             "drop_last_train_examples_per_round": sum(n // cfg["batch_size"] * cfg["batch_size"] for n in train_counts),
             "upstream_sha": pin["commit"], "model_evaluation_performed": False,
             "git_revision": git_revision(),
             "exact_duplicate_train_val_pairs": len(duplicates["train_vs_validation_exact_duplicate_pairs"]),
             "exact_duplicate_official_train_test_pairs": len(duplicates["official_train_vs_test_exact_duplicates"])}
    (out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    paths = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "prepared_artifacts.json").write_text(json.dumps({p.relative_to(out).as_posix(): sha256(p) for p in paths}, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def verify_prepared_data(cfg, labels, train, val):
    out = Path(cfg["prepared_data_dir"])
    for name, digest in json.loads((out / "prepared_artifacts.json").read_text(encoding="utf-8")).items():
        assert sha256(out / name) == digest, f"Prepared artifact changed: {name}"
    registered = json.loads((out / "registered_settings.json").read_text(encoding="utf-8"))
    assert registered == {k: cfg[k] for k in FROZEN_KEYS}, "Pre-registered settings changed"
    manifest = json.loads((out / "splits_manifest.json").read_text(encoding="utf-8"))
    expected_train, expected_val = validate_manifest(manifest, len(labels), 10000)
    assert train == expected_train and val == expected_val, "Prepared split does not match training"
    stats = json.loads((out / "dataset_statistics.json").read_text(encoding="utf-8"))
    assert array_hash(labels) == stats["clean_train_labels_sha256"], "Training source labels changed"
    actual = verify_official_files(cfg.get("data_root", "data_cache"))
    for name, record in stats["official_batch_files"].items():
        if name != CIFAR10.filename or name in actual:
            assert actual[name] == record, f"Official source bytes changed: {name}"
