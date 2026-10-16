#!/usr/bin/env python3
"""Fetch the byte-identical official archive; verify official MD5, keep archive/partials."""
import argparse
import hashlib
import json
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_cache")
    parser.add_argument("--source", choices=("toronto", "paddle"), default="toronto")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / CIFAR10.filename
    if archive.exists() and not check_integrity(archive, CIFAR10.tgz_md5):
        raise FileExistsError("Existing archive incomplete; preserve it under another filename first")
    url = CIFAR10.url if args.source == "toronto" else "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
    # Mirror appears in PaddlePaddle's own dataset loader; content must match Toronto MD5.
    download_and_extract_archive(url, root, filename=CIFAR10.filename, md5=CIFAR10.tgz_md5,
                                 remove_finished=False)
    dataset = CIFAR10(root, train=True, download=False)
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    report = {"transport_source": args.source, "url": url,
              "official_url": CIFAR10.url, "official_md5": CIFAR10.tgz_md5,
              "archive_md5_passed": True, "archive_sha256": digest.hexdigest(),
              "archive_bytes": archive.stat().st_size, "n_official_train": len(dataset),
              "data_content_modified": False, "model_evaluation_performed": False}
    (root / "download_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
