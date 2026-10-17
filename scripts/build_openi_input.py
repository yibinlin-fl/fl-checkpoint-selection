#!/usr/bin/env python3
"""Build a private transport package; no login, upload, push, training, or cleanup."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1.artifacts import sha256
from p1.openi import verify_input_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/openi_packages/p1_phase1a_input_v1.zip")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    staging = output.parent / (output.stem + "_staging")
    if output.exists() or staging.exists():
        raise FileExistsError("Use new package/staging names; never overwrite inputs")
    root = staging / "p1_phase1a_input"
    (root / "data_cache").mkdir(parents=True)
    shutil.copyfile(ROOT / "data_cache/cifar-10-python.tar.gz", root / "data_cache/cifar-10-python.tar.gz")
    prepared = ROOT / "outputs/phase1a_cifar10_prepared_seed42"
    for source in prepared.rglob("*"):
        if source.is_file():
            dest = root / "prepared" / source.relative_to(prepared)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    subprocess.run(["git", "-C", str(ROOT / "vendor/FNBench"), "bundle", "create", str(root / "FNBench.bundle"), "HEAD"], check=True)
    manifest = {"schema": "p1-openi-phase1a-input-v1", "official_dataset_modified": False,
                "prepared_model_evaluation_performed": False,
                "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in sorted(root.rglob("*")) if p.is_file()}}
    (root / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_input_tree(root)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as z:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(staging).as_posix())
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
                      "archive_root": "p1_phase1a_input", "uploaded": False}, indent=2))


if __name__ == "__main__":
    main()
