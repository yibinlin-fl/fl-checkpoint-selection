#!/usr/bin/env python3
"""Fetch exact upstream and apply only the reviewed import patch. No training/data download."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    pin = json.loads((ROOT / "configs/fnbench_upstream.json").read_text())
    dest = ROOT / "vendor/FNBench"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", pin["url"], str(dest)], check=True)
    sha = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if sha != pin["commit"]:
        status = subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"], text=True)
        if status:
            raise RuntimeError("Vendor contains changes; refusing checkout")
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", pin["commit"]], check=True)
    patch = str(ROOT / pin["patch"])
    applied = subprocess.run(["git", "-C", str(dest), "apply", "--reverse", "--check", patch], capture_output=True)
    if applied.returncode:
        subprocess.run(["git", "-C", str(dest), "apply", "--check", patch], check=True)
        subprocess.run(["git", "-C", str(dest), "apply", patch], check=True)
    (ROOT / "vendor/FNBENCH_PINNED_COMMIT.txt").write_text(pin["commit"] + "\n")
    print("Pinned FNBench prepared:", pin["commit"], "; LICENSE is MIT (README badge differs)")


if __name__ == "__main__":
    main()
