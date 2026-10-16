#!/usr/bin/env python3
"""Report verifiable local facts without modifying environment or installing packages."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=ROOT)
        return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def main():
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
              "torch_cuda_available": torch.cuda.is_available(),
              "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torchvision", "numpy", "pytest", "Pillow", "matplotlib", "scikit-learn", "tqdm")},
              "git": command(["git", "rev-parse", "HEAD"]),
              "nvidia_smi": command(["nvidia-smi"])}
    try:
        import psutil
        report["memory"] = psutil.virtual_memory()._asdict()
    except ImportError:
        report["memory"] = "unavailable: psutil not installed"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
