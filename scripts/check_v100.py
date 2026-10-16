#!/usr/bin/env python3
"""Read-only preflight plus one tiny GPU operation; no dataset/training."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1.phase1a import require_v100


def main():
    print(json.dumps(require_v100(), indent=2))
    print("GPU preflight passed; formal training has NOT started")


if __name__ == "__main__":
    main()
