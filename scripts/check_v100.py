#!/usr/bin/env python3
"""Read-only preflight plus one tiny GPU operation; no dataset/training."""
import torch


def main():
    print("torch", torch.__version__, "runtime CUDA", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled torch/GPU required")
    print("GPU", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    print("wheel architectures", torch.cuda.get_arch_list())
    if "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("Wheel lacks V100 sm_70; install the documented cu118 environment")
    x = torch.ones(2, device="cuda")
    assert (x + x).sum().item() == 4
    print("GPU preflight passed; formal training has NOT started")


if __name__ == "__main__":
    main()
