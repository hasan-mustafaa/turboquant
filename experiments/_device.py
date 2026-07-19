"""Shared device auto-detection: cuda > mps > cpu.

Centralized because three scripts previously checked `mps` without ever
checking `cuda` first, which would silently fall back to CPU on a RunPod
GPU box unless `--device cuda` was passed explicitly on every invocation.
"""

from __future__ import annotations

import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    """fp16 on accelerators (matches the paper's inference setup and halves
    memory traffic); fp32 on CPU, where fp16 kernels are often emulated and
    can be *slower* than fp32."""
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
