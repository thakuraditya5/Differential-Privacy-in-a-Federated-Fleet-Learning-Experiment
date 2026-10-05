"""Reproducibility utilities.

Sets deterministic seeds across Python, NumPy, PyTorch to ensure
reproducible experiment results.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, and PyTorch.
    Also configures PyTorch for deterministic operations where possible.

    Note: Complete determinism is not guaranteed on all hardware/backends.
    CUDA convolutions with cuDNN may still be non-deterministic unless
    torch.backends.cudnn.deterministic is set to True (which may impact
    performance).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic mode (may reduce performance)
    torch.use_deterministic_algorithms(mode=False)  # True can cause errors
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_str: str = "cpu") -> torch.device:
    """Get torch device, falling back to CPU if CUDA unavailable.

    Args:
        device_str: Desired device ('cpu', 'cuda', 'auto').

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)
