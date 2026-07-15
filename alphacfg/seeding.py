"""Global reproducibility helpers for AlphaCFG."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    """Random seed configuration shared by all runners."""

    seed: int = 1
    deterministic: bool = True


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy and PyTorch random seeds from one global seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_worker_seed(base_seed: int, index: int, iteration: int = 0) -> int:
    """Create stable per-game seeds without depending on global RNG state."""
    return int(base_seed) + int(iteration) * 1_000_003 + int(index)
