"""Centralized seeding for reproducible experiments.

Kept in its own lightweight module so data scripts can seed without importing
the heavier training utilities (mlflow / matplotlib / model definitions).
"""

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility.

    Args:
        seed: the random seed to apply across all libraries.
        deterministic: if True, also force cuDNN into deterministic mode and set
            ``PYTHONHASHSEED`` so runs are bit-reproducible (at a small speed
            cost on GPU).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
