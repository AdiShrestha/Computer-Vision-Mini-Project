"""Deterministic random seed setting and state retrieval for reproducibility.

Fulfills INV-012 and Constitution C27.
"""
import os
import random
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed across Python random, NumPy, PyTorch, and CUDA.

    Args:
        seed: Integer seed value to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA determinism flags (INV-012)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed_state() -> Dict[str, Any]:
    """Get current seed state summary for telemetry and logging.

    Returns:
        Dict[str, Any]: Summary dictionary of reproducibility settings.
    """
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", False),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", True),
    }
