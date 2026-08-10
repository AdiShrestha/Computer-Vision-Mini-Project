"""
Exponential Moving Average (EMA) Temporal Smoothing Module.

Invariants:
    INV-006: Default smoothing span is 5 time windows (~150 days at 30-day stride).
"""

import numpy as np


def ema_smooth(scores: np.ndarray, span: int = 5) -> np.ndarray:
    """Compute Exponential Moving Average (EMA) over 1D anomaly score sequence.
    
    Args:
        scores: (T,) raw 1D numpy array of anomaly scores
        span: EMA smoothing span (default 5, per INV-006)
        
    Returns:
        smoothed: (T,) smoothed anomaly score array
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return scores.copy()

    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(scores)
    smoothed[0] = scores[0]

    for i in range(1, len(scores)):
        smoothed[i] = alpha * scores[i] + (1.0 - alpha) * smoothed[i - 1]

    return smoothed.astype(np.float32)
