"""
Static-Threshold Baseline Detector for Lake Extent (CH-01).

Computes percentage change rate in CH-01 lake area between consecutive time windows:
    change_rate[t] = |area[t] - area[t-1]| / area[t-1]
Produces an anomaly score sequence for Protocol E4 comparison.
"""

import numpy as np


class ExtentThresholdDetector:
    """Static-threshold baseline detector on lake extent (CH-01 area)."""
    
    def __init__(self, threshold: float = 0.10):
        self.threshold = float(threshold)
        
    def score(self, area_series: np.ndarray) -> np.ndarray:
        """Compute per-window extent change rate anomaly scores.
        
        Args:
            area_series: (T,) 1D array of lake surface area (CH-01) or (T, C) feature matrix
            
        Returns:
            scores: (T,) 1D array of absolute percentage change rates
        """
        area = np.asarray(area_series, dtype=np.float64)
        if area.ndim == 2:
            area = area[:, 0]  # CH-01 is column 0
            
        T = len(area)
        if T == 0:
            return np.array([], dtype=np.float32)
            
        scores = np.zeros(T, dtype=np.float64)
        for t in range(1, T):
            prev = area[t - 1]
            curr = area[t]
            if prev > 1e-8:
                scores[t] = abs(curr - prev) / prev
            else:
                scores[t] = 0.0
                
        return scores.astype(np.float32)
    
    def predict(self, area_series: np.ndarray) -> np.ndarray:
        """Binary detection prediction (1=anomalous, 0=normal)."""
        scores = self.score(area_series)
        return (scores > self.threshold).astype(int)
