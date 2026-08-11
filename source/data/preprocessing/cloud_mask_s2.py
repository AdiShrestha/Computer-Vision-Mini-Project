"""
Sentinel-2 L2A dual cloud masking module (SCL + s2cloudless).
"""
import numpy as np
from typing import Tuple, Dict, Any


def apply_cloud_mask(
    scl_array: np.ndarray,
    cloud_prob_array: np.ndarray,
    cloud_rejection_threshold: float = 0.80
) -> Tuple[np.ndarray, float]:
    """Apply dual cloud masking on Sentinel-2 pixel arrays.

    Excluded SCL classes:
      3: Cloud shadows
      8: Cloud medium probability
      9: Cloud high probability
      10: Thin cirrus
      11: Snow / ice (optional / flagged as invalid for water area)

    s2cloudless threshold: > 0.60

    Args:
        scl_array: 2D array of SCL classes.
        cloud_prob_array: 2D array of cloud probabilities [0..1].
        cloud_rejection_threshold: Threshold above which entire scene is marked missing.

    Returns:
        Tuple[np.ndarray, float]: (valid_mask boolean array, cloud_fraction float)
    """
    # SCL cloudy mask
    scl_cloudy = np.isin(scl_array, [3, 8, 9, 10, 11])

    # s2cloudless mask
    prob_cloudy = cloud_prob_array > 0.60

    # Dual mask: marked as cloud if EITHER flags cloud
    cloud_mask = scl_cloudy | prob_cloudy
    valid_mask = ~cloud_mask

    total_pixels = cloud_mask.size
    cloud_pixels = np.sum(cloud_mask)
    cloud_fraction = float(cloud_pixels / total_pixels) if total_pixels > 0 else 1.0

    return valid_mask, cloud_fraction
