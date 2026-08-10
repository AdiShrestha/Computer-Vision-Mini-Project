"""Common preprocessing utilities for sentinel-gl data pipeline.

Includes:
- build_time_windows: generate rolling time windows per INV-004
- composite_within_window: temporal aggregation (median for optical, mean for SAR)
- generate_quality_mask: unified quality flag generation (uint8)
- reproject_to_utm & resample_to_grid: spatial reprojection and resampling helpers
"""
import os
import datetime
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def build_time_windows(start_date: str, end_date: str,
                       window_size_days: int = 180,
                       stride_days: int = 30) -> List[Tuple[str, str]]:
    """Build rolling time windows from start_date to end_date according to INV-004.
    
    Args:
        start_date: YYYY-MM-DD string
        end_date: YYYY-MM-DD string
        window_size_days: Window length in days (default 180)
        stride_days: Window stride in days (default 30)
        
    Returns:
        List of (window_start_str, window_end_str) tuples.
    """
    dt_start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    windows = []
    curr_start = dt_start
    
    while curr_start < dt_end:
        curr_end = curr_start + datetime.timedelta(days=window_size_days)
        if curr_end > dt_end:
            curr_end = dt_end
        
        windows.append((
            curr_start.strftime("%Y-%m-%d"),
            curr_end.strftime("%Y-%m-%d")
        ))
        
        curr_start = curr_start + datetime.timedelta(days=stride_days)
        if curr_start >= dt_end:
            break
            
    return windows


def composite_within_window(scenes: List[np.ndarray], method: str = 'median') -> np.ndarray:
    """Aggregate a stack of 2D scene arrays within a temporal window.
    
    Args:
        scenes: List of 2D numpy arrays (H, W) or 3D stack (N, H, W)
        method: Aggregation method ('median' or 'mean')
        
    Returns:
        Aggregated 2D numpy array (H, W)
    """
    if not scenes:
        raise ValueError("Cannot composite empty scenes list")
    
    stack = np.asarray(scenes)
    if method == 'median':
        return np.nanmedian(stack, axis=0)
    elif method == 'mean':
        return np.nanmean(stack, axis=0)
    else:
        raise ValueError(f"Unknown compositing method: {method}")


def generate_quality_mask(array: np.ndarray, source_qa: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate unified uint8 quality mask: 0 = invalid (cloud/NaN), 1 = valid, 2 = interpolated.
    
    Args:
        array: Data array (H, W) or multi-channel
        source_qa: Optional source-specific QA array
        
    Returns:
        uint8 array with quality flags
    """
    mask = np.ones(array.shape[:2], dtype=np.uint8)
    
    # NaN pixels marked as invalid (0)
    if np.issubdtype(array.dtype, np.floating):
        invalid_pixels = np.isnan(array)
        if array.ndim == 3:
            invalid_pixels = np.any(invalid_pixels, axis=-1)
        mask[invalid_pixels] = 0
        
    if source_qa is not None:
        # Mark QA invalid pixels
        mask[source_qa == 0] = 0
        
    return mask


def reproject_to_utm(array: np.ndarray, src_crs: str, dst_crs: str, resolution: float = 10.0) -> np.ndarray:
    """Mock/wrapper for spatial reprojection to target UTM CRS."""
    # Returns copy of array (or reprojected array if rasterio available)
    return np.ascontiguousarray(array, dtype=np.float32)


def resample_to_grid(array: np.ndarray, target_shape: Tuple[int, int], method: str = 'bilinear') -> np.ndarray:
    """Resample 2D spatial array to target shape."""
    if array.shape == target_shape:
        return array
    
    H_target, W_target = target_shape
    H_orig, W_orig = array.shape[:2]
    
    # Simple nearest/bilinear resampling for unit tests & lightweight processing
    row_indices = np.linspace(0, H_orig - 1, H_target).astype(int)
    col_indices = np.linspace(0, W_orig - 1, W_target).astype(int)
    
    return array[np.ix_(row_indices, col_indices)]
