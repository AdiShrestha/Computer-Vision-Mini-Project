"""
Shared Missing-Data Policy Module for Baseline Anomaly Detectors.

Contract ID: C08-01 (Chunk 08)
Objective: Define leak-free imputation and missingness-indicator transformation
for sklearn baselines (IsolationForest, OneClassSVM) operating on real feature matrices.

Invariants:
    INV-002: Imputation medians derived strictly from 15 training-role lakes.
    INV-004: 180-day sliding window, stride 30.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_normalization_stats(norm_path: Path = None) -> Dict[str, Any]:
    if norm_path is None:
        norm_path = PROJECT_ROOT / 'data' / 'features_real' / 'normalization_stats.json'
    with open(norm_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_training_medians(feature_dir: Path = None, training_lakes: List[str] = None) -> np.ndarray:
    """Compute per-channel medians from training-role lakes ONLY (INV-002)."""
    if feature_dir is None:
        feature_dir = PROJECT_ROOT / 'data' / 'features_real'
    
    if training_lakes is None:
        norm_stats = load_normalization_stats(feature_dir / 'normalization_stats.json')
        training_lakes = norm_stats['computed_from']

    all_features = []
    for lake_id in training_lakes:
        npz_path = feature_dir / lake_id / 'feature_matrix.npz'
        if npz_path.exists():
            data = np.load(npz_path)
            all_features.append(data['features'])  # (T, 13)

    if not all_features:
        raise RuntimeError("No training lake features found to compute medians.")

    stacked = np.concatenate(all_features, axis=0)  # (total_days, 13)
    
    # Compute median per channel, ignoring NaN
    with np.errstate(all='ignore'):
        medians = np.nanmedian(stacked, axis=0)  # (13,)
    
    # Fallback 0.0 if entire column is NaN (safety)
    medians = np.nan_to_num(medians, nan=0.0)
    return medians


def transform_window(window: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Transform one (180, 13) window: impute NaNs with training medians and append 13 missingness indicators.

    Returns:
        transformed: (180, 26) np.ndarray where columns 0..12 are imputed features, 
                     and columns 13..25 are binary missingness indicators (1 = imputed, 0 = observed).
    """
    missing_mask = np.isnan(window).astype(np.float32)  # (180, 13)
    imputed_window = np.where(np.isnan(window), medians[None, :], window)  # (180, 13)
    transformed = np.concatenate([imputed_window, missing_mask], axis=1)  # (180, 26)
    return transformed


def process_lake_features(
    features: np.ndarray,
    medians: np.ndarray,
    window_size: int = 180,
    stride: int = 30
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """Process a full lake feature matrix (T, 13) into imputed (180, 26) window arrays.

    Excludes windows where valid (non-NaN) ratio < 0.50.

    Returns:
        windows_26d: np.ndarray of shape (N_retained, 180, 26)
        valid_indices: list of starting indices of retained windows
        stats: dict recording total_windows, retained_windows, excluded_windows, imputed_element_count, total_element_count
    """
    T, C = features.shape
    retained_windows = []
    valid_indices = []

    total_windows = 0
    excluded_windows = 0
    imputed_element_count = 0
    total_element_count = 0

    for start in range(0, T - window_size + 1, stride):
        total_windows += 1
        window = features[start:start + window_size]  # (180, 13)
        valid_ratio = float((~np.isnan(window)).mean())

        if valid_ratio < 0.50:
            excluded_windows += 1
            continue

        n_nan = int(np.isnan(window).sum())
        n_total = window.size

        imputed_element_count += n_nan
        total_element_count += n_total

        transformed = transform_window(window, medians)  # (180, 26)
        retained_windows.append(transformed)
        valid_indices.append(start)

    if retained_windows:
        windows_26d = np.array(retained_windows, dtype=np.float32)
    else:
        windows_26d = np.empty((0, window_size, 26), dtype=np.float32)

    stats = {
        'total_windows': total_windows,
        'retained_windows': len(retained_windows),
        'excluded_windows': excluded_windows,
        'imputed_element_count': imputed_element_count,
        'total_element_count': total_element_count
    }

    return windows_26d, valid_indices, stats


def run_missing_data_policy(feature_dir: Path = None, output_json: Path = None) -> Dict[str, Any]:
    if feature_dir is None:
        feature_dir = PROJECT_ROOT / 'data' / 'features_real'
    if output_json is None:
        output_json = feature_dir / 'baseline_imputation_stats.json'

    norm_stats = load_normalization_stats(feature_dir / 'normalization_stats.json')
    training_lakes = norm_stats['computed_from']
    medians = compute_training_medians(feature_dir, training_lakes)

    reg_path = PROJECT_ROOT / 'source' / 'data' / 'registry' / 'lake_registry.json'
    with open(reg_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    total_windows = 0
    retained_windows = 0
    excluded_windows = 0
    total_imputed_elements = 0
    total_feature_elements = 0

    per_lake_summary = {}

    for lake in registry['lakes']:
        lake_id = lake['id']
        npz_path = feature_dir / lake_id / 'feature_matrix.npz'
        if not npz_path.exists():
            continue

        features = np.load(npz_path)['features']
        _, _, stats = process_lake_features(features, medians)

        total_windows += stats['total_windows']
        retained_windows += stats['retained_windows']
        excluded_windows += stats['excluded_windows']
        total_imputed_elements += stats['imputed_element_count']
        total_feature_elements += stats['total_element_count']

        per_lake_summary[lake_id] = stats

    imputed_fraction = float(total_imputed_elements / max(total_feature_elements, 1))

    summary = {
        'policy_version': 'v1.0',
        'training_lakes_used_for_medians': training_lakes,
        'n_training_lakes': len(training_lakes),
        'total_windows_evaluated': total_windows,
        'retained_windows': retained_windows,
        'excluded_windows': excluded_windows,
        'exclusion_threshold_valid_ratio': 0.50,
        'total_imputed_elements': total_imputed_elements,
        'total_feature_elements': total_feature_elements,
        'imputed_fraction': round(imputed_fraction, 6),
        'imputed_percentage': round(imputed_fraction * 100, 2),
        'per_channel_medians': medians.tolist(),
        'per_lake_summary': per_lake_summary
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    res = run_missing_data_policy()
    print(f"Missing Data Policy execution complete.")
    print(f"Imputed fraction: {res['imputed_percentage']}% ({res['total_imputed_elements']}/{res['total_feature_elements']})")
    print(f"Retained windows: {res['retained_windows']}/{res['total_windows_evaluated']} (Excluded: {res['excluded_windows']})")
