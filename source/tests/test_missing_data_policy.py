import os
import json
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from models.baseline.missing_data_policy import (
    load_normalization_stats,
    compute_training_medians,
    transform_window,
    process_lake_features,
    run_missing_data_policy
)


def test_medians_computed_from_training_lakes_only():
    """Assert imputation medians use strictly the 15 training-lake IDs from normalization_stats.json."""
    norm_path = PROJECT_ROOT / 'data' / 'features_real' / 'normalization_stats.json'
    assert norm_path.exists(), "normalization_stats.json missing"
    
    with open(norm_path, 'r', encoding='utf-8') as f:
        norm_stats = json.load(f)

    training_lakes = norm_stats['computed_from']
    assert len(training_lakes) == 15
    assert 'SGL-001' not in training_lakes
    assert 'SGL-002' not in training_lakes
    assert 'SGL-003' not in training_lakes
    assert 'SGL-004' not in training_lakes
    assert 'SGL-005' not in training_lakes

    medians = compute_training_medians(PROJECT_ROOT / 'data' / 'features_real', training_lakes)
    assert len(medians) == 13
    assert not np.isnan(medians).any()


def test_transform_window_shape_and_columns():
    """Assert transform_window converts (180, 13) to (180, 26) with binary indicators."""
    medians = np.zeros(13)
    window = np.ones((180, 13))
    # Inject NaNs in first column for first 10 steps
    window[:10, 0] = np.nan

    transformed = transform_window(window, medians)
    assert transformed.shape == (180, 26)

    # Columns 0..12 are imputed features
    # Columns 13..25 are missingness indicators
    assert np.all(transformed[:10, 13] == 1.0)  # Imputed
    assert np.all(transformed[10:, 13] == 0.0)  # Observed
    assert np.all(transformed[:10, 0] == 0.0)   # Imputed value from medians


def test_exclusion_threshold_50_percent():
    """Assert window with >50% NaN is excluded, and <=50% NaN is retained."""
    medians = np.zeros(13)
    
    # 300 days = 5 windows (180 size, 30 stride)
    features = np.ones((300, 13))
    
    # Window 0: 60% NaN -> excluded
    features[:180, :8] = np.nan
    
    windows_26d, valid_idx, stats = process_lake_features(features, medians, window_size=180, stride=30)
    assert stats['excluded_windows'] >= 1
    assert windows_26d.shape[2] == 26


def test_run_missing_data_policy_artifact():
    """Assert run_missing_data_policy creates baseline_imputation_stats.json."""
    summary = run_missing_data_policy()
    assert 'imputed_fraction' in summary
    assert 'imputed_percentage' in summary
    assert summary['n_training_lakes'] == 15
    assert summary['imputed_fraction'] >= 0.0 and summary['imputed_fraction'] <= 1.0

    artifact_path = PROJECT_ROOT / 'data' / 'features_real' / 'baseline_imputation_stats.json'
    assert artifact_path.exists()
