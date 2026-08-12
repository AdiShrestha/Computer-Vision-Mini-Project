"""
Unit test suite for CUSUM Baseline (Contract C08-04).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from models.baseline.cusum_baseline import (
    compute_cusum_series,
    process_lake_cusum_windows,
    evaluate_cusum
)


def test_cusum_nan_skipping_recursion():
    """Assert NaN entries in CH-01 series are skipped without state corruption or crash."""
    ch01 = np.ones(100) * 1.5
    # Inject NaNs
    ch01[20:30] = np.nan
    ch01[50:60] = np.nan
    # Inject jump after day 70
    ch01[70:] = 4.0

    scores = compute_cusum_series(ch01, k=0.50, baseline_days=10)
    assert len(scores) == 100
    assert not np.isnan(scores).any()
    assert scores[80] > scores[10]


def test_cusum_continuous_scores():
    """Assert score output is continuous float, not binary 0/1."""
    ch01 = np.random.normal(2.0, 0.2, 200)
    ch01[150:] += 1.5

    scores = compute_cusum_series(ch01, k=0.50)
    unique_vals = np.unique(scores)
    assert len(unique_vals) > 10, "Score array must be continuous float"


def test_cusum_evaluation_artifact_and_sweep():
    """Assert evaluate_cusum produces valid JSON with sensitivity sweep and all INV-010 metrics."""
    res = evaluate_cusum()
    assert res['is_univariate'] is True
    assert res['channel_used'] == 'CH-01_lake_area_km2'
    assert 'sensitivity_sweep_k' in res
    assert 'literature_citation' in res

    metrics = res['metrics']
    required_keys = [
        'lead_time_days',
        'peak_anomaly_magnitude',
        'false_positive_rate',
        'synthetic_detection_rate',
        'auc_roc',
        'auc_pr'
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric key: {k}"

    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'baseline_cusum.json'
    assert artifact_path.exists()
