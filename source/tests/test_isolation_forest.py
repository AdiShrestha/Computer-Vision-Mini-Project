"""
Unit test suite for Isolation Forest Baseline (Contract C08-02).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from models.baseline.isolation_forest import (
    train_isolation_forest,
    evaluate_isolation_forest,
    ema_smooth
)


def test_isolation_forest_train_leakage_boundary():
    """Assert IsolationForest fits strictly on training-role lakes (INV-002)."""
    clf, medians, training_lakes = train_isolation_forest(PROJECT_ROOT / 'data' / 'features_real', seed=42)
    assert len(training_lakes) == 15
    assert 'SGL-001' not in training_lakes
    assert 'SGL-002' not in training_lakes
    assert clf.n_features_in_ == 4680  # 180 windows * 26 columns


def test_ema_smooth_span5():
    """Assert EMA smoothing function behaves deterministically with span=5."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = ema_smooth(scores, span=5)
    assert len(smoothed) == 5
    assert smoothed[0] == 1.0
    assert smoothed[1] > 1.0 and smoothed[1] < 2.0


def test_isolation_forest_evaluation_artifact():
    """Assert evaluate_isolation_forest generates valid JSON artifact with all INV-010 metrics."""
    res = evaluate_isolation_forest()
    assert 'metrics' in res
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

    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'baseline_isolation_forest.json'
    assert artifact_path.exists()
