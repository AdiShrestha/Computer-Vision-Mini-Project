"""
Unit test suite for One-Class SVM Baseline (Contract C08-03).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from models.baseline.one_class_svm import (
    train_ocsvm,
    evaluate_ocsvm,
    ema_smooth
)


def test_ocsvm_train_leakage_boundary():
    """Assert OneClassSVM fits strictly on training-role lakes (INV-002)."""
    clf, medians, training_lakes = train_ocsvm(PROJECT_ROOT / 'data' / 'features_real')
    assert len(training_lakes) == 15
    assert 'SGL-001' not in training_lakes
    assert 'SGL-002' not in training_lakes


def test_ocsvm_evaluation_artifact_and_parameters():
    """Assert evaluate_ocsvm generates valid JSON artifact with all INV-010 metrics and parameter documentation."""
    res = evaluate_ocsvm()
    assert 'metrics' in res
    assert 'parameters' in res
    assert res['parameters']['nu'] == 0.1
    assert 'nu_justification' in res['parameters']
    
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

    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'baseline_ocsvm.json'
    assert artifact_path.exists()
