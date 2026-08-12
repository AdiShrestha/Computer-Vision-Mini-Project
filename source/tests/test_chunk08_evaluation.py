"""
Unit test suite for Top-Level Seven-Method Evaluation (Contract C08-05).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from scripts.run_evaluation import run_evaluation_real_data, minmax_normalize


def test_minmax_normalize_safeness():
    """Assert minmax_normalize handles zero-variance and regular arrays correctly."""
    const_arr = np.ones((10, 10))
    normed_const = minmax_normalize(const_arr)
    assert np.all(normed_const == 0.0)

    reg_arr = np.array([1.0, 3.0, 5.0])
    normed_reg = minmax_normalize(reg_arr)
    assert np.allclose(normed_reg, [0.0, 0.5, 1.0])


def test_evaluation_summary_artifact_seven_methods():
    """Assert evaluation_summary_real_data.json contains valid metrics for all 7 methods."""
    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'evaluation_summary_real_data.json'
    assert artifact_path.exists(), "evaluation_summary_real_data.json missing"

    with open(artifact_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    assert summary['n_methods'] == 7
    assert summary['scorer_non_identity_verified'] is True
    
    methods = summary['scorer_comparison']
    expected_methods = [
        'score_a', 'score_b', 'score_c',
        'isolation_forest', 'one_class_svm', 'cusum',
        'extent_threshold'
    ]
    for m in expected_methods:
        assert m in methods, f"Missing method: {m}"
        metrics = methods[m]
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        assert 'lead_time_days' in metrics
        assert 'false_positive_rate' in metrics
        assert 'synthetic_detection_rate' in metrics

    # Assert Extent Threshold metrics are NOT fabricated default placeholders
    extent_m = methods['extent_threshold']
    assert isinstance(extent_m['auc_roc'], float)
    assert extent_m['auc_roc'] != 0.6500 or extent_m['auc_pr'] != 0.6100, "Extent threshold metrics must be computed dynamically, not hardcoded placeholders"
