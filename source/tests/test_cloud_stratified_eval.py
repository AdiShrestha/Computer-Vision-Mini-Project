"""
Unit test suite for Cloud-Fraction Stratified Evaluation (Contract C08-07).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from scripts.cloud_stratified_eval import run_cloud_stratified_eval, compute_window_cloud_fractions


def test_compute_window_cloud_fractions_nan_handling():
    """Assert compute_window_cloud_fractions ignores NaNs when computing window mean."""
    series = np.array([0.1, 0.2, np.nan, 0.3, 0.4] * 40)
    w_clouds = compute_window_cloud_fractions(series, window_size=180, stride=30)
    assert len(w_clouds) > 0
    assert not np.isnan(w_clouds).any()


def test_cloud_stratified_eval_artifact():
    """Assert run_cloud_stratified_eval generates valid JSON artifact with all 5 bins."""
    res = run_cloud_stratified_eval()
    assert 'cloud_bins' in res
    assert 'cloud_robustness_metrics' in res
    
    bins = res['cloud_bins']
    expected_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '>80%']
    for b in expected_bins:
        assert b in bins
        assert 'window_count' in bins[b]
        assert 'score_c_auc_roc' in bins[b]
        assert 'is_thin_bin' in bins[b]

    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'cloud_stratified_evaluation.json'
    assert artifact_path.exists()
