"""Verify full-scale channel extraction produced feature matrices."""
import os
import sys
import json
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
DATA_FEATURES = os.path.join(repo_root, 'data', 'features')


def test_feature_summary_exists():
    """Feature extraction summary JSON exists."""
    summary_path = os.path.join(DATA_FEATURES, 'feature_summary.json')
    assert os.path.isfile(summary_path)


def test_south_lhonak_feature_matrix():
    """South Lhonak has a feature matrix with adequate temporal coverage."""
    matrix_path = os.path.join(DATA_FEATURES, 'SGL-001', 'feature_matrix.npz')
    assert os.path.isfile(matrix_path), "SGL-001 feature matrix missing"
    data = np.load(matrix_path, allow_pickle=True)
    assert 'features' in data
    assert 'window_dates' in data
    features = data['features']
    assert features.shape[0] >= 10, f"Only {features.shape[0]} windows for SGL-001"
    assert features.shape[1] >= 3, f"Only {features.shape[1]} channels for SGL-001"


def test_feature_matrix_count():
    """At least 16 lakes have feature matrices."""
    lake_dirs = [d for d in os.listdir(DATA_FEATURES) 
                 if d.startswith('SGL-') and 
                 os.path.isfile(os.path.join(DATA_FEATURES, d, 'feature_matrix.npz'))]
    assert len(lake_dirs) >= 16, f"Only {len(lake_dirs)} lakes have feature matrices"


def test_feature_matrix_shape_consistent():
    """All feature matrices have the same number of channels."""
    n_channels_set = set()
    for d in os.listdir(DATA_FEATURES):
        if d.startswith('SGL-'):
            matrix_path = os.path.join(DATA_FEATURES, d, 'feature_matrix.npz')
            if os.path.isfile(matrix_path):
                data = np.load(matrix_path, allow_pickle=True)
                n_channels_set.add(data['features'].shape[1])
    assert len(n_channels_set) == 1, (
        f"Inconsistent channel counts across lakes: {n_channels_set}"
    )
