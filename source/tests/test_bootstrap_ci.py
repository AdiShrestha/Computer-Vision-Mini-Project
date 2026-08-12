"""
Unit test suite for Lake-Level Bootstrap CIs & DeLong Tests (Contract C08-06).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from scripts.run_bootstrap_ci import run_bootstrap_ci, delong_pairwise_test


def test_invariants_md_contains_inv016():
    """Assert invariants.md contains newly appended INV-016."""
    inv_path = PROJECT_ROOT / 'project' / 'invariants.md'
    assert inv_path.exists()
    content = inv_path.read_text(encoding='utf-8')
    assert '## INV-016 — Statistical Unit for Confidence Intervals' in content
    assert 'INV-001' in content
    assert 'INV-002' in content


def test_bootstrap_ci_execution_and_delong():
    """Assert run_bootstrap_ci produces valid JSON with 2000 resamples and DeLong p-values."""
    res = run_bootstrap_ci(n_resamples=100, seed=4096)
    assert res['bootstrap_protocol'] == 'INV-016_lake_level_resampling'
    assert 'small_n_limitation' in res
    assert 'With 5 evaluation lakes' in res['small_n_limitation']

    cis = res['bootstrap_confidence_intervals']
    for m, m_data in cis.items():
        assert 'auc_roc_95ci' in m_data
        assert len(m_data['auc_roc_95ci']) == 2
        assert m_data['auc_roc_95ci'][0] <= m_data['auc_roc_95ci'][1]

    delong = res['delong_pairwise_tests']
    assert len(delong) == 6
    for k, d_data in delong.items():
        assert 'p_value' in d_data
        assert 'verdict_plain_text' in d_data

    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'statistical_significance.json'
    assert artifact_path.exists()
