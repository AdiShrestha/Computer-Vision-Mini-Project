"""
Adversarial verification tests for Chunk 05 — Sensitivity Analysis, Multi-Event Generalization, & Final Synthesis.

Tests are added progressively by each contract (C05-01 through C05-05).
"""
import os
import json
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

RQ2_DIR = os.path.join(repo_root, 'results', 'rq2')
ABLATION_DIR = os.path.join(repo_root, 'results', 'ablation')


# ============================================================
# C05-01: InSAR infeasibility documentation
# ============================================================

def test_insar_infeasibility_documented():
    """InSAR negative result file must exist."""
    assert os.path.isfile(os.path.join(RQ2_DIR, 'insar_infeasibility.md'))
    assert os.path.isfile(os.path.join(RQ2_DIR, 'insar_metadata.json'))


def test_insar_metadata_coherence_matches_decision001():
    """ADVERSARIAL: coherence value must match Decision 001 (0.24)."""
    with open(os.path.join(RQ2_DIR, 'insar_metadata.json')) as f:
        meta = json.load(f)
    assert abs(meta['mean_coherence_sgl001'] - 0.24) < 0.001, (
        f"mean_coherence_sgl001={meta['mean_coherence_sgl001']} != 0.24 (Decision 001)"
    )
    assert meta['insar_feasibility'] == 'infeasible'


def test_insar_metadata_has_required_fields():
    """All required fields present in insar_metadata.json."""
    with open(os.path.join(RQ2_DIR, 'insar_metadata.json')) as f:
        meta = json.load(f)
    required = ['insar_feasibility', 'mean_coherence_sgl001', 'coherence_threshold_for_feasibility',
                'decision_ref', 'active_channels', 'ablation_approach']
    for field in required:
        assert field in meta, f"Missing required field: {field}"


def test_active_channels_correct():
    """Active channels list must not include CH-06."""
    with open(os.path.join(RQ2_DIR, 'insar_metadata.json')) as f:
        meta = json.load(f)
    assert 'CH-06' not in meta['active_channels'], "CH-06 (InSAR) should be excluded"
    assert 'CH-01' in meta['active_channels'], "CH-01 should be active"
    assert len(meta['active_channels']) == 7, f"Expected 7 active channels, got {len(meta['active_channels'])}"
