"""
Adversarial verification tests for the channel ablation study (C05-02).
"""
import os
import sys
import json
import numpy as np
import pytest

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

ABLATION_DIR = os.path.join(repo_root, 'results', 'ablation')
SUMMARY_PATH = os.path.join(ABLATION_DIR, 'ablation_summary.json')

EXPECTED_CONFIGS = [
    'FULL_15CH', 'NO_CH01', 'NO_CH02', 'NO_CH03', 'NO_CH04',
    'NO_CH05', 'NO_CH07', 'NO_CH08',
    'OPTICAL_ONLY', 'SAR_ONLY', 'DYNAMIC_ONLY',
]
CHUNK04_SCORE_C_AUC = 0.9521053093284387


def _load_summary():
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def test_ablation_summary_exists():
    """ablation_summary.json must exist."""
    assert os.path.isfile(SUMMARY_PATH), "ablation_summary.json not found"


def test_ablation_version_tagged():
    """Must have ablation_version: C05-02."""
    s = _load_summary()
    assert s.get('ablation_version') == 'C05-02', (
        f"ablation_version={s.get('ablation_version')} — expected C05-02"
    )


def test_all_configs_computed():
    """ADVERSARIAL: All 11 ablation configs must be present in summary."""
    s = _load_summary()
    for cfg in EXPECTED_CONFIGS:
        assert cfg in s['configs'], f"Missing config: {cfg}"
        assert 'auc_roc' in s['configs'][cfg], f"No auc_roc for config {cfg}"


def test_full_config_matches_chunk04():
    """ADVERSARIAL: FULL_15CH AUC-ROC must be within ±0.01 of Chunk04 Score-C.

    If this fails, the ablation is using a different scorer or encoder than
    what produced the Chunk 04 results.
    """
    s = _load_summary()
    full_auc = s['configs']['FULL_15CH']['auc_roc']
    assert abs(full_auc - CHUNK04_SCORE_C_AUC) <= 0.01, (
        f"FULL_15CH AUC-ROC={full_auc:.4f} deviates from Chunk04 ({CHUNK04_SCORE_C_AUC:.4f}) "
        "by more than ±0.01 — different scorer or encoder used"
    )


def test_ablation_produces_variation():
    """ADVERSARIAL: Not all configs should have identical AUC-ROC.

    If all configs produce the same AUC-ROC, the channel masking is not working.
    """
    s = _load_summary()
    aucs = [s['configs'][cfg]['auc_roc'] for cfg in EXPECTED_CONFIGS if 'auc_roc' in s['configs'][cfg]]
    auc_range = max(aucs) - min(aucs)
    assert auc_range > 0.001, (
        f"All configs have nearly identical AUC-ROC (range={auc_range:.6f}) — "
        "channel masking is not affecting the scorer"
    )


def test_masked_channels_zeroed():
    """ADVERSARIAL: Verify masking logic directly."""
    # Import and test the apply_channel_mask function
    sys.path.insert(0, source_root)
    from evaluation.ablation import apply_channel_mask, ABLATION_CONFIGS

    dummy = np.ones((10, 15), dtype=np.float32)
    keep_cols = ABLATION_CONFIGS['NO_CH01']  # all except col 0

    masked = apply_channel_mask(dummy, keep_cols)
    assert masked[:, 0].sum() == 0.0, "Column 0 (CH-01) should be zeroed in NO_CH01 config"
    assert masked[:, 1].sum() == 10.0, "Column 1 should not be zeroed in NO_CH01 config"

    # Test OPTICAL_ONLY
    optical_keep = ABLATION_CONFIGS['OPTICAL_ONLY']  # cols 0-4 + 7
    masked_opt = apply_channel_mask(dummy, optical_keep)
    assert masked_opt[:, 5].sum() == 0.0, "Col 5 (CH-03) should be zeroed in OPTICAL_ONLY"
    assert masked_opt[:, 0].sum() == 10.0, "Col 0 (CH-01) should be kept in OPTICAL_ONLY"


def test_no_encoder_retraining():
    """ADVERSARIAL: Ablation summary must confirm no encoder retraining."""
    s = _load_summary()
    assert s.get('encoder_retrained') is False, (
        "encoder_retrained must be False — ablation must not modify the checkpoint"
    )


def test_channel_contributions_present():
    """Channel contribution dict must be present with 7 entries (one per active channel group)."""
    s = _load_summary()
    contrib = s.get('channel_contributions', {})
    assert len(contrib) >= 7, f"Expected ≥7 channel contributions, got {len(contrib)}"
    for ch in ['CH-01', 'CH-02', 'CH-03', 'CH-04', 'CH-05', 'CH-07', 'CH-08']:
        assert ch in contrib, f"Missing contribution for {ch}"
