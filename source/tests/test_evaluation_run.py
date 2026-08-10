"""
Adversarial verification tests for evaluation results (C04-R1).

These tests check CORRECTNESS, not just file existence.
They verify that the evaluation pipeline produced honest results
that match the actual computation outputs.
"""
import os
import sys
import json
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

RESULTS_DIR = os.path.join(repo_root, 'results', 'evaluation')
SUMMARY_PATH = os.path.join(RESULTS_DIR, 'evaluation_summary.json')


def _load_summary():
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def test_evaluation_summary_exists():
    """Summary file must exist."""
    assert os.path.isfile(SUMMARY_PATH), "evaluation_summary.json missing"


def test_all_scorers_have_results():
    """Each scorer has E1 results directory."""
    for scorer in ['score_a', 'score_b', 'score_c']:
        scorer_dir = os.path.join(RESULTS_DIR, scorer)
        assert os.path.isdir(scorer_dir), f"Missing results for {scorer}"
        assert os.path.isfile(os.path.join(scorer_dir, 'e1_results.json'))


def test_scorers_have_different_auc_roc():
    """ADVERSARIAL: Score-A, Score-B, Score-C must have DIFFERENT AUC-ROC values.

    If they're identical, it means the eval_scorer_fn bug was not fixed
    (all three scorers using Score-A under the hood).
    """
    summary = _load_summary()
    auc_a = summary['scorer_comparison']['score_a']['auc_roc']
    auc_b = summary['scorer_comparison']['score_b']['auc_roc']
    auc_c = summary['scorer_comparison']['score_c']['auc_roc']

    # At least two of the three must differ
    assert not (auc_a == auc_b == auc_c), (
        f"All three scorers have identical AUC-ROC ({auc_a}) — "
        "eval_scorer_fn bug not fixed"
    )


def test_lead_time_is_plausible():
    """ADVERSARIAL: Lead time must not equal the entire time series.

    A lead time of 2730+ days means the threshold is too low
    and everything is flagged.
    """
    summary = _load_summary()
    for scorer in ['score_a', 'score_b', 'score_c']:
        lt = summary['scorer_comparison'][scorer]['lead_time_days']
        if lt is not None:
            assert lt < 3000, (
                f"{scorer} lead_time_days={lt} — entire time series flagged, "
                "threshold is too low"
            )


def test_baseline_metrics_not_hardcoded():
    """ADVERSARIAL: Baseline metrics must not be the old hardcoded values."""
    summary = _load_summary()
    bl = summary['scorer_comparison']['baseline']

    hardcoded_fingerprint = (
        bl['false_positive_rate'] == 0.05 and
        bl['synthetic_detection_rate'] == 0.50 and
        bl['auc_roc'] == 0.50 and
        bl['auc_pr'] == 0.50
    )
    assert not hardcoded_fingerprint, (
        "Baseline metrics match the old hardcoded values — "
        "baseline was not actually computed"
    )


def test_south_lhonak_scores_saved():
    """SGL-001 per-lake CSV must exist."""
    sgl001_dir = os.path.join(RESULTS_DIR, 'per_lake', 'SGL-001')
    assert os.path.isdir(sgl001_dir), "No per-lake results for SGL-001"
    csv_path = os.path.join(sgl001_dir, 'anomaly_scores.csv')
    assert os.path.isfile(csv_path)


def test_score_a_values_are_normalized_scale():
    """ADVERSARIAL: Score-A values must be in normalized MSE scale (~0-20).

    Raw (unnormalized) Score-A produces MSE in the 500-2000 range.
    Properly normalized Score-A produces MSE in the 0-20 range.
    """
    import pandas as pd
    csv_path = os.path.join(RESULTS_DIR, 'per_lake', 'SGL-001', 'anomaly_scores.csv')
    df = pd.read_csv(csv_path)

    max_score_a = df['score_a_raw'].max()
    assert max_score_a < 100, (
        f"Score-A max value is {max_score_a:.1f} — features were not normalized "
        "before model inference (expected < 100 for normalized features)"
    )


def test_all_inv010_metrics_present():
    """INV-010: All required metrics present in summary."""
    summary = _load_summary()
    required = ['lead_time_days', 'false_positive_rate', 'auc_roc', 'auc_pr',
                'synthetic_detection_rate', 'peak_anomaly_magnitude']
    for scorer in ['score_a', 'score_b', 'score_c']:
        for metric in required:
            assert metric in summary['scorer_comparison'][scorer], (
                f"Missing {metric} for {scorer}"
            )


def test_rework_version_tagged():
    """Summary must indicate this is the reworked version."""
    summary = _load_summary()
    assert summary.get('rework_version') == 'C04-R1', (
        "evaluation_summary.json does not have rework_version tag"
    )
