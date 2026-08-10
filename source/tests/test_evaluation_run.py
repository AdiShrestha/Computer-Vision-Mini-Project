"""Verify full evaluation execution."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

RESULTS_DIR = os.path.join(repo_root, 'results', 'evaluation')


def test_evaluation_summary_exists():
    """Evaluation summary JSON exists."""
    assert os.path.isfile(os.path.join(RESULTS_DIR, 'evaluation_summary.json'))


def test_all_scorers_have_results():
    """Per-scorer directories exist with e1_results.json."""
    for scorer in ['score_a', 'score_b', 'score_c']:
        scorer_dir = os.path.join(RESULTS_DIR, scorer)
        assert os.path.isdir(scorer_dir), f"Missing results for {scorer}"
        assert os.path.isfile(os.path.join(scorer_dir, 'e1_results.json'))


def test_e1_has_lead_time():
    """Summary records lead time key for all scorers."""
    with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json')) as f:
        summary = json.load(f)
    for scorer in ['score_a', 'score_b', 'score_c']:
        assert 'lead_time_days' in summary['scorer_comparison'][scorer]


def test_south_lhonak_scores_saved():
    """Per-lake anomaly score time series saved for South Lhonak."""
    sgl001_dir = os.path.join(RESULTS_DIR, 'per_lake', 'SGL-001')
    assert os.path.isdir(sgl001_dir), "No per-lake results for SGL-001"


def test_all_inv010_metrics_present():
    """INV-010: All required metrics present in summary."""
    with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json')) as f:
        summary = json.load(f)
    required = ['lead_time_days', 'false_positive_rate', 'auc_roc', 'auc_pr']
    for scorer in ['score_a', 'score_b', 'score_c']:
        for metric in required:
            assert metric in summary['scorer_comparison'][scorer], (
                f"Missing {metric} for {scorer}"
            )
