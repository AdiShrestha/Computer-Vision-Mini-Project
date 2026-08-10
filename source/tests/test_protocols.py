"""Verify evaluation protocol implementations."""
import os
import sys
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_metrics_module_exists():
    """INV-010 metrics module exists."""
    from evaluation.protocols.metrics import compute_full_metrics
    assert callable(compute_full_metrics)


def test_lead_time_computation():
    """Lead time computation is correct."""
    from evaluation.protocols.metrics import compute_lead_time

    # Score that rises at window 80, event at window 90
    scores = np.zeros(108)
    scores[80:] = 1.0

    lead_time = compute_lead_time(scores, threshold=0.5, event_window_idx=90)
    assert lead_time is not None
    assert lead_time > 0


def test_lead_time_returns_none_when_no_detection():
    """Lead time returns None when no score exceeds threshold."""
    from evaluation.protocols.metrics import compute_lead_time

    scores = np.zeros(108)  # All below threshold
    lead_time = compute_lead_time(scores, threshold=0.5, event_window_idx=90)
    assert lead_time is None


def test_fp_rate_computation():
    """False positive rate computation is correct."""
    from evaluation.protocols.metrics import compute_false_positive_rate

    # 10% of windows above threshold
    scores = np.zeros(100)
    scores[:10] = 1.0

    fp_rate = compute_false_positive_rate({'lake1': scores}, threshold=0.5)
    assert abs(fp_rate - 0.10) < 0.01


def test_event_date_is_inv009():
    """Event date matches INV-009."""
    from evaluation.protocols.metrics import EVENT_DATE
    from datetime import datetime
    assert EVENT_DATE == datetime(2023, 10, 4)


def test_ema_span_is_inv006():
    """EMA span matches INV-006."""
    from evaluation.protocols.metrics import EMA_SPAN
    assert EMA_SPAN == 5


def test_e1_module_exists():
    """E1 retrospective protocol exists."""
    from evaluation.protocols.e1_retrospective import run_e1_retrospective
    assert callable(run_e1_retrospective)


def test_e2_module_exists():
    """E2 negative controls protocol exists."""
    from evaluation.protocols.e2_negative_controls import run_e2_negative_controls
    assert callable(run_e2_negative_controls)


def test_e3_module_exists():
    """E3 synthetic protocol exists."""
    from evaluation.protocols.e3_synthetic import run_e3_synthetic
    assert callable(run_e3_synthetic)


def test_e4_module_exists():
    """E4 baseline protocol exists."""
    from evaluation.protocols.e4_baseline import run_e4_baseline
    assert callable(run_e4_baseline)


def test_runner_exists():
    """Evaluation runner exists."""
    from evaluation.runner import run_full_evaluation
    assert callable(run_full_evaluation)


def test_auc_computation():
    """AUC computation handles normal case."""
    from evaluation.protocols.metrics import compute_auc

    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    auc = compute_auc(labels, scores)
    assert 0 <= auc['auc_roc'] <= 1
    assert 0 <= auc['auc_pr'] <= 1


def test_auc_handles_single_class():
    """AUC handles edge case with single class."""
    from evaluation.protocols.metrics import compute_auc

    labels = np.zeros(10)  # All negative
    scores = np.random.rand(10)

    auc = compute_auc(labels, scores)
    assert auc['auc_roc'] == 0.5  # Random baseline


def test_synthetic_detection_rate():
    """Synthetic detection rate computation is correct."""
    from evaluation.protocols.metrics import compute_synthetic_detection_rate

    detections = [True, True, False, True, False]
    rate = compute_synthetic_detection_rate(detections)
    assert abs(rate - 0.6) < 0.01


def test_peak_magnitude_computation():
    """Peak magnitude in pre-event window is correct."""
    from evaluation.protocols.metrics import compute_peak_magnitude

    scores = np.zeros(108)
    scores[85] = 5.0  # Spike before event at window 90
    scores[95] = 10.0  # Spike after event — should NOT be counted

    peak = compute_peak_magnitude(scores, event_window_idx=90)
    assert peak == 5.0
