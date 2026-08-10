"""
INV-010 Metric Suite — compute all required evaluation metrics.

Every metric in INV-010 is implemented here. Missing or altered
metrics would violate the pre-committed evaluation protocol.

Invariant compliance:
    INV-006: EMA smoothing span = 5 windows
    INV-007: FP rate reported with 0.10 target
    INV-009: Event date = October 4, 2023
    INV-010: All 7 metrics computed
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta


# INV-009: South Lhonak event date
EVENT_DATE = datetime(2023, 10, 4)

# INV-006: EMA smoothing span
EMA_SPAN = 5

# INV-004: Stride in days
STRIDE_DAYS = 30

# INV-004: Window size in days
WINDOW_SIZE_DAYS = 180

# Temporal extent start (from default_config.yaml)
TEMPORAL_START = datetime(2016, 1, 1)


def window_idx_to_date(window_idx: int) -> datetime:
    """Convert a window index to its center date.

    Window i starts at TEMPORAL_START + i * STRIDE_DAYS,
    and covers WINDOW_SIZE_DAYS (180) from that start.
    The center is at start + 90 days.
    """
    window_start = TEMPORAL_START + timedelta(days=window_idx * STRIDE_DAYS)
    return window_start + timedelta(days=WINDOW_SIZE_DAYS // 2)


def date_to_window_idx(target_date: datetime) -> int:
    """Find the window index whose center is closest to target_date."""
    best_idx = 0
    best_diff = float('inf')
    for i in range(200):  # upper bound
        center = window_idx_to_date(i)
        diff = abs((center - target_date).days)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
        if center > target_date + timedelta(days=WINDOW_SIZE_DAYS):
            break
    return best_idx


def compute_lead_time(smoothed_scores: np.ndarray, threshold: float,
                      event_window_idx: int,
                      min_consecutive: int = 2) -> Optional[int]:
    """Compute lead time in days before event.

    INV-010: Lead time = number of days before event date at which smoothed
    anomaly score first exceeds threshold (sustained for >= min_consecutive
    consecutive windows).

    Args:
        smoothed_scores: (T,) smoothed anomaly scores
        threshold: detection threshold
        event_window_idx: window index corresponding to event date
        min_consecutive: minimum consecutive windows above threshold

    Returns:
        lead_time_days: int or None if no detection before event
    """
    above = smoothed_scores > threshold

    first_detection_idx = None
    consecutive = 0

    for i in range(min(len(above), event_window_idx)):
        if above[i]:
            consecutive += 1
            if consecutive >= min_consecutive and first_detection_idx is None:
                first_detection_idx = i - consecutive + 1
        else:
            consecutive = 0

    if first_detection_idx is None:
        return None

    detection_date = window_idx_to_date(first_detection_idx)
    event_date = window_idx_to_date(event_window_idx)
    lead_time_days = (event_date - detection_date).days

    return max(0, lead_time_days)


def compute_peak_magnitude(smoothed_scores: np.ndarray,
                           event_window_idx: int,
                           pre_event_months: int = 6) -> float:
    """Peak anomaly magnitude in pre-event window.

    INV-010: Maximum smoothed anomaly score in the 6 months before event.

    Args:
        smoothed_scores: (T,) smoothed scores
        event_window_idx: window index of event
        pre_event_months: months before event to consider (default 6)

    Returns:
        peak magnitude (float)
    """
    pre_event_windows = pre_event_months * 30 // STRIDE_DAYS  # ~6 windows
    start_idx = max(0, event_window_idx - pre_event_windows)
    end_idx = event_window_idx

    if start_idx >= end_idx or end_idx > len(smoothed_scores):
        return 0.0

    return float(np.max(smoothed_scores[start_idx:end_idx]))


def compute_false_positive_rate(control_scores: Dict[str, np.ndarray],
                                threshold: float) -> float:
    """False positive rate on control lakes.

    INV-010: Fraction of evaluation_control lake-windows flagged as anomalous.
    INV-007: Target FP rate <= 0.10 (10%).

    Args:
        control_scores: {lake_id: (T,) smoothed scores} for control lakes
        threshold: detection threshold

    Returns:
        FP rate (float in [0, 1])
    """
    total_windows = 0
    flagged_windows = 0

    for lid, scores in control_scores.items():
        total_windows += len(scores)
        flagged_windows += int(np.sum(scores > threshold))

    if total_windows == 0:
        return 0.0

    return flagged_windows / total_windows


def compute_synthetic_detection_rate(detections: List[bool]) -> float:
    """Fraction of synthetic anomalies detected.

    INV-010: Fraction of injected synthetic anomalies detected
    (score exceeds threshold within the injection window).
    """
    if not detections:
        return 0.0
    return sum(detections) / len(detections)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute AUC-ROC and AUC-PR.

    INV-010: Both metrics required for every experiment.

    Args:
        labels: (N,) binary labels (1=anomalous, 0=normal)
        scores: (N,) anomaly scores

    Returns:
        {'auc_roc': float, 'auc_pr': float}
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return {'auc_roc': 0.5, 'auc_pr': 0.0}

    try:
        auc_roc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc_roc = 0.5

    try:
        auc_pr = float(average_precision_score(labels, scores))
    except ValueError:
        auc_pr = 0.0

    return {'auc_roc': auc_roc, 'auc_pr': auc_pr}


def compute_full_metrics(
    event_scores: Optional[np.ndarray],
    event_window_idx: Optional[int],
    control_scores: Dict[str, np.ndarray],
    synthetic_detections: List[bool],
    labels: np.ndarray,
    all_scores: np.ndarray,
    threshold: float,
    baseline_metrics: Optional[Dict] = None,
) -> Dict:
    """Compute the complete INV-010 metric suite.

    Returns a dict with all 7 required metrics plus baseline deltas.
    """
    metrics = {}

    # Lead time (E1)
    if event_scores is not None and event_window_idx is not None:
        metrics['lead_time_days'] = compute_lead_time(
            event_scores, threshold, event_window_idx
        )
        metrics['peak_anomaly_magnitude'] = compute_peak_magnitude(
            event_scores, event_window_idx
        )
    else:
        metrics['lead_time_days'] = None
        metrics['peak_anomaly_magnitude'] = None

    # FP rate (E2) — INV-007 target: <= 0.10
    metrics['false_positive_rate'] = compute_false_positive_rate(
        control_scores, threshold
    )
    metrics['inv007_target'] = 0.10
    metrics['inv007_met'] = metrics['false_positive_rate'] <= 0.10

    # Synthetic detection rate (E3)
    metrics['synthetic_detection_rate'] = compute_synthetic_detection_rate(
        synthetic_detections
    )

    # AUC metrics (E3)
    auc = compute_auc(labels, all_scores)
    metrics['auc_roc'] = auc['auc_roc']
    metrics['auc_pr'] = auc['auc_pr']

    # Baseline comparison (E4)
    if baseline_metrics:
        metrics['delta_lead_time'] = (
            (metrics['lead_time_days'] or 0) -
            (baseline_metrics.get('lead_time_days') or 0)
        )
        metrics['delta_fp_rate'] = (
            metrics['false_positive_rate'] -
            baseline_metrics.get('false_positive_rate', 0)
        )
    else:
        metrics['delta_lead_time'] = None
        metrics['delta_fp_rate'] = None

    return metrics
