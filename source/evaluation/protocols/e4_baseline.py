"""
Protocol E4: Baseline Comparison.

Compare the learned system against a static threshold on lake
extent change rate (the current operational standard).
"""

import numpy as np
import json
import os
from typing import Dict, Optional
from .metrics import (
    compute_lead_time, compute_peak_magnitude,
    compute_false_positive_rate,
    EVENT_DATE, date_to_window_idx,
)


def run_e4_baseline(
    baseline_detector,
    event_features: np.ndarray,
    control_features: Dict[str, np.ndarray],
    learned_metrics: Dict,
    output_dir: str,
) -> Dict:
    """Run Protocol E4: baseline vs. learned system comparison.

    Args:
        baseline_detector: ExtentThresholdDetector instance
        event_features: (T, C) features for event lake (CH-01 = column 0)
        control_features: {lake_id: (T, C)} for control lakes
        learned_metrics: Dict of learned system metrics for comparison
        output_dir: Where to save results

    Returns:
        Dict with baseline metrics and delta comparison
    """
    event_window_idx = date_to_window_idx(EVENT_DATE)

    # Score event lake with baseline
    event_area = event_features[:, 0]  # CH-01 is column 0
    baseline_event_scores = baseline_detector.score(event_area)

    # Find optimal threshold for baseline (median of non-zero scores)
    nonzero = baseline_event_scores[baseline_event_scores > 0]
    baseline_threshold = float(np.median(nonzero)) if len(nonzero) > 0 else 0.05

    baseline_lead_time = compute_lead_time(
        baseline_event_scores, baseline_threshold, event_window_idx
    )
    baseline_peak = compute_peak_magnitude(
        baseline_event_scores, event_window_idx
    )

    # Score control lakes with baseline
    control_baseline_scores = {}
    for lid, features in control_features.items():
        area = features[:, 0]
        control_baseline_scores[lid] = baseline_detector.score(area)

    baseline_fp_rate = compute_false_positive_rate(
        control_baseline_scores, baseline_threshold
    )

    baseline_metrics = {
        'lead_time_days': baseline_lead_time,
        'peak_anomaly_magnitude': float(baseline_peak),
        'false_positive_rate': float(baseline_fp_rate),
        'threshold': float(baseline_threshold),
    }

    # Compare with learned system
    learned_lead = learned_metrics.get('lead_time_days') or 0
    baseline_lead = baseline_lead_time or 0

    delta = {
        'delta_lead_time': learned_lead - baseline_lead,
        'delta_fp_rate': (
            learned_metrics.get('false_positive_rate', 0) - baseline_fp_rate
        ),
        'learned_better_lead_time': learned_lead > baseline_lead,
        'learned_better_fp_rate': (
            learned_metrics.get('false_positive_rate', 0) < baseline_fp_rate
        ),
    }

    results = {
        'baseline': baseline_metrics,
        'learned': {
            k: learned_metrics.get(k)
            for k in ['lead_time_days', 'false_positive_rate', 'peak_anomaly_magnitude']
        },
        'comparison': delta,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'e4_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results
