"""
Protocol E3: Synthetic Anomaly Injection.

Inject physically plausible synthetic anomalies (INV-011) into
otherwise-normal control lake time series. Measure detection rate.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Callable
from .metrics import compute_synthetic_detection_rate, compute_auc


def run_e3_synthetic(
    scorer_fn: Callable,
    control_features: Dict[str, np.ndarray],
    injector,
    threshold: float,
    output_dir: str,
) -> Dict:
    """Run Protocol E3: inject synthetic anomalies and measure detection.

    Args:
        scorer_fn: Callable(features) -> (T,) anomaly scores
        control_features: {lake_id: (T, C) features} for control lakes
        injector: SyntheticInjector instance
        threshold: Detection threshold
        output_dir: Where to save results

    Returns:
        Dict with detection rates and AUC metrics
    """
    all_detections = []
    all_labels = []
    all_scores = []
    per_type_detections = {}

    for lid, features in control_features.items():
        # Generate all injection scenarios for this lake
        injections = injector.generate_injections(features, lid)

        for modified_features, meta in injections:
            anomaly_type = meta['anomaly_type']
            injection_window = meta['window_idx']
            duration = meta.get('duration_windows', 1)

            # Score the modified features
            scores = scorer_fn(modified_features)

            # Check if detected within injection window
            injection_end = min(injection_window + duration, len(scores))
            detected = bool(np.any(
                scores[injection_window:injection_end] > threshold
            ))

            all_detections.append(detected)

            # For AUC computation: label injection windows as 1, rest as 0
            labels = np.zeros(len(scores))
            labels[injection_window:injection_end] = 1
            all_labels.extend(labels.tolist())
            all_scores.extend(scores.tolist())

            # Track per-type
            type_name = meta.get('name', str(anomaly_type))
            if type_name not in per_type_detections:
                per_type_detections[type_name] = []
            per_type_detections[type_name].append(detected)

    # Compute overall metrics
    overall_detection_rate = compute_synthetic_detection_rate(all_detections)

    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    auc = compute_auc(labels_arr, scores_arr)

    # Per-type detection rates
    per_type_rates = {
        name: compute_synthetic_detection_rate(dets)
        for name, dets in per_type_detections.items()
    }

    results = {
        'overall_detection_rate': float(overall_detection_rate),
        'total_injections': len(all_detections),
        'total_detected': sum(all_detections),
        'threshold': float(threshold),
        'auc_roc': float(auc['auc_roc']),
        'auc_pr': float(auc['auc_pr']),
        'per_type_detection_rates': per_type_rates,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'e3_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results
