"""
Protocol E2: Negative Controls.

Run the detector on evaluation_control lakes (dangerous but no
documented GLOF event in the study period). Report the false-positive
rate at the E3-derived detection threshold.

INV-007: Target FP rate <= 0.10 (10%).
"""

import numpy as np
import json
import os
from typing import Dict, List
from .metrics import compute_false_positive_rate


def run_e2_negative_controls(
    control_lake_ids: List[str],
    smoothed_scores: Dict[str, Dict[str, np.ndarray]],
    threshold: float,
    output_dir: str,
) -> Dict:
    """Run Protocol E2 on control lakes.

    Args:
        control_lake_ids: List of control lake IDs
        smoothed_scores: {scorer_name: {lake_id: (T,) scores}}
        threshold: Detection threshold (from E3)
        output_dir: Where to save results

    Returns:
        Dict with per-scorer FP rate results
    """
    results = {}
    for scorer_name, per_lake_scores in smoothed_scores.items():
        control_scores = {
            lid: per_lake_scores[lid]
            for lid in control_lake_ids
            if lid in per_lake_scores
        }

        fp_rate = compute_false_positive_rate(control_scores, threshold)

        # Per-lake breakdown
        per_lake_fp = {}
        for lid, scores in control_scores.items():
            flagged = int(np.sum(scores > threshold))
            per_lake_fp[lid] = {
                'total_windows': len(scores),
                'flagged_windows': flagged,
                'fp_rate': flagged / len(scores) if len(scores) > 0 else 0,
            }

        results[scorer_name] = {
            'overall_fp_rate': float(fp_rate),
            'threshold': float(threshold),
            'n_control_lakes': len(control_scores),
            'per_lake': per_lake_fp,
            'inv007_target': 0.10,
            'inv007_met': fp_rate <= 0.10,
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'e2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results
