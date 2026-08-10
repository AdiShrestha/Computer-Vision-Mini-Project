"""
Protocol E1: Retrospective Backtesting on South Lhonak (SGL-001).

Run the trained detector "as if live" over the months/years before
South Lhonak's October 2023 collapse. The encoder was never trained
on South Lhonak data (INV-002).

This is the project's signature result.
"""

import numpy as np
import json
import os
from typing import Dict, Optional
from .metrics import (
    EVENT_DATE, date_to_window_idx, window_idx_to_date,
    compute_lead_time, compute_peak_magnitude
)


def run_e1_retrospective(
    event_lake_id: str,
    smoothed_scores: Dict[str, np.ndarray],
    threshold: float,
    output_dir: str,
) -> Dict:
    """Run Protocol E1 on the event lake.

    Args:
        event_lake_id: Lake ID (should be SGL-001)
        smoothed_scores: {scorer_name: (T,) smoothed scores} for event lake
        threshold: Detection threshold (derived from E3 ROC)
        output_dir: Where to save per-scorer results

    Returns:
        Dict with per-scorer E1 results
    """
    event_window_idx = date_to_window_idx(EVENT_DATE)

    results = {}
    for scorer_name, scores in smoothed_scores.items():
        lead_time = compute_lead_time(scores, threshold, event_window_idx)
        peak_mag = compute_peak_magnitude(scores, event_window_idx)

        results[scorer_name] = {
            'event_lake': event_lake_id,
            'event_date': EVENT_DATE.isoformat(),
            'event_window_idx': int(event_window_idx),
            'threshold': float(threshold),
            'lead_time_days': lead_time,
            'peak_anomaly_magnitude': float(peak_mag),
            'detected': lead_time is not None,
            'anomaly_scores': scores.tolist(),
        }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'e1_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results
