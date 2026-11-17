"""Milestone SR-1 Evaluation Estimands E1-E5 and Bootstrap Inference.

Governed by plan.md SR-1.4 item 3 and SR-1.5 rows 10, 13, 14.
Estimands:
- E1: First post-event detection date and delay (days) per detector at frozen threshold.
- E2: Uneventful false-alarm alert episodes per eligible lake-year.
- E3: Seasonally matched out-of-sample percentile of autumn post-event scores vs autumn historical scores.
- E4: Zone-contrast statistic: Water/moraine-zone score vs background-zone score (internal negative control).
- E5: Incremental discrimination of T-MAE over physical water-area rule and baselines.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_zone_contrast(
    patch_scores: np.ndarray,      # (N_patches,) reconstruction or anomaly scores
    patch_zones: np.ndarray,       # (N_patches,) 0: water, 1: moraine, 2: background
    patch_observed: np.ndarray,    # (N_patches,) bool
) -> Dict[str, float]:
    """Estimand E4: Zone Contrast within a single scene.
    
    Target Zone (Water + Moraine: zones 0, 1) vs Negative Control Zone (Background: zone 2).
    """
    obs_water = patch_observed & np.isin(patch_zones, [0, 1])
    obs_bg = patch_observed & (patch_zones == 2)
    
    score_target = float(np.nanmedian(patch_scores[obs_water])) if obs_water.any() else 0.0
    score_bg = float(np.nanmedian(patch_scores[obs_bg])) if obs_bg.any() else 0.0
    contrast = score_target - score_bg
    
    return {
        "score_target_zone": score_target,
        "score_background_zone": score_bg,
        "zone_contrast": contrast,
    }


def compute_percentile_rank(
    query_val: float,
    reference_distribution: List[float],
) -> float:
    """Estimand E3: Percentile rank of query score against historical reference distribution."""
    valid_refs = [x for x in reference_distribution if math.isfinite(x)]
    if not valid_refs or not math.isfinite(query_val):
        return 50.0
    rank = sum(1.0 for x in valid_refs if x < query_val) + 0.5 * sum(1.0 for x in valid_refs if x == query_val)
    return float(100.0 * rank / len(valid_refs))


def block_bootstrap_ci(
    time_series_records: List[Dict[str, Any]],  # List of {'dt': datetime, 'score': float}
    block_days: int = 30,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Moving-block bootstrap confidence interval for time series median."""
    rng = np.random.RandomState(seed)
    scores = np.array([r["score"] for r in time_series_records if math.isfinite(r["score"])])
    if len(scores) < 3:
        med = float(np.median(scores)) if len(scores) > 0 else 0.0
        return med, med, med
        
    n = len(scores)
    block_size = max(1, min(n // 2, 5))
    n_blocks = int(math.ceil(n / block_size))
    
    boot_medians = []
    for _ in range(n_bootstraps):
        start_indices = rng.randint(0, n - block_size + 1, size=n_blocks)
        sample = []
        for s in start_indices:
            sample.extend(scores[s:s+block_size])
        sample = sample[:n]
        boot_medians.append(np.median(sample))
        
    low = float(np.percentile(boot_medians, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))
    med = float(np.median(scores))
    return med, low, high


def evaluate_detection_delay_e1(
    eval_timeline: List[Dict[str, Any]],  # sorted by datetime
    detector_key: str,
    threshold: float,
    event_start_utc: str = "2023-10-02T18:30:00Z",
) -> Dict[str, Any]:
    """Estimand E1: Detection delay (days from breach onset to first triggered alert)."""
    post_event = [r for r in eval_timeline if r["acquired_at"] >= event_start_utc]
    if not post_event:
        return {"detected": False, "detection_delay_days": None, "first_alert_date": None}
        
    t_event = datetime.fromisoformat(event_start_utc.replace("Z", "+00:00"))
    for r in post_event:
        val = r.get(detector_key, 0.0)
        if math.isfinite(val) and val >= threshold:
            t_det = datetime.fromisoformat(r["acquired_at"].replace("Z", "+00:00"))
            delay_days = max(0.0, (t_det - t_event).total_seconds() / 86400.0)
            return {
                "detected": True,
                "first_alert_date": r["acquired_at"],
                "detection_delay_days": float(delay_days),
                "alert_score": float(val),
                "threshold": float(threshold),
            }
            
    return {
        "detected": False,
        "detection_delay_days": None,
        "first_alert_date": None,
        "max_post_score": float(max((r.get(detector_key, 0.0) for r in post_event), default=0.0)),
        "threshold": float(threshold),
    }


def evaluate_false_alarm_burden_e2(
    control_timeline: List[Dict[str, Any]],
    detector_key: str,
    threshold: float,
) -> Dict[str, Any]:
    """Estimand E2: Uneventful alert episodes on control site / pre-event years."""
    alerts = 0
    in_episode = False
    episode_count = 0
    total_valid = 0
    
    for r in control_timeline:
        val = r.get(detector_key, 0.0)
        if not math.isfinite(val):
            continue
        total_valid += 1
        if val >= threshold:
            alerts += 1
            if not in_episode:
                episode_count += 1
                in_episode = True
        else:
            in_episode = False
            
    rate = float(episode_count / max(1, total_valid))
    return {
        "total_eval_scenes": total_valid,
        "alert_scenes_count": alerts,
        "uneventful_episodes_count": episode_count,
        "episode_rate_per_scene": rate,
    }
