"""Empirical threshold calibration on explicit reference IDs; empirical budgets are not guarantees.

Enforces strict disjointness between calibration lakes and final holdout lakes.
Returns immutable calibration artifacts with threshold, percentile, lake IDs,
and cryptographic state hashes.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, Sequence
import numpy as np
from .contracts import finite_vector, instance_ids


def calibrate_false_alert_threshold(
    scores: Sequence[float],
    calibration_ids: Sequence[str],
    final_ids: Sequence[str],
    target: float,
) -> Dict[str, Any]:
    """Empirical calibration of false alert threshold on explicit reference IDs."""
    s = finite_vector(scores)
    cal_ids, eval_ids = instance_ids(calibration_ids), instance_ids(final_ids)
    if len(cal_ids) != len(s) or not cal_ids or not eval_ids:
        raise ValueError("identify every calibration observation and the final evaluation instances")
    if set(cal_ids) & set(eval_ids):
        raise ValueError("calibration overlaps evaluation")
    if isinstance(target, bool) or not np.isfinite(target) or not 0 <= target < 1:
        raise ValueError("target false alert fraction must be in [0,1)")

    # Sorted unique thresholds and their exact tail counts avoid an O(n^2) repeated scan.
    candidates, counts = np.unique(s, return_counts=True)
    fractions = np.cumsum(counts[::-1])[::-1] / len(s)
    feasible = np.flatnonzero(fractions <= target)
    if feasible.size:
        index = int(feasible[0])
        threshold, fraction = float(candidates[index]), float(fractions[index])
    else:
        with np.errstate(over="ignore"):
            threshold = float(np.nextafter(s.max(), np.inf))
        if not np.isfinite(threshold):
            raise ValueError("no finite threshold can meet this empirical target")
        fraction = 0.0

    percentile = float(1.0 - target)
    state_repr = json.dumps(
        {
            "threshold": threshold,
            "target": float(target),
            "percentile": percentile,
            "calibration_ids": list(cal_ids),
            "n_windows": len(s),
        },
        sort_keys=True,
    ) + ":" + hashlib.sha256(np.sort(s).tobytes()).hexdigest()
    state_hash = hashlib.sha256(state_repr.encode("utf-8")).hexdigest()

    return {
        "threshold": threshold,
        "observed_false_alert_fraction": fraction,
        "target": float(target),
        "percentile": percentile,
        "n_calibration": len(s),
        "n_windows": len(s),
        "calibration_ids": cal_ids,
        "calibration_lake_ids": list(set(cal_ids)),
        "comparator": ">=",
        "state_hash": state_hash,
        "scope": "empirical calibration WINDOW fraction only; not episodes per lake-year or a population guarantee",
    }


def calibrate_percentile_threshold(
    scores: Sequence[float],
    calibration_lake_ids: Sequence[str],
    evaluation_lake_ids: Sequence[str],
    percentile: float,
) -> Dict[str, Any]:
    """Empirically calibrate threshold for given percentile (e.g. 0.95 or 95.0).

    Strictly enforces split disjointness between calibration and evaluation lakes.
    Returns immutable dictionary with threshold, percentile, lake IDs, n_windows,
    and state_hash.
    """
    cal_lakes = list(instance_ids(calibration_lake_ids))
    eval_lakes = list(instance_ids(evaluation_lake_ids))

    if not cal_lakes or not eval_lakes:
        raise ValueError("calibration_lake_ids and evaluation_lake_ids must be non-empty")
    if set(cal_lakes) & set(eval_lakes):
        raise ValueError("calibration overlaps evaluation")

    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence of numbers")
    if np.any(~np.isfinite(s)):
        raise ValueError("all calibration scores must be finite")

    p = float(percentile)
    if p > 1.0:
        p = p / 100.0
    if not (0.0 < p < 1.0):
        raise ValueError(f"percentile must be in (0, 1) or (0, 100), got {percentile}")

    # Compute empirical quantile corresponding to percentile p
    threshold = float(np.percentile(s, p * 100.0))

    state_repr = json.dumps(
        {
            "threshold": threshold,
            "percentile": p,
            "calibration_lake_ids": sorted(list(set(cal_lakes))),
            "n_windows": int(len(s)),
        },
        sort_keys=True,
    ) + ":" + hashlib.sha256(np.sort(s).tobytes()).hexdigest()
    state_hash = hashlib.sha256(state_repr.encode("utf-8")).hexdigest()

    return {
        "threshold": threshold,
        "percentile": p,
        "calibration_lake_ids": list(cal_lakes),
        "n_windows": int(len(s)),
        "state_hash": state_hash,
    }
