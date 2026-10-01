"""Calibration on explicit reference IDs; empirical budgets are not guarantees."""
import numpy as np
from .scoring import finite_vector


def calibrate_false_alert_threshold(scores, calibration_ids, final_ids, target: float):
    s = finite_vector(scores)
    calibration_ids, final_ids = tuple(calibration_ids), tuple(final_ids)
    if len(calibration_ids) != len(s) or not calibration_ids or not final_ids:
        raise ValueError("identify every calibration observation and the final evaluation instances")
    if any(not isinstance(x, str) or not x.strip() for x in calibration_ids+final_ids):
        raise ValueError("instance IDs must be nonempty strings")
    if len(set(calibration_ids)) != len(calibration_ids) or len(set(final_ids)) != len(final_ids):
        raise ValueError("duplicate instance IDs")
    if set(calibration_ids) & set(final_ids):
        raise ValueError("calibration overlaps evaluation")
    if isinstance(target, bool) or not np.isfinite(target) or not 0 <= target < 1:
        raise ValueError("target false alert fraction must be in [0,1)")
    # Sorted unique thresholds and their exact tail counts avoid an O(n^2)
    # repeated scan. The comparator and tie handling remain >= throughout.
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
        fraction = 0.
    return {"threshold": threshold, "observed_false_alert_fraction": fraction,
            "target": float(target), "n_calibration": len(s),
            "calibration_ids": calibration_ids, "comparator": ">=",
            "scope": "empirical calibration WINDOW fraction only; not episodes per lake-year or a population guarantee"}
