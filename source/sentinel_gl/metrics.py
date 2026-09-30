"""Ranking and alert metrics; undefined estimands remain explicit nulls."""
from datetime import date
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .scoring import finite_vector


def ranking_metrics(labels, scores):
    s = finite_vector(scores)
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape != s.shape or y.dtype.kind not in "iu" or not np.isin(y, [0,1]).all():
        raise ValueError("labels must be a matching integer 0/1 vector")
    counts = {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())}
    if len(np.unique(y)) != 2:
        return {"status": "NOT_ESTIMABLE", "reason": "SINGLE_CLASS", "auroc": None,
                "average_precision": None, "class_counts": counts}
    return {"status": "ESTIMABLE", "auroc": float(roc_auc_score(y,s)),
            "average_precision": float(average_precision_score(y,s)), "class_counts": counts}


def false_alert_fraction(scores, threshold: float):
    if isinstance(threshold, bool) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite numeric data")
    if not scores:
        return {"status": "NOT_ESTIMABLE", "reason": "NO_CONTROL_OBSERVATIONS", "fraction": None, "n": 0}
    arrays = [finite_vector(a) for a in scores.values()]
    flat = np.concatenate(arrays)
    return {"status": "ESTIMABLE", "fraction": float((flat >= threshold).mean()),
            "n": len(flat), "n_lakes": len(scores), "flagged": int((flat >= threshold).sum())}


def first_sustained_alarm(scores, decision_dates, threshold: float, event_date: str,
                          min_consecutive: int, horizon_days: int):
    """Alarm exists when the sustaining observation ARRIVES, not at streak start.

    This estimates a retrospective alarm timing, not physical precursor truth.
    Censored/dropped dates must be handled by the caller's frozen observation
    protocol; this primitive does not bridge missing rows automatically.
    """
    s = finite_vector(scores)
    ds = [date.fromisoformat(str(x)) for x in decision_dates]
    event = date.fromisoformat(event_date)
    if len(ds) != len(s) or any(b <= a for a,b in zip(ds,ds[1:])):
        raise ValueError("decision dates must be strictly ordered and match scores")
    if type(min_consecutive) is not int or min_consecutive < 1 or type(horizon_days) is not int or horizon_days < 1:
        raise ValueError("streak length and horizon must be positive integers")
    if isinstance(threshold, bool) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    streak = 0
    for score, day in zip(s, ds):
        if day >= event or (event-day).days > horizon_days:
            streak = 0
            continue
        streak = streak+1 if score >= threshold else 0
        if streak >= min_consecutive:
            return {"status": "DETECTED", "alarm_date": day.isoformat(), "lead_time_days": (event-day).days}
    return {"status": "NOT_DETECTED", "alarm_date": None, "lead_time_days": None}
