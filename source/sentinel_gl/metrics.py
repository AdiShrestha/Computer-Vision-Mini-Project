"""Ranking and alert metrics; undefined estimands remain explicit nulls."""
from datetime import date
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .contracts import finite_vector


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
                          min_consecutive: int, horizon_days: int, *, eligible,
                          max_gap_days: int):
    """Declare on the sustaining arrival; explicit abstentions/gaps reset streaks.

    This is a retrospective date-only calculation, not proof of a precursor.
    The caller supplies every scheduled decision, its eligibility, and the
    maximum permissible inter-decision gap from a frozen monitoring protocol.
    Noneligible scores may be NaN/None, never infinities. Event-day is excluded.
    """
    s = np.asarray(scores)
    if s.ndim != 1:
        raise ValueError("scores must be a one-dimensional vector")
    if s.dtype.kind not in "iuf" and not (s.dtype.kind == "O" and all(x is None or type(x) in (int,float) for x in s)):
        raise ValueError("scores must be numeric or null for ineligible decisions")
    s = s.astype(float)
    ok = np.asarray(eligible)
    ds = [date.fromisoformat(str(x)) for x in decision_dates]
    event = date.fromisoformat(event_date)
    if s.ndim != 1 or ok.shape != s.shape or (ok.size and ok.dtype != np.bool_):
        raise ValueError("explicit boolean eligibility must match the score vector")
    if np.isinf(s).any() or np.any(ok.astype(bool) & ~np.isfinite(s)):
        raise ValueError("eligible scores must be finite; infinities are never admissible")
    if len(ds) != len(s) or any(b <= a for a,b in zip(ds,ds[1:])):
        raise ValueError("decision dates must be strictly ordered and match scores")
    if any(type(v) is not int or v < 1 for v in (min_consecutive,horizon_days,max_gap_days)):
        raise ValueError("streak, horizon and maximum gap must be positive integers")
    if isinstance(threshold, bool) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    streak, support, previous = 0, 0, None
    for score, day, valid in zip(s, ds, ok):
        if previous is not None and (day-previous).days > max_gap_days:
            streak = 0
        previous = day
        if not valid or day >= event or (event-day).days > horizon_days:
            streak = 0
            continue
        support += 1
        streak = streak+1 if score >= threshold else 0
        if streak >= min_consecutive:
            return {"status": "DETECTED", "alarm_date": day.isoformat(),
                    "lead_time_days": (event-day).days, "eligible_decisions_seen": support}
    if not support:
        return {"status": "NOT_ESTIMABLE", "reason": "NO_ELIGIBLE_PRE_EVENT_DECISIONS",
                "alarm_date": None, "lead_time_days": None, "eligible_decisions_seen": 0}
    return {"status": "NOT_DETECTED", "alarm_date": None, "lead_time_days": None,
            "eligible_decisions_seen": support}
