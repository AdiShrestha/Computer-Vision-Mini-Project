"""Operational alert episode, exposure accounting, and bounded lead time estimands.

Tracks follow-up exposure Y_i in lake-years, implements sustained detection q
with hysteresis r and 60-day refractory periods, collapses persistent alarm
states to single episodes, and computes bounded lead times without averaging
missed events as zeros.
"""
from __future__ import annotations
import dataclasses
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


@dataclasses.dataclass(frozen=True)
class AlertEpisode:
    """A distinct alert episode defined by sustained crossing and refractory closure."""
    episode_id: str
    lake_id: str
    start_date: str          # ISO-8601 date of first qualifying alarm
    end_date: str            # ISO-8601 date of episode closure
    peak_score: float
    qualifying_decisions_count: int
    duration_days: int
    declaration_date: Optional[str] = None  # Arrival satisfying sustained_q; start_date is the candidate crossing.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "lake_id": self.lake_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "peak_score": self.peak_score,
            "qualifying_decisions_count": self.qualifying_decisions_count,
            "duration_days": self.duration_days,
            "declaration_date": self.declaration_date,
        }


def calculate_lake_exposure_years(*args, **kwargs):
    from .admission import blocked
    blocked("exposure from row counts without an interval ledger")



class AlertEpisodeEngine:
    """Stateful engine detecting sustained alert episodes with hysteresis and refractory limits."""

    def __init__(
        self,
        threshold: float,
        sustained_q: int = 2,
        hysteresis_r: int = 2,
        refractory_days: int = 60,
        max_gap_days: int = 45,
    ):
        if type(threshold) not in (int, float) or not math.isfinite(threshold):
            raise ValueError("threshold must be finite numeric data")
        if type(sustained_q) is not int or sustained_q < 1:
            raise ValueError(f"sustained_q must be >= 1, got {sustained_q}")
        if type(hysteresis_r) is not int or hysteresis_r < 1:
            raise ValueError(f"hysteresis_r must be >= 1, got {hysteresis_r}")
        if type(refractory_days) is not int or refractory_days < 0:
            raise ValueError(f"refractory_days must be >= 0, got {refractory_days}")
        if type(max_gap_days) is not int or max_gap_days < 1:
            raise ValueError("max_gap_days must be a positive integer")

        self.threshold = threshold
        self.sustained_q = sustained_q
        self.hysteresis_r = hysteresis_r
        self.refractory_days = refractory_days
        self.max_gap_days = max_gap_days

    def extract_episodes(
        self,
        lake_id: str,
        decisions: Sequence[Tuple[str, float, bool]],  # (date_str, score, is_eligible)
    ) -> List[AlertEpisode]:
        """Process chronological decisions and return distinct alert episodes.

        Persistent alarm states across multiple consecutive months collapse to
        exactly ONE episode. An episode remains open until hysteresis_r consecutive
        decisions fall below threshold or refractory_days elapses without new triggers.
        """
        # Keep every scheduled row: abstentions are part of the state transition.
        rows = []
        for day_text, score, eligible in decisions:
            day = date.fromisoformat(day_text)
            if type(eligible) is not bool:
                raise ValueError("eligibility must be boolean")
            if eligible and (type(score) not in (int, float) or not math.isfinite(score)):
                raise ValueError("eligible score must be finite numeric data")
            if not eligible and score is not None:
                raise ValueError("ineligible score must be null")
            if rows and day <= rows[-1][0]:
                raise ValueError("decisions must be strictly ordered with unique dates")
            rows.append((day, score, eligible))

        episodes = []
        candidate = []
        active = None
        non_alarms = 0
        previous = None

        def close_episode():
            nonlocal active, non_alarms
            if active is not None:
                start, declaration, last, peak, count = active
                episodes.append(AlertEpisode(
                    f"EP-{lake_id}-{len(episodes)+1:03d}", lake_id,
                    start.isoformat(), last.isoformat(), peak, count,
                    (last-start).days, declaration.isoformat()))
            active, non_alarms = None, 0

        for day, score, eligible in rows:
            if previous is not None and (day-previous).days > self.max_gap_days:
                candidate = []
                close_episode()
            previous = day
            if active is not None and (day-active[2]).days >= self.refractory_days:
                close_episode()
            if not eligible:
                candidate = []
                non_alarms = 0  # Unknown support cannot count as a below-threshold decision.
                continue
            alarm = score >= self.threshold
            if active is not None:
                if alarm:
                    active[2], active[3], active[4] = day, max(active[3], score), active[4]+1
                    non_alarms = 0
                else:
                    non_alarms += 1
                    if non_alarms >= self.hysteresis_r:
                        close_episode()
                continue
            if not alarm:
                candidate = []
                continue
            candidate.append((day, score))
            if len(candidate) >= self.sustained_q:
                active = [candidate[0][0], day, day, max(v for _, v in candidate), len(candidate)]
                candidate = []
                non_alarms = 0
        close_episode()
        return episodes


def compute_alert_burden_estimand(
    episodes: Sequence[AlertEpisode],
    lake_exposures_years: Mapping[str, float],
) -> Dict[str, Any]:
    """Compute alert burden lambda_alert = N_episodes / sum(Y_i)."""
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in lake_exposures_years.values()):
        raise ValueError("exposure must be finite nonnegative numeric data")
    if len({e.episode_id for e in episodes}) != len(episodes) or any(e.lake_id not in lake_exposures_years for e in episodes):
        raise ValueError("duplicate or orphan episode")
    if any(lake_exposures_years[e.lake_id] <= 0 for e in episodes):
        raise ValueError("episode has no positive exposure at its own site")
    total_exposure = sum(lake_exposures_years.values())
    n_episodes = len(episodes)

    if total_exposure <= 0.0:
        return {
            "lambda_alert": None,
            "status": "NOT_ESTIMABLE",
            "reason": "zero_follow_up_exposure",
            "n_episodes": n_episodes,
            "total_lake_years": 0.0 if lake_exposures_years else None,
        }

    lambda_alert = float(n_episodes / total_exposure)
    return {
        "lambda_alert": lambda_alert,
        "status": "ESTIMATED",
        "reason": None,
        "n_episodes": n_episodes,
        "total_lake_years": total_exposure,
    }


@dataclasses.dataclass(frozen=True)
class LeadTimeResult:
    """Case detection and bounded lead time estimand Delta t_lead."""
    event_id: str
    lake_id: str
    status: str  # DETECTED, NOT_DETECTED, NOT_ESTIMABLE
    lead_time_days: Optional[int]
    onset_date: str
    declaration_date: Optional[str]
    warning_horizon_days: int = 180

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "lake_id": self.lake_id,
            "status": self.status,
            "lead_time_days": self.lead_time_days,
            "onset_date": self.onset_date,
            "declaration_date": self.declaration_date,
            "warning_horizon_days": self.warning_horizon_days,
        }


def compute_bounded_lead_time(event_id, lake_id, event_onset_date, decisions, threshold,
                              sustained_q=2, warning_horizon_days=180, *, max_gap_days):
    """Date-only declaration; abstentions/gaps reset the sustaining streak."""
    from .metrics import first_sustained_alarm
    result=first_sustained_alarm([r[1] for r in decisions],[r[0] for r in decisions],threshold,
                                event_onset_date.split('T')[0],sustained_q,warning_horizon_days,
                                eligible=[r[2] for r in decisions],max_gap_days=max_gap_days)
    return LeadTimeResult(event_id,lake_id,result['status'],result['lead_time_days'],
                          event_onset_date.split('T')[0],result['alarm_date'],warning_horizon_days)
