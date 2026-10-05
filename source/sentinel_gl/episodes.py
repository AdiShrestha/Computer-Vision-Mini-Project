"""Operational alert episode, exposure accounting, and bounded lead time estimands.

Tracks follow-up exposure Y_i in lake-years, implements sustained detection q
with hysteresis r and 60-day refractory periods, collapses persistent alarm
states to single episodes, and computes bounded lead times without averaging
missed events as zeros.
"""
from __future__ import annotations
import dataclasses
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "lake_id": self.lake_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "peak_score": self.peak_score,
            "qualifying_decisions_count": self.qualifying_decisions_count,
            "duration_days": self.duration_days,
        }


def calculate_lake_exposure_years(
    decision_dates: Sequence[str],
    stride_days: int = 30,
) -> float:
    """Calculate follow-up exposure Y_i in lake-years over scheduled eligible decisions.

    Y_i = (N_decisions * stride_days) / 365.25.
    """
    if not decision_dates:
        return 0.0
    total_days = len(decision_dates) * stride_days
    return float(total_days / 365.25)


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
        if sustained_q < 1:
            raise ValueError(f"sustained_q must be >= 1, got {sustained_q}")
        if hysteresis_r < 1:
            raise ValueError(f"hysteresis_r must be >= 1, got {hysteresis_r}")
        if refractory_days < 0:
            raise ValueError(f"refractory_days must be >= 0, got {refractory_days}")

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
        # Sort chronologically by date
        sorted_decisions = sorted(
            [d for d in decisions if d[2]],  # Filter only eligible decisions
            key=lambda x: date.fromisoformat(x[0])
        )

        if not sorted_decisions:
            return []

        episodes: List[AlertEpisode] = []
        episode_idx = 1

        in_episode = False
        consecutive_alarms = 0
        consecutive_non_alarms = 0

        episode_start_date: Optional[date] = None
        last_alarm_date: Optional[date] = None
        last_decision_date: Optional[date] = None
        peak_score = -float("inf")
        qualifying_count = 0

        for d_str, score, _ in sorted_decisions:
            d_curr = date.fromisoformat(d_str)

            # Check for excessive gap between decisions
            if last_decision_date is not None:
                gap = (d_curr - last_decision_date).days
                if gap > self.max_gap_days:
                    # Gap too large: reset active candidate tracking
                    consecutive_alarms = 0

            last_decision_date = d_curr
            is_alarm = score >= self.threshold

            if in_episode:
                # Active episode is open
                days_since_last_alarm = (d_curr - last_alarm_date).days if last_alarm_date else 0

                if is_alarm:
                    qualifying_count += 1
                    consecutive_non_alarms = 0
                    last_alarm_date = d_curr
                    if score > peak_score:
                        peak_score = score
                else:
                    consecutive_non_alarms += 1

                # Check episode closure conditions:
                # 1. r consecutive non-alarms
                # 2. Refractory period exceeded since last alarm trigger
                if consecutive_non_alarms >= self.hysteresis_r or days_since_last_alarm >= self.refractory_days:
                    # Close current episode
                    end_date = last_alarm_date if last_alarm_date else d_curr
                    duration = (end_date - episode_start_date).days if episode_start_date else 0
                    episodes.append(
                        AlertEpisode(
                            episode_id=f"EP-{lake_id}-{episode_idx:03d}",
                            lake_id=lake_id,
                            start_date=episode_start_date.isoformat(),
                            end_date=end_date.isoformat(),
                            peak_score=peak_score,
                            qualifying_decisions_count=qualifying_count,
                            duration_days=max(0, duration),
                        )
                    )
                    episode_idx += 1
                    in_episode = False
                    consecutive_alarms = 0
                    consecutive_non_alarms = 0
                    episode_start_date = None
                    last_alarm_date = None
                    peak_score = -float("inf")
                    qualifying_count = 0

            else:
                # Not currently in an active episode: track candidate alarms for sustained entry
                if is_alarm:
                    consecutive_alarms += 1
                    if consecutive_alarms == 1:
                        candidate_start = d_curr
                    if score > peak_score:
                        peak_score = score

                    if consecutive_alarms >= self.sustained_q:
                        # Sustained alarm declaration satisfied -> open new episode
                        in_episode = True
                        episode_start_date = candidate_start
                        last_alarm_date = d_curr
                        qualifying_count = consecutive_alarms
                        consecutive_non_alarms = 0
                else:
                    consecutive_alarms = 0
                    peak_score = -float("inf")

        # If loop finishes with an open episode, close it at final trigger date
        if in_episode and episode_start_date and last_alarm_date:
            duration = (last_alarm_date - episode_start_date).days
            episodes.append(
                AlertEpisode(
                    episode_id=f"EP-{lake_id}-{episode_idx:03d}",
                    lake_id=lake_id,
                    start_date=episode_start_date.isoformat(),
                    end_date=last_alarm_date.isoformat(),
                    peak_score=peak_score,
                    qualifying_decisions_count=qualifying_count,
                    duration_days=max(0, duration),
                )
            )

        return episodes


def compute_alert_burden_estimand(
    episodes: Sequence[AlertEpisode],
    lake_exposures_years: Mapping[str, float],
) -> Dict[str, Any]:
    """Compute alert burden lambda_alert = N_episodes / sum(Y_i)."""
    total_exposure = sum(lake_exposures_years.values())
    n_episodes = len(episodes)

    if total_exposure <= 0.0:
        return {
            "lambda_alert": None,
            "status": "NOT_ESTIMABLE",
            "reason": "zero_follow_up_exposure",
            "n_episodes": n_episodes,
            "total_lake_years": 0.0,
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


def compute_bounded_lead_time(
    event_id: str,
    lake_id: str,
    event_onset_date: str,
    decisions: Sequence[Tuple[str, float, bool]],  # (date_str, score, is_eligible)
    threshold: float,
    sustained_q: int = 2,
    warning_horizon_days: int = 180,
) -> LeadTimeResult:
    """Compute bounded lead time Delta t_lead in [0, 180d] prior to event onset.

    Strict Invariants:
    1. Only decisions with t_decision < t_onset are evaluated.
    2. Decisions within [t_onset - H_max, t_onset) are checked for sustained alarm.
    3. If alarm was never sustained, returns NOT_DETECTED with lead_time_days = None
       (never average as 0!).
    4. If no eligible decisions exist in horizon, returns NOT_ESTIMABLE.
    """
    dt_onset = date.fromisoformat(event_onset_date.split("T")[0])
    dt_horizon_start = dt_onset - timedelta(days=warning_horizon_days)

    # Filter pre-event eligible decisions strictly before onset
    pre_event_decisions = [
        (date.fromisoformat(d[0].split("T")[0]), d[1])
        for d in decisions
        if d[2] and date.fromisoformat(d[0].split("T")[0]) < dt_onset
    ]

    # Check decisions inside warning horizon
    horizon_decisions = [
        d for d in pre_event_decisions
        if d[0] >= dt_horizon_start
    ]
    horizon_decisions.sort(key=lambda x: x[0])

    if not horizon_decisions:
        return LeadTimeResult(
            event_id=event_id,
            lake_id=lake_id,
            status="NOT_ESTIMABLE",
            lead_time_days=None,
            onset_date=dt_onset.isoformat(),
            declaration_date=None,
            warning_horizon_days=warning_horizon_days,
        )

    consecutive = 0
    declaration_date: Optional[date] = None

    for d_date, score in horizon_decisions:
        if score >= threshold:
            consecutive += 1
            if consecutive == sustained_q:
                declaration_date = d_date
                break
        else:
            consecutive = 0

    if declaration_date is not None:
        lead_time = (dt_onset - declaration_date).days
        return LeadTimeResult(
            event_id=event_id,
            lake_id=lake_id,
            status="DETECTED",
            lead_time_days=lead_time,
            onset_date=dt_onset.isoformat(),
            declaration_date=declaration_date.isoformat(),
            warning_horizon_days=warning_horizon_days,
        )
    else:
        return LeadTimeResult(
            event_id=event_id,
            lake_id=lake_id,
            status="NOT_DETECTED",
            lead_time_days=None,  # Explicitly None: MUST NOT BE 0
            onset_date=dt_onset.isoformat(),
            declaration_date=None,
            warning_horizon_days=warning_horizon_days,
        )
