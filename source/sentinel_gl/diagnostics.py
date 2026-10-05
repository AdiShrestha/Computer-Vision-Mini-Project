"""Operational failure taxonomy engine for Sentinel-GL predictive diagnostics.

Classifies false alarms, missed events, and unestimable decision windows into
formal failure taxonomy categories:
1. ERR_CLOUD_OBSCURATION: >= 80% missing optical observations in preceding 60 days.
2. ERR_SAR_GEOMETRIC_DISTORTION: radar backscatter anomaly coinciding with extreme spatial variance (> 5.0).
3. ERR_SPURIOUS_SEASONAL_ANOMALY: false alarms during freeze-up/breakup transition months (Nov, Dec, Apr, May) or ~0°C.
4. ERR_MISSED_RAPID_TRIGGER: event had zero precursor signal within satellite revisit (< 6 days).
5. ERR_INSUFFICIENT_OBSERVATIONS: window fails C_obs eligibility (< 2 observations per active modality).
"""
from __future__ import annotations
import dataclasses
from datetime import date
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .features import MultiModalPanel
from .predictions import PredictionRecord


# ---------------------------------------------------------------------------
# Taxonomy Categories
# ---------------------------------------------------------------------------

ERR_CLOUD_OBSCURATION: str = "ERR_CLOUD_OBSCURATION"
ERR_SAR_GEOMETRIC_DISTORTION: str = "ERR_SAR_GEOMETRIC_DISTORTION"
ERR_SPURIOUS_SEASONAL_ANOMALY: str = "ERR_SPURIOUS_SEASONAL_ANOMALY"
ERR_MISSED_RAPID_TRIGGER: str = "ERR_MISSED_RAPID_TRIGGER"
ERR_INSUFFICIENT_OBSERVATIONS: str = "ERR_INSUFFICIENT_OBSERVATIONS"
ERR_UNCLASSIFIED: str = "ERR_UNCLASSIFIED"

ALL_ERROR_CATEGORIES: Tuple[str, ...] = (
    ERR_CLOUD_OBSCURATION,
    ERR_SAR_GEOMETRIC_DISTORTION,
    ERR_SPURIOUS_SEASONAL_ANOMALY,
    ERR_MISSED_RAPID_TRIGGER,
    ERR_INSUFFICIENT_OBSERVATIONS,
    ERR_UNCLASSIFIED,
)


@dataclasses.dataclass(frozen=True)
class DiagnosticReport:
    """Immutable report summarizing operational diagnostics and failure taxonomy rates."""
    total_evaluations: int
    total_failures: int
    failure_counts: Dict[str, int]
    failure_rates: Dict[str, float]
    cases_by_category: Dict[str, List[str]]
    report_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "total_failures": self.total_failures,
            "failure_counts": dict(self.failure_counts),
            "failure_rates": dict(self.failure_rates),
            "cases_by_category": {k: list(v) for k, v in self.cases_by_category.items()},
            "report_hash": self.report_hash,
        }


def classify_failure(context: Mapping[str, Any]) -> str:
    """Classify a single failure event into the formal failure taxonomy based on context flags."""
    # 1. Ineligible / Insufficient observations
    if not context.get("eligible", True) or context.get("is_eligible") is False:
        return ERR_INSUFFICIENT_OBSERVATIONS
    if context.get("obs_count") is not None and int(context["obs_count"]) < 2:
        return ERR_INSUFFICIENT_OBSERVATIONS

    # 2. Optical Cloud Obscuration (>= 80% missing optical in preceding 60 days)
    opt_missing_frac = context.get("optical_missing_fraction_last_60d")
    if opt_missing_frac is not None and float(opt_missing_frac) >= 0.80:
        return ERR_CLOUD_OBSCURATION
    cloud_pct = context.get("cloud_obscuration_pct")
    if cloud_pct is not None and float(cloud_pct) >= 80.0:
        return ERR_CLOUD_OBSCURATION

    # 3. SAR Geometric / Layover Distortion
    sar_var = context.get("sar_vv_spatial_variance")
    if sar_var is not None and float(sar_var) > 5.0:
        return ERR_SAR_GEOMETRIC_DISTORTION
    if context.get("sar_distortion_flag") is True:
        return ERR_SAR_GEOMETRIC_DISTORTION

    # 4. Spurious Seasonal Transition Anomaly (Freeze-up / Breakup)
    if context.get("is_false_alarm") is True or context.get("failure_type") == "false_alarm":
        # Check transition months: Nov, Dec, Apr, May
        cal_month = context.get("calendar_month")
        if cal_month in (4, 5, 11, 12) or context.get("is_transition_season") is True:
            return ERR_SPURIOUS_SEASONAL_ANOMALY
        temp_c = context.get("era5_temp_2m_c")
        if temp_c is not None and -2.0 <= float(temp_c) <= 2.0:
            return ERR_SPURIOUS_SEASONAL_ANOMALY

    # 5. Missed Rapid Trigger (< 6 days before event onset)
    if context.get("is_missed_event") is True or context.get("failure_type") == "missed_event":
        lead_time = context.get("lead_time_days")
        if lead_time is not None and int(lead_time) < 6:
            return ERR_MISSED_RAPID_TRIGGER
        days_before = context.get("precursor_days_before_onset")
        if days_before is not None and int(days_before) < 6:
            return ERR_MISSED_RAPID_TRIGGER
        if context.get("zero_precursor_signal") is True:
            return ERR_MISSED_RAPID_TRIGGER

    return ERR_UNCLASSIFIED


def diagnose_panel_window(
    record: PredictionRecord,
    panel: Optional[MultiModalPanel] = None,
    event_onset: Optional[str] = None,
    threshold: float = 0.70,
) -> str:
    """Diagnose a decision window using joined panel observations and event metadata."""
    if not record.eligible:
        return ERR_INSUFFICIENT_OBSERVATIONS

    if panel is not None:
        # Check optical missingness in the last 60 days of the 180-day window
        # Rows [120, 180), channels 0..3 (optical)
        opt_mask_60d = panel.mask[120:180, 0:4]
        valid_opt_days = np.sum(np.any(opt_mask_60d, axis=1))
        missing_opt_frac = 1.0 - (float(valid_opt_days) / 60.0)
        if missing_opt_frac >= 0.80:
            return ERR_CLOUD_OBSCURATION

        # Check SAR spatial variance (channel 7)
        sar_var_vals = panel.values[:, 7][panel.mask[:, 7] & np.isfinite(panel.values[:, 7])]
        if sar_var_vals.size > 0 and np.max(sar_var_vals) > 5.0:
            return ERR_SAR_GEOMETRIC_DISTORTION

        # Check false alarm seasonal anomaly (freeze/thaw around 0°C or transition months)
        if record.score >= threshold:
            dt_decision = date.fromisoformat(record.decision_date.split("T")[0])
            if dt_decision.month in (4, 5, 11, 12):
                return ERR_SPURIOUS_SEASONAL_ANOMALY
            # Check temperature on decision day (channel 8)
            temp_val = panel.values[-1, 8]
            if np.isfinite(temp_val) and -2.0 <= float(temp_val) <= 2.0:
                return ERR_SPURIOUS_SEASONAL_ANOMALY

        # Check missed rapid trigger if this is an event lake
        if event_onset is not None and record.score < threshold:
            dt_onset = date.fromisoformat(event_onset.split("T")[0])
            dt_decision = date.fromisoformat(record.decision_date.split("T")[0])
            days_to_onset = (dt_onset - dt_decision).days
            if 0 <= days_to_onset < 6:
                return ERR_MISSED_RAPID_TRIGGER

    return ERR_UNCLASSIFIED


def diagnose_prediction_ledger(
    records: Sequence[PredictionRecord],
    panels: Optional[Mapping[str, MultiModalPanel]] = None,
    events: Optional[Mapping[str, str]] = None,  # lake_id -> onset_date
    threshold: float = 0.70,
) -> DiagnosticReport:
    """Audit prediction ledger and categorize failures into operational diagnostic categories."""
    panels_map = panels or {}
    events_map = events or {}

    failure_counts: Dict[str, int] = {cat: 0 for cat in ALL_ERROR_CATEGORIES}
    cases_by_cat: Dict[str, List[str]] = {cat: [] for cat in ALL_ERROR_CATEGORIES}
    total_evals = len(records)
    total_failures = 0

    for r in records:
        panel = panels_map.get(f"{r.lake_id}_{r.decision_date}") or panels_map.get(r.lake_id)
        onset = events_map.get(r.lake_id)

        is_event_lake = r.lake_id in events_map
        is_false_alarm = (not is_event_lake) and (r.score >= threshold)
        is_missed = is_event_lake and (r.score < threshold)
        is_ineligible = not r.eligible

        if is_false_alarm or is_missed or is_ineligible:
            total_failures += 1
            cat = diagnose_panel_window(r, panel, onset, threshold)
            failure_counts[cat] += 1
            cases_by_cat[cat].append(r.sample_id)

    # Compute failure rates
    failure_rates: Dict[str, float] = {}
    for cat in ALL_ERROR_CATEGORIES:
        rate = float(failure_counts[cat] / total_failures) if total_failures > 0 else 0.0
        failure_rates[cat] = rate

    # Deterministic cryptographic report hash
    hash_payload = {
        "total_evaluations": total_evals,
        "total_failures": total_failures,
        "failure_counts": failure_counts,
        "cases_digest": hashlib.sha256(json.dumps(cases_by_cat, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    report_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

    return DiagnosticReport(
        total_evaluations=total_evals,
        total_failures=total_failures,
        failure_counts=failure_counts,
        failure_rates=failure_rates,
        cases_by_category=cases_by_cat,
        report_hash=report_hash,
    )
