"""Research claim generation is disabled pending admitted measurements and task inference."""
from __future__ import annotations
import dataclasses
from datetime import date
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .episodes import (
    AlertEpisode,
    AlertEpisodeEngine,
    LeadTimeResult,
    calculate_lake_exposure_years,
    compute_alert_burden_estimand,
    compute_bounded_lead_time,
)
from .inference import (
    HolmBonferroniResult,
    SignFlipTestResult,
    cluster_conditional_metrics,
    exact_sign_flip_test,
    holm_bonferroni_correction,
    poisson_rate_confidence_interval,
)
from .predictions import PredictionRecord, compute_paired_score_differences


# ---------------------------------------------------------------------------
# Claims Table Data Structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BaselineComparisonClaim:
    """Statistical claim comparing model against a specific baseline."""
    baseline_id: str
    n_paired_windows: int
    mean_delta_s: Optional[float]
    abs_statistic: Optional[float]
    unadjusted_p_value: Optional[float]
    adjusted_p_value: Optional[float]
    claim_finding: str  # "FAIL_TO_REJECT", "REJECT_NULL_SUPERIOR", "REJECT_NULL_INFERIOR", "NOT_ESTIMABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "n_paired_windows": self.n_paired_windows,
            "mean_delta_s": self.mean_delta_s,
            "abs_statistic": self.abs_statistic,
            "unadjusted_p_value": self.unadjusted_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "claim_finding": self.claim_finding,
        }


@dataclasses.dataclass(frozen=True)
class ClaimsTable:
    """Comprehensive claims table embodying conservative evaluation standards."""
    case_detections: List[Dict[str, Any]]
    alert_burden: Dict[str, Any]
    baseline_comparisons: List[BaselineComparisonClaim]
    cluster_aggregation: Optional[Dict[str, Any]]
    alpha: float
    status: str
    report_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_detections": list(self.case_detections),
            "alert_burden": dict(self.alert_burden),
            "baseline_comparisons": [b.to_dict() for b in self.baseline_comparisons],
            "cluster_aggregation": self.cluster_aggregation,
            "alpha": self.alpha,
            "status": self.status,
            "report_hash": self.report_hash,
        }


def generate_claims_table(*args, **kwargs):
    from .admission import blocked
    blocked("generate_claims_table")
