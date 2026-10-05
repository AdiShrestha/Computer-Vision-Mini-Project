"""Conservative claims table and independent negative-result evaluation engine.

Recomputes prospective case detection, alert burden (with exact Poisson 95% CI),
and paired contrasts against the four fair baselines under Holm-Bonferroni
multiplicity correction.

Explicitly marks non-significant findings as FAIL_TO_REJECT / INCONCLUSIVE.
Never asserts model superiority without statistically significant empirical evidence.
"""
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


def generate_claims_table(
    model_records: Sequence[PredictionRecord],
    baseline_records: Mapping[str, Sequence[PredictionRecord]],
    event_registry: Optional[Mapping[str, str]] = None,  # lake_id -> onset_date
    cluster_map: Optional[Mapping[str, str]] = None,      # lake_id -> cluster_id
    threshold: float = 0.70,
    alpha: float = 0.05,
    stride_days: int = 30,
) -> ClaimsTable:
    """Recompute headline metrics from raw prediction records through an independent execution path.

    Evaluates:
    1. Prospective case detection & bounded lead times on historical events.
    2. Operational alert burden (lambda_alert) with exact Garwood Poisson 95% CI.
    3. Paired score differences (Delta S) vs all baselines with Holm-Bonferroni correction.
    4. Conservative claims reporting: non-significant results are marked FAIL_TO_REJECT.
    """
    events_map = event_registry or {}
    model_by_lake: Dict[str, List[PredictionRecord]] = {}
    for r in model_records:
        if r.lake_id not in model_by_lake:
            model_by_lake[r.lake_id] = []
        model_by_lake[r.lake_id].append(r)

    # 1. Prospective Case Detection & Lead Times
    case_detections: List[Dict[str, Any]] = []
    for l_id, onset_date in sorted(events_map.items()):
        lake_recs = model_by_lake.get(l_id, [])
        decisions = [(r.decision_date, r.score, r.eligible) for r in lake_recs]
        lead_res = compute_bounded_lead_time(
            event_id=f"EV-{l_id}",
            lake_id=l_id,
            event_onset_date=onset_date,
            decisions=decisions,
            threshold=threshold,
            sustained_q=2,
            warning_horizon_days=180,
        )
        case_detections.append(lead_res.to_dict())

    # 2. Alert Burden (lambda_alert) & Exact Poisson 95% CI
    control_lakes = [l_id for l_id in sorted(model_by_lake.keys()) if l_id not in events_map]
    engine = AlertEpisodeEngine(threshold=threshold, sustained_q=2, hysteresis_r=2, refractory_days=60)

    all_control_episodes: List[AlertEpisode] = []
    lake_exposures: Dict[str, float] = {}

    for l_id in control_lakes:
        lake_recs = model_by_lake[l_id]
        decisions = [(r.decision_date, r.score, r.eligible) for r in lake_recs]
        episodes = engine.extract_episodes(l_id, decisions)
        exp_y = calculate_lake_exposure_years([d[0] for d in decisions if d[2]], stride_days=stride_days)
        lake_exposures[l_id] = exp_y
        all_control_episodes.extend(episodes)

    burden_res = compute_alert_burden_estimand(all_control_episodes, lake_exposures)
    total_exposure = burden_res["total_lake_years"]
    n_episodes = burden_res["n_episodes"]

    if burden_res["status"] == "ESTIMATED" and total_exposure > 0.0:
        ci_lower, ci_upper = poisson_rate_confidence_interval(n_episodes, total_exposure, alpha=alpha)
        alert_burden_payload = {
            "status": "ESTIMATED",
            "reason": None,
            "lambda_alert": burden_res["lambda_alert"],
            "poisson_ci_lower": ci_lower,
            "poisson_ci_upper": ci_upper,
            "confidence_level": 1.0 - alpha,
            "n_episodes": n_episodes,
            "total_lake_years": total_exposure,
        }
    else:
        alert_burden_payload = {
            "status": "NOT_ESTIMABLE",
            "reason": burden_res.get("reason", "zero_exposure"),
            "lambda_alert": None,
            "poisson_ci_lower": None,
            "poisson_ci_upper": None,
            "confidence_level": 1.0 - alpha,
            "n_episodes": n_episodes,
            "total_lake_years": total_exposure,
        }

    # 3. Paired Differences & Exact Tests vs Baselines
    baseline_names = sorted(baseline_records.keys())
    unadj_p_values: List[float] = []
    comparisons_pre: List[Dict[str, Any]] = []

    for b_id in baseline_names:
        b_recs = baseline_records[b_id]
        try:
            paired_diffs = compute_paired_score_differences(model_records, b_recs)
            diff_vals = [p["delta_s"] for p in paired_diffs]
            sign_res = exact_sign_flip_test(diff_vals)
            p_val = sign_res.p_value if sign_res.status == "ESTIMATED" else None
            stat = sign_res.statistic if sign_res.status == "ESTIMATED" else None
            abs_stat = sign_res.abs_statistic if sign_res.status == "ESTIMATED" else None
            n_paired = len(diff_vals)
        except ValueError:
            p_val = None
            stat = None
            abs_stat = None
            n_paired = 0

        comparisons_pre.append({
            "baseline_id": b_id,
            "n_paired_windows": n_paired,
            "mean_delta_s": stat,
            "abs_statistic": abs_stat,
            "unadjusted_p_value": p_val,
        })
        if p_val is not None:
            unadj_p_values.append(p_val)

    # 4. Multiplicity Correction across Baselines
    baseline_comparisons: List[BaselineComparisonClaim] = []

    if len(unadj_p_values) == len(comparisons_pre) and len(unadj_p_values) > 0:
        hb_res = holm_bonferroni_correction(unadj_p_values, alpha=alpha)
        for idx, item in enumerate(comparisons_pre):
            adj_p = hb_res.adjusted_p_values[idx]
            reject = hb_res.reject[idx]
            mean_diff = item["mean_delta_s"]

            if reject and mean_diff is not None and mean_diff > 0:
                claim_finding = "REJECT_NULL_SUPERIOR"
            elif reject and mean_diff is not None and mean_diff < 0:
                claim_finding = "REJECT_NULL_INFERIOR"
            else:
                # Conservative non-rejection: fails Holm-Bonferroni correction
                claim_finding = "FAIL_TO_REJECT"

            baseline_comparisons.append(
                BaselineComparisonClaim(
                    baseline_id=item["baseline_id"],
                    n_paired_windows=item["n_paired_windows"],
                    mean_delta_s=item["mean_delta_s"],
                    abs_statistic=item["abs_statistic"],
                    unadjusted_p_value=item["unadjusted_p_value"],
                    adjusted_p_value=adj_p,
                    claim_finding=claim_finding,
                )
            )
    else:
        for item in comparisons_pre:
            baseline_comparisons.append(
                BaselineComparisonClaim(
                    baseline_id=item["baseline_id"],
                    n_paired_windows=item["n_paired_windows"],
                    mean_delta_s=item["mean_delta_s"],
                    abs_statistic=item["abs_statistic"],
                    unadjusted_p_value=item["unadjusted_p_value"],
                    adjusted_p_value=None,
                    claim_finding="NOT_ESTIMABLE" if item["unadjusted_p_value"] is None else "FAIL_TO_REJECT",
                )
            )

    # 5. Spatial Cluster Conditional Aggregation (if provided)
    cluster_agg = None
    if cluster_map:
        lake_metric_map: Dict[str, Dict[str, Any]] = {}
        for l_id, recs in model_by_lake.items():
            valid_scores = [r.score for r in recs if r.eligible]
            lake_metric_map[l_id] = {
                "mean_score": float(np.mean(valid_scores)) if valid_scores else None,
                "n_windows": len(valid_scores),
            }
        cluster_agg = cluster_conditional_metrics(lake_metric_map, cluster_map)

    # Cryptographic report hash
    hash_payload = {
        "case_detections": case_detections,
        "alert_burden": alert_burden_payload,
        "baseline_comparisons": [b.to_dict() for b in baseline_comparisons],
        "alpha": alpha,
    }
    report_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()

    return ClaimsTable(
        case_detections=case_detections,
        alert_burden=alert_burden_payload,
        baseline_comparisons=baseline_comparisons,
        cluster_aggregation=cluster_agg,
        alpha=alpha,
        status="COMPLETE",
        report_hash=report_hash,
    )
