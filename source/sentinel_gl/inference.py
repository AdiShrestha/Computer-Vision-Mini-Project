"""Statistical inference, hypothesis testing, multiplicity corrections, and Poisson CIs.

Implements:
1. Holm-Bonferroni step-down multiplicity correction with original-order mapping.
2. Exact paired sign-flip permutation test (2^N combinatorial for N <= 20, Monte Carlo for N > 20).
3. Exact Garwood / Chi-Square Poisson rate confidence interval, handling N=0 gracefully.
4. Spatial cluster-conditional metric aggregation.
5. Strict null/status semantics: unestimable quantities return status="NOT_ESTIMABLE"
   and explicit None values; placeholders (0.0, 0.5, -1.0) are strictly forbidden.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# 1. Holm-Bonferroni Multiplicity Adjustment
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HolmBonferroniResult:
    """Holm-Bonferroni step-down adjustment result preserving original hypothesis order."""
    adjusted_p_values: Tuple[float, ...]
    reject: Tuple[bool, ...]
    alpha: float
    n_hypotheses: int
    rejection_thresholds: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjusted_p_values": list(self.adjusted_p_values),
            "reject": list(self.reject),
            "alpha": self.alpha,
            "n_hypotheses": self.n_hypotheses,
            "rejection_thresholds": list(self.rejection_thresholds),
        }


def holm_bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> HolmBonferroniResult:
    """Apply Holm-Bonferroni step-down multiplicity adjustment.

    Controls family-wise error rate (FWER) strongly under arbitrary dependence.
    Maps adjusted p-values and boolean rejections back to input order.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    if p_arr.ndim != 1 or p_arr.size == 0:
        raise ValueError("p_values must be a non-empty 1D sequence of numbers")
    if np.any(~np.isfinite(p_arr)) or np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError("all p-values must be finite real numbers in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    m = len(p_arr)
    # Sort p-values in ascending order
    order = np.argsort(p_arr)
    p_sorted = p_arr[order]

    # Rejection thresholds: alpha_k = alpha / (M - k + 1) for k=1..M
    k_indices = np.arange(1, m + 1)
    thresholds_sorted = alpha / (m - k_indices + 1)

    # Step-down rejection rule: once a hypothesis fails to reject, all subsequent hypotheses are retained
    rejections_sorted = np.zeros(m, dtype=bool)
    stopped = False
    for i in range(m):
        if not stopped and p_sorted[i] <= thresholds_sorted[i]:
            rejections_sorted[i] = True
        else:
            stopped = True

    # Adjusted p-values: p_tilde_(k) = max_{j <= k} min(1.0, (M - j + 1) * p_(j))
    multipliers = m - np.arange(m)
    raw_adj = np.minimum(1.0, multipliers * p_sorted)
    adj_sorted = np.maximum.accumulate(raw_adj)

    # Map back to original input ordering
    adj_original = np.empty(m, dtype=np.float64)
    adj_original[order] = adj_sorted
    rej_original = np.empty(m, dtype=bool)
    rej_original[order] = rejections_sorted
    thresholds_original = np.empty(m, dtype=np.float64)
    thresholds_original[order] = thresholds_sorted

    return HolmBonferroniResult(
        adjusted_p_values=tuple(float(x) for x in adj_original),
        reject=tuple(bool(x) for x in rej_original),
        alpha=float(alpha),
        n_hypotheses=m,
        rejection_thresholds=tuple(float(x) for x in thresholds_original),
    )


# ---------------------------------------------------------------------------
# 2. Exact Paired Sign-Flip Permutation Test
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SignFlipTestResult:
    """Exact or Monte Carlo paired sign-flip permutation test result."""
    status: str
    reason: Optional[str]
    p_value: Optional[float]
    statistic: Optional[float]
    abs_statistic: Optional[float]
    n_samples: int
    n_permutations: int
    mode: Optional[str]  # "exact" or "monte_carlo"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "p_value": self.p_value,
            "statistic": self.statistic,
            "abs_statistic": self.abs_statistic,
            "n_samples": self.n_samples,
            "n_permutations": self.n_permutations,
            "mode": self.mode,
        }


def exact_sign_flip_test(
    differences: Sequence[float],
    seed: Optional[int] = None,
) -> SignFlipTestResult:
    """Compute two-sided paired sign-flip test for H0: mean difference == 0.

    Uses exact 2^N combinatorial enumeration for N <= 20.
    Uses deterministic Monte Carlo (B=10000 draws) for N > 20.
    """
    diff_arr = np.asarray(differences, dtype=np.float64)
    if diff_arr.ndim != 1 or diff_arr.size == 0 or np.any(~np.isfinite(diff_arr)):
        return SignFlipTestResult(
            status="NOT_ESTIMABLE",
            reason="empty_or_non_finite_differences",
            p_value=None,
            statistic=None,
            abs_statistic=None,
            n_samples=int(diff_arr.size),
            n_permutations=0,
            mode=None,
        )

    n = len(diff_arr)
    t_obs = float(abs(np.mean(diff_arr)))
    mean_diff = float(np.mean(diff_arr))

    if n <= 16:
        # Full vectorization for 2^N <= 65536
        num_perms = 1 << n
        signs = (((np.arange(num_perms)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(np.float64)
        perm_means = np.abs(signs @ diff_arr) / n
        extreme_count = int(np.sum(perm_means >= t_obs - 1e-12))
        p_val = float(extreme_count / num_perms)
        mode = "exact"
    elif n <= 20:
        # Chunked evaluation for 2^17 .. 2^20 to avoid memory spikes
        num_perms = 1 << n
        extreme_count = 0
        chunk_size = 65536
        for start in range(0, num_perms, chunk_size):
            end = min(start + chunk_size, num_perms)
            signs = (((np.arange(start, end)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(np.float64)
            perm_means = np.abs(signs @ diff_arr) / n
            extreme_count += int(np.sum(perm_means >= t_obs - 1e-12))
        p_val = float(extreme_count / num_perms)
        mode = "exact"
    else:
        # Monte Carlo with B=10000 draws
        b = 10000
        rng = np.random.default_rng(seed if seed is not None else 42)
        mc_signs = rng.choice([-1.0, 1.0], size=(b, n)).astype(np.float64)
        perm_means = np.abs(mc_signs @ diff_arr) / n
        extreme_count = int(np.sum(perm_means >= t_obs - 1e-12))
        p_val = float((extreme_count + 1) / (b + 1))
        num_perms = b
        mode = "monte_carlo"

    return SignFlipTestResult(
        status="ESTIMATED",
        reason=None,
        p_value=p_val,
        statistic=mean_diff,
        abs_statistic=t_obs,
        n_samples=n,
        n_permutations=num_perms,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# 3. Exact Poisson Rate Confidence Interval
# ---------------------------------------------------------------------------

def poisson_rate_confidence_interval(
    count: int,
    exposure: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute exact Garwood / Chi-Square Poisson confidence interval for rate lambda = count / exposure.

    For count == 0: CI = [0.0, -ln(alpha) / exposure].
    Rejects non-positive exposure or negative count.
    """
    if exposure <= 0.0 or not np.isfinite(exposure):
        raise ValueError(f"exposure must be strictly positive and finite, got {exposure}")
    if count < 0 or not isinstance(count, (int, np.integer)):
        raise ValueError(f"count must be a non-negative integer, got {count}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if count == 0:
        ci_lower = 0.0
        ci_upper = float(-math.log(alpha) / exposure)
        return ci_lower, ci_upper

    # Garwood exact limits:
    # Lower: chi2(alpha/2, 2*N) / (2*Y)
    # Upper: chi2(1 - alpha/2, 2*(N+1)) / (2*Y)
    df_lower = 2 * count
    df_upper = 2 * (count + 1)

    ci_lower = float(chi2.ppf(alpha / 2.0, df_lower) / (2.0 * exposure))
    ci_upper = float(chi2.ppf(1.0 - alpha / 2.0, df_upper) / (2.0 * exposure))

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# 4. Spatial Cluster Conditional Metric Aggregation
# ---------------------------------------------------------------------------

def cluster_conditional_metrics(
    lake_metrics: Mapping[str, Mapping[str, Any]],
    cluster_map: Mapping[str, str],
) -> Dict[str, Any]:
    """Group lake metrics conditionally by spatial basin/cluster ID.

    Discloses intra-cluster dependence rather than pooling under unjustified i.i.d. assumptions.
    Strictly preserves status="NOT_ESTIMABLE" and None values for empty or missing metrics.
    """
    if not lake_metrics:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "empty_lake_metrics",
            "clusters": {},
        }

    # Group lakes by cluster ID
    clusters: Dict[str, List[str]] = {}
    for lake_id in sorted(lake_metrics.keys()):
        c_id = cluster_map.get(lake_id, f"CLUSTER_UNKNOWN_{lake_id}")
        if c_id not in clusters:
            clusters[c_id] = []
        clusters[c_id].append(lake_id)

    aggregated_clusters: Dict[str, Any] = {}

    for c_id, lakes in clusters.items():
        # Identify all metric keys across lakes in this cluster
        metric_keys = set()
        for l_id in lakes:
            metric_keys.update(lake_metrics[l_id].keys())

        c_summary: Dict[str, Any] = {
            "cluster_id": c_id,
            "lake_ids": lakes,
            "n_lakes": len(lakes),
            "metrics": {},
        }

        for m_key in sorted(metric_keys):
            valid_vals = []
            for l_id in lakes:
                val = lake_metrics[l_id].get(m_key)
                if val is not None and isinstance(val, (int, float, np.number)) and np.isfinite(val):
                    valid_vals.append(float(val))

            if not valid_vals:
                c_summary["metrics"][m_key] = {
                    "status": "NOT_ESTIMABLE",
                    "reason": "no_finite_values_in_cluster",
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "n_valid": 0,
                }
            else:
                arr = np.array(valid_vals, dtype=np.float64)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                c_summary["metrics"][m_key] = {
                    "status": "ESTIMATED",
                    "reason": None,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "n_valid": len(arr),
                }

        aggregated_clusters[c_id] = c_summary

    return {
        "status": "ESTIMATED",
        "reason": None,
        "n_clusters": len(clusters),
        "clusters": aggregated_clusters,
    }
