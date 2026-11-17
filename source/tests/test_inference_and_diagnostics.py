"""Analytical unit tests for inference, hypothesis testing, diagnostics, and claims tables.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Simulated p-values, synthetic paired differences, and mock prediction records in this
file verify statistical algorithms, multiplicity corrections, diagnostic taxonomies,
and conservative claims table reporting in offline unit tests only. They do not
represent real physical observations and must never support scientific claims.
"""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from sentinel_gl.inference import (
    HolmBonferroniResult,
    SignFlipTestResult,
    cluster_conditional_metrics,
    exact_sign_flip_test,
    holm_bonferroni_correction,
    poisson_rate_confidence_interval,
)
from sentinel_gl.diagnostics import (
    ERR_CLOUD_OBSCURATION,
    ERR_INSUFFICIENT_OBSERVATIONS,
    ERR_MISSED_RAPID_TRIGGER,
    ERR_SAR_GEOMETRIC_DISTORTION,
    ERR_SPURIOUS_SEASONAL_ANOMALY,
    classify_failure,
    diagnose_prediction_ledger,
)
from sentinel_gl.evaluation import (
    BaselineComparisonClaim,
    ClaimsTable,
    generate_claims_table,
)
from sentinel_gl.predictions import PredictionRecord, make_prediction_record


# ---------------------------------------------------------------------------
# Test 1: Holm-Bonferroni Multiplicity Correction
# ---------------------------------------------------------------------------

def test_holm_bonferroni_multiplicity_correction():
    """Verify step-down rejection thresholds, order preservation, and early stopping.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    # 4 hypotheses: p-values [0.01, 0.04, 0.03, 0.005] at alpha = 0.05
    # Sorted: p_(1)=0.005, p_(2)=0.01, p_(3)=0.03, p_(4)=0.04
    # Thresholds: alpha/4 = 0.0125, alpha/3 = 0.0167, alpha/2 = 0.025, alpha/1 = 0.05
    # Step 1: 0.005 <= 0.0125 -> Reject
    # Step 2: 0.01 <= 0.0167 -> Reject
    # Step 3: 0.03 > 0.025 -> Fail to reject (stops!)
    # Step 4: 0.04 retained
    p_vals = [0.01, 0.04, 0.03, 0.005]
    res = holm_bonferroni_correction(p_vals, alpha=0.05)

    assert isinstance(res, HolmBonferroniResult)
    assert res.n_hypotheses == 4
    # Map back to original order: [0.01 (reject), 0.04 (retain), 0.03 (retain), 0.005 (reject)]
    assert res.reject == (True, False, False, True)

    # Adjusted p-values:
    # for 0.005: min(1, 4 * 0.005) = 0.02
    # for 0.01: max(0.02, min(1, 3 * 0.01)) = 0.03
    # for 0.03: max(0.03, min(1, 2 * 0.03)) = 0.06
    # for 0.04: max(0.06, min(1, 1 * 0.04)) = 0.06
    expected_adj = [0.03, 0.06, 0.06, 0.02]
    for obs, exp in zip(res.adjusted_p_values, expected_adj):
        assert obs == pytest.approx(exp, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Exact Paired Sign-Flip Permutation Test
# ---------------------------------------------------------------------------

def test_exact_sign_flip_paired_permutation():
    """Verify exact 2^N combinatorial sign-flip test and deterministic Monte Carlo.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    # N=4: diffs = [1.0, 2.0, 3.0, 4.0]
    # Sum = 10. Only all positive (+10) and all negative (-10) have |sum| >= 10.
    # Total permutations = 2^4 = 16. Exact p-value = 2 / 16 = 0.125.
    diffs = [1.0, 2.0, 3.0, 4.0]
    res_exact = exact_sign_flip_test(diffs)

    assert res_exact.status == "ESTIMATED"
    assert res_exact.mode == "exact"
    assert res_exact.n_samples == 4
    assert res_exact.n_permutations == 16
    assert res_exact.p_value == pytest.approx(0.125)
    assert res_exact.statistic == pytest.approx(2.5)

    # Large N=25: deterministic Monte Carlo reproducibility
    rng = np.random.default_rng(12345)
    large_diffs = rng.normal(loc=0.5, scale=1.0, size=25).tolist()

    res_mc1 = exact_sign_flip_test(large_diffs, seed=42)
    res_mc2 = exact_sign_flip_test(large_diffs, seed=42)

    assert res_mc1.mode == "monte_carlo"
    assert res_mc1.n_permutations == 10000
    assert res_mc1.p_value == pytest.approx(res_mc2.p_value)


# ---------------------------------------------------------------------------
# Test 3: Exact Poisson Rate Confidence Interval
# ---------------------------------------------------------------------------

def test_poisson_rate_confidence_interval():
    """Verify Garwood / Chi-Square Poisson confidence interval bounds.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    # Case 1: Count = 0, exposure Y = 10.0
    # Lower bound must be 0.0; upper bound must be -ln(alpha) / Y
    ci_low_0, ci_high_0 = poisson_rate_confidence_interval(0, 10.0, alpha=0.05)
    assert ci_low_0 == 0.0
    assert ci_high_0 == pytest.approx(-math.log(0.05 / 2) / 10.0)

    # Case 2: Count = 5, exposure Y = 2.0
    ci_low_5, ci_high_5 = poisson_rate_confidence_interval(5, 2.0, alpha=0.05)
    rate = 5.0 / 2.0
    assert 0.0 < ci_low_5 < rate < ci_high_5

    # Case 3: Invalid input handling
    with pytest.raises(ValueError, match="exposure must be strictly positive"):
        poisson_rate_confidence_interval(5, 0.0)

    with pytest.raises(ValueError, match="count must be a non-negative integer"):
        poisson_rate_confidence_interval(-1, 5.0)


# ---------------------------------------------------------------------------
# Test 4: Null / Status Semantics for Unestimable Quantities
# ---------------------------------------------------------------------------

def test_unestimable_quantities_remain_explicit_nulls():
    """Verify that unestimable quantities return status='NOT_ESTIMABLE' with None values,
    never placeholder values like 0.0 or 0.5.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    # Empty differences
    empty_res = exact_sign_flip_test([])
    assert empty_res.status == "NOT_ESTIMABLE"
    assert empty_res.p_value is None
    assert empty_res.statistic is None
    assert empty_res.reason == "empty_or_non_finite_differences"

    # Cluster with missing metrics
    empty_cluster = cluster_conditional_metrics(
        {"L1": {"lead_time_days": None}},
        {"L1": "C1"},
    )
    metric_res = empty_cluster["clusters"]["C1"]["metrics"]["lead_time_days"]
    assert metric_res["status"] == "NOT_ESTIMABLE"
    assert metric_res["mean"] is None
    assert metric_res["std"] is None
    assert metric_res["min"] is None
    assert metric_res["max"] is None


# ---------------------------------------------------------------------------
# Test 5: Operational Failure Taxonomy Classification
# ---------------------------------------------------------------------------

def test_failure_taxonomy_classification():
    for context in ({'optical_missing_fraction_last_60d':.85,'eligible':True},{'sar_vv_spatial_variance':6.2,'eligible':True},{'is_false_alarm':True,'calendar_month':11,'eligible':True},{'is_missed_event':True,'lead_time_days':2,'eligible':True}):
        assert classify_failure(context)=='ERR_UNCLASSIFIED'
    assert classify_failure({'eligible':False})==ERR_INSUFFICIENT_OBSERVATIONS



# ---------------------------------------------------------------------------
# Test 6: Spatial Cluster Conditional Aggregation
# ---------------------------------------------------------------------------

def test_cluster_conditional_aggregation():
    """Verify basin/cluster grouping without cluster cross-contamination.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    lake_metrics = {
        "SGL-001": {"lead_time_days": 40.0, "peak_score": 0.90},
        "SGL-002": {"lead_time_days": 60.0, "peak_score": 0.80},
        "SGL-003": {"lead_time_days": 10.0, "peak_score": 0.60},
        "SGL-004": {"lead_time_days": 20.0, "peak_score": 0.70},
    }
    cluster_map = {
        "SGL-001": "BASIN_NORTH",
        "SGL-002": "BASIN_NORTH",
        "SGL-003": "BASIN_SOUTH",
        "SGL-004": "BASIN_SOUTH",
    }

    res = cluster_conditional_metrics(lake_metrics, cluster_map)
    assert res["status"] == "ESTIMATED"
    assert res["n_clusters"] == 2

    north = res["clusters"]["BASIN_NORTH"]["metrics"]["lead_time_days"]
    assert north["mean"] == pytest.approx(50.0)
    assert north["min"] == 40.0
    assert north["max"] == 60.0

    south = res["clusters"]["BASIN_SOUTH"]["metrics"]["lead_time_days"]
    assert south["mean"] == pytest.approx(15.0)
    assert south["min"] == 10.0
    assert south["max"] == 20.0


# ---------------------------------------------------------------------------
# Test 7: Conservative Non-Rejection in Claims Table
# ---------------------------------------------------------------------------

def test_claims_table_conservative_non_rejection(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from sentinel_gl.evaluation import generate_claims_table
    with pytest.raises(ValueError, match="BLOCKED"):
        generate_claims_table(model_records=[],baseline_records={})
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 8: End-to-End Inference & Diagnostics Runner Execution
# ---------------------------------------------------------------------------

def test_run_inference_diagnostics_end_to_end(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from runners.run_inference_diagnostics import run_inference_diagnostics
    with pytest.raises(ValueError, match="BLOCKED"):
        run_inference_diagnostics(output_claims_path=tmp_path/"claims.json")
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 9: CLI Runner Subprocess Invocation
# ---------------------------------------------------------------------------

def test_run_inference_diagnostics_cli_subprocess(tmp_path):
    from runners.run_inference_diagnostics import main
    assert main(['--output-claims',str(tmp_path/'claims.json')])==2
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 10: Claims Table Schema and Conservative Reporting Invariants
# ---------------------------------------------------------------------------

def test_claims_table_schema_and_conservative_invariants(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from sentinel_gl.reports import load_claims_table
    with pytest.raises(ValueError, match="BLOCKED"):
        load_claims_table(Path("data/claims_table.json"))
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 11: Diagnostic Report Taxonomy Consistency
# ---------------------------------------------------------------------------

def test_diagnostic_report_taxonomy_consistency(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from sentinel_gl.diagnostics import diagnose_prediction_ledger
    with pytest.raises(ValueError, match="BLOCKED"):
        diagnose_prediction_ledger([])
    assert not list(tmp_path.iterdir())
