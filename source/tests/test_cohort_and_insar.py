"""Analytical unit tests for regional cohort expansion, spatial clustering, and InSAR feasibility.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Simulated radar geometry parameters, mock split allocations, synthetic terrain slopes,
and statistical sample size calculations in this file verify schema invariants,
haversine clustering, radar layover/shadow physics, coherence decay, storage budgeting,
and power floors in offline unit tests only. They do not represent real physical observations
and must never support scientific claims.
"""
from __future__ import annotations
import math
from pathlib import Path
import pytest

from sentinel_gl.cohort import (
    audit_cohort_integrity,
    cluster_lakes_spatially,
    get_expanded_event_registry,
    get_expanded_lake_registry,
    haversine_distance_km,
)
from sentinel_gl.insar import (
    check_insar_storage_budget,
    classify_geometric_distortion,
    generate_insar_feasibility_dossier,
    model_insar_coherence,
    MAX_LOCAL_STORAGE_BYTES,
    PHASE_UNWRAPPING_COHERENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Test 1: Expanded Lake Registry Schema and Invariants
# ---------------------------------------------------------------------------

def test_expanded_lake_registry_schema_and_invariants():
    """Verify that all expanded lakes satisfy schema, physical bounds, and minimum area."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    lakes = get_expanded_lake_registry()
    events = get_expanded_event_registry()

    assert len(lakes) == 8, "Expected exactly 8 curated candidate lakes"
    assert len(events) == 4, "Expected exactly 4 curated historical events"

    required_lake_keys = {
        "lake_id", "rgi_id", "glims_id", "name", "centroid_lat", "centroid_lon",
        "elevation_m", "dam_type", "area_km2_2020", "basin", "country", "cluster_id", "cohort_role",
    }
    valid_dam_types = {"moraine_dammed", "ice_dammed", "landslide_dammed"}

    for lk in lakes:
        assert required_lake_keys.issubset(lk.keys())
        assert lk["area_km2_2020"] >= 0.05, f"Lake {lk['lake_id']} area must be >= 0.05 km2"
        assert lk["elevation_m"] > 0, f"Lake {lk['elevation_m']} elevation must be positive"
        assert lk["dam_type"] in valid_dam_types
        assert -90.0 <= lk["centroid_lat"] <= 90.0
        assert -180.0 <= lk["centroid_lon"] <= 180.0
        assert lk["cohort_role"] in ("event", "control")

    # Run cohort integrity audit
    audit = audit_cohort_integrity(lakes, events)
    assert audit["status"] == "PASS"
    assert len(audit["errors"]) == 0
    assert audit["selection_bias"]["n_lakes"] == 8
    assert audit["selection_bias"]["n_events"] == 4


# ---------------------------------------------------------------------------
# Test 2: Spatial Clustering Enforces Isolation Across Splits
# ---------------------------------------------------------------------------

def test_spatial_clustering_enforces_isolation():
    """Verify that lakes within 50 km share a cluster and split boundaries never leak clusters."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    lakes = get_expanded_lake_registry()
    events = get_expanded_event_registry()

    # Dynamic spatial clustering with 50 km buffer
    computed_clusters = cluster_lakes_spatially(lakes, buffer_km=50.0)

    # Shishper and Passu in Hunza are within 20 km; must share cluster
    dist_hunza = haversine_distance_km(36.355, 74.721, 36.467, 74.882)
    assert dist_hunza < 50.0
    assert computed_clusters["SGL-003"] == computed_clusters["SGL-006"]

    # Verify that valid split allocation has zero leakage
    valid_splits = {
        "train": ["SGL-003", "SGL-006"],
        "val": ["SGL-004", "SGL-007"],
        "test": ["SGL-001", "SGL-002", "SGL-005", "SGL-008"],
    }
    audit_valid = audit_cohort_integrity(lakes, events, splits=valid_splits)
    assert audit_valid["status"] == "PASS"
    assert audit_valid["split_leakage_detected"] is False

    # Verify that invalid split allocation (splitting Hunza cluster) is flagged as leak
    leaky_splits = {
        "train": ["SGL-003"], # Shishper in train
        "test": ["SGL-006"],  # Passu in test (same basin/cluster!)
    }
    audit_leaky = audit_cohort_integrity(lakes, events, splits=leaky_splits)
    assert audit_leaky["status"] == "FAIL"
    assert audit_leaky["split_leakage_detected"] is True


# ---------------------------------------------------------------------------
# Test 3: InSAR Geometric Distortion Calculation (Layover & Shadow)
# ---------------------------------------------------------------------------

def test_insar_geometric_distortion_calculation():
    """Verify that steep radar-facing slopes cause layover and backslopes cause shadow."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    # 1. Steep slope (45 deg) facing radar (aspect 78 deg for look azimuth 78 deg)
    # With nominal incidence angle 38 deg, alpha (45 deg) >= theta_inc (38 deg) -> LAYOVER
    res_layover = classify_geometric_distortion(
        slope_deg=45.0, aspect_deg=78.0, inc_angle_deg=38.0, heading_deg=348.0
    )
    assert res_layover["distortion_class"] == "LAYOVER"
    assert res_layover["is_distorted"] is True

    # 2. Backslope facing away from radar (aspect 258 deg, opposite look direction)
    # Slope 60 deg facing away -> local incidence >= 90 deg -> SHADOW
    res_shadow = classify_geometric_distortion(
        slope_deg=60.0, aspect_deg=258.0, inc_angle_deg=38.0, heading_deg=348.0
    )
    assert res_shadow["distortion_class"] == "SHADOW"
    assert res_shadow["is_distorted"] is True

    # 3. Flat terrain (slope 2 deg) -> NORMAL
    res_normal = classify_geometric_distortion(
        slope_deg=2.0, aspect_deg=100.0, inc_angle_deg=38.0, heading_deg=348.0
    )
    assert res_normal["distortion_class"] == "NORMAL"
    assert res_normal["is_distorted"] is False


# ---------------------------------------------------------------------------
# Test 4: InSAR Temporal Coherence Decorrelation Modeling
# ---------------------------------------------------------------------------

def test_insar_temporal_coherence_decorrelation():
    """Verify that monsoon precipitation decays interferometric coherence below 0.25."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    # Monsoon regime: tau = 3 days, revisit = 12 days
    res_monsoon = model_insar_coherence(b_temp_days=12.0, b_perp_meters=50.0, season="monsoon")
    assert res_monsoon["gamma_temp"] < 0.05
    assert res_monsoon["gamma_total"] < 0.25
    assert res_monsoon["gamma_total"] < PHASE_UNWRAPPING_COHERENCE_THRESHOLD
    assert res_monsoon["is_phase_unwrapping_viable"] is False

    # Winter regime: tau = 45 days, revisit = 12 days
    res_winter = model_insar_coherence(b_temp_days=12.0, b_perp_meters=50.0, season="dry_winter")
    assert res_winter["gamma_temp"] > 0.70
    assert res_winter["gamma_total"] > 0.60
    assert res_winter["is_phase_unwrapping_viable"] is True


# ---------------------------------------------------------------------------
# Test 5: InSAR Storage Budget Bounds & Rejection
# ---------------------------------------------------------------------------

def test_insar_storage_budget_bounds():
    """Verify that multi-temporal full-scene SLC archives exceed storage budget and are rejected."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    # Full scene: 8 lakes x 30 scenes x 4.2 GB = ~1008 GB >> 114 GB budget -> REJECT
    res_full = check_insar_storage_budget(lake_count=8, n_scenes_per_lake=30, burst_crop_mode=False)
    assert res_full["is_feasible"] is False
    assert res_full["total_required_bytes"] > MAX_LOCAL_STORAGE_BYTES
    assert res_full["exceeded_bytes"] > 0

    # Burst crop: 8 lakes x 10 scenes x 1.0 GB = 80 GB <= 114 GB budget -> ACCEPT
    res_burst = check_insar_storage_budget(lake_count=8, n_scenes_per_lake=10, burst_crop_mode=True)
    assert res_burst["is_feasible"] is True
    assert res_burst["total_required_bytes"] <= MAX_LOCAL_STORAGE_BYTES
    assert res_burst["exceeded_bytes"] == 0


# ---------------------------------------------------------------------------
# Test 6: Statistical Power Floor Calculation Amendment
# ---------------------------------------------------------------------------

def test_statistical_power_calculation_amendment():
    """Verify analytical statistical power calculations confirming N <= 5 is underpowered (< 0.20)

    and identifying the N >= 30 sample size floor for 80% power at alpha=0.05.
    """
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    delta_auc = 0.10
    alpha = 0.05
    z_alpha = 1.96  # Two-sided 5% critical threshold

    # 1. Under N = 4 event lakes and N = 4 controls:
    # SE(Delta) approx 0.185 (Hanley & McNeil standard error)
    se_n4 = 0.185
    z_beta_n4 = (delta_auc / se_n4) - z_alpha
    # Standard normal cumulative distribution approximation
    power_n4 = 0.5 * (1.0 + math.erf(z_beta_n4 / math.sqrt(2.0)))

    # Statistical power must be strictly less than 0.20 (severely underpowered)
    assert power_n4 < 0.20, f"Expected power < 0.20 for N=4, got {power_n4:.4f}"

    # 2. Sample size floor for 80% power:
    # Requires z_beta = 0.842 => delta / SE >= 1.96 + 0.842 = 2.802
    target_z_total = 1.96 + 0.8416
    target_se = delta_auc / target_z_total  # ~ 0.0357

    # Since SE ~ 1 / sqrt(N), N_required = N_current * (SE_current / target_SE)^2
    n_required = 4.0 * (se_n4 / target_se) ** 2

    # Confirms minimum floor of N >= 30 independent events
    assert n_required >= 30.0, f"Sample size floor {n_required:.1f} should be >= 30"
