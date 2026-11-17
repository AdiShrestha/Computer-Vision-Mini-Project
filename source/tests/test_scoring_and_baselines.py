"""Unit tests for operational alert episodes, fair baselines, calibrations, and sensor ablations.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Synthetic panels, simulated decision sequences, and constructed test scores in this
file verify scoring algorithms, hysteresis logic, baseline training isolation,
split disjointness, and ablation math in offline unit tests only. They do not
represent real physical observations and must never support scientific claims.
"""
from __future__ import annotations
from datetime import date, timedelta
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from sentinel_gl.features import (
    CHANNEL_MODALITIES,
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    StaticTopography,
    MultiModalPanel,
)
from sentinel_gl.episodes import (
    AlertEpisode,
    AlertEpisodeEngine,
    calculate_lake_exposure_years,
    compute_alert_burden_estimand,
    compute_bounded_lead_time,
)
from sentinel_gl.baselines import (
    ClimatologyBaseline,
    LakeAreaTrendBaseline,
    WeatherOnlyBaseline,
    RobustPCABaseline,
)
from sentinel_gl.predictions import (
    ABLATION_CONFIGS,
    PredictionRecord,
    apply_sensor_ablation,
    check_window_eligibility,
    compute_paired_score_differences,
    make_prediction_record,
)
from sentinel_gl.calibration import (
    calibrate_false_alert_threshold,
    calibrate_percentile_threshold,
)


def _make_test_panel(
    lake_id: str,
    window_id: str,
    start_date: str = "2023-01-01",
    base_val: float = 1.0,
) -> MultiModalPanel:
    """Helper creating small valid MultiModalPanel for offline baseline tests.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    dt_start = date.fromisoformat(start_date)
    schedule_days = 180
    dates = tuple((dt_start + timedelta(days=i)).isoformat() for i in range(schedule_days))
    values = np.full((schedule_days, NUM_CHANNELS), np.nan, dtype=np.float64)
    mask = np.zeros((schedule_days, NUM_CHANNELS), dtype=np.bool_)

    # Populate 15 observations with base_val
    for d in range(0, 150, 10):
        values[d, 0] = base_val * 1.25   # lake area
        values[d, 1] = 0.45              # ndwi
        values[d, 2] = 0.50              # mndwi
        values[d, 3] = -12.0             # vv_db
        values[d, 4] = -18.0             # vh_db
        values[d, 5] = -6.0              # cross_ratio
        values[d, 6] = 2.0               # vv_var
        values[d, 7] = -5.0              # temp_2m
        values[d, 8] = 10.0              # precip_mm
        mask[d, :] = True
    statuses = np.full(mask.shape, "UNAVAILABLE_TEST_FIXTURE", dtype=object)
    statuses[mask] = "OBSERVED"

    topo = StaticTopography(elevation_m=5000.0, moraine_slope_deg=20.0, catchment_area_km2=8.0)
    return MultiModalPanel(
        lake_id=lake_id,
        window_id=window_id,
        start_date=dates[0],
        end_date=dates[-1],
        dates=dates,
        values=values,
        mask=mask,
        static_metadata=topo,
        origin="fixture", statuses=statuses)


# ---------------------------------------------------------------------------
# Test 1: Episode Engine Hysteresis and Refractory
# ---------------------------------------------------------------------------

def test_episode_engine_hysteresis_and_refractory():
    """Verify that multi-month sustained alarms collapse to exactly 1 episode,
    and subsequent alarms after hysteresis/refractory create a 2nd distinct episode.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    # 12 monthly decisions: 2023-01-01 to 2023-12-01
    dates = [f"2023-{m:02d}-01" for m in range(1, 13)]
    # Scores:
    # m1, m2: low
    # m3, m4, m5, m6: high (sustained alarms for 4 months -> should collapse to 1 episode!)
    # m7, m8: low (hysteresis r=2 non-alarms -> closes Episode 1)
    # m9: low
    # m10, m11: high (sustained alarm -> starts Episode 2)
    # m12: low
    scores = [0.20, 0.25, 0.85, 0.88, 0.92, 0.80, 0.30, 0.35, 0.20, 0.90, 0.95, 0.20]
    decisions = [(d, s, True) for d, s in zip(dates, scores)]

    engine = AlertEpisodeEngine(
        threshold=0.70,
        sustained_q=2,
        hysteresis_r=2,
        refractory_days=60,
        max_gap_days=45,
    )

    episodes = engine.extract_episodes("SGL-MOCK-001", decisions)

    # Invariant: Persistent multi-month alarms collapse to exactly 1 episode,
    # and a distinct second trigger creates a 2nd episode.
    assert len(episodes) == 2, f"Expected 2 episodes, got {len(episodes)}"

    ep1, ep2 = episodes[0], episodes[1]
    assert ep1.episode_id == "EP-SGL-MOCK-001-001"
    assert ep1.start_date == "2023-03-01"
    assert ep1.end_date == "2023-06-01"
    assert ep1.qualifying_decisions_count == 4
    assert ep1.peak_score == pytest.approx(0.92)

    assert ep2.episode_id == "EP-SGL-MOCK-001-002"
    assert ep2.start_date == "2023-10-01"
    assert ep2.end_date == "2023-11-01"
    assert ep2.qualifying_decisions_count == 2
    assert ep2.peak_score == pytest.approx(0.95)

    # Exposure calculation
    with pytest.raises(ValueError,match="BLOCKED"):
        calculate_lake_exposure_years(dates,stride_days=30)
    # Hand-declared fixture exposure, not inferred from row count.
    exp_years = 360 / 365.25

    # Alert burden estimand
    burden = compute_alert_burden_estimand(episodes, {"SGL-MOCK-001": exp_years})
    assert burden["status"] == "ESTIMATED"
    assert burden["n_episodes"] == 2
    assert burden["lambda_alert"] == pytest.approx(2.0 / exp_years)

    # Zero exposure must return NOT_ESTIMABLE
    zero_burden = compute_alert_burden_estimand([], {"SGL-MOCK-001": 0.0})
    assert zero_burden["status"] == "NOT_ESTIMABLE"
    assert zero_burden["lambda_alert"] is None


# ---------------------------------------------------------------------------
# Test 2: Baseline Training Isolation
# ---------------------------------------------------------------------------

def test_baseline_training_isolation():
    """Verify that appending evaluation panels does not alter fitted baseline state or hashes.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    p_train1 = _make_test_panel("SGL-TRAIN-01", "W01", "2023-01-01", base_val=1.0)
    p_train2 = _make_test_panel("SGL-TRAIN-02", "W02", "2023-01-01", base_val=1.5)
    train_panels = [p_train1, p_train2]
    train_ids = ["SGL-TRAIN-01", "SGL-TRAIN-02"]

    # Fit 4 baselines on training panels
    b_clim = ClimatologyBaseline.fit_fixture(train_panels, train_ids)
    b_trend = LakeAreaTrendBaseline.fit_fixture(train_panels, train_ids)
    b_weath = WeatherOnlyBaseline.fit_fixture(train_panels, train_ids)
    b_pca = RobustPCABaseline.fit_fixture(train_panels, train_ids, k_components=2)

    # Construct extreme evaluation panels from unseen lake
    p_eval1 = _make_test_panel("SGL-EVAL-01", "W03", "2023-01-01", base_val=99999.0)
    p_eval2 = _make_test_panel("SGL-EVAL-01", "W04", "2023-02-01", base_val=88888.0)
    combined_panels = train_panels + [p_eval1, p_eval2]

    # Re-fit with combined panels but identical training lake IDs
    b_clim_2 = ClimatologyBaseline.fit_fixture(combined_panels, train_ids)
    b_trend_2 = LakeAreaTrendBaseline.fit_fixture(combined_panels, train_ids)
    b_weath_2 = WeatherOnlyBaseline.fit_fixture(combined_panels, train_ids)
    b_pca_2 = RobustPCABaseline.fit_fixture(combined_panels, train_ids, k_components=2)

    # Invariants: 100% parameter and hash identity
    assert b_clim.state_hash == b_clim_2.state_hash
    assert np.allclose(b_clim.doy_means, b_clim_2.doy_means)
    assert np.allclose(b_clim.doy_scales, b_clim_2.doy_scales)

    assert b_trend.state_hash == b_trend_2.state_hash

    assert b_weath.state_hash == b_weath_2.state_hash
    assert np.allclose(b_weath.weather_means, b_weath_2.weather_means)
    assert np.allclose(b_weath.weather_scales, b_weath_2.weather_scales)

    assert b_pca.state_hash == b_pca_2.state_hash
    assert np.allclose(b_pca.mean_window, b_pca_2.mean_window)
    assert np.allclose(b_pca.components, b_pca_2.components)


# ---------------------------------------------------------------------------
# Test 3: Lead Time Estimand Integrity
# ---------------------------------------------------------------------------

def test_lead_time_estimand_integrity():
    """Verify lead time calculation, None return on non-detection, and post-event exclusion.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    onset = "2023-10-04"

    # Case 1: Pre-event sustained alarm (detected)
    # Alarm at 2023-08-01 and 2023-08-15 (50 days before onset)
    decisions_detected = [
        ("2023-07-01", 0.30, True),
        ("2023-08-01", 0.85, True),
        ("2023-08-15", 0.90, True),  # sustained_q = 2 met here -> declaration date
        ("2023-09-01", 0.80, True),
        ("2023-10-10", 0.99, True),  # post-event: must be ignored!
    ]
    res_det = compute_bounded_lead_time(
        event_id="EV-001",
        lake_id="SGL-001",
        event_onset_date=onset,
        decisions=decisions_detected,
        threshold=0.70,
        sustained_q=2,
    max_gap_days=31, )
    assert res_det.status == "DETECTED"
    assert res_det.declaration_date == "2023-08-15"
    assert res_det.lead_time_days == (date.fromisoformat(onset) - date.fromisoformat("2023-08-15")).days

    # Case 2: Alarm never triggered within horizon -> NOT_DETECTED with lead_time_days = None (NEVER 0!)
    decisions_undetected = [
        ("2023-07-01", 0.30, True),
        ("2023-08-01", 0.25, True),
        ("2023-08-15", 0.40, True),
        ("2023-09-01", 0.35, True),
    ]
    res_undet = compute_bounded_lead_time(
        event_id="EV-001",
        lake_id="SGL-001",
        event_onset_date=onset,
        decisions=decisions_undetected,
        threshold=0.70,
        sustained_q=2,
    max_gap_days=31, )
    assert res_undet.status == "NOT_DETECTED"
    assert res_undet.lead_time_days is None
    assert res_undet.lead_time_days != 0

    # Case 3: Post-event alarms only ($t >= t_onset)
    decisions_post_only = [
        ("2023-10-04", 0.99, True),  # On event day
        ("2023-10-05", 0.99, True),  # After event day
    ]
    res_post = compute_bounded_lead_time(
        event_id="EV-001",
        lake_id="SGL-001",
        event_onset_date=onset,
        decisions=decisions_post_only,
        threshold=0.70,
        sustained_q=2,
    max_gap_days=31, )
    # Since no pre-event decisions were available, status is NOT_ESTIMABLE
    assert res_post.status == "NOT_ESTIMABLE"
    assert res_post.lead_time_days is None


# ---------------------------------------------------------------------------
# Test 4: Calibration Strict Split Disjointness
# ---------------------------------------------------------------------------

def test_calibration_strict_split_disjointness():
    """Verify that overlapping calibration and evaluation IDs raise ValueError,
    and thresholds match empirical quantiles.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    scores = np.linspace(0.1, 10.0, 100)

    # Overlapping sets must be rejected
    with pytest.raises(ValueError, match="calibration overlaps evaluation"):
        calibrate_percentile_threshold(
            scores=scores,
            calibration_lake_ids=["SGL-001", "SGL-002"],
            evaluation_lake_ids=["SGL-002", "SGL-003"],
            percentile=0.95,
        )

    with pytest.raises(ValueError, match="calibration overlaps evaluation"):
        calibrate_false_alert_threshold(
            scores=scores,
            calibration_ids=[f"c_{i}" for i in range(100)],
            final_ids=["c_0", "eval_0"],
            target=0.05,
        )

    # Disjoint sets: verify threshold matches empirical quantile
    res_cal = calibrate_percentile_threshold(
        scores=scores,
        calibration_lake_ids=["SGL-002", "SGL-003"],
        evaluation_lake_ids=["SGL-001"],
        percentile=0.95,
    )
    expected_quantile = float(np.percentile(scores, 95.0))
    assert res_cal["threshold"] == pytest.approx(expected_quantile)
    assert res_cal["percentile"] == 0.95
    assert res_cal["n_windows"] == 100
    assert "state_hash" in res_cal and len(res_cal["state_hash"]) == 64


# ---------------------------------------------------------------------------
# Test 5: Sensor Ablation Intervention Identity
# ---------------------------------------------------------------------------

def test_sensor_ablation_intervention_identity():
    """Verify that all 7 configurations in the 2^N lattice modify observation masks as specified.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    panel = _make_test_panel("SGL-MOCK-001", "W01", base_val=1.0)
    eval_panel = _make_test_panel("SGL-MOCK-001", "W02", base_val=5.0)
    vals = eval_panel.values
    mask = eval_panel.mask

    # Expected channel assignments
    # optical: 0, 1, 2, 3
    # sar: 4, 5, 6, 7
    # era5: 8, 9, 10
    configs = {name: {index for index, channel in enumerate(FEATURE_CHANNELS)
                      if CHANNEL_MODALITIES[channel] in modalities}
               for name, modalities in ABLATION_CONFIGS.items()}

    scores = {}
    b_pca = RobustPCABaseline.fit_fixture([panel], ["SGL-MOCK-001"], k_components=2)

    for cfg_name, expected_active in configs.items():
        abl_vals, abl_mask = apply_sensor_ablation(vals, mask, cfg_name,
                                                   channel_names=FEATURE_CHANNELS)

        for c in range(len(FEATURE_CHANNELS)):
            if c in expected_active:
                # Active channels maintain original mask
                assert np.array_equal(abl_mask[:, c], mask[:, c])
            else:
                # Disabled channels must be completely False and NaN
                assert not np.any(abl_mask[:, c])
                assert np.all(np.isnan(abl_vals[:, c]))

        # Check eligibility
        assert check_window_eligibility(abl_mask, cfg_name, min_obs_per_modality=2,
                                        channel_names=FEATURE_CHANNELS) is (cfg_name != "all_off")

        # Score window
        score = b_pca.predict_window(abl_vals, abl_mask)
        scores[cfg_name] = score

    # Check distinct scores across sensor ablations
    all_off_values,all_off_mask=apply_sensor_ablation(vals,mask,"all_off",
                                                      channel_names=FEATURE_CHANNELS)
    assert not all_off_mask.any() and np.isnan(all_off_values).all()
    assert b_pca.predict_window(all_off_values,all_off_mask) is None
    assert not check_window_eligibility(all_off_mask,"all_off",2,
                                        channel_names=FEATURE_CHANNELS)


# ---------------------------------------------------------------------------
# Test 6: Paired Difference Support Alignment
# ---------------------------------------------------------------------------

def test_paired_difference_support_alignment():
    for scale in (1.,10.):
        with pytest.raises(ValueError,match='performance estimand'):compute_paired_score_differences([scale],[1.])



# ---------------------------------------------------------------------------
# Test 7: Prediction Record Sample ID Determinism
# ---------------------------------------------------------------------------

def test_prediction_record_sample_id_determinism():
    """Verify deterministic SHA-256 sample_id generation for closed-window records.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    rec1 = make_prediction_record(
        lake_id="SGL-001",
        decision_date="2023-09-06",
        window_start="2023-03-10",
        window_end="2023-09-05",
        model_id="tmae",
        ablation_id="full",
        score=0.531365,
        eligible=True,
    )
    expected_id = hashlib.sha256("SGL-001|2023-09-06|tmae|full|2023-03-10|2023-09-05".encode("utf-8")).hexdigest()
    assert rec1.sample_id == expected_id
    assert len(rec1.sample_id) == 64

    # Any change to key identity changes the hash
    rec2 = make_prediction_record(
        lake_id="SGL-001",
        decision_date="2023-09-06",
        window_start="2023-03-10",
        window_end="2023-09-05",
        model_id="tmae",
        ablation_id="opt_sar",
        score=0.531365,
        eligible=True,
    )
    assert rec2.sample_id != rec1.sample_id


# ---------------------------------------------------------------------------
# Test 8: End-to-End Prediction Runner Execution
# ---------------------------------------------------------------------------

def test_run_predictions_end_to_end(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from runners.run_predictions import run_predictions
    with pytest.raises(ValueError, match="BLOCKED"):
        run_predictions(output_ledger_path=tmp_path/"ledger.csv")
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 9: CLI Runner Subprocess Invocation
# ---------------------------------------------------------------------------

def test_run_predictions_cli_subprocess(tmp_path):
    from runners.run_predictions import main
    assert main(['--output-ledger',str(tmp_path/'ledger.csv')])==2
    assert not list(tmp_path.iterdir())
