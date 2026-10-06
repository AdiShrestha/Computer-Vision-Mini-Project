"""Split integrity, spatial disjointness, and temporal anti-leakage tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
These synthetic lake coordinates, observation timestamps, and revisit sequences
verify mathematical invariants, spatial clustering, and boundary purging in offline
unit tests only. They do not represent real observational records and must never be
used to support scientific or research findings.
"""
from __future__ import annotations
import datetime
import json
import math
import pytest

from sentinel_gl.splits import (
    haversine_distance,
    parse_utc_timestamp,
    cluster_lakes_by_distance,
    verify_spatial_disjointness,
    DecisionWindow,
    AvailabilityScheduler,
    TemporalBoundaryPurger,
    SplitEngine,
    SplitManifest,
)


def test_haversine_distance_properties():
    # 1. Zero distance to self
    assert haversine_distance(27.915, 88.204, 27.915, 88.204) == 0.0

    # 2. Symmetry
    d1 = haversine_distance(27.915, 88.204, 27.985, 88.752)
    d2 = haversine_distance(27.985, 88.752, 27.915, 88.204)
    assert abs(d1 - d2) < 1e-6

    # 3. Known baseline: 1 degree latitude is approximately 111.19 km
    d_lat = haversine_distance(0.0, 0.0, 1.0, 0.0)
    assert 111.0 < d_lat < 111.4

    # 4. South Lhonak to Khangchung Tsho distance is ~54.39 km
    d_sikkim = haversine_distance(27.915, 88.204, 27.985, 88.752)
    assert 54.0 < d_sikkim < 55.0


def test_spatial_clustering_and_buffer_isolation():
    # Construct 3 synthetic test lakes
    # Lake A and Lake B are ~11.1 km apart (<50 km buffer) -> must cluster together
    # Lake C is ~111 km away (>50 km buffer) -> must form separate cluster
    lakes = [
        {"lake_id": "SYN-001", "centroid_lat": 28.00, "centroid_lon": 88.00},
        {"lake_id": "SYN-002", "centroid_lat": 28.10, "centroid_lon": 88.00},
        {"lake_id": "SYN-003", "centroid_lat": 29.00, "centroid_lon": 88.00},
    ]

    clusters = cluster_lakes_by_distance(lakes, buffer_km=50.0)
    assert len(clusters) == 2

    # Find which cluster has SYN-001
    c1 = next(c for c in clusters.values() if "SYN-001" in c)
    assert "SYN-002" in c1
    assert "SYN-003" not in c1

    c2 = next(c for c in clusters.values() if "SYN-003" in c)
    assert len(c2) == 1

    coords = {l["lake_id"]: (l["centroid_lat"], l["centroid_lon"]) for l in lakes}

    # Verify disjointness passes between c1 and c2
    verify_spatial_disjointness(c1, c2, coords, min_buffer_km=50.0)

    # Violate disjointness: put SYN-001 in Train and SYN-002 in Test
    with pytest.raises(ValueError, match="Spatial leakage detected"):
        verify_spatial_disjointness(["SYN-001"], ["SYN-002"], coords, min_buffer_km=50.0)


def test_zero_future_data_leakage_into_past_decisions():
    scheduler = AvailabilityScheduler(window_days=180, stride_days=30)

    # Observations at varying times relative to decision timestamp
    observations = [
        {"record_id": "REC-01", "acquisition_timestamp": "2023-09-01T10:00:00Z"},
        {"record_id": "REC-02", "acquisition_timestamp": "2023-09-15T12:00:00Z"},
        {"record_id": "REC-03", "acquisition_timestamp": "2023-09-30T23:59:59Z"},
        # Exact decision second: MUST BE EXCLUDED (strict inequality t_acq < t_decision)
        {"record_id": "REC-04", "acquisition_timestamp": "2023-10-01T00:00:00Z"},
        # Future observations: MUST BE EXCLUDED
        {"record_id": "REC-05", "acquisition_timestamp": "2023-10-01T00:00:01Z"},
        {"record_id": "REC-06", "acquisition_timestamp": "2023-10-15T12:00:00Z"},
        # Too old (prior to 180-day context window: 2023-10-01 - 180 days = 2023-04-04): MUST BE EXCLUDED
        {"record_id": "REC-00", "acquisition_timestamp": "2023-03-01T00:00:00Z"},
    ]

    t_decision = "2023-10-01T00:00:00Z"
    filtered = scheduler.filter_retrospective_observations(
        observations=observations,
        decision_timestamp=t_decision,
    )

    valid_ids = [o["record_id"] for o in filtered]
    assert valid_ids == ["REC-01", "REC-02", "REC-03"]
    assert "REC-04" not in valid_ids
    assert "REC-05" not in valid_ids
    assert "REC-06" not in valid_ids
    assert "REC-00" not in valid_ids

    # Double-check that all included records satisfy strict inequality
    dt_decision = parse_utc_timestamp(t_decision)
    for obs in filtered:
        assert parse_utc_timestamp(obs["acquisition_timestamp"]) < dt_decision


def test_pre_event_quarantining():
    scheduler = AvailabilityScheduler(window_days=180, stride_days=30)

    # South Lhonak pre-event cutoff at 2023-10-03T00:00:00Z
    cutoff = "2023-10-03T00:00:00Z"

    observations = [
        {"record_id": "S1-SEP", "acquisition_timestamp": "2023-09-25T12:00:00Z"},
        {"record_id": "S2-OCT01", "acquisition_timestamp": "2023-10-01T04:30:00Z"},
        {"record_id": "S2-OCT02", "acquisition_timestamp": "2023-10-02T18:00:00Z"},
        # At or after cutoff: MUST BE QUARANTINED
        {"record_id": "S1-OCT03", "acquisition_timestamp": "2023-10-03T00:00:00Z"},
        {"record_id": "S1-OCT03-LATE", "acquisition_timestamp": "2023-10-03T12:00:00Z"},
        {"record_id": "S2-OCT04", "acquisition_timestamp": "2023-10-04T05:00:00Z"},
    ]

    # Evaluate decision after cutoff (e.g. 2023-10-05)
    filtered = scheduler.filter_retrospective_observations(
        observations=observations,
        decision_timestamp="2023-10-05T00:00:00Z",
        event_cutoff=cutoff,
    )

    ids = [o["record_id"] for o in filtered]
    assert ids == ["S1-SEP", "S2-OCT01", "S2-OCT02"]
    assert "S1-OCT03" not in ids
    assert "S1-OCT03-LATE" not in ids
    assert "S2-OCT04" not in ids


def test_temporal_boundary_purging_eliminates_autocorrelation_overlap():
    purger = TemporalBoundaryPurger()
    split_date = "2023-01-01T00:00:00Z"

    # Construct synthetic decision windows
    # Window A: Dec 2022 decision (support: June 2022 - Dec 2022) -> Pre-split (Train)
    win_pre = DecisionWindow(
        window_id="WIN-PRE",
        lake_id="LAKE-01",
        decision_timestamp="2022-12-15T00:00:00Z",
        context_start="2022-06-18T00:00:00Z",
        context_end="2022-12-14T00:00:00Z",
        observation_ids=("OBS-01", "OBS-02"),
        is_eligible=True,
        modalities_present=("optical", "sar"),
    )

    # Window B: Feb 2023 decision (support: Aug 2022 - Feb 2023)
    # Context start (Aug 2022) is before split_date (Jan 2023) -> OVERLAPS TRAINING DATA -> MUST BE PURGED
    win_overlap = DecisionWindow(
        window_id="WIN-OVERLAP",
        lake_id="LAKE-01",
        decision_timestamp="2023-02-15T00:00:00Z",
        context_start="2022-08-19T00:00:00Z",
        context_end="2023-02-14T00:00:00Z",
        observation_ids=("OBS-03", "OBS-04"),
        is_eligible=True,
        modalities_present=("optical", "sar"),
    )

    # Window C: Aug 2023 decision (support: Feb 2023 - Aug 2023)
    # Context start (Feb 2023) is after split_date (Jan 2023) -> Post-split clean evaluation
    win_clean_eval = DecisionWindow(
        window_id="WIN-CLEAN-POST",
        lake_id="LAKE-01",
        decision_timestamp="2023-08-15T00:00:00Z",
        context_start="2023-02-16T00:00:00Z",
        context_end="2023-08-14T00:00:00Z",
        observation_ids=("OBS-05", "OBS-06"),
        is_eligible=True,
        modalities_present=("optical", "sar"),
    )

    pre, post, purged = purger.purge_split_boundary(
        decision_windows=[win_pre, win_overlap, win_clean_eval],
        split_timestamp=split_date,
        window_days=180,
    )

    assert [w.window_id for w in pre] == ["WIN-PRE"]
    assert [w.window_id for w in purged] == ["WIN-OVERLAP"]
    assert [w.window_id for w in post] == ["WIN-CLEAN-POST"]

    # Verify zero overlap between pre-split training support and clean post-split evaluation support
    pre_context_end = parse_utc_timestamp(win_pre.decision_timestamp)
    post_context_start = parse_utc_timestamp(win_clean_eval.context_start)
    assert post_context_start > pre_context_end


def test_leap_years_and_irregular_cadence():
    scheduler = AvailabilityScheduler(window_days=180, stride_days=30, min_obs_per_modality=2)

    # 2024 is a leap year (2024-02-29 exists)
    start = "2024-01-01T00:00:00Z"
    end = "2024-04-01T00:00:00Z"

    # Synthetic observations with irregular satellite revisits over leap period
    observations = [
        # Optical: 2 passes
        {"record_id": "OPT-01", "modality": "optical", "acquisition_timestamp": "2024-01-15T04:30:00Z"},
        {"record_id": "OPT-02", "modality": "optical", "acquisition_timestamp": "2024-02-29T04:30:00Z"},  # Leap day
        # SAR: 2 passes
        {"record_id": "SAR-01", "modality": "sar", "acquisition_timestamp": "2024-02-10T12:00:00Z"},
        {"record_id": "SAR-02", "modality": "sar", "acquisition_timestamp": "2024-03-05T12:00:00Z"},
        # Weather: 3 daily records
        {"record_id": "WTH-01", "modality": "weather", "acquisition_timestamp": "2024-01-20T12:00:00Z"},
        {"record_id": "WTH-02", "modality": "weather", "acquisition_timestamp": "2024-02-28T12:00:00Z"},
        {"record_id": "WTH-03", "modality": "weather", "acquisition_timestamp": "2024-03-20T12:00:00Z"},
    ]

    windows = scheduler.generate_decision_schedule(
        lake_id="LEAP-LAKE",
        start_date=start,
        end_date=end,
        observations=observations,
        required_modalities=("optical", "sar", "weather"),
    )

    assert len(windows) > 0
    # The last window (end of March / April) sees all observations and should be eligible
    last_win = windows[-1]
    assert "optical" in last_win.modalities_present
    assert "sar" in last_win.modalities_present
    assert "weather" in last_win.modalities_present
    assert last_win.is_eligible is True

    # The first window (2024-01-01) sees 0 observations and must be ineligible
    first_win = windows[0]
    assert first_win.is_eligible is False
    assert "INSUFFICIENT_OBSERVATIONS" in (first_win.missingness_reason or "")


def test_split_engine_end_to_end_manifest():
    engine = SplitEngine(buffer_km=50.0, window_days=180, stride_days=30)

    lakes = [
        {"lake_id": "SGL-001", "centroid_lat": 27.915, "centroid_lon": 88.204},
        {"lake_id": "SGL-002", "centroid_lat": 27.985, "centroid_lon": 88.752},
    ]

    # Observations for SGL-001 and SGL-002
    obs_map = {
        "SGL-001": [
            {"record_id": "SGL1-O1", "modality": "optical", "acquisition_timestamp": "2023-01-15T04:30:00Z"},
            {"record_id": "SGL1-O2", "modality": "optical", "acquisition_timestamp": "2023-02-15T04:30:00Z"},
            {"record_id": "SGL1-S1", "modality": "sar", "acquisition_timestamp": "2023-01-20T12:00:00Z"},
            {"record_id": "SGL1-S2", "modality": "sar", "acquisition_timestamp": "2023-02-20T12:00:00Z"},
            {"record_id": "SGL1-W1", "modality": "weather", "acquisition_timestamp": "2023-01-10T12:00:00Z"},
            {"record_id": "SGL1-W2", "modality": "weather", "acquisition_timestamp": "2023-02-10T12:00:00Z"},
        ],
        "SGL-002": [
            {"record_id": "SGL2-O1", "modality": "optical", "acquisition_timestamp": "2023-01-15T04:30:00Z"},
            {"record_id": "SGL2-O2", "modality": "optical", "acquisition_timestamp": "2023-02-15T04:30:00Z"},
            {"record_id": "SGL2-S1", "modality": "sar", "acquisition_timestamp": "2023-01-20T12:00:00Z"},
            {"record_id": "SGL2-S2", "modality": "sar", "acquisition_timestamp": "2023-02-20T12:00:00Z"},
            {"record_id": "SGL2-W1", "modality": "weather", "acquisition_timestamp": "2023-01-10T12:00:00Z"},
            {"record_id": "SGL2-W2", "modality": "weather", "acquisition_timestamp": "2023-02-10T12:00:00Z"},
        ],
    }

    manifest = engine.build_spatiotemporal_split(
        split_id="SPLIT-PILOT-01",
        lakes=lakes,
        observations_by_lake=obs_map,
        start_date="2023-01-01T00:00:00Z",
        end_date="2023-06-01T00:00:00Z",
        split_date="2023-03-01T00:00:00Z",
        event_cutoffs={"SGL-001": "2023-10-03T00:00:00Z"},
    )

    # Check manifest invariants
    assert manifest.split_id == "SPLIT-PILOT-01"
    assert manifest.protocol == "spatiotemporal_purged_cluster"
    assert len(manifest.clusters) == 2
    assert "SGL-001" in manifest.train_lakes
    assert "SGL-002" in manifest.train_lakes

    d = manifest.to_dict()
    assert "summary" in d
    assert d["summary"]["total_train_windows"] == len(manifest.train_windows)
    assert d["summary"]["total_purged_windows"] == len(manifest.purged_windows)

    # Test JSON serialization roundtrip
    serialized = manifest.to_json()
    loaded = json.loads(serialized)
    assert loaded["split_id"] == "SPLIT-PILOT-01"


def test_verify_retrospective_invariants_rejects_future_data():
    from sentinel_gl.splits import verify_retrospective_invariants

    # Create a valid decision window
    win = DecisionWindow(
        window_id="WIN-TEST-01",
        lake_id="LAKE-01",
        decision_timestamp="2023-10-01T00:00:00Z",
        context_start="2023-04-04T00:00:00Z",
        context_end="2023-09-30T12:00:00Z",
        observation_ids=("REC-01", "REC-02"),
        is_eligible=True,
        modalities_present=("optical", "sar"),
    )

    valid_obs = [
        {"record_id": "REC-01", "acquisition_timestamp": "2023-09-01T00:00:00Z"},
        {"record_id": "REC-02", "acquisition_timestamp": "2023-09-25T12:00:00Z"},
    ]
    # Should pass without error
    verify_retrospective_invariants(win, valid_obs)

    # Invariant 1 violation: future observation (t_acq >= t_decision)
    future_obs = [
        {"record_id": "REC-01", "acquisition_timestamp": "2023-09-01T00:00:00Z"},
        {"record_id": "REC-02", "acquisition_timestamp": "2023-10-01T00:00:00Z"},  # Equal to decision second!
    ]
    with pytest.raises(ValueError, match="Future observation leakage detected"):
        verify_retrospective_invariants(win, future_obs)

    # Invariant 1 violation: observation prior to context start
    too_old_obs = [
        {"record_id": "REC-01", "acquisition_timestamp": "2023-01-01T00:00:00Z"},  # Before context_start
        {"record_id": "REC-02", "acquisition_timestamp": "2023-09-25T12:00:00Z"},
    ]
    with pytest.raises(ValueError, match="Temporal boundary violation"):
        verify_retrospective_invariants(win, too_old_obs)


def test_deterministic_partitioning_future_append_invariance(tmp_path):
    from runners.run_split_manifest import run_split_manifest
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    dossier_path = repo_root / "data" / "pilot_dossier.json"
    lake_reg_path = repo_root / "data" / "lake_registry.csv"
    event_reg_path = repo_root / "data" / "event_registry.csv"

    out_base = tmp_path / "manifest_base.json"
    manifest_base = run_split_manifest(
        dossier_path=dossier_path,
        lake_registry_path=lake_reg_path,
        event_registry_path=event_reg_path,
        output_path=out_base,
        split_id="SPLIT-APPEND-TEST",
        stride_days=5,
    )

    # Construct an extended dossier with 10 future evaluation observations in late October 2023
    with open(dossier_path, "r", encoding="utf-8") as f:
        dossier_data = json.load(f)

    extended_data = json.loads(json.dumps(dossier_data))
    for i in range(1, 11):
        extended_data["record_provenance"].append({
            "record_id": f"FUTURE-OBS-{i:02d}",
            "lake_id": "SGL-002",
            "modality": "optical",
            "provider_name": "Copernicus Data Space Ecosystem",
            "product_identifier": f"S2_FUTURE_{i}.SAFE",
            "acquisition_timestamp": f"2023-10-{20 + (i % 5):02d}T10:00:00Z",
            "sha256": f"futurehash{i:058d}",
            "storage_bytes": 1024,
            "metadata": {},
        })

    extended_dossier_path = tmp_path / "pilot_dossier_extended.json"
    with open(extended_dossier_path, "w", encoding="utf-8") as f:
        json.dump(extended_data, f, indent=2)

    out_extended = tmp_path / "manifest_extended.json"
    manifest_extended = run_split_manifest(
        dossier_path=extended_dossier_path,
        lake_registry_path=lake_reg_path,
        event_registry_path=event_reg_path,
        output_path=out_extended,
        split_id="SPLIT-APPEND-TEST",
        stride_days=5,
        start_date="2023-09-01T00:00:00Z",
        end_date="2023-10-03T00:00:00Z",  # Identical schedule evaluation horizon
    )

    # Invariant: Earlier windows, IDs, and split assignments must remain 100% bit-for-bit identical
    assert len(manifest_base.test_windows) == len(manifest_extended.test_windows)
    for w_base, w_ext in zip(manifest_base.test_windows, manifest_extended.test_windows):
        assert w_base.window_id == w_ext.window_id
        assert w_base.lake_id == w_ext.lake_id
        assert w_base.decision_timestamp == w_ext.decision_timestamp
        assert w_base.is_eligible == w_ext.is_eligible
        assert w_base.observation_ids == w_ext.observation_ids
        assert w_base.metadata.get("observation_hashes") == w_ext.metadata.get("observation_hashes")


def test_split_manifest_runner_on_authentic_pilot_dossier(tmp_path):
    from runners.run_split_manifest import run_split_manifest
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    dossier_path = repo_root / "data" / "pilot_dossier.json"
    lake_reg_path = repo_root / "data" / "lake_registry.csv"
    event_reg_path = repo_root / "data" / "event_registry.csv"
    output_path = tmp_path / "split_manifest.json"

    manifest = run_split_manifest(
        dossier_path=dossier_path,
        lake_registry_path=lake_reg_path,
        event_registry_path=event_reg_path,
        output_path=output_path,
        split_id="SPLIT-PILOT-01",
        stride_days=5,
    )

    # 1. Output file exists and valid
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["split_id"] == "SPLIT-PILOT-01"
    assert data["protocol"] == "spatiotemporal_purged_cluster"
    assert "CLS-SIKKIM-01" in data["clusters"]
    assert data["clusters"]["CLS-SIKKIM-01"] == ["SGL-001", "SGL-002"]

    # 2. Both pilot lakes in test partition (evaluation roles)
    assert set(data["test_lakes"]) == {"SGL-001", "SGL-002"}
    assert len(data["test_windows"]) == 14  # 7 per lake with 5-day stride

    # 3. Cryptographic provenance bindings
    meta = data["metadata"]
    assert "dossier_sha256" in meta and len(meta["dossier_sha256"]) == 64
    assert "lake_registry_sha256" in meta and len(meta["lake_registry_sha256"]) == 64
    assert meta["total_input_records"] == 123
    assert meta["quarantined_records_count"] == 1
    assert "ERA5-SGL-001-2023-10-03" in meta["quarantined_record_ids"]

    # 4. Strict retrospective availability for all generated windows
    for win_dict in data["test_windows"]:
        dt_dec = parse_utc_timestamp(win_dict["decision_timestamp"])
        dt_start = parse_utc_timestamp(win_dict["context_start"])
        assert dt_start < dt_dec
        # Check status and eligibility alignment
        if win_dict["is_eligible"]:
            assert win_dict["status"] == "ELIGIBLE"
            assert win_dict["missingness_reason"] is None
        else:
            assert win_dict["status"] == "NOT_ESTIMABLE"
            assert win_dict["missingness_reason"] is not None

        # Check observation hashes bound to metadata
        obs_hashes = win_dict["metadata"]["observation_hashes"]
        assert len(obs_hashes) == len(win_dict["observation_ids"])


def test_cli_runner_subprocess_execution(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    runner_script = repo_root / "source" / "runners" / "run_split_manifest.py"
    output_path = tmp_path / "cli_manifest.json"

    cmd = [
        sys.executable,
        "-B",
        str(runner_script),
        "--dossier", str(repo_root / "data" / "pilot_dossier.json"),
        "--lake-registry", str(repo_root / "data" / "lake_registry.csv"),
        "--event-registry", str(repo_root / "data" / "event_registry.csv"),
        "--output", str(output_path),
        "--split-id", "SPLIT-CLI-TEST",
        "--stride-days", "7",
    ]

    env = dict(os.environ) if "os" in dir() else {}
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "source")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"CLI runner failed: {proc.stderr}"

    summary = json.loads(proc.stdout)
    assert summary["status"] == "SUCCESS"
    assert summary["split_id"] == "SPLIT-CLI-TEST"
    assert output_path.exists()

