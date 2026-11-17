"""Offline schedule and split-integrity checks using declared synthetic fixtures only."""
from __future__ import annotations

import hashlib
import json
import pytest

from sentinel_gl.splits import (
    SCOPE_ROLES,
    AvailabilityScheduler,
    DecisionWindow,
    SplitEngine,
    build_fitting_provenance_graph,
    build_physical_sampling_graph,
    cluster_lakes_by_distance,
    haversine_distance,
    parse_utc_timestamp,
    verify_retrospective_invariants,
    verify_spatial_disjointness,
)

PHYSICAL_FIELDS = ("glacier_ids", "basin_ids", "era5_cell_ids", "sentinel_frame_ids",
                   "event_time_support_ids")


def _physical(prefix):
    return {field: [prefix + ":" + field] for field in PHYSICAL_FIELDS}


def _bindings(lakes):
    support, coordinates = {}, {}
    for row in lakes:
        lake_id = row["lake_id"]
        support_bytes = json.dumps(row["physical_support"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()
        coord_bytes = json.dumps([row["centroid_lat"], row["centroid_lon"]],
            separators=(",", ":"), allow_nan=False).encode()
        support[lake_id] = {"artifact_path": "fixture/physical.json", "artifact_sha256": "a" * 64,
            "support_sha256": hashlib.sha256(support_bytes).hexdigest(), "origin": "fixture"}
        coordinates[lake_id] = {"artifact_path": "fixture/coordinates.json", "artifact_sha256": "b" * 64,
            "coordinate_sha256": hashlib.sha256(coord_bytes).hexdigest(), "origin": "fixture"}
    return support, coordinates


def _four_lakes():
    coords = ((0., 0.), (0., 2.), (0., 4.), (0., 6.))
    return [{"lake_id": f"SYN-{i+1}", "centroid_lat": lat, "centroid_lon": lon,
             "physical_support_status": "VERIFIED", "physical_support": _physical(f"unit-{i+1}")}
            for i, (lat, lon) in enumerate(coords)]


def _scope_map(lakes):
    return {row["lake_id"]: role for row, role in zip(lakes, SCOPE_ROLES)}


def _source(record_id, *, start="2023-09-01T00:00:00Z", end="2023-09-01T01:00:00Z",
            available="2023-09-02T00:00:00Z", modality="optical"):
    evidence_time = "2023-09-02T00:00:00Z"
    return {"record_id": record_id, "modality": modality, "support_start": start,
            "support_end": end, "sha256": "a" * 64,
            "availability": {"status": "PROVIDER_EVIDENCE", "at": available,
                "evidence_id": "fixture-receipt-" + record_id, "evidence_sha256": "b" * 64,
                "reason": None},
            "time_axes": {
                "acquisition": {"status": "VERIFIED", "at": end, "reason": None},
                "product_generation": {"status": "VERIFIED", "at": evidence_time, "reason": None},
                "publication": {"status": "UNKNOWN", "at": None, "reason": "fixture metadata not supplied"},
                "retrieval": {"status": "VERIFIED", "at": available, "reason": None},
                "historical_availability": {"status": "VERIFIED", "at": available, "reason": None},
                "execution": {"status": "VERIFIED", "at": available, "reason": None}}}


def test_haversine_and_geometry_only_clustering_are_deterministic():
    assert haversine_distance(27.915, 88.204, 27.915, 88.204) == 0.0
    assert haversine_distance(0., 0., 1., 0.) == pytest.approx(111.195, abs=.2)
    lakes = [
        {"lake_id": "SYN-A", "centroid_lat": 28., "centroid_lon": 88.},
        {"lake_id": "SYN-B", "centroid_lat": 28.1, "centroid_lon": 88.},
        {"lake_id": "SYN-C", "centroid_lat": 29., "centroid_lon": 88.},
    ]
    clusters = cluster_lakes_by_distance(lakes, buffer_km=50.)
    assert sorted(sorted(v) for v in clusters.values()) == [["SYN-A", "SYN-B"], ["SYN-C"]]
    coords = {row["lake_id"]: (row["centroid_lat"], row["centroid_lon"]) for row in lakes}
    verify_spatial_disjointness(["SYN-A"], ["SYN-C"], coords, min_buffer_km=50.)
    with pytest.raises(ValueError, match="Spatial leakage detected"):
        verify_spatial_disjointness(["SYN-A"], ["SYN-B"], coords, min_buffer_km=50.)


def test_time_parser_requires_timezone_and_normalizes_offsets():
    assert parse_utc_timestamp("2023-10-03T00:00:00+05:30").isoformat() == "2023-10-02T18:30:00+00:00"
    with pytest.raises(ValueError, match="timezone is required"):
        parse_utc_timestamp("2023-10-03T00:00:00")


def test_availability_filters_full_support_and_retrieval_time_separately():
    scheduler = AvailabilityScheduler(window_days=180, stride_days=30,
        minimum_observations_by_modality={"optical": 1})
    rows = [
        _source("valid", start="2023-09-01T00:00:00Z", end="2023-09-01T01:00:00Z",
                available="2023-10-01T00:00:00Z"),
        _source("late", available="2023-10-01T00:00:01Z"),
        _source("future-support", end="2023-10-01T00:00:00Z"),
        _source("outside-context", start="2023-01-01T00:00:00Z", end="2023-01-01T01:00:00Z"),
        _source("post-cutoff", end="2023-09-30T23:45:00Z", available="2023-10-01T00:00:00Z"),
    ]
    accepted = scheduler.filter_retrospective_observations(rows, "2023-10-01T00:00:00Z",
        event_cutoff="2023-09-30T23:30:00Z")
    assert [row["record_id"] for row in accepted] == ["valid"]
    assert accepted[0]["availability"]["at"] != accepted[0]["time_axes"]["product_generation"]["at"]
    assert accepted[0]["availability"]["at"] == accepted[0]["time_axes"]["retrieval"]["at"]


def test_schedule_retains_all_rows_when_eligibility_rule_is_undeclared():
    scheduler = AvailabilityScheduler(window_days=30, stride_days=1)
    rows = scheduler.generate_decision_schedule("SYN-A", "2023-10-01T00:00:00Z",
        "2023-10-03T00:00:00Z", observations=(), required_modalities=("optical",),
        scope="fitting")
    assert len(rows) == 3
    assert [row.status for row in rows] == ["NOT_ASSESSED"] * 3
    assert all(not row.is_eligible and row.missingness_reason == "ELIGIBILITY_RULE_NOT_DECLARED" for row in rows)
    assert all(not row.observation_ids for row in rows)


def test_declared_support_rule_is_candidate_specific_and_causal():
    scheduler = AvailabilityScheduler(window_days=30, stride_days=1,
        minimum_observations_by_modality={"optical": 1})
    source = _source("OPT-1", start="2023-09-03T00:00:00Z", end="2023-09-03T01:00:00Z",
                     available="2023-09-04T00:00:00Z")
    rows = scheduler.generate_decision_schedule("SYN-A", "2023-10-01T00:00:00Z",
        "2023-10-02T00:00:00Z", observations=(source,), required_modalities=("optical",),
        scope="fitting")
    assert len(rows) == 2
    assert all(row.is_eligible and row.status == "ELIGIBLE" for row in rows)
    assert all(row.observation_ids == ("OPT-1",) for row in rows)
    assert rows[0].metadata["source_supports"][0]["availability"] == source["availability"]


def test_retrospective_invariant_rejects_late_and_future_support():
    window = DecisionWindow("W-1", "SYN-A", "2023-10-01T00:00:00Z",
        "2023-09-01T00:00:00Z", "2023-09-30T00:00:00Z", ("SRC",), False, (),
        scope="fitting", missingness_reason="fixture", status="NOT_ESTIMABLE")
    valid = _source("SRC", start="2023-09-20T00:00:00Z", end="2023-09-20T01:00:00Z",
                    available="2023-09-21T00:00:00Z")
    verify_retrospective_invariants(window, [valid])
    late = {**valid, "availability": {**valid["availability"], "at": "2023-10-01T00:00:01Z"}}
    with pytest.raises(ValueError, match="Late-available"):
        verify_retrospective_invariants(window, [late])
    future = {**valid, "support_end": "2023-10-01T00:00:00Z"}
    with pytest.raises(ValueError, match="Future or open"):
        verify_retrospective_invariants(window, [future])


def test_physical_gs_and_provenance_gp_do_not_merge_each_other():
    lakes = _four_lakes()
    roles = _scope_map(lakes)
    support, coordinates = _bindings(lakes)
    gs = build_physical_sampling_graph(lakes, roles, verified_support_bindings=support,
        verified_coordinate_bindings=coordinates)
    provenance = {row["lake_id"]: {"fit_input_ids": ["fit-" + row["lake_id"]],
                                   "checkpoint_ids": ["shared-weights"]} for row in lakes}
    gp = build_fitting_provenance_graph(provenance, roles)
    assert gs["graph_id"] == "G_S" and gs["edges"] == []
    assert gs["role_isolation_status"] == "FIXTURE_ONLY_NOT_RESEARCH_VERIFIED"
    assert gp["graph_id"] == "G_P" and gp["edges"]
    assert all(edge["shared_checkpoint_ids"] == ["shared-weights"] for edge in gp["edges"])
    assert gs["components"] == [[row["lake_id"]] for row in lakes]


def test_shared_physical_support_creates_gs_edge_even_when_far_apart():
    lakes = _four_lakes()
    lakes[1]["physical_support"]["era5_cell_ids"] = lakes[0]["physical_support"]["era5_cell_ids"]
    support, coordinates = _bindings(lakes)
    gs = build_physical_sampling_graph(lakes, _scope_map(lakes), verified_support_bindings=support,
        verified_coordinate_bindings=coordinates)
    assert any(reason["kind"] == "SHARED_ERA5_CELL_IDS"
               for edge in gs["edges"] for reason in edge["reasons"])
    assert gs["role_isolation_status"] == "CROSS_ROLE_PHYSICAL_ADJACENCY"


def test_unknown_physical_membership_stays_unresolved():
    lakes = _four_lakes()
    lakes[2].pop("physical_support")
    gs = build_physical_sampling_graph(lakes, _scope_map(lakes))
    assert "SYN-3" in gs["unresolved_units"]
    assert gs["role_isolation_status"] == "UNRESOLVED_UNKNOWN_PHYSICAL_MEMBERSHIP"


def test_verified_flags_and_query_coordinates_without_byte_bindings_stay_unknown():
    lakes = _four_lakes()
    graph = build_physical_sampling_graph(lakes, _scope_map(lakes))
    assert graph["unresolved_units"] == [row["lake_id"] for row in lakes]
    assert all(node["coordinate_status"] == "UNKNOWN" for node in graph["nodes"])


def test_four_role_manifest_keeps_complete_schedule_and_binds_graphs():
    lakes = _four_lakes()
    roles = _scope_map(lakes)
    observations = {row["lake_id"]: [_source("SRC-" + row["lake_id"],
        start="2023-09-04T00:00:00Z", end="2023-09-04T01:00:00Z",
        available="2023-09-05T00:00:00Z")] for row in lakes}
    provenance = {row["lake_id"]: {"fit_input_ids": ["fit-" + row["lake_id"]],
                                   "checkpoint_ids": ["checkpoint-" + roles[row["lake_id"]]]}
                  for row in lakes}
    engine = SplitEngine(window_days=30, stride_days=1,
        minimum_observations_by_modality={"optical": 1})
    support, coordinates = _bindings(lakes)
    manifest = engine.build_spatiotemporal_split(split_id="fixture-split", lakes=lakes,
        observations_by_lake=observations, start_date="2023-10-01T00:00:00Z",
        end_date="2023-10-03T00:00:00Z", scope_by_lake=roles,
        required_modalities=("optical",), provenance_by_lake=provenance,
        verified_support_bindings=support, verified_coordinate_bindings=coordinates)
    serialized = manifest.to_dict()
    assert set(serialized["scope_membership"]) == set(SCOPE_ROLES)
    assert "train_lakes" not in serialized and "train_windows" not in serialized
    assert len(serialized["schedule_rows"]) == 12
    assert all(row["scope"] in SCOPE_ROLES for row in serialized["schedule_rows"])
    assert serialized["graphs"]["G_S"]["graph_id"] == "G_S"
    assert serialized["graphs"]["G_P"]["graph_id"] == "G_P"
    assert serialized["summary"]["schedule_rows"] == 12
    assert all(row["metadata"]["source_supports"] for row in serialized["schedule_rows"])


def test_non_executable_draft_allows_empty_scope_roles_but_keeps_four_role_keys():
    lakes = _four_lakes()
    roles = _scope_map(lakes)
    engine = SplitEngine(window_days=30, stride_days=1)
    manifest = engine.build_spatiotemporal_split(split_id="empty-roles", lakes=lakes,
        observations_by_lake={}, start_date="2023-10-01T00:00:00Z",
        end_date="2023-10-02T00:00:00Z", scope_by_lake={key: "fitting" for key in roles},
        required_modalities=("optical",))
    serialized = manifest.to_dict()
    assert set(serialized["scope_membership"]) == set(SCOPE_ROLES)
    assert serialized["scope_membership"]["development_validation"] == {"lake_ids": [], "window_ids": []}
    assert serialized["scope_membership"]["evaluation"] == {"lake_ids": [], "window_ids": []}


def test_future_append_does_not_change_prior_decision_membership():
    scheduler = AvailabilityScheduler(window_days=10, stride_days=1,
        minimum_observations_by_modality={"optical": 1})
    early = _source("early", start="2023-09-25T00:00:00Z", end="2023-09-25T01:00:00Z",
        available="2023-09-26T00:00:00Z")
    later = _source("later", start="2023-10-02T00:00:00Z", end="2023-10-02T01:00:00Z",
        available="2023-10-02T02:00:00Z")
    base = scheduler.generate_decision_schedule("SYN-A", "2023-10-01T00:00:00Z",
        "2023-10-02T00:00:00Z", (early,), ("optical",), scope="fitting")[0]
    appended = scheduler.generate_decision_schedule("SYN-A", "2023-10-01T00:00:00Z",
        "2023-10-02T00:00:00Z", (early, later), ("optical",), scope="fitting")[0]
    assert (appended.observation_ids, appended.status, appended.is_eligible,
            appended.context_start, appended.context_end) == (
                base.observation_ids, base.status, base.is_eligible,
                base.context_start, base.context_end)
