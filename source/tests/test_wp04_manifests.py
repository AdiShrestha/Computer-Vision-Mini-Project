"""Integrity checks for WP04 manifests; synthetic rows are explicit test fixtures."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from sentinel_gl.wp04_manifests import (
    ROLES,
    WP04Error,
    _verify_exposure,
    union_interval_seconds,
    build_bundle,
    verify_bundle,
    verify_scope_partition,
    _make_current_decode,
)
from sentinel_gl.admission import AdmissionError
from sentinel_gl import wp04_manifests as wp04
from sentinel_gl.admission import validate_schema


def _roles():
    return {role: {"candidate_site_ids": [], "schedule_row_ids": [],
        "source_ids": [], "context_ids": [], "status": "EMPTY_DRAFT_NO_SCIENTIFIC_MEMBERSHIP"}
        for role in ROLES}


def test_interval_union_recomputes_overlap_leap_day_and_partial_days():
    intervals = [
        {"start": "2024-02-28T23:30:00Z", "end": "2024-02-29T01:30:00Z"},
        {"start": "2024-02-29T00:30:00Z", "end": "2024-02-29T03:00:00Z"},
        {"start": "2024-02-29T03:00:00+00:00", "end": "2024-02-29T03:15:00Z"},
        {"start": "2024-02-29T09:00:00+05:30", "end": "2024-02-29T10:00:00+05:30"},
    ]
    assert union_interval_seconds(intervals) == 4 * 3600 + 45 * 60
    assert union_interval_seconds([]) == 0


def test_unassessed_exposure_preserves_nulls_and_reason():
    _verify_exposure({"status": "NOT_ASSESSED", "intervals": None, "union_seconds": None,
        "reason": "No operational intervals in the retained evidence."})
    with pytest.raises(WP04Error, match="unassessed exposure"):
        _verify_exposure({"status": "NOT_ASSESSED", "intervals": [], "union_seconds": 0,
            "reason": "No operational intervals in the retained evidence."})


def test_assessed_exposure_is_independently_recomputed():
    record = {"status": "ASSESSED", "intervals": [
        {"start": "2023-09-01T23:00:00Z", "end": "2023-09-02T00:30:00Z"},
        {"start": "2023-09-02T00:00:00Z", "end": "2023-09-02T02:00:00Z"}],
        "union_seconds": 10800, "reason": None}
    _verify_exposure(record)
    corrupt = deepcopy(record)
    corrupt["union_seconds"] = 12600
    with pytest.raises(WP04Error, match="does not independently recompute"):
        _verify_exposure(corrupt)


def test_four_scope_partition_rejects_overlap_and_orphan_membership():
    roles = _roles()
    roles["development_validation"]["candidate_site_ids"] = ["PILOT-A"]
    roles["development_validation"]["schedule_row_ids"] = ["ANCHOR-A"]
    row = {"schedule_row_id": "ANCHOR-A", "scope": "development_validation"}
    verify_scope_partition(roles, [row], {"PILOT-A"})

    overlapping = deepcopy(roles)
    overlapping["fitting"]["source_ids"] = ["SRC-1"]
    overlapping["development_validation"]["source_ids"] = ["SRC-1"]
    with pytest.raises(WP04Error, match="source_ids overlaps"):
        verify_scope_partition(overlapping, [row], {"PILOT-A"})

    orphan = deepcopy(roles)
    orphan["evaluation"]["schedule_row_ids"] = ["MISSING"]
    with pytest.raises(WP04Error, match="schedule membership is incomplete"):
        verify_scope_partition(orphan, [row], {"PILOT-A"})


def test_roles_are_exactly_four_and_empty_roles_are_permitted_in_draft():
    roles = _roles()
    roles["development_validation"]["candidate_site_ids"] = ["PILOT-A"]
    row = {"schedule_row_id": "ANCHOR-A", "scope": "development_validation"}
    roles["development_validation"]["schedule_row_ids"] = ["ANCHOR-A"]
    verify_scope_partition(roles, [row], {"PILOT-A"})
    roles.pop("evaluation")
    with pytest.raises(WP04Error, match="exactly four"):
        verify_scope_partition(roles, [row], {"PILOT-A"})


def test_failed_sar_receipt_is_hashed_without_admitting_sar_measurement():
    root = Path(__file__).resolve().parents[2]
    output_parent = root / "project/architecture/wp04_local_2026-10-14"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / ("test_failed_receipt_" + uuid.uuid4().hex)
    build_bundle(root, output)

    sources = json.loads((output / "sources.v2.json").read_text())
    failed = sources["failed_attempts"][0]
    inventory = {row["path"]: row for row in sources["asset_hash_inventory"]}
    receipt = Path(failed["receipt_path"])
    receipt_bytes = (root / receipt).read_bytes()
    expected = inventory[failed["receipt_path"]]
    raw_inventory = json.loads((root / "data/raw/wp03/attempt_2026-10-14_stage_001/outputs_sha256.json").read_text())
    relative = failed["receipt_path"].removeprefix("data/raw/wp03/attempt_2026-10-14_stage_001/")
    original = {row["path"]: row for row in raw_inventory["files"]}[relative]

    assert expected["sha256"] == hashlib.sha256(receipt_bytes).hexdigest() == original["sha256"]
    assert expected["bytes"] == len(receipt_bytes) == original["bytes"]
    assert failed["received_bytes"] == 0
    assert failed["source_bytes_sha256"] is None
    assert sources["observations"] == []
    assert sources["measurement_admission"] == "BLOCKED"
    assert all(item["source_kind"] != "SENTINEL_1_GRD_PRODUCT" for item in sources["source_items"])
    assert len(sources["source_items"]) == 2
    for item in sources["source_items"]:
        assert item["historical_inspection"]["inspection"] == {
            "status": "INSPECTION_FAILED",
            "error_type": "AdmissionError",
            "literal_error": "query centroid is outside the selected C1 green tile",
            "research_admitted": False,
        }
        current = item["current_decode"]
        assert current["decode_status"] == "ROI_PIXEL_DECODED_QUARANTINED"
        assert current["query_roi_alignment"]["status"] == "ROI_COVERED_BY_ALL_FOUR_SOURCE_ASSETS_WITH_NO_ALIGNMENT_HALO"
        assert current["research_admissible"] is False
        assert len(current["source_asset_bindings"]) == 4
        assert len(current["decoded_output_bindings"]) == 25
        assert current["limits"]["lake_geometry_status"] == "NOT_VERIFIED_CENTROID_ONLY"
        assert current["limits"]["mask_accuracy_status"] == "UNVERIFIED"
        assert current["limits"]["co_registration_status"] == "UNRESOLVED"
    registry = json.loads((output / "registry.v1.json").read_text())
    south = next(site for site in registry["sites"] if site["site_id"] == "PILOT-SOUTH-LHONAK")
    khangchung = next(site for site in registry["sites"] if site["site_id"] == "PILOT-KHANGCHUNG-TSHO")
    assert south["geometry"]["status"] == "UNAVAILABLE"
    assert south["aliases"]["status"] == "UNRESOLVED"
    assert south["event"]["classification_status"] == "UNVERIFIED"
    assert khangchung["control_status"] == "NOT_A_CONTROL"
    schedule = json.loads((output / "schedule.v1.json").read_text())
    assert schedule["cutoff"]["original"] == "2023-10-03T00:00:00+05:30"
    assert schedule["cutoff"]["utc"] == "2023-10-02T18:30:00+00:00"
    assert len(schedule["rows"]) == 2
    assert all(row["status"] == "NOT_ASSESSED" and row["eligibility"] is None and
               row["context_start"] is None and not row["observation_ids"] for row in schedule["rows"])
    scopes = json.loads((output / "scopes.v1.json").read_text())
    assert set(scopes["roles"]) == set(ROLES)
    assert all(not role["source_ids"] and not role["context_ids"] for role in scopes["roles"].values())
    assert scopes["graphs"]["G_S"]["role_isolation_status"] == "UNRESOLVED_UNKNOWN_PHYSICAL_MEMBERSHIP"
    assert scopes["graphs"]["G_P"]["status"] == "PROVENANCE_UNRESOLVED"
    assert verify_bundle(root, output)["research_admissible"] is False
    shutil.rmtree(output)


def _build_bundle_for_mutation(root):
    output_parent = root / "project/architecture/wp04_local_2026-10-14"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / ("test_decode_binding_" + uuid.uuid4().hex)
    build_bundle(root, output)
    return output


def _rewrite_sources_and_bundle_hash(output, mutate):
    sources_path = output / "sources.v2.json"
    sources = json.loads(sources_path.read_text())
    mutate(sources)
    sources_path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    bundle_path = output / "bundle.v2.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["sections"]["sources"]["sha256"] = hashlib.sha256(sources_path.read_bytes()).hexdigest()
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mutation", ["omit_output", "corrupt_output_hash"])
def test_current_decode_outputs_must_match_complete_hashed_inventory(mutation):
    root = Path(__file__).resolve().parents[2]
    output = _build_bundle_for_mutation(root)
    try:
        def mutate(sources):
            bindings = sources["source_items"][0]["current_decode"]["decoded_output_bindings"]
            if mutation == "omit_output":
                bindings.pop()
            else:
                bindings[0]["sha256"] = "0" * 64
        _rewrite_sources_and_bundle_hash(output, mutate)
        with pytest.raises(WP04Error, match="current decode join"):
            verify_bundle(root, output)
    finally:
        shutil.rmtree(output)


@pytest.mark.parametrize("mutation", ["stale_failure", "unsupported_success", "wrong_product", "wrong_site", "wrong_raw_asset"])
def test_current_decode_rejects_stale_success_or_cross_product_bindings(mutation):
    root = Path(__file__).resolve().parents[2]
    output = _build_bundle_for_mutation(root)
    try:
        def mutate(sources):
            current = sources["source_items"][0]["current_decode"]
            if mutation == "stale_failure":
                current["decode_status"] = "INSPECTION_FAILED"
            elif mutation == "unsupported_success":
                current["research_admissible"] = True
            elif mutation == "wrong_product":
                current["product_id"] = "S2A_T45RXM_20230921T043839_L2A"
            elif mutation == "wrong_site":
                current["site_id"] = "PILOT-KHANGCHUNG-TSHO"
            else:
                current["source_asset_bindings"][0]["sha256"] = "f" * 64
        _rewrite_sources_and_bundle_hash(output, mutate)
        with pytest.raises(AdmissionError):
            verify_bundle(root, output)
    finally:
        shutil.rmtree(output)


@pytest.mark.parametrize("coordinate_case", ["missing", "short", "reordered"])
def test_current_decode_builder_guard_rejects_missing_short_and_reordered_coordinates(coordinate_case):
    root = Path(__file__).resolve().parents[2]
    site_id = "PILOT-SOUTH-LHONAK"
    pilot = json.loads((root / "project/wp03_pilot_manifest.json").read_text())
    pilot_site = next(site for site in pilot["sites"] if site["site_id"] == site_id)
    stage = json.loads((root / "data/raw/wp03/attempt_2026-10-14_stage_001/stage_report.json").read_text())
    stage_inventory_doc = json.loads((root / "data/raw/wp03/attempt_2026-10-14_stage_001/outputs_sha256.json").read_text())
    decode_report_rel = "data/raw/wp03/attempt_2026-10-14_roi_decode_005/stage_report.json"
    decode_inventory_rel = "data/raw/wp03/attempt_2026-10-14_roi_decode_005/outputs_sha256.json"
    decode_report = json.loads((root / decode_report_rel).read_text())
    decode_inventory_doc = json.loads((root / decode_inventory_rel).read_text())
    decode_product = deepcopy(decode_report["sentinel2_c1"][site_id])
    coordinates = decode_product["pixel_inspection"]["query_centroid_wgs84"]
    if coordinate_case == "missing":
        del decode_product["pixel_inspection"]["query_centroid_wgs84"]
    elif coordinate_case == "short":
        decode_product["pixel_inspection"]["query_centroid_wgs84"] = coordinates[:1]
    else:
        decode_product["pixel_inspection"]["query_centroid_wgs84"] = list(reversed(coordinates))
    stage_inventory = {row["path"]: row for row in stage_inventory_doc["files"]}
    decode_inventory = {row["path"]: row for row in decode_inventory_doc["files"]}
    report_binding = {"path": decode_report_rel,
        "sha256": hashlib.sha256((root / decode_report_rel).read_bytes()).hexdigest()}
    inventory_binding = {"path": decode_inventory_rel,
        "sha256": hashlib.sha256((root / decode_inventory_rel).read_bytes()).hexdigest()}
    with pytest.raises(WP04Error, match="query coordinate differs"):
        _make_current_decode(site_id, pilot_site, stage["sentinel2_c1"][site_id], decode_product,
            stage_inventory, decode_inventory, report_binding, inventory_binding)


def test_builder_and_independent_verifier_reject_three_coordinate_input(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    decode_report_rel = "data/raw/wp03/attempt_2026-10-14_roi_decode_005/stage_report.json"
    original_read_json = wp04._read_json

    def read_with_extra_coordinate(evidence_root, relative):
        value = original_read_json(evidence_root, relative)
        if relative == decode_report_rel:
            value["sentinel2_c1"]["PILOT-SOUTH-LHONAK"]["pixel_inspection"]["query_centroid_wgs84"].append(0.0)
        return value

    monkeypatch.setattr(wp04, "_read_json", read_with_extra_coordinate)
    failed_output = root / ("project/architecture/wp04_local_2026-10-14/test_builder_extra_coordinate_" + uuid.uuid4().hex)
    with pytest.raises(WP04Error, match="query coordinate differs"):
        build_bundle(root, failed_output)
    monkeypatch.setattr(wp04, "_read_json", original_read_json)

    output = _build_bundle_for_mutation(root)
    try:
        sources_path = output / "sources.v2.json"
        sources = json.loads(sources_path.read_text())
        current = sources["source_items"][0]["current_decode"]
        current["query_roi_alignment"]["query_centroid_wgs84_lon_lat"].append(0.0)
        validate_schema(current, "wp04_current_decode")
        sources_path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        bundle_path = output / "bundle.v2.json"
        bundle = json.loads(bundle_path.read_text())
        bundle["sections"]["sources"]["sha256"] = hashlib.sha256(sources_path.read_bytes()).hexdigest()
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        with pytest.raises(WP04Error, match="current decode join"):
            verify_bundle(root, output)
    finally:
        shutil.rmtree(output)
