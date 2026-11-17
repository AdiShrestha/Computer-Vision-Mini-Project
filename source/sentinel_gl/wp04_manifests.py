"""Bounded offline WP04 manifest production and independent verification.

This module records retained source bytes and design uncertainty. It does not
decode imagery, admit measurements, or make the research pipeline executable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .admission import AdmissionError, validate_schema
from .integrity import digest, file_hash, loads_strict, verified_file
from .splits import build_fitting_provenance_graph, build_physical_sampling_graph, parse_utc_timestamp


ROLES = ("fitting", "development_validation", "threshold_calibration", "evaluation")
SITE_IDS = ("PILOT-SOUTH-LHONAK", "PILOT-KHANGCHUNG-TSHO")
STAGE_ATTEMPT = "attempt_2026-10-14_stage_001"
DECODE_ATTEMPT = "attempt_2026-10-14_roi_decode_005"
STAGE_ROOT = f"data/raw/wp03/{STAGE_ATTEMPT}"
DECODE_ROOT = f"data/raw/wp03/{DECODE_ATTEMPT}"
RAW_BAND_IDENTITIES = {"green": "B03", "nir": "B08", "swir16": "B11", "scl": "SCL"}
OUTPUT_FILES = {
    "registry": "registry.v1.json",
    "sources": "sources.v2.json",
    "schedule": "schedule.v1.json",
    "scopes": "scopes.v1.json",
}


class WP04Error(AdmissionError):
    """The retained evidence or WP04 bundle is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WP04Error(message)


def _read_json(root: Path, relative: str) -> dict:
    path = verified_file(root, relative, file_hash(root / relative))
    value = loads_strict(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {relative}")
    return value


def _binding(root: Path, relative: str, *, evidence_type: str, status: str = "VERIFIED") -> dict:
    path = root / relative
    _require(path.is_file() and not path.is_symlink(), f"required local evidence is unavailable: {relative}")
    return {"path": relative, "sha256": file_hash(path), "evidence_type": evidence_type, "status": status}


def _iso(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be an explicit timestamp")
    parse_utc_timestamp(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _validate_output_inventory(root: Path, inventory_path: str, inventory: Mapping[str, Any],
                               stage: Mapping[str, Any]) -> dict[str, dict]:
    _require(inventory.get("file_count") == len(inventory.get("files", [])),
             "retained output inventory file count is inconsistent")
    _require(stage.get("output_file_count") == inventory.get("file_count"),
             "stage report and output inventory counts differ")
    _require(stage.get("output_manifest_sha256") == file_hash(root / inventory_path),
             "stage report does not bind the retained output inventory bytes")
    entries = {}
    for item in inventory.get("files", []):
        _require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
                 "output inventory entries require exact path/bytes/SHA-256 fields")
        path = item["path"]
        _require(isinstance(path, str) and path not in entries, "duplicate or invalid output inventory path")
        _require(type(item["bytes"]) is int and item["bytes"] >= 0, "invalid inventoried byte count")
        _require(re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is not None,
                 "invalid output SHA-256")
        checked = verified_file(root / "data/raw/wp03/attempt_2026-10-14_stage_001", path, item["sha256"])
        _require(checked.stat().st_size == item["bytes"], f"inventoried byte count mismatch: {path}")
        entries[path] = item
    return entries


def _unknown_axis(reason: str) -> dict:
    return {"status": "UNKNOWN", "at": None, "reason": reason}


def _verified_axis(value: str) -> dict:
    return {"status": "VERIFIED", "at": _iso(value, "source time axis"), "reason": None}


def _validate_decode_output_inventory(root: Path, inventory_path: str, inventory: Mapping[str, Any],
                                      decode_report: Mapping[str, Any], stage: Mapping[str, Any],
                                      stage_inventory_path: str) -> dict[str, dict]:
    _require(inventory.get("file_count") == len(inventory.get("files", [])),
             "ROI decode output inventory file count is inconsistent")
    _require(decode_report.get("output_file_count") == inventory.get("file_count") and
             decode_report.get("output_manifest_sha256") == file_hash(root / inventory_path),
             "ROI decode report does not bind the retained output inventory")
    _require(decode_report.get("stage_output_manifest_sha256") == file_hash(root / stage_inventory_path) ==
             stage.get("output_manifest_sha256"),
             "ROI decode is not chained to the retained stage output inventory")
    _require(decode_report.get("input_sha256") == stage.get("input_sha256"),
             "ROI decode pilot/selection/source inputs differ from the retained stage")
    _require(decode_report.get("status") == "OPTICAL_ROI_DECODED_QUARANTINED_SAR_UNAVAILABLE" and
             decode_report.get("mode") == "decode-only" and
             decode_report.get("network_call_status") == "NOT_INVOKED_REUSED_STAGED_PROVIDER_BYTES" and
             decode_report.get("research_admitted") is False and
             decode_report.get("research_status") == "RESEARCH_NOT_READY" and
             decode_report.get("release_status") == "BLOCKED",
             "ROI decode execution status/admission differs from the accepted retained attempt")
    entries: dict[str, dict] = {}
    counts = {site_id: 0 for site_id in SITE_IDS}
    for item in inventory.get("files", []):
        _require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
                 "ROI decode inventory entries require exact path/bytes/SHA-256 fields")
        path = item["path"]
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        _require(len(parts) == 4 and parts[0] == "c1" and parts[1] in SITE_IDS and
                 parts[2] == "roi" and parts[3] not in ("", ".") and path not in entries,
                 "ROI decode output path is duplicate or outside a named pilot ROI")
        _require(type(item["bytes"]) is int and item["bytes"] >= 0 and
                 re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is not None,
                 "ROI decode output byte count or SHA-256 is malformed")
        checked = verified_file(root / DECODE_ROOT, path, item["sha256"])
        _require(checked.stat().st_size == item["bytes"], f"ROI decode output byte count mismatch: {path}")
        entries[path] = item
        counts[parts[1]] += 1
    _require(set(decode_report.get("sentinel2_c1", {})) == set(SITE_IDS) and
             all(counts[site_id] > 0 for site_id in SITE_IDS),
             "ROI decode outputs do not cover exactly the two declared pilot sites")
    return entries


def _make_current_decode(site_id: str, pilot_site: Mapping[str, Any], stage_product: Mapping[str, Any],
                         decode_product: Mapping[str, Any], stage_inventory: Mapping[str, dict],
                         decode_inventory: Mapping[str, dict], decode_report_binding: Mapping[str, Any],
                         decode_inventory_binding: Mapping[str, Any]) -> dict:
    pixel = decode_product.get("pixel_inspection")
    _require(decode_product.get("item_id") == stage_product.get("item_id") and
             decode_product.get("product_uri") == stage_product.get("product_uri") and
             decode_product.get("site_join") == [site_id] and
             decode_product.get("site_scope") == "DEVELOPMENT_EXPOSED" and
             decode_product.get("operational_availability") == "UNVERIFIED_REPROCESSED_IN_2025" and
             isinstance(pixel, dict) and pixel.get("site_id") == site_id and
             pixel.get("site_scope") == "DEVELOPMENT_EXPOSED" and
             pixel.get("site_item_id") == stage_product.get("item_id") and
             pixel.get("product_uri") == stage_product.get("product_uri") and
             pixel.get("status") == "AUTHENTIC_PROVIDER_COGS_INSPECTED_NOT_RESEARCH_ADMITTED" and
             pixel.get("research_admissible") is False,
             f"current ROI decode identity or admission is unsupported: {site_id}")
    roi = pixel.get("roi")
    _require(isinstance(roi, dict) and roi.get("status") == "ROI_PIXEL_DECODED_QUARANTINED" and
             roi.get("site_id") == site_id and roi.get("site_item_id") == stage_product.get("item_id") and
             roi.get("product_uri") == stage_product.get("product_uri") and
             roi.get("site_scope") == "DEVELOPMENT_EXPOSED" and
             roi.get("research_admissible") is False and
             roi.get("2023_operational_availability") == "UNVERIFIED_REPROCESSED_IN_2025" and
             roi.get("site_geometry_status") == "PI_QUERY_COORDINATE_ONLY_LAKE_GEOMETRY_UNVERIFIED" and
             pixel.get("lake_geometry_status") == "NOT_VERIFIED_CENTROID_ONLY",
             f"current ROI decode status or limits are unsupported: {site_id}")
    query = pixel.get("query_centroid_wgs84")
    expected_query = [pilot_site.get("longitude"), pilot_site.get("latitude")]
    _require(isinstance(query, list) and len(query) == 2 and query == expected_query and
             roi.get("centroid_wgs84_lon_lat") == query,
             f"ROI decode query coordinate differs from the pilot identity: {site_id}")
    coverage = roi.get("source_coverage")
    _require(isinstance(coverage, dict) and set(coverage) == set(RAW_BAND_IDENTITIES),
             f"ROI decode source coverage roles differ: {site_id}")
    coverage_by_band = {}
    source_asset_bindings = []
    decode_assets = pixel.get("assets")
    stage_assets = stage_product.get("assets")
    _require(isinstance(decode_assets, dict) and set(decode_assets) == set(RAW_BAND_IDENTITIES) and
             isinstance(stage_assets, dict) and set(stage_assets) == set(RAW_BAND_IDENTITIES),
             f"ROI decode raw asset roles differ: {site_id}")
    for band, identity in RAW_BAND_IDENTITIES.items():
        covered = coverage[band].get("full_requested_roi_covered")
        _require(covered is True, f"requested ROI is not fully covered by {band}: {site_id}")
        coverage_by_band[band] = covered
        relative = f"{STAGE_ROOT}/c1/{site_id}/{band}.tif"
        stage_record = stage_inventory.get(f"c1/{site_id}/{band}.tif")
        stage_asset = stage_assets[band]
        decoded_asset = decode_assets[band]
        _require(stage_record is not None and stage_asset.get("expected_sha256") == stage_record["sha256"] and
                 stage_asset.get("received_bytes") == stage_record["bytes"] == decoded_asset.get("bytes") and
                 decoded_asset.get("source_sha256") == stage_record["sha256"] and
                 decoded_asset.get("band_identity") == identity,
                 f"ROI decode raw asset does not join to the verified stage bytes: {site_id}/{band}")
        source_asset_bindings.append({"band_role": band, "band_identity": identity, "path": relative,
            "bytes": stage_record["bytes"], "sha256": stage_record["sha256"],
            "status": "MATCHED_VERIFIED_STAGE_RAW_ASSET"})
    halo = roi.get("source_alignment_halo_in_output_m")
    _require(type(halo) in (int, float) and halo == 0,
             f"ROI decode reports an unsupported alignment halo: {site_id}")
    output_prefix = f"c1/{site_id}/roi/"
    output_rows = [row for path, row in sorted(decode_inventory.items()) if path.startswith(output_prefix)]
    _require(output_rows and set(roi.get("artifacts", [])) <=
             {PurePosixPath(row["path"]).name for row in output_rows},
             f"ROI decode's declared products are not bound by the current output inventory: {site_id}")
    return {
        "attempt_id": DECODE_ATTEMPT,
        "report_path": decode_report_binding["path"],
        "report_sha256": decode_report_binding["sha256"],
        "output_inventory_path": decode_inventory_binding["path"],
        "output_inventory_sha256": decode_inventory_binding["sha256"],
        "site_id": site_id,
        "product_id": stage_product["item_id"],
        "product_uri": stage_product["product_uri"],
        "site_scope": "DEVELOPMENT_EXPOSED",
        "decode_status": roi["status"],
        "pixel_inspection_status": pixel["status"],
        "research_admissible": pixel["research_admissible"],
        "operational_availability": decode_product["operational_availability"],
        "query_roi_alignment": {
            "status": "ROI_COVERED_BY_ALL_FOUR_SOURCE_ASSETS_WITH_NO_ALIGNMENT_HALO",
            "query_centroid_wgs84_lon_lat": query,
            "coverage_by_band": coverage_by_band,
            "source_alignment_halo_in_output_m": halo,
        },
        "source_asset_bindings": source_asset_bindings,
        "decoded_output_bindings": [{
            "path": f"{DECODE_ROOT}/{row['path']}", "bytes": row["bytes"],
            "sha256": row["sha256"], "site_id": site_id, "product_id": stage_product["item_id"],
        } for row in output_rows],
        "limits": {
            "lake_geometry_status": pixel["lake_geometry_status"],
            "roi_geometry_status": roi["site_geometry_status"],
            "mask_accuracy_status": "UNVERIFIED",
            "mask_accuracy_reason": "No independently verified lake boundary or held-out labeled pixels are bound to the decoded ROI.",
            "co_registration_status": "UNRESOLVED",
            "co_registration_reason": "No independent control geometry or tie-point error assessment is bound.",
        },
    }


def _make_source_items(stage: Mapping[str, Any], inventory: Mapping[str, dict],
                       stage_binding: Mapping[str, Any], pilot_sites: Mapping[str, Any],
                       decode_report: Mapping[str, Any], decode_inventory: Mapping[str, dict],
                       decode_report_binding: Mapping[str, Any],
                       decode_inventory_binding: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    c1 = stage.get("sentinel2_c1")
    _require(isinstance(c1, dict) and set(c1) == set(SITE_IDS),
             "retained C1 site identities differ from the two-site pilot manifest")
    decode_c1 = decode_report.get("sentinel2_c1")
    _require(isinstance(decode_c1, dict) and set(decode_c1) == set(SITE_IDS),
             "accepted ROI decode site identities differ from the two-site pilot manifest")
    items, asset_hash_records = [], []
    for site_id in SITE_IDS:
        product = c1[site_id]
        historical_pixel = product.get("pixel_inspection")
        _require(product.get("site_scope") == "DEVELOPMENT_EXPOSED" and product.get("site_join") == [site_id],
                 f"C1 item is not uniquely joined to the exposed pilot site {site_id}")
        _require(product.get("research_admitted") is False and
                 product.get("operational_availability") == "UNVERIFIED_REPROCESSED_IN_2025",
                 f"C1 item admission/availability status is absent or changed: {site_id}")
        _require(isinstance(historical_pixel, dict) and historical_pixel.get("status") == "INSPECTION_FAILED" and
                 historical_pixel.get("literal_error") == "query centroid is outside the selected C1 green tile" and
                 historical_pixel.get("research_admitted") is False,
                 f"original stage inspection failure is not preserved as historical evidence: {site_id}")
        acquisition = _iso(product.get("acquisition_datetime"), "acquisition_datetime")
        generation = _iso(product.get("generation_time"), "generation_time")
        assets = product.get("assets")
        _require(isinstance(assets, dict) and set(assets) == {"green", "nir", "swir16", "scl"},
                 f"C1 product has an incomplete asset set: {site_id}")
        asset_rows, finishes = [], []
        for band in ("green", "nir", "swir16", "scl"):
            asset = assets[band]
            rel = f"c1/{site_id}/{band}.tif"
            receipt_rel = rel + ".receipt.json"
            _require(rel in inventory and receipt_rel in inventory,
                     f"retained asset or transfer receipt is missing from the byte inventory: {rel}")
            _require(asset.get("status") == "COMPLETE_HASH_VERIFIED" and
                     asset.get("expected_sha256") == inventory[rel]["sha256"] and
                     asset.get("received_bytes") == inventory[rel]["bytes"] and
                     asset.get("expected_bytes") == inventory[rel]["bytes"],
                     f"stage receipt metadata differs from actual inventoried bytes: {rel}")
            request_start = _iso(asset.get("request_started_utc"), f"{band} retrieval start")
            request_end = _iso(asset.get("request_finished_utc"), f"{band} retrieval end")
            _require(parse_utc_timestamp(request_start) <= parse_utc_timestamp(request_end),
                     f"retrieval interval is reversed: {rel}")
            finishes.append(request_end)
            row = {
                "asset_id": f"{site_id}:{product['item_id']}:{band}",
                "band_role": band,
                "path": f"data/raw/wp03/attempt_2026-10-14_stage_001/{rel}",
                "bytes": inventory[rel]["bytes"],
                "sha256": inventory[rel]["sha256"],
                "receipt_path": f"data/raw/wp03/attempt_2026-10-14_stage_001/{receipt_rel}",
                "receipt_bytes": inventory[receipt_rel]["bytes"],
                "receipt_sha256": inventory[receipt_rel]["sha256"],
                "retrieval_started_at": request_start,
                "retrieval_finished_at": request_end,
                "origin": "PROVIDER_BYTES_RETAINED",
            }
            asset_rows.append(row)
            asset_hash_records.append({"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]})
            asset_hash_records.append({"path": row["receipt_path"], "sha256": row["receipt_sha256"],
                                       "bytes": row["receipt_bytes"]})
        available = {
            "status": "UNVERIFIED",
            "at": None,
            "evidence_id": None,
            "evidence_sha256": None,
            "reason": "Exact product version was generated after the pinned 2023 cutoff; historical publication/availability evidence is absent.",
        }
        current_decode = _make_current_decode(site_id, pilot_sites[site_id], product, decode_c1[site_id],
            inventory, decode_inventory, decode_report_binding, decode_inventory_binding)
        items.append({
            "source_id": f"SRC-{product['item_id']}",
            "source_kind": "SENTINEL_2_C1_L2A_PRODUCT",
            "product_id": product["item_id"],
            "product_uri": product["product_uri"],
            "site_id": site_id,
            "site_scope": "DEVELOPMENT_EXPOSED",
            "site_join_status": current_decode["query_roi_alignment"]["status"],
            "historical_inspection": {
                "attempt_id": STAGE_ATTEMPT,
                "report_path": stage_binding["path"],
                "report_sha256": stage_binding["sha256"],
                "site_id": site_id,
                "product_id": product["item_id"],
                "inspection": historical_pixel,
            },
            "current_decode": current_decode,
            "acquisition_support": {"status": "VERIFIED_PRODUCT_METADATA", "start": acquisition, "end": acquisition},
            "time_axes": {
                "acquisition": _verified_axis(acquisition),
                "product_generation": _verified_axis(generation),
                "publication": _unknown_axis("Provider publication time for this exact version is not retained."),
                "retrieval": _verified_axis(max(finishes, key=parse_utc_timestamp)),
                "historical_availability": _unknown_axis("No source-backed 2023 historical availability record is retained."),
                "execution": _verified_axis(_iso(stage.get("finished_utc"), "stage execution finish")),
            },
            "availability": available,
            "rights": {"status": "UNVERIFIED_SOURCE_SPECIFIC_TERMS", "reason":
                "Retained provenance does not include reviewed source-specific redistribution or publication terms."},
            "admission": "SOURCE_BYTES_ACCEPTED_FOR_LOCAL_DEVELOPMENT_PROVENANCE_ONLY_MEASUREMENTS_NOT_ADMITTED",
            "assets": asset_rows,
        })
    return items, asset_hash_records


def _site_registry(pilot: Mapping[str, Any], pilot_binding: Mapping[str, str],
                   stage_binding: Mapping[str, str]) -> dict:
    sites_by_id = {site.get("site_id"): site for site in pilot.get("sites", []) if isinstance(site, dict)}
    _require(set(sites_by_id) == set(SITE_IDS), "pilot manifest site IDs are unsupported")
    sites = []
    for site_id in SITE_IDS:
        source = sites_by_id[site_id]
        khangchung = site_id == "PILOT-KHANGCHUNG-TSHO"
        sites.append({
            "site_id": site_id,
            "display_name": source.get("name"),
            "identity_status": source.get("identity_status"),
            "candidate_lake_id": None,
            "lake_identity_status": "UNVERIFIED_INDEPENDENT_INVENTORY_JOIN_REQUIRED",
            "query_point": {"latitude": source.get("latitude"), "longitude": source.get("longitude"),
                "crs": pilot.get("crs"), "origin": pilot.get("coordinate_origin"),
                "status": "PI_QUERY_POINT_ONLY_NOT_LAKE_GEOMETRY_OR_GS_COORDINATE"},
            "geometry": {"status": "UNAVAILABLE", "artifact_path": None, "sha256": None,
                "mapping_date": None, "crs": None, "reason": "No independently admitted dated lake polygon is retained."},
            "aliases": {"status": "UNRESOLVED", "values": None,
                "reason": "No evidence-backed alias crosswalk is retained."},
            "event": {"classification": None, "classification_status": "UNVERIFIED",
                "onset_earliest": None, "onset_latest": None,
                "reason": "The pilot manifest is a query-site design record and does not establish event class or phase-specific onset."},
            "control_status": "NOT_A_CONTROL" if khangchung else "NOT_ASSESSED",
            "development_exposure": "DEVELOPMENT_EXPOSED",
            "physical_support": {"status": "UNRESOLVED", "glacier_ids": None, "basin_ids": None,
                "era5_cell_ids": None, "sentinel_frame_ids": None, "event_time_support_ids": None,
                "reason": "Independent lake identity, polygon, basin, grid, frame and temporal joins are unavailable."},
            "evidence": [dict(pilot_binding), dict(stage_binding)],
        })
    return {"schema_version": 1, "registry_id": "WP04-RETAINED-PILOT-REGISTRY-DRAFT-001",
        "status": "DEVELOPMENT_DRAFT_IDENTITY_AND_PHYSICAL_JOINS_UNRESOLVED",
        "sites": sites, "research_admissible": False}


def build_bundle(root: str | Path, output_dir: str | Path) -> Path:
    """Build a new local bundle from the accepted retained WP03 records only."""
    root = Path(root).resolve(strict=True)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    _require(output_dir.resolve(strict=False).is_relative_to(root), "output directory must stay inside repository root")
    _require(not output_dir.exists(), "WP04 output directory already exists; preserve prior attempts")

    pilot_rel = "project/wp03_pilot_manifest.json"
    stage_rel = f"{STAGE_ROOT}/stage_report.json"
    inventory_rel = f"{STAGE_ROOT}/outputs_sha256.json"
    decode_report_rel = f"{DECODE_ROOT}/stage_report.json"
    decode_inventory_rel = f"{DECODE_ROOT}/outputs_sha256.json"
    acceptance_rel = "project/contracts/WP03_optical_offline_acceptance.md"
    selection_rel = "project/architecture/wp03_2026-10-13/attempt_13_provider_pixels/selection.json"
    legacy_rel = "data/split_manifest.json"
    pilot = _read_json(root, pilot_rel)
    stage = _read_json(root, stage_rel)
    inventory = _read_json(root, inventory_rel)
    decode_report = _read_json(root, decode_report_rel)
    decode_inventory_doc = _read_json(root, decode_inventory_rel)
    _require(pilot.get("research_executable") is False, "pilot query manifest must remain non-executable")
    _require(stage.get("research_status") == "RESEARCH_NOT_READY" and stage.get("release_status") == "BLOCKED",
             "retained stage does not preserve the research/release block")
    _require(stage.get("mode") == "stage" and stage.get("research_admitted") is False,
             "retained stage is outside the authorized offline source scope")
    _require(stage.get("input_sha256", {}).get("manifest") == file_hash(root / pilot_rel),
             "stage run does not bind the current pilot manifest")
    _require(stage.get("input_sha256", {}).get("selection") == file_hash(root / selection_rel),
             "stage run does not bind the retained source selection bytes")
    inventory_entries = _validate_output_inventory(root, inventory_rel, inventory, stage)
    decode_inventory_entries = _validate_decode_output_inventory(root, decode_inventory_rel,
        decode_inventory_doc, decode_report, stage, inventory_rel)
    pilot_binding = _binding(root, pilot_rel, evidence_type="PI_DECLARED_QUERY_SITE_MANIFEST")
    stage_binding = _binding(root, stage_rel, evidence_type="RETAINED_PROVIDER_STAGE_REPORT")
    decode_report_binding = _binding(root, decode_report_rel, evidence_type="ACCEPTED_QUARANTINED_ROI_DECODE_REPORT")
    decode_inventory_binding = _binding(root, decode_inventory_rel,
        evidence_type="ROI_DECODE_OUTPUT_SHA256_INVENTORY")
    inputs = {
        "pilot_manifest": pilot_binding,
        "stage_report": stage_binding,
        "output_inventory": _binding(root, inventory_rel, evidence_type="LOCAL_BYTE_SHA256_INVENTORY"),
        "roi_decode_report": decode_report_binding,
        "roi_decode_inventory": decode_inventory_binding,
        "accepted_scope": _binding(root, acceptance_rel, evidence_type="ARCHITECT_ACCEPTED_DEVELOPMENT_SCOPE"),
        "selection_bytes": _binding(root, selection_rel, evidence_type="HASH_ONLY_SELECTION_INPUT"),
        "legacy_split": _binding(root, legacy_rel, evidence_type="QUARANTINED_LEGACY_NOT_PARSED", status="QUARANTINED_NOT_IMPORTED"),
    }
    pilot_sites = {site["site_id"]: site for site in pilot.get("sites", []) if isinstance(site, dict)}
    _require(set(pilot_sites) == set(SITE_IDS), "pilot manifest site identities are unsupported")
    source_items, asset_hashes = _make_source_items(stage, inventory_entries, stage_binding,
        pilot_sites, decode_report, decode_inventory_entries, decode_report_binding, decode_inventory_binding)
    failed = stage.get("sentinel1_asf_grd")
    _require(isinstance(failed, dict), "retained failed SAR attempt is absent")
    failed_receipt_rel = "asf/" + failed["receipt_path"]
    failed_receipt_path = "data/raw/wp03/attempt_2026-10-14_stage_001/" + failed_receipt_rel
    _require(failed_receipt_rel in inventory_entries and failed.get("received_bytes") == 0 and
             failed.get("partial_sha256") is None, "failed SAR retrieval record is inconsistent")
    failed_attempt = {
        "attempt_id": failed.get("selection_id"),
        "product_id": failed.get("fileID"),
        "site_ids": failed.get("site_joins"),
        "status": "FAILED_NO_MEASUREMENT_BYTES",
        "literal_error": failed.get("literal_error"),
        "received_bytes": 0,
        "source_bytes_sha256": None,
        "receipt_path": failed_receipt_path,
        "receipt_bytes": inventory_entries[failed_receipt_rel]["bytes"],
        "receipt_sha256": inventory_entries[failed_receipt_rel]["sha256"],
        "acquisition_start": failed.get("startTime"),
        "acquisition_end": failed.get("stopTime"),
        "request_started_at": failed.get("request_started_utc"),
        "request_finished_at": failed.get("request_finished_utc"),
        "catalogue_expected_bytes_not_received": failed.get("expected_bytes"),
        "admission": "ATTEMPT_EVIDENCE_ONLY_NO_OBSERVATION",
    }
    asset_hashes.append({"path": failed_receipt_path,
        "sha256": inventory_entries[failed_receipt_rel]["sha256"],
        "bytes": inventory_entries[failed_receipt_rel]["bytes"]})
    source_doc = {"schema_version": 2, "source_manifest_id": "WP04-RETAINED-SOURCES-DRAFT-002",
        "status": "SOURCE_BYTES_HASHED_MEASUREMENT_AND_RIGHTS_ADMISSION_BLOCKED",
        "source_items": source_items, "observations": [], "failed_attempts": [failed_attempt],
        "asset_hash_inventory": asset_hashes, "measurement_admission": "BLOCKED",
        "research_admissible": False}

    cutoff_local = _iso(pilot.get("pre_event_cutoff_exclusive"), "pre-event cutoff")
    cutoff_utc = parse_utc_timestamp(cutoff_local).isoformat()
    schedule_rows = []
    for site_id in SITE_IDS:
        site_sources = [item for item in source_items if item["site_id"] == site_id]
        exclusions = []
        for item in site_sources:
            generation = parse_utc_timestamp(item["time_axes"]["product_generation"]["at"])
            decision = parse_utc_timestamp(cutoff_utc)
            reason = "PRODUCT_GENERATED_AFTER_DECISION" if generation > decision else "HISTORICAL_AVAILABILITY_UNVERIFIED"
            exclusions.append({"source_id": item["source_id"], "reason": reason})
        for attempt in failed_attempts_for_site(failed_attempt, site_id):
            exclusions.append({"source_id": attempt["attempt_id"], "reason": "NO_RECEIVED_MEASUREMENT_BYTES"})
        schedule_rows.append({
            "schedule_row_id": f"ANCHOR-{site_id}-20231003",
            "row_kind": "PINNED_PRE_EVENT_CUTOFF_ANCHOR",
            "site_id": site_id,
            "scope": "development_validation",
            "decision_timestamp_original": cutoff_local,
            "decision_timestamp_utc": cutoff_utc,
            "context_window_days_candidate": None,
            "context_start": None,
            "context_end": None,
            "context_status": "UNRESOLVED",
            "observation_ids": [],
            "eligibility": None,
            "status": "NOT_ASSESSED",
            "missingness_reason": "CONTEXT_DURATION_AND_HISTORICAL_SOURCE_AVAILABILITY_NOT_ESTABLISHED",
            "source_exclusions": exclusions,
            "alarm_declaration": {"status": "NOT_RUN", "value": None, "reason": "No model or alarm task is admitted."},
        })
    schedule_doc = {"schema_version": 1, "schedule_id": "WP04-PINNED-CUTOFF-SCHEDULE-DRAFT-001",
        "cutoff": {"original": cutoff_local, "utc": cutoff_utc,
            "meaning": "STRICT_PRE_EVENT_SUPPORT_CUTOFF_EXCLUSIVE", "phase_specific_onset_asserted": False},
        "cadence_and_context": {"status": "UNRESOLVED", "context_window_days": None,
            "stride_days": None, "reason": "Development candidates remain 90/180/365 days and 1/5 days; pilot evidence does not select them."},
        "rows": schedule_rows,
        "exposure": {"calendar": {"status": "NOT_ASSESSED", "intervals": None, "union_seconds": None,
                "reason": "No reviewed operational schedule intervals exist."},
            "eligible": {"status": "NOT_ASSESSED", "intervals": None, "union_seconds": None,
                "reason": "No admitted eligible decision intervals exist."}},
        "research_admissible": False}

    candidate_rows = [{"lake_id": site_id, "centroid_lat": None, "centroid_lon": None,
                       "physical_support_status": "UNKNOWN", "physical_support": None} for site_id in SITE_IDS]
    scope_by_lake = {site_id: "development_validation" for site_id in SITE_IDS}
    gs = build_physical_sampling_graph(candidate_rows, scope_by_lake)
    gp = build_fitting_provenance_graph({}, scope_by_lake)
    memberships = {role: {"candidate_site_ids": [], "schedule_row_ids": [],
                           "source_ids": [], "context_ids": [],
                           "status": "EMPTY_DRAFT_NO_SCIENTIFIC_MEMBERSHIP"} for role in ROLES}
    memberships["development_validation"] = {
        "candidate_site_ids": list(SITE_IDS),
        "schedule_row_ids": [row["schedule_row_id"] for row in schedule_rows],
        "source_ids": [], "context_ids": [],
        "status": "EXPOSURE_ONLY_CUTOFF_ANCHORS_NO_ADMITTED_MEASUREMENTS",
    }
    scopes_doc = {"schema_version": 1, "scope_manifest_id": "WP04-FOUR-SCOPE-DRAFT-001",
        "roles": memberships, "graphs": {"G_S": gs, "G_P": gp},
        "role_separation": "EXPLICIT_FOUR_ROLE_KEYS_EMPTY_ROLES_ALLOWED_IN_NON_EXECUTABLE_DRAFT",
        "future_append_invariant": "Earlier source/context memberships are immutable; amendments require a new version.",
        "research_admissible": False}

    sections = {"registry": _site_registry(pilot, pilot_binding, stage_binding),
        "sources": source_doc, "schedule": schedule_doc, "scopes": scopes_doc}
    for name, section in sections.items():
        validate_schema(section, "wp04_" + name)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, section in sections.items():
        _write_json(output_dir / OUTPUT_FILES[name], section)
    section_bindings = {name: {"path": OUTPUT_FILES[name], "sha256": file_hash(output_dir / OUTPUT_FILES[name])}
                        for name in OUTPUT_FILES}
    bundle = {"schema_version": 2, "bundle_id": "WP04-OFFLINE-DEVELOPMENT-DRAFT-002",
        "status": "LOCAL_DEVELOPMENT_DRAFT_NOT_RESEARCH_ADMISSIBLE",
        "sections": section_bindings, "inputs": inputs,
        "research_status": "RESEARCH_NOT_READY", "plan_executable": False,
        "release_status": "BLOCKED", "research_admissible": False}
    validate_schema(bundle, "wp04_bundle")
    _write_json(output_dir / "bundle.v2.json", bundle)
    verification = verify_bundle(root, output_dir)
    _write_json(output_dir / "verification.json", verification)
    return output_dir


def failed_attempts_for_site(attempt: Mapping[str, Any], site_id: str) -> list[Mapping[str, Any]]:
    return [attempt] if site_id in attempt.get("site_ids", []) else []


def union_interval_seconds(intervals: list[Mapping[str, str]]) -> int:
    """Return independent UTC union duration; overlaps and adjacency count once."""
    spans = []
    for row in intervals:
        _require(isinstance(row, Mapping) and set(row) == {"start", "end"},
                 "exposure interval requires exact start/end timestamps")
        start, end = parse_utc_timestamp(row["start"]), parse_utc_timestamp(row["end"])
        _require(start < end, "exposure interval must have positive duration")
        spans.append((start, end))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(int((end - start).total_seconds()) for start, end in merged)


def _verify_exposure(record: Mapping[str, Any]) -> None:
    _require(set(record) == {"status", "intervals", "union_seconds", "reason"},
             "exposure record fields are incomplete")
    if record["status"] == "NOT_ASSESSED":
        _require(record["intervals"] is None and record["union_seconds"] is None and
                 isinstance(record["reason"], str) and bool(record["reason"].strip()),
                 "unassessed exposure must preserve null and a reason")
    elif record["status"] == "ASSESSED":
        _require(isinstance(record["intervals"], list), "assessed exposure requires interval rows")
        actual = union_interval_seconds(record["intervals"])
        _require(type(record["union_seconds"]) is int and record["union_seconds"] == actual and
                 record["reason"] is None, "exposure union does not independently recompute")
    else:
        raise WP04Error("unknown exposure status")


def verify_scope_partition(roles: Mapping[str, Mapping[str, Any]], schedule_rows: list[Mapping[str, Any]],
                          candidate_ids: set[str]) -> None:
    """Independently reject duplicate, overlapping, omitted, and orphan memberships."""
    _require(set(roles) == set(ROLES), "exactly four explicit scope roles are required")
    rows = {row["schedule_row_id"]: row for row in schedule_rows}
    _require(len(rows) == len(schedule_rows), "duplicate schedule row IDs")
    owners: dict[str, dict[str, str]] = {key: {} for key in
        ("candidate_site_ids", "schedule_row_ids", "source_ids", "context_ids")}
    for role, membership in roles.items():
        for field, owner in owners.items():
            values = membership[field]
            _require(len(values) == len(set(values)), f"duplicate {field} in {role}")
            for value in values:
                _require(value not in owner, f"{field} overlaps between {owner.get(value)} and {role}")
                owner[value] = role
    _require(set(owners["candidate_site_ids"]) == candidate_ids,
             "candidate site membership is incomplete or has an orphan")
    _require(set(owners["schedule_row_ids"]) == set(rows),
             "schedule membership is incomplete or has an orphan")
    _require(all(rows[row_id]["scope"] == role for row_id, role in owners["schedule_row_ids"].items()),
             "schedule role differs from explicit scope membership")
    _require(not owners["source_ids"] and not owners["context_ids"],
             "this offline draft admits no source or context membership")


def verify_bundle(root: str | Path, bundle_dir: str | Path) -> dict:
    """Recompute memberships, source bytes, chronology, cutoff, and exposure."""
    root = Path(root).resolve(strict=True)
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_absolute():
        bundle_dir = root / bundle_dir
    bundle_dir = bundle_dir.resolve(strict=True)
    _require(bundle_dir.is_relative_to(root), "bundle directory must remain inside repository root")
    bundle_path = bundle_dir / "bundle.v2.json"
    bundle = loads_strict(bundle_path.read_text(encoding="utf-8"))
    validate_schema(bundle, "wp04_bundle")
    _require(bundle["research_status"] == "RESEARCH_NOT_READY" and bundle["plan_executable"] is False and
             bundle["release_status"] == "BLOCKED" and bundle["research_admissible"] is False,
             "WP04 draft cannot change research/readiness admission")
    sections = {}
    for name, relative in OUTPUT_FILES.items():
        binding = bundle["sections"].get(name)
        _require(isinstance(binding, dict) and binding.get("path") == relative,
                 f"bundle section path mismatch: {name}")
        path = verified_file(bundle_dir, relative, binding.get("sha256", ""))
        sections[name] = loads_strict(path.read_text(encoding="utf-8"))
        validate_schema(sections[name], "wp04_" + name)
    inputs = bundle["inputs"]
    _require(set(inputs) == {"pilot_manifest", "stage_report", "output_inventory", "roi_decode_report",
                             "roi_decode_inventory", "accepted_scope", "selection_bytes", "legacy_split"},
             "bundle input set is incomplete")
    for name, row in inputs.items():
        _require(isinstance(row, dict) and set(row) == {"path", "sha256", "evidence_type", "status"},
                 f"invalid input binding: {name}")
        if name == "legacy_split":
            _require(row["status"] == "QUARANTINED_NOT_IMPORTED", "legacy split must remain quarantined")
        verified_file(root, row["path"], row["sha256"])

    pilot = _read_json(root, inputs["pilot_manifest"]["path"])
    stage = _read_json(root, inputs["stage_report"]["path"])
    output_inventory_doc = _read_json(root, inputs["output_inventory"]["path"])
    decode_report = _read_json(root, inputs["roi_decode_report"]["path"])
    decode_inventory_doc = _read_json(root, inputs["roi_decode_inventory"]["path"])
    _require(stage.get("input_sha256", {}).get("manifest") == inputs["pilot_manifest"]["sha256"] and
             stage.get("output_manifest_sha256") == inputs["output_inventory"]["sha256"],
             "retained stage is not chained to the declared pilot and output inventory")
    _require(file_hash(root / inputs["selection_bytes"]["path"]) ==
             stage.get("input_sha256", {}).get("selection"),
             "hash-only retained selection input differs from the stage binding")
    stage_inventory_entries = _validate_output_inventory(root, inputs["output_inventory"]["path"],
        output_inventory_doc, stage)
    decode_inventory_entries = _validate_decode_output_inventory(root, inputs["roi_decode_inventory"]["path"],
        decode_inventory_doc, decode_report, stage, inputs["output_inventory"]["path"])

    registry, sources, schedule, scopes = (sections[name] for name in ("registry", "sources", "schedule", "scopes"))
    registry_sites = {site["site_id"]: site for site in registry["sites"]}
    _require(set(registry_sites) == set(SITE_IDS), "registry candidate membership changed")
    pilot_sites = {site["site_id"]: site for site in pilot["sites"]}
    _require(set(pilot_sites) == set(SITE_IDS), "retained pilot manifest site identities changed")
    expected_source_items, _ = _make_source_items(stage, stage_inventory_entries,
        inputs["stage_report"], pilot_sites, decode_report, decode_inventory_entries,
        inputs["roi_decode_report"], inputs["roi_decode_inventory"])
    expected_source_by_site = {item["site_id"]: item for item in expected_source_items}
    for site_id, site in registry_sites.items():
        original = pilot_sites[site_id]
        _require(site["display_name"] == original["name"] and
                 site["query_point"]["latitude"] == original["latitude"] and
                 site["query_point"]["longitude"] == original["longitude"] and
                 site["query_point"]["status"] == "PI_QUERY_POINT_ONLY_NOT_LAKE_GEOMETRY_OR_GS_COORDINATE" and
                 site["candidate_lake_id"] is None and site["aliases"]["status"] == "UNRESOLVED" and
                 site["geometry"]["status"] == "UNAVAILABLE" and
                 site["event"]["classification_status"] == "UNVERIFIED" and
                 site["development_exposure"] == "DEVELOPMENT_EXPOSED",
                 "registry identity or uncertainty differs from the retained pilot source")
    _require(registry_sites["PILOT-KHANGCHUNG-TSHO"]["control_status"] == "NOT_A_CONTROL",
             "Khangchung must not be admitted as a no-event control")
    source_ids = {item["source_id"] for item in sources["source_items"]}
    _require(len(source_ids) == len(sources["source_items"]), "duplicate product source IDs")
    stage_c1 = stage.get("sentinel2_c1", {})
    _require(len(sources["source_items"]) == len(SITE_IDS), "source products are missing or multiplied")
    for item in sources["source_items"]:
        retained = stage_c1.get(item["site_id"])
        _require(isinstance(retained, dict) and
                 item["product_id"] == retained.get("item_id") and
                 item["product_uri"] == retained.get("product_uri") and
                 item["site_scope"] == retained.get("site_scope") == "DEVELOPMENT_EXPOSED" and
                 item["acquisition_support"]["start"] == retained.get("acquisition_datetime") and
                 item["time_axes"]["product_generation"]["at"] == retained.get("generation_time") and
                 item["historical_inspection"] == expected_source_by_site[item["site_id"]]["historical_inspection"] and
                 item["current_decode"] == expected_source_by_site[item["site_id"]]["current_decode"] and
                 item["site_join_status"] == expected_source_by_site[item["site_id"]]["site_join_status"] and
                 item["rights"]["status"] == "UNVERIFIED_SOURCE_SPECIFIC_TERMS" and
                 retained.get("research_admitted") is False,
                 "source identity, historical inspection, current decode join, or admission differs from retained evidence")
    inventory = {row["path"]: row for row in sources["asset_hash_inventory"]}
    _require(len(inventory) == len(sources["asset_hash_inventory"]), "duplicate source byte paths")
    for item in sources["source_items"]:
        _require(item["site_id"] in registry_sites, "source has an orphan site join")
        _require(item["site_scope"] == "DEVELOPMENT_EXPOSED", "pilot source exposure was lost")
        _require(item["availability"]["status"] == "UNVERIFIED", "historical availability is not admitted")
        acquisition = parse_utc_timestamp(item["acquisition_support"]["end"])
        generation = parse_utc_timestamp(item["time_axes"]["product_generation"]["at"])
        for asset in item["assets"]:
            for key, path_key, hash_key, size_key in (("source", "path", "sha256", "bytes"),
                    ("receipt", "receipt_path", "receipt_sha256", "receipt_bytes")):
                path = asset[path_key]
                inv = inventory.get(path)
                _require(inv is not None and inv["sha256"] == asset[hash_key] and inv["bytes"] == asset[size_key],
                         "source/receipt inventory differs from source manifest")
                verified = verified_file(root, path, asset[hash_key])
                _require(verified.stat().st_size == asset[size_key], "source or receipt byte count mismatch")
        _require(acquisition < generation, "retained optical product generation must remain distinct from acquisition")
    for attempt in sources["failed_attempts"]:
        receipt = inventory.get(attempt["receipt_path"])
        retained_attempt = stage.get("sentinel1_asf_grd", {})
        _require(attempt["status"] == "FAILED_NO_MEASUREMENT_BYTES" and
                 attempt["received_bytes"] == 0 and attempt["source_bytes_sha256"] is None and
                 receipt is not None and receipt["sha256"] == attempt["receipt_sha256"] and
                 receipt["bytes"] == attempt["receipt_bytes"] and
                 attempt["product_id"] == retained_attempt.get("fileID") and
                 attempt["literal_error"] == retained_attempt.get("literal_error") and
                 attempt["request_started_at"] == retained_attempt.get("request_started_utc") and
                 attempt["request_finished_at"] == retained_attempt.get("request_finished_utc") and
                 attempt["catalogue_expected_bytes_not_received"] == retained_attempt.get("expected_bytes"),
                 "failed retrieval receipt does not match its zero-byte attempt record")
        verified_file(root, attempt["receipt_path"], attempt["receipt_sha256"])
    _require(not sources["observations"] and sources["measurement_admission"] == "BLOCKED",
             "retained source bytes cannot become measurements in WP04")

    cutoff = schedule["cutoff"]
    _require(parse_utc_timestamp(cutoff["original"]) == parse_utc_timestamp(cutoff["utc"]),
             "original Asia/Kolkata cutoff and UTC conversion differ")
    _require(cutoff["original"] == pilot.get("pre_event_cutoff_exclusive") and
             cutoff["meaning"] == "STRICT_PRE_EVENT_SUPPORT_CUTOFF_EXCLUSIVE",
             "schedule cutoff differs from the retained pilot cutoff")
    rows_by_id = {row["schedule_row_id"]: row for row in schedule["rows"]}
    _require(len(rows_by_id) == len(schedule["rows"]), "duplicate schedule row IDs")
    _require({row["site_id"] for row in schedule["rows"]} == set(SITE_IDS) and len(rows_by_id) == 2,
             "pinned cutoff schedule must retain one candidate row per pilot site")
    for row in schedule["rows"]:
        _require(row["site_id"] in registry_sites and row["scope"] == "development_validation",
                 "schedule row has orphan or unauthorized scope membership")
        _require(row["decision_timestamp_original"] == cutoff["original"] and
                 row["decision_timestamp_utc"] == cutoff["utc"], "schedule row altered the pinned cutoff")
        _require(row["status"] == "NOT_ASSESSED" and row["eligibility"] is None and
                 row["context_start"] is None and row["context_end"] is None and
                 row["observation_ids"] == [], "unresolved candidate row must retain null, not fabricated context/result")
        _require(row["context_status"] == "UNRESOLVED" and row["missingness_reason"],
                 "candidate context uncertainty is not recorded")
        _require(not row["observation_ids"], "unverified source cannot enter schedule observations")
        expected_exclusions = {}
        for item in sources["source_items"]:
            if item["site_id"] != row["site_id"]:
                continue
            generated = parse_utc_timestamp(item["time_axes"]["product_generation"]["at"])
            decision = parse_utc_timestamp(row["decision_timestamp_utc"])
            expected_exclusions[item["source_id"]] = (
                "PRODUCT_GENERATED_AFTER_DECISION" if generated > decision
                else "HISTORICAL_AVAILABILITY_UNVERIFIED")
        for attempt in sources["failed_attempts"]:
            if row["site_id"] in attempt["site_ids"]:
                expected_exclusions[attempt["attempt_id"]] = "NO_RECEIVED_MEASUREMENT_BYTES"
        actual_exclusions = {entry["source_id"]: entry["reason"] for entry in row["source_exclusions"]}
        _require(actual_exclusions == expected_exclusions,
                 "candidate schedule exclusions are incomplete, orphaned, or misclassified")
    for key in ("calendar", "eligible"):
        _verify_exposure(schedule["exposure"][key])
    roles = scopes["roles"]
    _require(set(roles) == set(ROLES), "exactly four explicit scope roles are required")
    verify_scope_partition(roles, schedule["rows"], set(SITE_IDS))
    _require(roles["development_validation"]["candidate_site_ids"] == list(SITE_IDS),
             "pilot exposure candidates must retain their declared source order")
    _require(scopes["graphs"]["G_S"]["graph_id"] == "G_S" and
             scopes["graphs"]["G_P"]["graph_id"] == "G_P", "physical/provenance graph identity mismatch")
    _require(scopes["graphs"]["G_S"]["role_isolation_status"] == "UNRESOLVED_UNKNOWN_PHYSICAL_MEMBERSHIP",
             "unknown physical joins must remain unresolved")
    _require(all(node["physical_support_status"] == "UNKNOWN" and node["coordinate_status"] == "UNKNOWN"
                 for node in scopes["graphs"]["G_S"]["nodes"]),
             "unverified query points or physical fields cannot certify G_S")
    _require(scopes["graphs"]["G_P"]["status"] == "PROVENANCE_UNRESOLVED",
             "no fitted provenance may be implied")
    output_root = f"{STAGE_ROOT}/"
    expected_inventory_paths = {output_root + record["path"] for record in output_inventory_doc.get("files", [])}
    _require(set(inventory) == expected_inventory_paths,
             "WP04 source inventory must retain every and only the files in the WP03 output hash manifest")
    expected_decode_paths = {
        f"{DECODE_ROOT}/{record['path']}" for record in decode_inventory_doc.get("files", [])
    }
    actual_decode_paths = {
        row["path"] for item in sources["source_items"]
        for row in item["current_decode"]["decoded_output_bindings"]
    }
    _require(actual_decode_paths == expected_decode_paths,
             "WP04 current decode bindings must retain every and only inventoried ROI output")
    return {"schema_version": 1, "status": "PASS_INTEGRITY_LOCAL_DEVELOPMENT_DRAFT_ONLY",
        "verified_sections": list(OUTPUT_FILES), "verified_source_and_receipt_files": len(inventory),
        "verified_decoded_output_files": len(decode_inventory_entries),
        "verified_historical_failed_inspections": len(SITE_IDS),
        "verified_current_decode_products": len(sources["source_items"]),
        "verified_product_records": len(source_ids), "schedule_rows": len(rows_by_id),
        "scope_roles": list(ROLES), "source_membership_count": 0, "context_membership_count": 0,
        "G_S": "UNRESOLVED", "G_P": "UNRESOLVED", "research_admissible": False,
        "research_status": "RESEARCH_NOT_READY", "plan_executable": False, "release_status": "BLOCKED"}
