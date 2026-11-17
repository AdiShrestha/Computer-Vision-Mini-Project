"""Validate retained WP03 inputs or stage a bounded, quarantined acquisition.

Every path, source binding, and output directory is supplied on the command line.
Validation mode is offline and never invokes a provider adapter. Staging mode is
kept separate so a future, separately authorized run can use the same checked
input wiring without editing this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from sentinel_gl.admission import AdmissionError  # noqa: E402
from sentinel_gl.integrity import loads_strict  # noqa: E402
from sentinel_gl.ingestion import asf_grd, earth_search_c1  # noqa: E402
from sentinel_gl.ingestion.asf_grd import inspect_safe_roles, stage_feature  # noqa: E402
from sentinel_gl.ingestion.earth_search_c1 import (  # noqa: E402
    decode_c1_bundle,
    stage_asset,
    validate_item,
    validate_native_metadata,
)

MAX_PRODUCTS = 4
MAX_PRODUCT_BYTES = 2 * 1024**3
AUTHORIZED_RECEIVED_BYTE_CAP = 1_600_000_000
REQUIRED_C1_ROLES = tuple(earth_search_c1.ASSET_ROLES)
INTEGRITY_FAILURES = {
    "GET_ETAG_DIFFERS_FROM_RETAINED_HEAD",
    "LENGTH_OR_SHA256_MISMATCH",
}


class PreflightError(ValueError):
    """A retained input failed its declared identity or scope checks."""


class OutputBindingError(AdmissionError):
    """Staged bytes are absent or differ from their transfer receipt."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime() -> dict:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "runner": "run_wp03_alternate_access.py",
    }


def _safe_error(exc: BaseException) -> str:
    value = str(exc)
    value = re.sub(r"(https?://)[^/@\s:]+:[^/@\s]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(https?://[^\s?#]+)\?[^\s#]*", r"\1?[REDACTED]", value)
    value = re.sub(
        r"(?i)(authorization|cookie|token|password|credential)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2[REDACTED]", value)
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    value = loads_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"JSON input is not an object: {path.name}")
    return value


def _resolve_file(repository_root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreflightError(f"{label} path is missing")
    unresolved = repository_root / relative
    if unresolved.is_symlink():
        raise PreflightError(f"{label} must not be a symbolic link")
    candidate = unresolved.resolve(strict=True)
    root = repository_root.resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
        raise PreflightError(f"{label} must be a regular file inside repository root")
    return candidate


def _resolve_source(source_root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreflightError(f"{label} path is missing")
    unresolved = source_root / relative
    if unresolved.is_symlink():
        raise PreflightError(f"{label} must not be a symbolic link")
    candidate = unresolved.resolve(strict=True)
    root = source_root.resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
        raise PreflightError(f"{label} must be a regular file inside the retained source root")
    return candidate


def _assert_hash(path: Path, expected: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
        raise PreflightError(f"{label} SHA-256 binding is malformed")
    actual = _sha256(path)
    if actual != expected:
        raise PreflightError(f"{label} SHA-256 differs from its private input binding")
    return actual


def _utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PreflightError(f"{label} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise PreflightError(f"{label} lacks an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _source_inventory(repository_root: Path, inventory: dict) -> tuple[Path, dict[str, str]]:
    if type(inventory.get("schema_version")) is not int or inventory["schema_version"] != 1:
        raise PreflightError("private source inventory schema version is unsupported")
    source_root_rel = inventory.get("source_root")
    if not isinstance(source_root_rel, str):
        raise PreflightError("private source inventory lacks source_root")
    source_root = (repository_root / source_root_rel).resolve(strict=True)
    if not source_root.is_dir() or not source_root.is_relative_to(repository_root.resolve()):
        raise PreflightError("retained source root must be a directory inside repository root")
    records = inventory.get("source_files")
    if not isinstance(records, list) or not records:
        raise PreflightError("private source inventory has no immutable source-file membership")
    verified = {}
    for record in records:
        if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
            raise PreflightError("source inventory entry must bind path and SHA-256")
        relative = record["path"]
        if relative in verified:
            raise PreflightError(f"duplicate source inventory path: {relative}")
        path = _resolve_source(source_root, relative, label="retained source")
        verified[relative] = _assert_hash(path, record["sha256"], f"retained source {relative}")
    return source_root, verified


def _require_source(relative: str, verified: dict[str, str], label: str) -> None:
    if relative not in verified:
        raise PreflightError(f"{label} is outside the pinned source inventory")


def _head_map(evidence: dict) -> dict[tuple[str, str, str], dict]:
    requests = evidence.get("requests")
    if not isinstance(requests, list):
        raise PreflightError("retained asset summary has no request list")
    heads = {}
    for entry in requests:
        if entry.get("method") != "HEAD":
            continue
        key = (entry.get("site_id"), entry.get("item_id"), entry.get("role"))
        if key in heads:
            raise PreflightError(f"retained HEAD identity is duplicated: {key}")
        heads[key] = entry
    return heads


def _validate_c1(repository_root: Path, source_root: Path, inventory_paths: dict[str, str],
                 evidence: dict, manifest: dict, selection: dict) -> dict:
    sites = manifest.get("sites")
    selected_sites = selection.get("sentinel2_c1")
    if not isinstance(sites, list) or not isinstance(selected_sites, dict):
        raise PreflightError("pilot sites or selected C1 site records are absent")
    site_by_id = {site.get("site_id"): site for site in sites if isinstance(site, dict)}
    if not site_by_id or set(selected_sites) != set(site_by_id):
        raise PreflightError("selected C1 site membership differs from the PI manifest")
    start, end, cutoff = (manifest.get("acquisition_start"),
                          manifest.get("acquisition_end_exclusive"),
                          manifest.get("pre_event_cutoff_exclusive"))
    _utc(start, "PI acquisition_start")
    _utc(end, "PI acquisition_end_exclusive")
    _utc(cutoff, "PI pre_event_cutoff_exclusive")
    heads = _head_map(evidence)
    validated = {}
    products = {}

    for site_id, selected in selected_sites.items():
        site = site_by_id[site_id]
        if site.get("scope") != "DEVELOPMENT_EXPOSED" or selected.get("site_scope") != "DEVELOPMENT_EXPOSED":
            raise PreflightError(f"C1 site scope is not development-exposed: {site_id}")
        if selected.get("site_join") != [site_id]:
            raise PreflightError(f"C1 site join is missing, ambiguous, or cross-site: {site_id}")
        item_rel = selected.get("item_path")
        _require_source(item_rel, inventory_paths, f"C1 item for {site_id}")
        item_path = _resolve_source(source_root, item_rel, label="retained C1 item")
        response = _load_json(item_path)
        matches = [item for item in response.get("features", [])
                   if item.get("id") == selected.get("item_id")]
        if len(matches) != 1:
            raise PreflightError(f"retained C1 item match count is {len(matches)} for {site_id}")
        item = validate_item(matches[0], site_id, cutoff,
                             acquisition_start=start, acquisition_end_exclusive=end,
                             selection_record=selected)

        metadata_paths = {}
        for role, field in (("product_metadata", "product_metadata_path"),
                            ("granule_metadata", "granule_metadata_path")):
            relative = selected.get(field)
            _require_source(relative, inventory_paths, f"C1 {role} for {site_id}")
            path = _resolve_source(source_root, relative, label=f"C1 {role}")
            entries = [row for row in evidence.get("requests", [])
                       if row.get("site_id") == site_id and row.get("item_id") == selected["item_id"]
                       and row.get("role") == role and row.get("method") == "GET"]
            if (len(entries) != 1 or entries[0].get("http_status") != 200 or
                    entries[0].get("path") != relative or
                    entries[0].get("bytes") != path.stat().st_size or
                    entries[0].get("sha256") != inventory_paths[relative]):
                raise PreflightError(f"retained native {role} receipt does not bind bytes for {site_id}")
            metadata_paths[role] = path
        native = validate_native_metadata(
            item, metadata_paths["product_metadata"], metadata_paths["granule_metadata"],
            acquisition_start=start, acquisition_end_exclusive=end, cutoff=cutoff)

        assets = selected.get("assets")
        if not isinstance(assets, dict) or set(assets) != set(REQUIRED_C1_ROLES):
            raise PreflightError(f"selected C1 role set is incomplete or ambiguous for {site_id}")
        product_uri = selected.get("product_uri")
        if not isinstance(product_uri, str):
            raise PreflightError(f"selected C1 product URI is absent for {site_id}")
        product_group = products.setdefault(product_uri, {})
        for role in REQUIRED_C1_ROLES:
            asset = item["assets"][role]
            selected_asset = assets[role]
            if (selected_asset.get("href") != asset.get("href") or
                    selected_asset.get("file_checksum_multihash") != asset.get("file:checksum") or
                    selected_asset.get("file_size") != asset.get("file:size") or
                    selected_asset.get("band") != earth_search_c1.ASSET_ROLES[role]):
                raise PreflightError(f"C1 {role} selection differs from retained item for {site_id}")
            key = (site_id, selected["item_id"], role)
            head = heads.get(key)
            embedded_head = selected_asset.get("head")
            if (head is None or embedded_head != head or head.get("http_status") != 200 or
                    head.get("url") != asset.get("href") or
                    int(head.get("response_headers", {}).get("Content-Length", -1)) != asset["file:size"]):
                raise PreflightError(f"retained C1 HEAD does not bind item/site/role asset for {site_id}/{role}")
            etag = head.get("response_headers", {}).get("ETag")
            if not isinstance(etag, str) or not etag:
                raise PreflightError(f"retained C1 HEAD lacks opaque ETag identity for {site_id}/{role}")
            href = asset["href"]
            if href in product_group and product_group[href] != asset["file:size"]:
                raise PreflightError(f"one C1 product asset has conflicting byte sizes: {href}")
            product_group[href] = asset["file:size"]
        validated[site_id] = {
            "site_scope": site["scope"],
            "site_join": selected["site_join"],
            "item": item,
            "selection": selected,
            "native_metadata": native,
            "product_metadata_path": metadata_paths["product_metadata"],
            "granule_metadata_path": metadata_paths["granule_metadata"],
        }
    return {"sites": validated, "products": products, "heads": heads}


def _validate_asf(source_root: Path, inventory_paths: dict[str, str], manifest: dict,
                  selection: dict) -> dict:
    selected = selection.get("sentinel1_asf_grd")
    if not isinstance(selected, dict):
        raise PreflightError("selected ASF product record is absent")
    site_ids = [site["site_id"] for site in manifest.get("sites", [])]
    if (selected.get("site_scope") != "DEVELOPMENT_EXPOSED" or
            set(selected.get("site_join", [])) != set(site_ids) or
            set(selected.get("query_files", {})) != set(site_ids)):
        raise PreflightError("ASF site scope/query membership does not match declared development sites")
    selected_feature = None
    for site_id in site_ids:
        relative = selected["query_files"][site_id]
        _require_source(relative, inventory_paths, f"ASF query response for {site_id}")
        response = _load_json(_resolve_source(source_root, relative, label="retained ASF response"))
        matches = [feature for feature in response.get("features", [])
                   if feature.get("properties", {}).get("fileID") == selected.get("fileID")]
        if len(matches) != 1:
            raise PreflightError(f"retained ASF feature match count is {len(matches)} for {site_id}")
        feature = matches[0]
        properties = asf_grd._validate_feature(feature)
        if selected_feature is None:
            selected_feature = feature
        else:
            prior = selected_feature["properties"]
            for field in ("fileID", "sceneName", "fileName", "url", "bytes", "md5sum",
                          "startTime", "stopTime", "polarization"):
                if properties.get(field) != prior.get(field):
                    raise PreflightError("ASF site queries do not identify the same complete product")
    props = selected_feature["properties"]
    for field in ("fileID", "sceneName", "fileName", "url", "bytes", "md5sum",
                  "startTime", "stopTime", "polarization"):
        if selected.get(field) != props.get(field):
            raise PreflightError(f"ASF selected {field} differs from retained feature")
    start = _utc(manifest["acquisition_start"], "PI acquisition_start")
    end = _utc(manifest["acquisition_end_exclusive"], "PI acquisition_end_exclusive")
    cutoff = _utc(manifest["pre_event_cutoff_exclusive"], "PI pre_event_cutoff_exclusive")
    scene_start, scene_stop = _utc(props["startTime"], "ASF startTime"), _utc(props["stopTime"], "ASF stopTime")
    if scene_stop < scene_start or scene_start < start or scene_stop >= end or scene_stop >= cutoff:
        raise PreflightError("retained ASF sensing support falls outside the PI window or strict cutoff")
    return {"feature": selected_feature, "properties": props}


def preflight(repository_root: Path, manifest_path: Path, selection_path: Path,
              evidence_path: Path, inventory_path: Path) -> dict:
    inventory = _load_json(inventory_path)
    inventory_hash = _sha256(inventory_path)
    expected_inputs = inventory.get("bound_inputs")
    if not isinstance(expected_inputs, dict):
        raise PreflightError("private source inventory lacks bound_inputs")
    inputs = {"manifest": manifest_path, "selection": selection_path, "evidence": evidence_path}
    hashes = {}
    for key, path in inputs.items():
        binding = expected_inputs.get(key)
        if not isinstance(binding, dict):
            raise PreflightError(f"private input binding is absent: {key}")
        expected_path = _resolve_file(repository_root, binding.get("path"), label=f"{key} input")
        unresolved = path if path.is_absolute() else repository_root / path
        if unresolved.is_symlink():
            raise PreflightError(f"CLI {key} input must not be a symbolic link")
        actual_path = unresolved.resolve(strict=True)
        if actual_path != expected_path:
            raise PreflightError(f"CLI {key} path differs from its private input binding")
        hashes[key] = _assert_hash(actual_path, binding.get("sha256"), key)
    source_root, inventory_paths = _source_inventory(repository_root, inventory)
    manifest = _load_json(manifest_path)
    selection = _load_json(selection_path)
    evidence = _load_json(evidence_path)
    if manifest.get("research_executable") is not False:
        raise PreflightError("PI manifest is not explicitly non-executable")
    if selection.get("manifest_path") != expected_inputs["manifest"]["path"] or selection.get(
            "manifest_sha256") != hashes["manifest"]:
        raise PreflightError("selected acquisition does not bind the unchanged PI manifest")
    c1 = _validate_c1(repository_root, source_root, inventory_paths, evidence, manifest, selection)
    asf = _validate_asf(source_root, inventory_paths, manifest, selection)
    products = {uri: sum(group.values()) for uri, group in c1["products"].items()}
    products[asf["properties"]["fileID"]] = asf["properties"]["bytes"]
    declared_product_limit = selection.get("product_limit")
    product_limit = selection.get("per_product_byte_limit")
    if (type(declared_product_limit) is not int or
            not 1 <= declared_product_limit <= MAX_PRODUCTS):
        raise PreflightError(f"selected logical product limit must be between 1 and {MAX_PRODUCTS}")
    if type(product_limit) is not int or not 0 < product_limit <= MAX_PRODUCT_BYTES:
        raise PreflightError("selected per-product byte limit must be positive and no greater than 2 GiB")
    if len(products) > declared_product_limit:
        raise PreflightError("selected logical product count exceeds its declared product limit")
    declared_count = selection.get("logical_product_count")
    if type(declared_count) is not int or declared_count != len(products) or declared_count > MAX_PRODUCTS:
        raise PreflightError("declared logical product count differs from recomputed unique products")
    oversized = {name: size for name, size in products.items() if size > product_limit}
    if oversized:
        raise PreflightError(f"grouped logical product bytes exceed the selected bound: {oversized}")
    return {
        "manifest": manifest,
        "selection": selection,
        "evidence": evidence,
        "inventory": inventory,
        "source_root": source_root,
        "inventory_paths": inventory_paths,
        "c1": c1,
        "asf": asf,
        "input_sha256": {**hashes, "source_inventory": inventory_hash},
        "logical_product_bytes": products,
        "logical_product_count": len(products),
    }


def _read_adapter_receipt(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = _load_json(path)
    except Exception:
        return None
    value["receipt_path"] = path.name
    return value


def _record_output(output_dir: Path, path: Path, receipt: dict) -> None:
    if not path.is_file() or path.is_symlink():
        raise OutputBindingError("adapter marked output complete but no regular output file exists")
    actual_size, actual_hash = path.stat().st_size, _sha256(path)
    if receipt.get("output_bytes") != actual_size or receipt.get("output_sha256") != actual_hash:
        raise OutputBindingError(
            "adapter output differs from its byte/hash receipt "
            f"(expected_bytes={receipt.get('output_bytes')!r}, actual_bytes={actual_size}, "
            f"expected_sha256={receipt.get('output_sha256')!r}, actual_sha256={actual_hash})")
    receipt["verified_output_path"] = path.relative_to(output_dir).as_posix()
    receipt["verified_output_bytes"] = actual_size
    receipt["verified_output_sha256"] = actual_hash


def _output_binding_failure(receipt: dict | None, path: Path, exc: OutputBindingError) -> dict:
    retained = dict(receipt or {})
    output_exists = path.is_file() and not path.is_symlink()
    actual_bytes = path.stat().st_size if output_exists else None
    actual_hash = _sha256(path) if output_exists else None
    return {
        "status": "OUTPUT_BINDING_FAILED",
        "output_binding_status": "MISSING_OR_CHANGED",
        "transfer_receipt_status": retained.get("status"),
        "transfer_receipt": retained,
        "expected_output_bytes": retained.get("output_bytes"),
        "expected_output_sha256": retained.get("output_sha256"),
        "actual_output_bytes": actual_bytes,
        "actual_output_sha256": actual_hash,
        "literal_error": _safe_error(exc),
        "cause_status": "OUTPUT_BYTES_DO_NOT_MATCH_RETAINED_TRANSFER_RECEIPT",
        "underlying_cause_status": "UNKNOWN",
    }


def _c1_site_complete(site: dict) -> bool:
    inspection = site.get("pixel_inspection", {})
    accepted_inspection = {
        "AUTHENTIC_PROVIDER_COGS_INSPECTED_NOT_RESEARCH_ADMITTED",
        "CONSTRUCTED_FIXTURE_INSPECTED_NOT_RESEARCH_ADMITTED",
    }
    return (
        set(site.get("assets", {})) == set(REQUIRED_C1_ROLES)
        and all(record.get("status") == "COMPLETE_HASH_VERIFIED"
                for record in site.get("assets", {}).values())
        and inspection.get("status") in accepted_inspection
        and inspection.get("research_admissible") is False
        and inspection.get("origin") in {"provider_download", "constructed_fixture"})


def _asf_complete(receipt: dict | None) -> bool:
    safe = (receipt or {}).get("safe_roles", {})
    return (
        (receipt or {}).get("status") == "COMPLETE_ARCHIVE_BYTES_VERIFIED"
        and (receipt or {}).get("verified_output_bytes") == (receipt or {}).get("output_bytes")
        and (receipt or {}).get("verified_output_sha256") == (receipt or {}).get("output_sha256")
        and isinstance((receipt or {}).get("verified_output_path"), str)
        and bool((receipt or {}).get("verified_output_path"))
        and safe.get("archive_crc_status") == "ALL_ZIP_MEMBER_CRC_CHECKS_PASSED"
        and safe.get("manifest_identity_status") ==
            "SCOPED_MISSION_PRODUCT_ORBIT_TIME_POLARIZATION_AND_MEMBER_JOIN_MATCHED"
        and safe.get("annotation_identity_status") == "MISSION_PRODUCT_TIME_POLARIZATION_MATCHED"
        and safe.get("role_mapping_status") == "SAFE_SCOPED_MEMBER_AND_ROLE_IDENTITIES_MATCHED"
        and safe.get("research_admitted") is False)


def _stage(prepared: dict, output_dir: Path) -> tuple[dict, int]:
    selection = prepared["selection"]
    manifest = prepared["manifest"]
    report = {
        "status": "IN_PROGRESS",
        "started_utc": _now(),
        "runtime": _runtime(),
        "mode": "stage",
        "research_admitted": False,
        "research_status": "RESEARCH_NOT_READY",
        "release_status": "BLOCKED",
        "pilot_manifest_mutated": False,
        "selection_sha256": prepared["input_sha256"]["selection"],
        "input_sha256": prepared["input_sha256"],
        "logical_product_count": prepared["logical_product_count"],
        "logical_product_bytes": prepared["logical_product_bytes"],
        "site_exposure": {site["site_id"]: site["scope"] for site in manifest["sites"]},
        "sentinel1_asf_grd": None,
        "sentinel2_c1": {},
        "failures": [],
        "provider_adapters_invoked": True,
        "network_call_status": "RECORDED_PER_ADAPTER_RECEIPT",
    }
    optical_consecutive_failures = 0
    stopped = False

    asf_selection = selection["sentinel1_asf_grd"]
    asf_target = output_dir / "asf" / asf_selection["fileName"]
    try:
        asf_receipt = stage_feature(
            prepared["asf"]["feature"], asf_selection["site_join"], asf_target,
            context={"selection_id": selection["attempt"],
                     "selection_sha256": report["selection_sha256"],
                     "site_scope_id": "|".join(asf_selection["site_join"])})
        report["sentinel1_asf_grd"] = asf_receipt
        if asf_receipt.get("status") == "COMPLETE_ARCHIVE_BYTES_VERIFIED":
            _record_output(output_dir, asf_target, asf_receipt)
            safe = inspect_safe_roles(asf_target, asf_selection["sceneName"], "VV")
            _record_output(output_dir, asf_target, asf_receipt)
            asf_receipt["safe_roles"] = safe
    except Exception as exc:
        receipt = (_read_adapter_receipt(asf_target.with_name(asf_target.name + ".download.json"))
                   or report.get("sentinel1_asf_grd"))
        if isinstance(exc, OutputBindingError):
            report["sentinel1_asf_grd"] = _output_binding_failure(receipt, asf_target, exc)
        else:
            report["sentinel1_asf_grd"] = receipt or {
            "status": "ONE_SHOT_ATTEMPT_FAILED",
            "error_type": type(exc).__name__,
            "literal_error": _safe_error(exc),
            }
        report["failures"].append({"branch": "ASF_GRD",
            "status": report["sentinel1_asf_grd"].get("status"),
            "error_type": type(exc).__name__, "literal_error": _safe_error(exc),
            "cause_status": ("OUTPUT_BYTES_DO_NOT_MATCH_RETAINED_TRANSFER_RECEIPT"
                             if isinstance(exc, OutputBindingError) else
                             "UNKNOWN_UNLESS_RETAINED_RECEIPT_ESTABLISHES_CAUSE")})

    for site_id, prepared_site in prepared["c1"]["sites"].items():
        selected = prepared_site["selection"]
        item = prepared_site["item"]
        asset_results = {}
        asset_paths = {}
        for role in REQUIRED_C1_ROLES:
            asset = item["assets"][role]
            selected_asset = selected["assets"][role]
            head = prepared["c1"]["heads"][(site_id, selected["item_id"], role)]
            target = output_dir / "c1" / site_id / f"{role}.tif"
            context = {"selection_id": selection["attempt"],
                       "selection_sha256": report["selection_sha256"],
                       "product_id": selected["item_id"], "site_id": site_id,
                       "site_scope_id": selected["site_scope"], "asset_role": role}
            try:
                receipt = stage_asset(asset, head, target, context=context)
                if receipt.get("status") == "COMPLETE_HASH_VERIFIED":
                    _record_output(output_dir, target, receipt)
                    asset_paths[role] = target
                    optical_consecutive_failures = 0
                else:
                    optical_consecutive_failures += 1
                    report["failures"].append({
                        "branch": "C1_ASSET", "site_id": site_id, "role": role,
                        "status": receipt.get("status", "MISSING_STATUS"),
                        "literal_error": _safe_error(RuntimeError(
                            receipt.get("literal_error", "asset adapter returned a non-complete status"))),
                        "cause_status": "RECEIPT_STATUS_REQUIRES_REVIEW"})
                asset_results[role] = receipt
            except Exception as exc:
                receipt = _read_adapter_receipt(target.with_name(target.name + ".receipt.json"))
                if isinstance(exc, OutputBindingError):
                    receipt = _output_binding_failure(receipt, target, exc)
                if receipt is None:
                    receipt = {"status": "ATTEMPT_FAILED_NO_RECEIPT",
                               "error_type": type(exc).__name__,
                               "literal_error": _safe_error(exc)}
                receipt.setdefault("literal_error", _safe_error(exc))
                receipt.setdefault("error_type", type(exc).__name__)
                asset_results[role] = receipt
                report["failures"].append({
                    "branch": "C1_ASSET", "site_id": site_id, "role": role,
                    "status": receipt.get("status"), "error_type": type(exc).__name__,
                    "literal_error": _safe_error(exc),
                    "cause_status": "UNKNOWN_UNLESS_RETAINED_RECEIPT_ESTABLISHES_CAUSE"})
                optical_consecutive_failures += 1
            if optical_consecutive_failures >= 2:
                stopped = True
                report["status"] = "STOPPED_AFTER_TWO_CONSECUTIVE_C1_FAILURES"
                break

        site_report = {
            "item_id": selected["item_id"],
            "product_uri": selected["product_uri"],
            "acquisition_datetime": selected["acquisition_datetime"],
            "generation_time": selected["generation_time"],
            "site_scope": selected["site_scope"],
            "site_join": selected["site_join"],
            "assets": asset_results,
            "collection_packaging": "DERIVED_COG_BUNDLE_NOT_ORIGINAL_SAFE_OR_JP2",
            "operational_availability": "UNVERIFIED_REPROCESSED_IN_2025",
            "research_admitted": False,
        }
        if len(asset_paths) == len(REQUIRED_C1_ROLES):
            try:
                inspected = decode_c1_bundle(
                    item, site_id, prepared_site["product_metadata_path"],
                    prepared_site["granule_metadata_path"], asset_paths,
                    manifest["pre_event_cutoff_exclusive"],
                    acquisition_start=manifest["acquisition_start"],
                    acquisition_end_exclusive=manifest["acquisition_end_exclusive"],
                    selection_record=selected)
                if (not isinstance(inspected, dict) or inspected.get("research_admissible") is not False or
                        inspected.get("origin") not in {"provider_download", "constructed_fixture"}):
                    raise AdmissionError("C1 inspection did not return an explicit quarantined origin/status")
                _write_json(output_dir / "c1" / site_id / "verified_windows.json", inspected)
                site_report["pixel_inspection"] = inspected
            except Exception as exc:
                site_report["pixel_inspection"] = {
                    "status": "INSPECTION_FAILED", "error_type": type(exc).__name__,
                    "literal_error": _safe_error(exc), "research_admitted": False}
                report["failures"].append({"branch": "C1_PIXEL_INSPECTION", "site_id": site_id,
                    "error_type": type(exc).__name__, "literal_error": _safe_error(exc),
                    "cause_status": "UNKNOWN_UNLESS_RETAINED_RECEIPT_ESTABLISHES_CAUSE"})
        else:
            site_report["pixel_inspection"] = {
                "status": "NOT_RUN_INCOMPLETE_ASSET_SET",
                "staged_asset_count": len(asset_paths),
                "required_asset_count": len(REQUIRED_C1_ROLES),
                "research_admitted": False}
        report["sentinel2_c1"][site_id] = site_report
        if stopped:
            break

    current_asf_receipt = report.get("sentinel1_asf_grd") or {}
    if current_asf_receipt.get("status") == "COMPLETE_ARCHIVE_BYTES_VERIFIED":
        try:
            _record_output(output_dir, asf_target, current_asf_receipt)
        except OutputBindingError as exc:
            report["sentinel1_asf_grd"] = _output_binding_failure(current_asf_receipt, asf_target, exc)
            report["failures"].append({"branch": "ASF_GRD_FINAL_OUTPUT_CHECK",
                "status": "OUTPUT_BINDING_FAILED", "error_type": type(exc).__name__,
                "literal_error": _safe_error(exc),
                "cause_status": "OUTPUT_BYTES_DO_NOT_MATCH_RETAINED_TRANSFER_RECEIPT"})

    optical_complete = (
        len(report["sentinel2_c1"]) == len(prepared["c1"]["sites"])
        and all(_c1_site_complete(site) for site in report["sentinel2_c1"].values()))
    asf_complete = _asf_complete(report.get("sentinel1_asf_grd"))
    if not stopped:
        report["status"] = ("COMPLETE_STAGED_QUARANTINED" if optical_complete and asf_complete
                            else "PARTIAL_MODALITY_OR_STAGE")
    report["modality_status"] = {
        "sentinel1_asf_grd": "COMPLETE" if asf_complete else "INCOMPLETE_OR_UNAVAILABLE",
        "sentinel2_c1": "COMPLETE" if optical_complete else "INCOMPLETE_OR_UNAVAILABLE",
    }
    report["finished_utc"] = _now()
    report["quarantine_status"] = "PROVIDER_BYTES_AND_INSPECTION_RETAINED_NOT_RESEARCH_ADMITTED"
    return report, (0 if report["status"] == "COMPLETE_STAGED_QUARANTINED" else 2)


def _output_hash_manifest(output_dir: Path) -> dict:
    entries = []
    excluded = {"stage_report.json", "outputs_sha256.json"}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name in excluded or path.name.endswith(".tmp"):
            continue
        entries.append({"path": path.relative_to(output_dir).as_posix(),
                        "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"files": entries, "file_count": len(entries)}


def _verified_staged_c1_inputs(prepared: dict, staged_dir: Path) -> dict:
    if staged_dir.is_symlink() or not staged_dir.is_dir():
        raise PreflightError("staged input directory must be an existing regular directory")
    stage_report = _load_json(staged_dir / "stage_report.json")
    if (stage_report.get("mode") != "stage" or stage_report.get("research_admitted") is not False or
            stage_report.get("research_status") != "RESEARCH_NOT_READY" or
            stage_report.get("release_status") != "BLOCKED"):
        raise PreflightError("staged input report is not a quarantined WP03 stage attempt")
    if stage_report.get("input_sha256") != prepared["input_sha256"]:
        raise PreflightError("staged input report does not bind current manifest/selection/evidence/inventory")
    if stage_report.get("selection_sha256") != prepared["input_sha256"].get("selection"):
        raise PreflightError("staged input report selection hash differs from current selection")
    if stage_report.get("logical_product_count") != prepared["logical_product_count"]:
        raise PreflightError("staged input report product count differs from current selection")
    recorded_manifest_path = staged_dir / "outputs_sha256.json"
    recorded_manifest = _load_json(recorded_manifest_path)
    actual_manifest = _output_hash_manifest(staged_dir)
    if recorded_manifest != actual_manifest:
        raise PreflightError("staged input bytes differ from their retained output SHA-256 manifest")
    if stage_report.get("output_manifest_sha256") != _sha256(recorded_manifest_path):
        raise PreflightError("stage report does not bind its output SHA-256 manifest")
    if stage_report.get("output_file_count") != actual_manifest["file_count"]:
        raise PreflightError("stage report output file count differs from verified manifest")
    reported_sites = stage_report.get("sentinel2_c1")
    if not isinstance(reported_sites, dict) or set(reported_sites) != set(prepared["c1"]["sites"]):
        raise PreflightError("staged C1 site membership differs from current PI selection")

    verified = {"asset_paths": {}, "asset_sha256": {}, "asset_bytes": {},
                "stage_report": stage_report, "stage_output_manifest_sha256": _sha256(recorded_manifest_path)}
    for site_id, prepared_site in prepared["c1"]["sites"].items():
        selected = prepared_site["selection"]
        site_record = reported_sites[site_id]
        if (site_record.get("item_id") != selected["item_id"] or
                site_record.get("product_uri") != selected["product_uri"] or
                site_record.get("site_scope") != "DEVELOPMENT_EXPOSED" or
                site_record.get("site_join") != [site_id] or
                site_record.get("research_admitted") is not False):
            raise PreflightError(f"staged C1 product/site/scope join differs for {site_id}")
        stage_assets = site_record.get("assets")
        if not isinstance(stage_assets, dict) or set(stage_assets) != set(REQUIRED_C1_ROLES):
            raise PreflightError(f"staged C1 role membership is incomplete for {site_id}")
        verified["asset_paths"][site_id] = {}
        verified["asset_sha256"][site_id] = {}
        verified["asset_bytes"][site_id] = {}
        for role in REQUIRED_C1_ROLES:
            item_asset = prepared_site["item"]["assets"][role]
            report_asset = stage_assets[role]
            asset_path = staged_dir / "c1" / site_id / f"{role}.tif"
            receipt_path = asset_path.with_name(asset_path.name + ".receipt.json")
            if asset_path.is_symlink() or not asset_path.is_file() or receipt_path.is_symlink():
                raise PreflightError(f"staged C1 {role} provider bytes or receipt are unavailable for {site_id}")
            receipt = _load_json(receipt_path)
            expected_hash = earth_search_c1._multihash_sha256(item_asset["file:checksum"])
            actual_hash, actual_bytes = _sha256(asset_path), asset_path.stat().st_size
            expected_identity = {
                "origin": "provider_download", "provider": "Earth Search C1",
                "status": "COMPLETE_HASH_VERIFIED", "selection_sha256": prepared["input_sha256"]["selection"],
                "product_id": selected["item_id"], "site_id": site_id, "asset_role": role,
                "href": item_asset["href"], "expected_bytes": item_asset["file:size"],
                "expected_sha256": expected_hash, "output_bytes": actual_bytes,
                "output_sha256": actual_hash, "received_bytes": actual_bytes,
            }
            if any(receipt.get(key) != value for key, value in expected_identity.items()):
                raise PreflightError(f"staged C1 {role} receipt identity/bytes/hash differ for {site_id}")
            if any(report_asset.get(key) != value for key, value in {
                    "origin": "provider_download", "provider": "Earth Search C1",
                    "status": "COMPLETE_HASH_VERIFIED", "product_id": selected["item_id"],
                    "site_id": site_id, "asset_role": role, "href": item_asset["href"],
                    "expected_bytes": item_asset["file:size"], "expected_sha256": expected_hash,
                    "output_bytes": actual_bytes, "output_sha256": actual_hash,
                    "received_bytes": actual_bytes}.items()):
                raise PreflightError(f"stage report C1 {role} row differs from receipt for {site_id}")
            if (actual_bytes != item_asset["file:size"] or actual_hash != expected_hash or
                    receipt.get("request_finished_utc") is None):
                raise PreflightError(f"staged C1 {role} source checksum/length is not verified for {site_id}")
            verified["asset_paths"][site_id][role] = asset_path
            verified["asset_sha256"][site_id][role] = actual_hash
            verified["asset_bytes"][site_id][role] = actual_bytes

    asf_selection = prepared["selection"]["sentinel1_asf_grd"]
    payload_paths = []
    for site_id in prepared["c1"]["sites"]:
        for role in REQUIRED_C1_ROLES:
            path = staged_dir / "c1" / site_id / f"{role}.tif"
            payload_paths.extend((path, path.with_name(path.name + ".partial")))
    asf_path = staged_dir / "asf" / asf_selection["fileName"]
    payload_paths.extend((asf_path, asf_path.with_name(asf_path.name + ".partial")))
    received_payload_bytes = sum(path.stat().st_size for path in payload_paths if path.is_file())
    if received_payload_bytes > AUTHORIZED_RECEIVED_BYTE_CAP:
        raise PreflightError("retained provider payload bytes exceed the Human-authorized aggregate cap")
    verified["received_payload_bytes"] = received_payload_bytes
    verified["authorized_received_byte_cap"] = AUTHORIZED_RECEIVED_BYTE_CAP
    return verified


def _decode_staged_optical(prepared: dict, staged_dir: Path, output_dir: Path) -> dict:
    staged = _verified_staged_c1_inputs(prepared, staged_dir)
    manifest_sites = {site["site_id"]: site for site in prepared["manifest"]["sites"]}
    decoded_sites = {}
    for site_id, prepared_site in prepared["c1"]["sites"].items():
        site = manifest_sites[site_id]
        latitude, longitude = float(site["latitude"]), float(site["longitude"])
        if earth_search_c1.CENTROIDS.get(site_id) != (latitude, longitude):
            raise PreflightError(f"decoder centroid tuple differs from named PI manifest fields for {site_id}")
        selected = prepared_site["selection"]
        item = prepared_site["item"]
        roi_dir = output_dir / "c1" / site_id / "roi"
        decoded = decode_c1_bundle(
            item, site_id, prepared_site["product_metadata_path"],
            prepared_site["granule_metadata_path"], staged["asset_paths"][site_id],
            prepared["manifest"]["pre_event_cutoff_exclusive"],
            acquisition_start=prepared["manifest"]["acquisition_start"],
            acquisition_end_exclusive=prepared["manifest"]["acquisition_end_exclusive"],
            selection_record=selected, origin="provider_download",
            roi_output_dir=roi_dir, roi_halfwidth_m=5000)
        roi_report = decoded.get("roi")
        if (decoded.get("origin") != "provider_download" or
                decoded.get("research_admissible") is not False or
                not isinstance(roi_report, dict) or
                roi_report.get("origin") != "provider_download"):
            raise AdmissionError(f"C1 ROI decode lost provider origin or quarantine for {site_id}")
        for role in REQUIRED_C1_ROLES:
            path = staged["asset_paths"][site_id][role]
            if path.stat().st_size != staged["asset_bytes"][site_id][role] or \
                    _sha256(path) != staged["asset_sha256"][site_id][role]:
                raise PreflightError(f"staged provider C1 {role} bytes changed during decode for {site_id}")
        decoded_sites[site_id] = {
            "item_id": selected["item_id"], "product_uri": selected["product_uri"],
            "site_scope": "DEVELOPMENT_EXPOSED", "site_join": [site_id],
            "operational_availability": selected.get("historical_2023_availability"),
            "pixel_inspection": decoded,
        }
    old_asf = staged["stage_report"].get("sentinel1_asf_grd") or {}
    asf_status = {key: old_asf.get(key) for key in
                  ("origin", "provider", "status", "error_type", "literal_error",
                   "cause_status", "received_bytes", "request_started_utc", "request_finished_utc")}
    # The retained transfer receipt records the redirect guard failure, but not
    # the upstream reason for that redirect. Keep that uncertainty explicit.
    if asf_status.get("cause_status") is None:
        asf_status["cause_status"] = "UNKNOWN"
    asf_status["underlying_cause_status"] = "UNKNOWN"
    asf_status["retry_status"] = "NOT_RETRIED_OFFLINE_CONTINUATION"
    return {
        "status": "OPTICAL_ROI_DECODED_QUARANTINED_SAR_UNAVAILABLE",
        "mode": "decode-only", "started_utc": _now(), "finished_utc": _now(),
        "runtime": _runtime(), "network_call_status": "NOT_INVOKED_REUSED_STAGED_PROVIDER_BYTES",
        "input_sha256": prepared["input_sha256"],
        "staged_input_directory": str(staged_dir),
        "stage_output_manifest_sha256": staged["stage_output_manifest_sha256"],
        "received_provider_payload_bytes": staged["received_payload_bytes"],
        "authorized_received_byte_cap": staged["authorized_received_byte_cap"],
        "logical_product_count": prepared["logical_product_count"],
        "logical_product_bytes": prepared["logical_product_bytes"],
        "site_exposure": {site["site_id"]: site["scope"] for site in prepared["manifest"]["sites"]},
        "sentinel2_c1": decoded_sites,
        "sentinel1_asf_grd": asf_status,
        "modality_status": {"sentinel2_c1": "ROI_DECODED_QUARANTINED",
                            "sentinel1_asf_grd": "INCOMPLETE_OR_UNAVAILABLE"},
        "research_admitted": False, "research_status": "RESEARCH_NOT_READY",
        "release_status": "BLOCKED",
        "claims_limits": [
            "one development-exposed image per site; no temporal-change or event-detection result",
            "candidate water area is restricted to the PI-centred ROI and is not whole-lake truth or accuracy",
            "2025 reprocessing does not establish operational 2023 availability",
            "lake geometry, terrain correction, shadow/layover, co-registration uncertainty and SAR remain unavailable",
            "Khangchung is development-exposed and is not a control",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="PI pilot manifest JSON")
    parser.add_argument("--selection", type=Path, required=True, help="bounded product selection JSON")
    parser.add_argument("--evidence", type=Path, required=True, help="retained asset-access summary JSON")
    parser.add_argument("--source-inventory", type=Path, required=True,
                        help="private path/hash bindings for retained source bytes")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new, empty output directory for this invocation")
    parser.add_argument("--mode", choices=("validate-only", "stage", "decode-only"), default="validate-only")
    parser.add_argument("--staged-input-dir", type=Path,
                        help="existing immutable provider stage directory for offline decode-only mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "decode-only" and args.staged_input_dir is None:
        parser.error("--mode decode-only requires --staged-input-dir")
    repository_root = args.repository_root.resolve(strict=True)
    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else repository_root / path

    manifest_path = rooted(args.manifest)
    selection_path = rooted(args.selection)
    evidence_path = rooted(args.evidence)
    inventory_path = rooted(args.source_inventory)
    output_dir = args.output_dir if args.output_dir.is_absolute() else repository_root / args.output_dir
    output_dir = output_dir.resolve()
    staged_input_dir = None
    if args.mode == "decode-only":
        staged_raw = rooted(args.staged_input_dir)
        if staged_raw.is_symlink():
            print(json.dumps({"status": "STAGED_INPUT_REJECTED", "network_call_status": "NOT_INVOKED",
                              "literal_error": "staged input directory must not be a symbolic link"}))
            return 2
        try:
            staged_input_dir = staged_raw.resolve(strict=True)
        except OSError as exc:
            print(json.dumps({"status": "STAGED_INPUT_REJECTED", "network_call_status": "NOT_INVOKED",
                              "error_type": type(exc).__name__, "literal_error": _safe_error(exc)}))
            return 2
        if (not staged_input_dir.is_dir() or
                not staged_input_dir.is_relative_to(repository_root)):
            print(json.dumps({"status": "STAGED_INPUT_REJECTED", "network_call_status": "NOT_INVOKED",
                              "literal_error": "staged input directory must be inside the repository"}))
            return 2
    if output_dir.exists():
        print(json.dumps({"status": "OUTPUT_DIRECTORY_ALREADY_EXISTS", "exit_code": 2}))
        return 2
    try:
        inventory = _load_json(inventory_path)
        if isinstance(inventory.get("source_root"), str):
            source_root_hint = (repository_root / inventory["source_root"]).resolve()
        else:
            raise PreflightError("private source inventory lacks source_root")
    except Exception as exc:
        print(json.dumps({"status": "PREFLIGHT_REJECTED", "error_type": type(exc).__name__,
                          "literal_error": _safe_error(exc), "network_call_status": "NOT_INVOKED"}))
        return 2
    if (output_dir == source_root_hint or source_root_hint in output_dir.parents
            or output_dir in source_root_hint.parents):
        print(json.dumps({"status": "PREFLIGHT_REJECTED",
                          "literal_error": "output directory must not overlap the retained source root",
                          "network_call_status": "NOT_INVOKED"}))
        return 2
    if staged_input_dir is not None and (
            output_dir == staged_input_dir or staged_input_dir in output_dir.parents or
            output_dir in staged_input_dir.parents):
        print(json.dumps({"status": "PREFLIGHT_REJECTED",
                          "literal_error": "decode output must not overlap immutable staged inputs",
                          "network_call_status": "NOT_INVOKED"}))
        return 2
    output_dir.mkdir(parents=True, exist_ok=False)
    started = _now()
    try:
        prepared = preflight(repository_root, manifest_path, selection_path,
                             evidence_path, inventory_path)
    except Exception as exc:
        report = {"status": "PREFLIGHT_REJECTED", "mode": args.mode,
                  "started_utc": started, "finished_utc": _now(), "runtime": _runtime(),
                  "network_call_status": "NOT_INVOKED", "research_admitted": False,
                  "research_status": "RESEARCH_NOT_READY", "release_status": "BLOCKED",
                  "error_type": type(exc).__name__, "literal_error": _safe_error(exc),
                  "cause_status": "SOURCE_VALIDATION_FAILURE"}
        _write_json(output_dir / "stage_report.json", report)
        print(json.dumps({"status": report["status"], "network_call_status": "NOT_INVOKED",
                          "error_type": report["error_type"], "literal_error": report["literal_error"]}))
        return 2

    if args.mode == "validate-only":
        report = {
            "status": "VALIDATED_INPUTS_ONLY_NO_TRANSFER",
            "mode": "validate-only",
            "started_utc": started,
            "finished_utc": _now(),
            "runtime": _runtime(),
            "network_call_status": "NOT_INVOKED",
            "research_admitted": False,
            "research_status": "RESEARCH_NOT_READY",
            "release_status": "BLOCKED",
            "input_sha256": prepared["input_sha256"],
            "logical_product_count": prepared["logical_product_count"],
            "logical_product_bytes": prepared["logical_product_bytes"],
            "site_exposure": {site["site_id"]: site["scope"] for site in prepared["manifest"]["sites"]},
            "site_native_time_and_join_status": {
                site_id: {"time_join_status": item["native_metadata"]["time_join_status"],
                          "tile_join_status": item["native_metadata"]["tile_join_status"],
                          "scope": item["site_scope"]}
                for site_id, item in prepared["c1"]["sites"].items()},
        }
        _write_json(output_dir / "stage_report.json", report)
        print(json.dumps({"status": report["status"], "network_call_status": "NOT_INVOKED",
                          "logical_product_count": report["logical_product_count"],
                          "research_admitted": False, "release_status": "BLOCKED"}))
        return 0

    if args.mode == "decode-only":
        try:
            report = _decode_staged_optical(prepared, staged_input_dir, output_dir)
            report["started_utc"] = started
            exit_code = 0
        except Exception as exc:
            report = {"status": "ROI_DECODE_FAILED", "mode": "decode-only",
                      "started_utc": started, "finished_utc": _now(), "runtime": _runtime(),
                      "network_call_status": "NOT_INVOKED_REUSED_STAGED_PROVIDER_BYTES",
                      "research_admitted": False, "research_status": "RESEARCH_NOT_READY",
                      "release_status": "BLOCKED", "error_type": type(exc).__name__,
                      "literal_error": _safe_error(exc),
                      "cause_status": "UNKNOWN_UNLESS_SOURCE_BOUND_CHECK_ESTABLISHES_CAUSE",
                      "staged_input_directory": str(staged_input_dir),
                      "input_sha256": prepared["input_sha256"]}
            exit_code = 2
        report["finished_utc"] = _now()
    else:
        report, exit_code = _stage(prepared, output_dir)
    manifest = _output_hash_manifest(output_dir)
    _write_json(output_dir / "outputs_sha256.json", manifest)
    report["output_manifest_sha256"] = _sha256(output_dir / "outputs_sha256.json")
    report["output_file_count"] = manifest["file_count"]
    _write_json(output_dir / "stage_report.json", report)
    print(json.dumps({"status": report["status"],
                      "network_call_status": report["network_call_status"],
                      "exit_code": exit_code, "failure_count": len(report.get("failures", [])),
                      "research_admitted": False, "release_status": "BLOCKED"}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
