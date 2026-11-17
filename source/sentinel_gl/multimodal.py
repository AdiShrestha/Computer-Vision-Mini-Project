"""Quarantined asynchronous WP05/WP06 development panel and ragged loader.

This module assembles only retained, hash-bound optical and Sentinel-1
diagnostic products. It does not admit measurements for research, fit transforms,
or train a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rasterio

from .admission import AdmissionError, validate_schema
from .integrity import file_hash, loads_strict, verified_file
from .wp05_optical_features import verify_feature_bundle


PANEL_SCHEMA_VERSION = 1
PANEL_BUNDLE_ID = "WP05-WP06-ASYNC-DEVELOPMENT-DIAGNOSTIC-V1"
OPTICAL_CHANNELS = ("B03", "B08", "B11", "NDWI", "MNDWI")
SAR_CHANNELS = ("SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR")
OPTICAL_LAYER_BY_CHANNEL = {
    "B03": "b03_reflectance_20m", "B08": "b08_reflectance_20m",
    "B11": "b11_reflectance_20m", "NDWI": "ndwi", "MNDWI": "mndwi",
}
OPTICAL_VALID_LAYER_BY_CHANNEL = {
    "B03": "b03_valid_mask", "B08": "b08_valid_mask",
    "B11": "b11_valid_mask",
}
OPTICAL_UNITS = {
    "B03": "dimensionless_surface_reflectance",
    "B08": "dimensionless_surface_reflectance",
    "B11": "dimensionless_surface_reflectance",
    "NDWI": "dimensionless_index",
    "MNDWI": "dimensionless_index",
}
SAR_UNITS = "dimensionless_linear_sigma0_power_ratio"
INDEX_ATOL = 2e-6
# Source dB and linear GeoTIFFs are rounded independently to float32.  Four
# micro-decibels covers the measured two-ULP serialization difference while
# keeping the positive-finite domain check exact.
DB_FLOAT32_ROUNDING_ATOL = 4e-6
CONTEXT_POLICY_VERSION = "OPTICAL_INDEX_DEPENDENCY_CLOSURE_V1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _sha_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _root_file(root: Path, path: str, digest: str, byte_count: int | None = None) -> Path:
    file = verified_file(root, path, digest)
    if byte_count is not None:
        _require(file.stat().st_size == byte_count, f"source byte count changed: {path}")
    return file


def _read_json(path: Path) -> dict:
    value = loads_strict(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, f"expected JSON object: {path}")
    return value


def _raster(root: Path, binding: Mapping[str, Any], expected_grid: Mapping[str, Any] | None = None):
    path = _root_file(root, binding["path"], binding["sha256"], binding["bytes"])
    with rasterio.open(path) as ds:
        _require(ds.count == 1, f"expected one raster band: {path}")
        _require(ds.crs is not None and ds.crs.to_epsg() == 32645,
                 f"raster CRS differs from accepted EPSG:32645 grid: {path}")
        array = ds.read(1, masked=False)
        grid = {"crs": ds.crs.to_string(), "shape": [ds.height, ds.width],
                "transform": [float(x) for x in list(ds.transform)[:6]],
                "coordinate_units": "metre", "dtype": ds.dtypes[0],
                "nodata": None if ds.nodata is None else float(ds.nodata)}
    _require(grid["shape"] == [499, 499], f"raster shape differs from accepted ROI grid: {path}")
    if expected_grid is not None:
        _require(grid["crs"] == expected_grid["crs"] and
                 grid["shape"] == expected_grid["shape"] and
                 grid["transform"] == expected_grid["transform"],
                 f"raster does not match the source-bound target grid: {path}")
    return array, grid, path


def _asset_map(record: Mapping[str, Any]) -> dict[str, dict]:
    assets = record["lineage"]["decoded_assets"]
    result = {}
    for item in assets:
        name = item["layer"]
        _require(name not in result, f"duplicate accepted optical decoded asset: {name}")
        result[name] = {"path": item["path"], "sha256": item["sha256"],
                        "bytes": item["bytes"], "source_sha256": item.get("source_sha256")}
    return result


def _verify_optical_inputs(root: Path, feature_path: Path, receipt_path: Path,
                           feature_bundle: dict) -> tuple[dict, dict]:
    receipt = _read_json(receipt_path)
    _require(receipt.get("status") == "PASS_LOCAL_DEVELOPMENT_FEATURE_INTEGRITY_ONLY" and
             receipt.get("research_admissible") is False and
             receipt.get("plan_executable") is False and receipt.get("release_status") == "BLOCKED",
             "optical feature receipt is not a quarantined accepted development execution")
    feature_hash = file_hash(feature_path)
    matching = [x for x in receipt.get("output_files", []) if x.get("path") ==
                feature_path.relative_to(root).as_posix()]
    _require(len(matching) == 1 and matching[0].get("sha256") == feature_hash and
             matching[0].get("bytes") == feature_path.stat().st_size,
             "optical feature bundle does not match its retained execution receipt")
    wp04_path = root / feature_bundle["input_bundle_path"]
    _require(wp04_path.is_file() and not wp04_path.is_symlink() and
             file_hash(wp04_path) == feature_bundle["input_bundle_sha256"],
             "accepted WP04 optical bundle path/hash is unavailable or changed")
    verified = verify_feature_bundle(root, wp04_path, feature_bundle)
    _require(verified.get("status") == "PASS_LOCAL_DEVELOPMENT_FEATURE_INTEGRITY_ONLY" and
             verified.get("research_admissible") is False,
             "independent WP05 optical source replay did not pass at quarantined scope")
    return receipt, verified


def _workspace_member_path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        try:
            return relative.relative_to(root)
        except ValueError:
            # Older accepted execution records retain extraction paths as
            # absolute workstation paths. Rebase only the canonical data
            # subtree in a disposable repository copy, then verify exact
            # byte count and SHA-256 at the rebased in-workspace path.
            parts = relative.parts
            anchors = [i for i in range(len(parts) - 2)
                       if parts[i:i + 3] == ("data", "raw", "wp03")]
            _require(len(anchors) == 1,
                     "retained SAR source member absolute path has no unique workspace data anchor")
            return Path(*parts[anchors[0]:])
    return relative


def _archive_members(root: Path, execution: Mapping[str, Any]) -> tuple[str, dict]:
    members = execution.get("extracted_members", {})
    _require(len(members) == 11, "retained SAR execution does not bind exactly eleven role-bound extracted members")
    safe_names = set()
    verified = {}
    for member, item in members.items():
        relative = _workspace_member_path(root, item["relative_path"])
        file = _root_file(root, relative.as_posix(), item["sha256"], item["extracted_bytes"])
        _require(item["archive_stream_bytes"] == item["extracted_bytes"],
                 "retained SAR member archive/extracted byte count differs")
        safe_names.add(member.split("/", 1)[0])
        verified[member] = {"path": relative.as_posix(), "bytes": file.stat().st_size,
                            "sha256": item["sha256"], "roles": item["roles"],
                            "zip_crc32": item["zip_crc32"]}
    _require(len(safe_names) == 1, "VV/VH SAFE source product identities are not unique")
    return next(iter(safe_names)), verified


def _source_output_map(execution: Mapping[str, Any]) -> dict[str, dict]:
    out = {}
    for item in execution.get("outputs", []):
        path = item.get("path")
        _require(path not in out, f"duplicate retained SAR output path: {path}")
        out[path] = item
    return out


def _attach_optical_group(root: Path, record: dict, arrays: dict, array_prefix: str,
                          sar_product: Mapping[str, Any], site_id: str) -> dict:
    assets = _asset_map(record)
    required_layers = set(OPTICAL_LAYER_BY_CHANNEL.values()) | set(OPTICAL_VALID_LAYER_BY_CHANNEL.values()) | {
        "scl_raw_20m_classes", "scl_valid_mask", "scl_clear_4_5_6_mask"}
    _require(required_layers <= assets.keys(), "accepted optical panel lacks an observed channel or QA source asset")
    grid = record["grid"]
    ref = {"crs": grid["crs"], "shape": grid["shape"], "transform": grid["transform"]}
    loaded, bindings = {}, {}
    for name in sorted(required_layers):
        value, actual_grid, _ = _raster(root, assets[name], expected_grid=ref)
        loaded[name] = value
        bindings[name] = {"path": assets[name]["path"], "sha256": assets[name]["sha256"],
                          "bytes": assets[name]["bytes"], "layer": name,
                          "source_sha256": assets[name].get("source_sha256"),
                          "grid_sha256": _sha_json({k: actual_grid[k] for k in ("crs", "shape", "transform")})}
    source_valid = {channel: loaded[layer].astype(bool) for channel, layer in
                    OPTICAL_VALID_LAYER_BY_CHANNEL.items()}
    b03, b08, b11 = (loaded[OPTICAL_LAYER_BY_CHANNEL[name]].astype(np.float32)
                     for name in ("B03", "B08", "B11"))
    expected_index = {}
    for channel, left, right in (("NDWI", b03, b08), ("MNDWI", b03, b11)):
        denominator = left.astype(np.float64) + right.astype(np.float64)
        numerator = left.astype(np.float64) - right.astype(np.float64)
        support = (np.isfinite(left) & np.isfinite(right) &
                   (source_valid["B03"] & source_valid["B08"] if channel == "NDWI"
                    else source_valid["B03"] & source_valid["B11"]) &
                   np.isfinite(denominator) & (denominator != 0))
        index = np.full(left.shape, np.nan, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            index[support] = (numerator[support] / denominator[support]).astype(np.float32)
        expected_index[channel] = (index, support)
        stored = loaded[OPTICAL_LAYER_BY_CHANNEL[channel]].astype(np.float32)
        _require(np.allclose(stored, index, rtol=0, atol=INDEX_ATOL, equal_nan=True),
                 f"retained {channel} array does not replay from its source reflectances")
    channel_values, channel_valid = [], []
    for channel in OPTICAL_CHANNELS:
        source = loaded[OPTICAL_LAYER_BY_CHANNEL[channel]].astype(np.float32)
        if channel in OPTICAL_VALID_LAYER_BY_CHANNEL:
            valid = source_valid[channel] & np.isfinite(source)
        else:
            source, valid = expected_index[channel]
            # Retain the actual accepted derived index after independently checking it.
            source = loaded[OPTICAL_LAYER_BY_CHANNEL[channel]].astype(np.float32)
        canonical = source.copy()
        canonical[~valid] = np.nan
        channel_values.append(canonical)
        channel_valid.append(valid)
    values = np.stack(channel_values).astype(np.float32, copy=False)
    observed = np.stack(channel_valid).astype(bool, copy=False)
    research = np.zeros(observed.shape, dtype=bool)
    values_key, valid_key, research_key = (f"{array_prefix}_{suffix}" for suffix in
                                            ("values", "observed_valid", "research_admissible"))
    arrays[values_key], arrays[valid_key], arrays[research_key] = values, observed, research
    qa = {
        "scl_classes": loaded["scl_raw_20m_classes"].astype(np.uint8),
        "scl_valid": loaded["scl_valid_mask"].astype(bool),
        "scl_clear_4_5_6": loaded["scl_clear_4_5_6_mask"].astype(bool),
    }
    qa_keys = {}
    for name, value in qa.items():
        key = f"{array_prefix}_qa_{name}"
        arrays[key] = value
        qa_keys[name] = key
    time_axes = record["time_axes"]
    acquisition = time_axes["acquisition"]
    _require(acquisition["status"] == "VERIFIED" and isinstance(acquisition["at"], str),
             f"optical acquisition timestamp is not source-verified for {site_id}")
    group_id = f"{site_id}__OPTICAL__{acquisition['at'][:10]}"
    channels = []
    for channel in OPTICAL_CHANNELS:
        record_channel = next((x for x in record["channels"] if x["channel_id"] ==
                               ("MNDWI_NDSI" if channel == "MNDWI" else channel)), None)
        _require(record_channel is not None, f"accepted optical feature record lacks channel {channel}")
        channels.append({"channel_id": channel, "array_index": len(channels), "units": OPTICAL_UNITS[channel],
            "status": "OBSERVED_SOURCE_DERIVED_QUARANTINED", "validity": "PER_COORDINATE_MASK",
            "radiometry": (record_channel.get("calibration") if channel in {"B03", "B08", "B11"}
                           else {"formula": "NDWI=(B03-B08)/(B03+B08)" if channel == "NDWI"
                                 else "MNDWI=(B03-B11)/(B03+B11)",
                                 "source_channels": ["B03", "B08"] if channel == "NDWI" else ["B03", "B11"],
                                 "stored_channel_aliases": ["NDSI"] if channel == "MNDWI" else []}),
            "source_asset_layers": [OPTICAL_LAYER_BY_CHANNEL[channel]] +
                ([OPTICAL_VALID_LAYER_BY_CHANNEL[channel]] if channel in OPTICAL_VALID_LAYER_BY_CHANNEL else [])})
    unavailable = [
        {"channel_id": name, "status": "UNAVAILABLE", "value": None,
         "reason_code": "NOT_STAGED_BY_AUTHORIZED_SCOPE", "reason": f"{name} source bytes were not staged."}
        for name in ("B02", "B04", "B12")
    ]
    return {
        "group_id": group_id, "sensor": "SENTINEL_2_OPTICAL", "site_id": site_id,
        "acquisition": {"at": acquisition["at"], "timezone_status": "EXPLICIT_UTC",
            "support_interval": {"start": acquisition["at"], "end": acquisition["at"],
                "status": "ACQUISITION_INSTANT_ONLY_PIXEL_TIME_SPAN_UNAVAILABLE"},
            "historical_availability": {"at": time_axes["historical_availability"]["at"],
                "status": time_axes["historical_availability"]["status"],
                "reason": time_axes["historical_availability"]["reason"]},
            "product_generation": time_axes["product_generation"],
            "publication": time_axes["publication"], "retrieval": time_axes["retrieval"],
            "processing": time_axes["execution"]},
        "site_exposure": "DEVELOPMENT_EXPOSED", "scope_membership": {
            "role": None, "status": "UNASSIGNED_NO_ADMITTED_SCIENTIFIC_SOURCE_MEMBERSHIP",
            "scientific_source_memberships": []},
        "sampling_graph": {"graph_id": None, "physical_unit_id": None, "status": "UNRESOLVED",
            "reason": "No verified lake geometry or independent physical sampling unit is admitted."},
        "provenance_graph": {"graph_id": None, "fitted_state_id": None,
            "status": "NO_FITTED_STATE_OR_ROLE_MEMBERSHIP"},
        "support_id": record["support_id"], "source_id": record["source_id"],
        "product_id": record["product_id"], "product_uri": record["product_uri"],
        "grid": {"crs": ref["crs"], "shape": ref["shape"], "transform": ref["transform"],
            "transform_sha256": _sha_json({k: ref[k] for k in ("crs", "shape", "transform")}),
            "source_grid_version": record["grid"]["roi_grid_version"],
            "cross_sensor_registration": "SHARED_GRID_ONLY_PIXEL_REGISTRATION_UNVERIFIED",
            "geometric_uncertainty": {"value": None, "status": "UNQUANTIFIED",
                "reason": "The accepted source does not quantify lake-geometry or SAR-optical registration uncertainty."}},
        "channels": channels, "unavailable_channels": unavailable,
        "arrays": {"file_role": "SOURCE_DERIVED_PIXEL_VALUES_AND_MASKS", "values_key": values_key,
            "observed_valid_key": valid_key, "research_admissible_key": research_key,
            "channel_axis_order": [x["channel_id"] for x in channels], "shape": list(values.shape),
            "qa_keys": qa_keys},
        "qa": {"status": "SCL_SOURCE_QA_AVAILABLE", "observed_validity_is_separate": True,
            "clear_classes": [4, 5, 6], "qa_assets": [bindings[x] for x in
                ("scl_raw_20m_classes", "scl_valid_mask", "scl_clear_4_5_6_mask")]},
        "lineage": {"optical_feature_bundle_path": str(record["lineage"]["wp04_bundle_path"]),
            "optical_feature_record_id": record["record_id"], "accepted_wp04_bundle_sha256": record["lineage"]["wp04_bundle_sha256"],
            "decoded_assets": [bindings[name] for name in sorted(required_layers)]},
        "admission": {"origin": "RETAINED_PROVIDER_BYTES_DEVELOPMENT_ONLY",
            "observed_valid_mask_available": True, "research_admissible": False,
            "training_admitted": False, "status": "DEVELOPMENT_DIAGNOSTIC_QUARANTINED"},
    }


def _attach_sar_group(root: Path, execution: dict, output_map: dict, safe_product_id: str,
                      site_id: str, array_prefix: str, arrays: dict, optical_grid: dict) -> dict:
    roi = execution["roi_extraction"][site_id]
    target_grid = roi["target_grid"]
    expected = {"crs": target_grid["crs"], "shape": target_grid["shape"],
                "transform": target_grid["transform"]}
    _require(expected == optical_grid, f"optical and SAR target grid metadata differ for {site_id}")
    raw_values, raw_valids, coverage_values, db_values, db_valids = [], [], [], [], []
    channels, lineage_assets = [], []
    for pol, channel in (("VV", SAR_CHANNELS[0]), ("VH", SAR_CHANNELS[1])):
        diagnostic = roi["diagnostic_products"][pol]
        _require(diagnostic["status"] == "ELLIPSOID_TIEPOINT_DIAGNOSTIC_ONLY_NO_TERRAIN_CORRECTION",
                 f"SAR product has unexpected geolocation or admission status: {site_id}/{pol}")
        _require(diagnostic["grid"]["crs"] == expected["crs"] and
                 diagnostic["grid"]["shape"] == expected["shape"] and
                 diagnostic["grid"]["transform"] == expected["transform"],
                 f"SAR diagnostic grid differs from its declared exact target grid: {site_id}/{pol}")
        bindings = {}
        loaded = {}
        for key in ("linear", "valid", "coverage", "db"):
            item = diagnostic["outputs"][key]
            output = output_map.get(item["path"])
            _require(output is not None and output["sha256"] == item["sha256"] and
                     output["bytes"] == item["bytes"], f"SAR receipt output join mismatch: {item['path']}")
            loaded[key], grid, _ = _raster(root, item, expected_grid=expected)
            bindings[key] = {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
        power = loaded["linear"].astype(np.float32)
        valid = loaded["valid"].astype(bool) & np.isfinite(power) & (loaded["coverage"] > 0)
        power = power.copy(); power[~valid] = np.nan
        db = loaded["db"].astype(np.float32)
        db_valid = valid & np.isfinite(db) & (power > 0)
        db = db.copy(); db[~db_valid] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_db = np.full(power.shape, np.nan, dtype=np.float32)
            expected_db[valid & (power > 0)] = (10.0 * np.log10(power[valid & (power > 0)])).astype(np.float32)
        _require(np.allclose(db, expected_db, rtol=0, atol=DB_FLOAT32_ROUNDING_ATOL, equal_nan=True),
                 f"SAR dB source does not obey strict positive-finite conversion: {site_id}/{pol}")
        _require(np.array_equal(np.isfinite(db), valid & (power > 0)),
                 f"SAR dB missingness differs from positive-finite power support: {site_id}/{pol}")
        raw_values.append(power); raw_valids.append(valid)
        coverage_values.append(loaded["coverage"].astype(np.float32))
        db_values.append(db); db_valids.append(np.isfinite(db))
        lineage_assets.extend(bindings.values())
        channels.append({"channel_id": channel, "array_index": len(channels), "units": SAR_UNITS,
            "status": "OBSERVED_RAW_SIGMA0_TIEPOINT_DIAGNOSTIC_QUARANTINED",
            "validity": "SOURCE_VALID_MASK_AND_FINITE_LINEAR_POWER_AND_POSITIVE_COVERAGE",
            "radiometry": {"representation": "RAW_SIGMA0_LINEAR", "calibration": "DN^2/A_sigma^2",
                "noise_subtraction": "NOT_APPLIED_TO_THIS_PANEL_CHANNEL", "thermal_noise_correction_in_product": False,
                "db_alternative": {"representation": "RAW_SIGMA0_DB_POSITIVE_FINITE_ONLY",
                    "path": bindings["db"]["path"], "sha256": bindings["db"]["sha256"],
                    "validity": "FINITE_AND_RAW_LINEAR_POWER_GT_ZERO", "epsilon_clamp": False}},
            "source_product_id": safe_product_id, "polarization": pol,
            "source_asset_layers": [bindings["linear"]["path"], bindings["valid"]["path"]]})
    values = np.stack(raw_values).astype(np.float32, copy=False)
    observed = np.stack(raw_valids).astype(bool, copy=False)
    coverage = np.stack(coverage_values).astype(np.float32, copy=False)
    research = np.zeros(observed.shape, dtype=bool)
    db_values_array = np.stack(db_values).astype(np.float32, copy=False)
    db_valid_array = np.stack(db_valids).astype(bool, copy=False)
    values_key, valid_key, research_key, coverage_key, db_key, db_valid_key = (
        f"{array_prefix}_{suffix}" for suffix in
        ("values", "observed_valid", "research_admissible", "coverage", "raw_db_alternative", "raw_db_valid"))
    arrays.update({values_key: values, valid_key: observed, research_key: research,
                   coverage_key: coverage, db_key: db_values_array, db_valid_key: db_valid_array})
    product = execution["products"]["VV"]
    source_start, source_stop = product["start_time"], product["stop_time"]
    _require(execution["products"]["VH"]["start_time"] == source_start and
             execution["products"]["VH"]["stop_time"] == source_stop,
             "VV/VH product acquisition intervals differ")
    group_id = f"{site_id}__SENTINEL1__{source_start[:10]}"
    signed_native = []
    for item in execution["outputs"]:
        if "sigma0_noise_subtracted_signed_linear_native.tif" in item["path"] or "sigma0_noise_subtracted_signed_db_native.tif" in item["path"]:
            signed_native.append({"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]})
    return {
        "group_id": group_id, "sensor": "SENTINEL_1_IW_GRD", "site_id": site_id,
        "acquisition": {"start_raw": source_start, "end_raw": source_stop,
            "timezone_status": "NOT_EXPLICIT_IN_RETAINED_START_STOP_STRINGS",
            "support_interval": {"start_raw": source_start, "end_raw": source_stop,
                "status": "SOURCE_PRODUCT_INTERVAL_RETAINED_VERBATIM_TIMEZONE_UNSPECIFIED"},
            "historical_availability": {"at": None, "status": "UNKNOWN",
                "reason": "No source-backed 2023 historical availability record is bound to this SAR acquisition."},
            "processing": {"at": None, "status": "NOT_RECORDED_IN_NATIVE_EXECUTION_RECEIPT",
                "elapsed_seconds": execution.get("elapsed_seconds")},
            "retrieval": {"at": None, "status": "NOT_BOUND_IN_NATIVE_EXECUTION_RECEIPT"}},
        "site_exposure": "DEVELOPMENT_EXPOSED", "scope_membership": {
            "role": None, "status": "UNASSIGNED_NO_ADMITTED_SCIENTIFIC_SOURCE_MEMBERSHIP",
            "scientific_source_memberships": []},
        "sampling_graph": {"graph_id": None, "physical_unit_id": None, "status": "UNRESOLVED",
            "reason": "No verified lake geometry or independent physical sampling unit is admitted."},
        "provenance_graph": {"graph_id": None, "fitted_state_id": None,
            "status": "NO_FITTED_STATE_OR_ROLE_MEMBERSHIP"},
        "support_id": f"SAR-SUPPORT:{site_id}:{safe_product_id}",
        "source_id": f"SAR-SOURCE:{safe_product_id}", "product_id": safe_product_id,
        "grid": {"crs": expected["crs"], "shape": expected["shape"], "transform": expected["transform"],
            "transform_sha256": _sha_json(expected), "source_grid_version": "WP03_ACCEPTED_ROI_DECODE_005",
            "cross_sensor_registration": "SHARED_GRID_ONLY_PIXEL_REGISTRATION_UNVERIFIED",
            "geometric_uncertainty": {"value": None, "status": "UNQUANTIFIED",
                "reason": "Tiepoint inversion is ellipsoid-only; terrain registration and its uncertainty are unavailable."}},
        "channels": channels, "unavailable_channels": [],
        "arrays": {"file_role": "RAW_SIGMA0_DIAGNOSTIC_VALUES_AND_MASKS", "values_key": values_key,
            "observed_valid_key": valid_key, "research_admissible_key": research_key,
            "channel_axis_order": [x["channel_id"] for x in channels], "shape": list(values.shape),
            "coverage_key": coverage_key, "raw_db_alternative_key": db_key,
            "raw_db_valid_key": db_valid_key},
        "qa": {"status": "VALIDITY_AND_WEIGHT_COVERAGE_AVAILABLE_TERRAIN_QA_UNAVAILABLE",
            "observed_validity_is_separate": True,
            "coverage_source_assets": [coverage_values and {"path": d["outputs"]["coverage"]["path"],
                "sha256": d["outputs"]["coverage"]["sha256"], "bytes": d["outputs"]["coverage"]["bytes"]}
                for d in [execution["roi_extraction"][site_id]["diagnostic_products"][pol]
                          for pol in ("VV", "VH")]],
            "terrain_correction": "NOT_PERFORMED", "layover_shadow_qa": "UNAVAILABLE",
            "registration_qa": "UNAVAILABLE", "water_join": "NOT_PERFORMED"},
        "lineage": {"native_sar_execution_path": "data/derived/wp03/native_sar_attempt_20261015_restart005/native_sar_execution.json",
            "archive_sha256": execution["archive"]["sha256"], "processing_flags": product["processing_flags"],
            "ipf_version": product["ipf_version"], "product_type": product["product_type"],
            "geolocation_method": "scipy LinearNDInterpolator inverse mapping of annotation tiepoints; ellipsoid diagnostic only",
            "diagnostic_assets": lineage_assets,
            "signed_noise_subtracted_native_variants_excluded_from_roi_panel": {
                "status": "NATIVE_GRID_ONLY_NOT_CO_REGISTERED_TO_TARGET_GRID",
                "reason": "These are signed noise-subtracted native-grid products, not the raw sigma0 tiepoint diagnostic channels.",
                "assets": signed_native}},
        "admission": {"origin": "RETAINED_PROVIDER_BYTES_DEVELOPMENT_ONLY",
            "observed_valid_mask_available": True, "research_admissible": False,
            "training_admitted": False, "status": "DEVELOPMENT_DIAGNOSTIC_QUARANTINED"},
    }


def _input_bindings(root: Path, feature_path: Path, receipt_path: Path,
                    feature_bundle: Mapping[str, Any], sar_path: Path,
                    sar_members: Mapping[str, Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> list[dict]:
    paths: dict[str, dict] = {}
    candidates = [(feature_path, "accepted_optical_feature_bundle"),
                  (receipt_path, "accepted_optical_feature_execution_receipt"),
                  (root / feature_bundle["input_bundle_path"], "accepted_wp04_bundle"),
                  (sar_path, "accepted_native_sar_execution_record")]
    candidates.extend((root / item["path"], "role_bound_native_sar_input") for item in sar_members.values())
    for group in groups:
        for layer in group["lineage"].get("decoded_assets", []):
            candidates.append((root/layer["path"], "retained_optical_decoded_asset"))
        candidates.extend((root/x["path"], "retained_sar_diagnostic_asset")
                          for x in group["lineage"].get("diagnostic_assets", []))
        for x in group["lineage"].get("signed_noise_subtracted_native_variants_excluded_from_roi_panel", {}).get("assets", []):
            candidates.append((root/x["path"], "excluded_native_signed_sar_variant"))
    for path, role in candidates:
        relative = path.resolve(strict=True).relative_to(root).as_posix()
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": file_hash(path), "role": role}
        prior = paths.get(relative)
        if prior:
            _require(prior["sha256"] == record["sha256"] and prior["bytes"] == record["bytes"],
                     f"source binding changed during assembly: {relative}")
        else:
            paths[relative] = record
    return [paths[key] for key in sorted(paths)]


def build_multimodal_panel(root: str | Path, feature_path: str | Path,
                           feature_receipt_path: str | Path, sar_execution_path: str | Path,
                           output_dir: str | Path, *, mode: str = "research") -> dict:
    """Build a fresh, hash-bound diagnostic artifact; research mode always fails closed."""
    root = Path(root).resolve(strict=True)
    if mode != "development-diagnostic":
        raise AdmissionError("normal research panel build is blocked; explicit development-diagnostic mode is required")
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    _require(out.resolve(strict=False).is_relative_to(root), "panel output path must stay inside the workspace")
    _require(not out.exists(), f"refusing to overwrite prior panel attempt: {out}")
    feature_file = Path(feature_path)
    receipt_file = Path(feature_receipt_path)
    sar_file = Path(sar_execution_path)
    if not feature_file.is_absolute(): feature_file = root / feature_file
    if not receipt_file.is_absolute(): receipt_file = root / receipt_file
    if not sar_file.is_absolute(): sar_file = root / sar_file
    feature_file = feature_file.resolve(strict=True); receipt_file = receipt_file.resolve(strict=True)
    sar_file = sar_file.resolve(strict=True)
    _require(feature_file.is_relative_to(root) and receipt_file.is_relative_to(root) and sar_file.is_relative_to(root),
             "panel inputs must be retained workspace files")
    feature_bundle = _read_json(feature_file)
    receipt, optical_verification = _verify_optical_inputs(root, feature_file, receipt_file, feature_bundle)
    execution = _read_json(sar_file)
    _require(execution.get("research_admissible") is False and execution.get("network_requests") == 0,
             "native SAR inputs must be quarantined retained offline outputs")
    safe_product, sar_members = _archive_members(root, execution)
    output_map = _source_output_map(execution)
    feature_records = {row["site_id"]: row for row in feature_bundle.get("records", [])}
    site_records = execution.get("roi_extraction", {})
    _require(set(feature_records) == set(site_records) and len(feature_records) == 2,
             "optical and SAR pilot site memberships differ or are incomplete")
    arrays: dict[str, np.ndarray] = {}
    groups, panel_rows = [], []
    for site_index, site_id in enumerate(sorted(feature_records)):
        optical_record = feature_records[site_id]
        optical_grid = {"crs": optical_record["grid"]["crs"],
                        "shape": optical_record["grid"]["shape"],
                        "transform": optical_record["grid"]["transform"]}
        prefix = f"g{2*site_index:02d}"
        optical = _attach_optical_group(root, optical_record, arrays, prefix,
                                        execution["products"]["VV"], site_id)
        sar = _attach_sar_group(root, execution, output_map, safe_product,
                               site_id, f"g{2*site_index+1:02d}", arrays, optical_grid)
        # Keep distinct sensor observations and order by retained acquisition date/source string.
        site_groups = sorted((sar, optical), key=lambda g: (
            g["acquisition"].get("at", g["acquisition"].get("start_raw", "")), g["group_id"]))
        _require(site_groups[0]["group_id"] != site_groups[1]["group_id"] and
                 site_groups[0]["sensor"] != site_groups[1]["sensor"],
                 "asynchronous sensor groups were merged or duplicated")
        groups.extend(site_groups)
        panel_rows.append({"panel_id": f"{site_id}__RAGGED_DIAGNOSTIC_PANEL_V1",
            "site_id": site_id, "site_exposure": "DEVELOPMENT_EXPOSED",
            "group_ids": [g["group_id"] for g in site_groups],
            "sampling_graph": {"graph_id": None, "physical_unit_id": None, "status": "UNRESOLVED"},
            "provenance_graph": {"graph_id": None, "fitted_state_id": None,
                "status": "NO_FITTED_STATE_OR_ROLE_MEMBERSHIP"},
            "scope_membership": {"role": None, "status": "UNASSIGNED_NO_ADMITTED_MEMBERSHIPS"}})
    missing_dependencies = [
        {"channel_id": "WEATHER_TEMPERATURE", "status": "UNAVAILABLE", "value": None, "reason": "No retained source-bound weather measurement."},
        {"channel_id": "WEATHER_PRECIPITATION", "status": "UNAVAILABLE", "value": None, "reason": "No retained source-bound weather measurement."},
        {"channel_id": "DEM_ELEVATION", "status": "UNAVAILABLE", "value": None, "reason": "No retained DEM measurement."},
        {"channel_id": "TERRAIN_CORRECTION", "status": "UNAVAILABLE", "value": None, "reason": "No DEM/terrain correction; SAR remains ellipsoid tiepoint diagnostic."},
        {"channel_id": "LAYOVER_SHADOW_QA", "status": "UNAVAILABLE", "value": None, "reason": "No verified terrain/layover/shadow QA."},
        {"channel_id": "VELOCITY", "status": "UNAVAILABLE", "value": None, "reason": "No retained source-bound velocity measurement."},
        {"channel_id": "COHERENCE", "status": "UNAVAILABLE", "value": None, "reason": "GRD inputs contain no interferometric phase/coherence."},
    ]
    arrays_buffer = io.BytesIO()
    np.savez_compressed(arrays_buffer, **arrays)
    packed = arrays_buffer.getvalue()
    out.mkdir(parents=True, exist_ok=False)
    arrays_path = out / "panel_arrays.npz"
    arrays_path.write_bytes(packed)
    arrays_digest = hashlib.sha256(packed).hexdigest()
    array_keys = sorted(arrays)
    document = {
        "schema_version": PANEL_SCHEMA_VERSION, "bundle_id": PANEL_BUNDLE_ID,
        "panel_bundle_id": f"{PANEL_BUNDLE_ID}:{arrays_digest[:16]}",
        "origin": "RETAINED_PROVIDER_BYTES_DEVELOPMENT_ONLY",
        "mode": "DEVELOPMENT_DIAGNOSTIC", "status": "RAGGED_ASYNCHRONOUS_DIAGNOSTIC_PANEL",
        "research_admissible": False, "training_admitted": False,
        "plan_executable": False, "research_status": "RESEARCH_NOT_READY", "release_status": "BLOCKED",
        "fit_transform_state": None, "fit_transform_status": "RAW_VALUES_NO_FITTED_TRANSFORM",
        "arrays_file": {"path": arrays_path.relative_to(root).as_posix(), "bytes": len(packed),
            "sha256": arrays_digest, "keys": array_keys, "compression": "NUMPY_NPZ_DEFLATE"},
        "inputs": _input_bindings(root, feature_file, receipt_file, feature_bundle, sar_file,
                                  sar_members, groups),
        "accepted_inputs": {"optical_features": {"path": feature_file.relative_to(root).as_posix(),
                "sha256": file_hash(feature_file), "bytes": feature_file.stat().st_size,
                "receipt_path": receipt_file.relative_to(root).as_posix(), "receipt_status": receipt["status"],
                "independent_replay_status": optical_verification["status"]},
            "native_sar_execution": {"path": sar_file.relative_to(root).as_posix(),
                "sha256": file_hash(sar_file), "bytes": sar_file.stat().st_size,
                "research_admissible": False, "archive_sha256": execution["archive"]["sha256"]}},
        "groups": groups, "panels": panel_rows,
        "graph_separation": {"G_S": {"status": "UNRESOLVED_PHYSICAL_MEMBERSHIP",
                "reason": "No verified lake polygon or independently checked spatial adjacency."},
            "G_P": {"status": "NO_FIT_MEMBERSHIP_OR_FITTED_STATE",
                "reason": "All source memberships are empty; no fitting was run."},
            "shared_grid_is_physical_registration": False},
        "independent_support": {"value": None, "status": "UNKNOWN",
            "reason": "Four sensor-site acquisition groups are computational tokens, not independent lake/event supports."},
        "unavailable_dependencies": missing_dependencies,
        "missingness_contract": {"invalid_source_values_canonicalized_to_nan": True,
            "observed_validity_masks_separate_from_QA": True,
            "research_admissibility_mask_all_false": True,
            "absent_channels_omitted_with_explicit_status": True,
            "no_interpolation_forward_fill_or_repeated_source": True,
            "tensor_padding_neutral_value": 0.0,
            "padding_neutral_value_is_never_observation": True,
            "natural_missingness_and_artificial_targets_are_distinct": True},
        "limitations": ["The shared EPSG:32645 grid does not verify SAR-optical physical registration.",
            "SAR arrays are raw sigma0 ellipsoid/tiepoint diagnostics, not denoised or terrain-corrected ROI measurements.",
            "September 1 SAR and September 4/21 optical acquisitions remain separate asynchronous groups.",
            "Historical 2023 product availability is unknown; product generation/reprocessing/retrieval does not establish it.",
            "Scientific ROI/water summaries remain null; later optical masks do not define causal SAR water.",
            "No fitted transform, model, score, checkpoint, scientific label, or independent sample claim is produced."],
    }
    validate_multimodal_panel_document(document)
    from .admission import validate_schema as _validate_schema
    _validate_schema(document, "wp05_multimodal_panel")
    panel_path = out / "panel.v1.json"
    panel_path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {"schema_version": 1, "status": "DEVELOPMENT_DIAGNOSTIC_PANEL_BUILT",
        "command": "run_wp05_wp06_panel build", "mode": "DEVELOPMENT_DIAGNOSTIC",
        "research_admissible": False, "training_admitted": False,
        "network_requests": 0, "randomness": "NONE_DETERMINISTIC",
        "runtime": {"python": __import__("sys").version, "executable": __import__("sys").executable,
            "numpy": np.__version__, "rasterio": rasterio.__version__},
        "input_count": len(document["inputs"]),
        "outputs": [{"path": arrays_path.relative_to(root).as_posix(), "bytes": arrays_path.stat().st_size,
                     "sha256": file_hash(arrays_path)},
                    {"path": panel_path.relative_to(root).as_posix(), "bytes": panel_path.stat().st_size,
                     "sha256": file_hash(panel_path)}]}
    (out / "execution_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replay = verify_multimodal_panel(root, panel_path)
    replay_path = out / "source_array_replay.json"
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["independent_source_array_replay"] = replay
    receipt["schema_sha256"] = file_hash(Path(__file__).with_name("schemas") / "wp05_multimodal_panel.json")
    receipt["implementation_sha256"] = file_hash(Path(__file__))
    receipt["outputs"].append({"path": replay_path.relative_to(root).as_posix(),
        "bytes": replay_path.stat().st_size, "sha256": file_hash(replay_path)})
    (out / "execution_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def validate_multimodal_panel_document(document: dict) -> None:
    validate_schema(document, "wp05_multimodal_panel")
    _require(document["schema_version"] == PANEL_SCHEMA_VERSION and document["bundle_id"] == PANEL_BUNDLE_ID,
             "unsupported multimodal panel schema/bundle identity")
    _require(document["mode"] == "DEVELOPMENT_DIAGNOSTIC" and
             document["research_admissible"] is False and document["training_admitted"] is False and
             document["plan_executable"] is False and document["release_status"] == "BLOCKED",
             "multimodal panel must remain quarantined and training-blocked")
    _require(document["fit_transform_state"] is None and
             document["fit_transform_status"] == "RAW_VALUES_NO_FITTED_TRANSFORM",
             "development panel must not contain a fitted transform")
    _require(document["graph_separation"]["G_S"]["status"] == "UNRESOLVED_PHYSICAL_MEMBERSHIP" and
             document["graph_separation"]["G_P"]["status"] == "NO_FIT_MEMBERSHIP_OR_FITTED_STATE" and
             document["graph_separation"]["shared_grid_is_physical_registration"] is False,
             "physical sampling and provenance/fitting graph status is invalid")
    _require(document["independent_support"]["value"] is None and
             document["independent_support"]["status"] == "UNKNOWN",
             "computational sensor groups cannot be promoted to independent support")
    groups = document["groups"]
    ids = [x["group_id"] for x in groups]
    _require(len(ids) == len(set(ids)) and len(groups) == 4,
             "expected four distinct sensor acquisition groups, without padding/duplicates")
    for panel in document["panels"]:
        _require(len(panel["group_ids"]) == 2 and len(set(panel["group_ids"])) == 2,
                 "each exposed site must preserve its two separate asynchronous acquisitions")
        members = [next(g for g in groups if g["group_id"] == x) for x in panel["group_ids"]]
        _require({g["sensor"] for g in members} == {"SENTINEL_1_IW_GRD", "SENTINEL_2_OPTICAL"},
                 "panel merges sensors or pads a missing sensor group")
        _require(members[0]["site_id"] == panel["site_id"] == members[1]["site_id"],
                 "panel group membership crosses pilot site")
        _require(members[0]["group_id"] != members[1]["group_id"], "duplicate sensor group membership")
    for group in groups:
        expected_channels = OPTICAL_CHANNELS if group["sensor"] == "SENTINEL_2_OPTICAL" else SAR_CHANNELS
        channels = [x["channel_id"] for x in group["channels"]]
        _require(channels == list(expected_channels), f"wrong or duplicate actual channel membership in {group['group_id']}")
        if group["sensor"] == "SENTINEL_2_OPTICAL":
            unavailable = group["unavailable_channels"]
            _require([(x["channel_id"], x["status"], x["value"]) for x in unavailable] ==
                     [(name, "UNAVAILABLE", None) for name in ("B02", "B04", "B12")],
                     "unstaged optical channels must remain explicitly unavailable")
            for channel in group["channels"]:
                _require(channel["units"] == OPTICAL_UNITS[channel["channel_id"]] and
                         channel["validity"] == "PER_COORDINATE_MASK" and
                         channel["status"] == "OBSERVED_SOURCE_DERIVED_QUARANTINED",
                         f"optical channel unit, validity or quarantine status changed: {channel['channel_id']}")
                if channel["channel_id"] == "NDWI":
                    _require(channel["radiometry"]["formula"] == "NDWI=(B03-B08)/(B03+B08)" and
                             channel["radiometry"]["source_channels"] == ["B03", "B08"],
                             "NDWI formula or source channels changed")
                if channel["channel_id"] == "MNDWI":
                    _require(channel["radiometry"]["formula"] == "MNDWI=(B03-B11)/(B03+B11)" and
                             channel["radiometry"]["source_channels"] == ["B03", "B11"] and
                             channel["radiometry"]["stored_channel_aliases"] == ["NDSI"],
                             "MNDWI formula/alias metadata changed or was duplicated")
        else:
            _require(group["unavailable_channels"] == [],
                     "SAR group cannot declare optical channels as observed")
            for channel in group["channels"]:
                _require(channel["units"] == SAR_UNITS and
                         channel["status"] == "OBSERVED_RAW_SIGMA0_TIEPOINT_DIAGNOSTIC_QUARANTINED" and
                         channel["radiometry"]["representation"] == "RAW_SIGMA0_LINEAR" and
                         channel["radiometry"]["noise_subtraction"] == "NOT_APPLIED_TO_THIS_PANEL_CHANNEL" and
                         channel["radiometry"]["db_alternative"]["epsilon_clamp"] is False,
                         f"SAR raw radiometry or diagnostic status changed: {channel['channel_id']}")
        _require(group["scope_membership"]["role"] is None and
                 group["scope_membership"]["scientific_source_memberships"] == [] and
                 group["admission"]["research_admissible"] is False and
                 group["admission"]["training_admitted"] is False,
                 "unresolved scope/admission was assigned a role or promoted")
        _require(group["site_exposure"] == "DEVELOPMENT_EXPOSED" and
                 group["admission"]["status"] == "DEVELOPMENT_DIAGNOSTIC_QUARANTINED",
                 "exposed diagnostic inputs cannot be relabeled or admitted")
        _require(group["sampling_graph"]["physical_unit_id"] is None and
                 group["sampling_graph"]["status"] == "UNRESOLVED" and
                 group["provenance_graph"]["fitted_state_id"] is None,
                 "G_S/G_P identifiers cannot be invented from site/pixel/token count")
        _require(group["acquisition"]["historical_availability"]["status"] == "UNKNOWN" and
                 group["acquisition"]["historical_availability"]["at"] is None,
                 "unknown historical availability cannot be imputed")


def _verify_input_bindings(root: Path, document: Mapping[str, Any]) -> dict:
    inputs = document["inputs"]
    paths = [x["path"] for x in inputs]
    _require(len(paths) == len(set(paths)) and bool(paths),
             "panel input bindings must be nonempty and path-unique")
    missing_retired = []
    verified_input_count = 0
    for item in inputs:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise AdmissionError("panel input paths must be root-relative and traversal-free")
        candidate = root / path
        exists = candidate.is_file() and not any(
            parent.is_symlink() for parent in (candidate, *candidate.parents)
            if parent != root)
        if exists:
            verified = _root_file(root, item["path"], item["sha256"], item["bytes"])
            _require(verified.stat().st_size == item["bytes"],
                     f"panel input byte count changed: {item['path']}")
            verified_input_count += 1
            continue
        if item.get("role") != "excluded_native_signed_sar_variant":
            raise AdmissionError(f"unlisted or active panel input is missing: {item['path']}")
        missing_retired.append(item)
    required = {
        document["accepted_inputs"]["optical_features"]["path"],
        document["accepted_inputs"]["optical_features"]["receipt_path"],
        document["accepted_inputs"]["native_sar_execution"]["path"],
    }
    _require(required <= set(paths), "accepted source records are missing from panel input bindings")
    if not missing_retired:
        return {"status": "PASS_ALL_PANEL_INPUT_BYTES_PRESENT",
                "verified_input_count": verified_input_count,
                "retired_excluded_input_count": 0,
                "retired_excluded_inputs": [],
                "fresh_retired_native_pixel_replay": False}

    retention_rel = "project/artifact_retention_2023.json"
    retention_path = _root_file(root, retention_rel, file_hash(root / retention_rel))
    retention = _read_json(retention_path)
    _require(retention.get("schema_version") == 1 and
             type(retention.get("schema_version")) is int and
             retention.get("authority") == "HUMAN_PI_2026_10_15_EXPLICIT_NATIVE_INTERMEDIATE_CLEANUP" and
             retention.get("disposition") == "ALL_LISTED_NATIVE_BYTES_HASH_VERIFIED_BEFORE_AUTHORIZED_DELETION",
             "native intermediate retirement manifest lacks the declared PI authorization")
    _require(retention.get("raw_archive_retained") is True,
             "retirement manifest does not retain the source archive")
    entries = retention.get("entries")
    roi_bindings = retention.get("retained_roi_bindings")
    _require(type(entries) is list and type(roi_bindings) is list,
             "retirement manifest lacks retired-entry or retained-ROI inventories")
    retired_by_path = {}
    for entry in entries:
        _require(type(entry) is dict and set(entry) >= {"path", "bytes", "sha256", "disposition"},
                 "retirement manifest entry is missing path, size, SHA-256 or disposition")
        rel = entry["path"]
        _require(isinstance(rel, str) and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
                 "retirement manifest contains an unsafe path")
        _require(type(entry["bytes"]) is int and entry["bytes"] >= 0 and
                 isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64 and
                 all(char in "0123456789abcdef" for char in entry["sha256"]) and
                 entry["disposition"] == "RETIRED_VERIFIED_RECOMPUTABLE_INTERMEDIATE",
                 "retirement manifest entry has invalid former identity or disposition")
        _require(rel not in retired_by_path, "retirement manifest repeats a retired path")
        retired_by_path[rel] = entry

    execution_binding = document["accepted_inputs"]["native_sar_execution"]
    _require(retention.get("native_execution_record") == execution_binding["path"],
             "retirement manifest points to a different native execution record")
    execution_path = _root_file(root, execution_binding["path"],
                                execution_binding["sha256"], execution_binding["bytes"])
    execution = _read_json(execution_path)
    execution_outputs = {}
    for output in execution.get("outputs", []):
        _require(type(output) is dict and isinstance(output.get("path"), str),
                 "native execution output inventory is malformed")
        _require(output["path"] not in execution_outputs,
                 "native execution output inventory repeats a path")
        execution_outputs[output["path"]] = output
    for rel, entry in retired_by_path.items():
        output = execution_outputs.get(rel)
        _require(output is not None and output.get("bytes") == entry["bytes"] and
                 output.get("sha256") == entry["sha256"],
                 f"retirement identity differs from native execution record: {rel}")

    retained_by_path = {}
    for item in roi_bindings:
        _require(type(item) is dict and set(item) >= {"path", "bytes", "sha256"},
                 "retained ROI binding is malformed")
        rel = item["path"]
        _require(isinstance(rel, str) and rel not in retained_by_path,
                 "retained ROI inventory contains a duplicate or invalid path")
        _require(type(item["bytes"]) is int and item["bytes"] >= 0 and
                 isinstance(item["sha256"], str) and len(item["sha256"]) == 64 and
                 all(char in "0123456789abcdef" for char in item["sha256"]),
                 "retained ROI binding has invalid byte identity")
        output = execution_outputs.get(rel)
        _require(output is not None and output.get("bytes") == item["bytes"] and
                 output.get("sha256") == item["sha256"],
                 f"retained ROI identity differs from native execution record: {rel}")
        roi_file = _root_file(root, rel, item["sha256"], item["bytes"])
        retained_by_path[rel] = item
    if "retained_outputs" in retention:
        _require(retention["retained_outputs"] == roi_bindings,
                 "retirement manifest retained ROI inventories disagree")

    verified_retirements = []
    for missing in missing_retired:
        rel = missing["path"]
        entry = retired_by_path.get(rel)
        _require(entry is not None and entry["bytes"] == missing["bytes"] and
                 entry["sha256"] == missing["sha256"],
                 f"missing excluded panel input is not exactly listed in the retirement manifest: {rel}")
        references = []
        for group in document.get("groups", []):
            lineage = group.get("lineage", {})
            signed = lineage.get("signed_noise_subtracted_native_variants_excluded_from_roi_panel", {})
            for asset in signed.get("assets", []):
                if asset.get("path") == rel:
                    _require(asset.get("bytes") == missing["bytes"] and
                             asset.get("sha256") == missing["sha256"],
                             f"retired input lineage identity differs: {rel}")
                    _require(group.get("sensor") == "SENTINEL_1_IW_GRD",
                             f"retired native signed input is attached to a non-SAR group: {rel}")
                    references.append(group)
        _require(bool(references),
                 f"retired input is not referenced only as an excluded native SAR variant: {rel}")
        affected_sites = sorted({group["site_id"] for group in references})
        site_roi = {}
        for site_id in affected_sites:
            site_prefix = site_id.replace(" ", "-") + "__"
            selected = {name: item for name, item in retained_by_path.items()
                        if Path(name).name.startswith(site_prefix)}
            required_markers = ("_VV_sigma0_raw_linear_", "_VH_sigma0_raw_linear_",
                                "_VV_sigma0_raw_db_", "_VH_sigma0_raw_db_",
                                "_VV_valid_", "_VH_valid_",
                                "_VV_linear_valid_weight_coverage_",
                                "_VH_linear_valid_weight_coverage_")
            names = [Path(name).name for name in selected]
            missing_markers = [marker for marker in required_markers
                               if not any(marker in name for name in names)]
            _require(not missing_markers,
                     f"retained ROI proof set is incomplete for {site_id}: {missing_markers}")
            site_roi[site_id] = sorted(selected)
        verified_retirements.append({
            "path": rel, "former_bytes": missing["bytes"],
            "former_sha256": missing["sha256"],
            "disposition": entry["disposition"],
            "identity_status": "FORMER_BYTES_MATCH_AUTHORIZED_RETIREMENT_RECORD",
            "retained_roi_bindings_by_site": site_roi,
            "fresh_retired_native_pixel_replay": False,
        })
    return {
        "status": "PASS_EXPLICITLY_RETIRED_EXCLUDED_NATIVE_INPUTS",
        "retention_manifest": {"path": retention_rel,
            "bytes": retention_path.stat().st_size, "sha256": file_hash(retention_path)},
        "verified_input_count": verified_input_count,
        "retired_excluded_input_count": len(verified_retirements),
        "retired_excluded_inputs": verified_retirements,
        "retained_roi_binding_count": len(retained_by_path),
        "fresh_retired_native_pixel_replay": False,
        "retention_limit": "Former SHA-256 identities are verified from the authorized manifest; deleted native arrays are not read or replayed.",
    }


def _read_panel(root: Path, panel_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    document = _read_json(panel_path)
    validate_multimodal_panel_document(document)
    _verify_input_bindings(root, document)
    arrays_binding = document["arrays_file"]
    arrays_path = _root_file(root, arrays_binding["path"], arrays_binding["sha256"], arrays_binding["bytes"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        _require(sorted(archive.files) == sorted(arrays_binding["keys"]), "panel array membership changed")
        arrays = {name: archive[name] for name in archive.files}
    return document, arrays


def verify_multimodal_panel(root: str | Path, panel_path: str | Path) -> dict:
    """Independently replay each emitted pixel channel from accepted source arrays."""
    root = Path(root).resolve(strict=True)
    path = Path(panel_path)
    if not path.is_absolute(): path = root / path
    path = path.resolve(strict=True)
    _require(path.is_relative_to(root), "panel manifest must be a retained workspace file")
    document, arrays = _read_panel(root, path)
    input_retention = _verify_input_bindings(root, document)
    accepted = document["accepted_inputs"]
    feature_path = _root_file(root, accepted["optical_features"]["path"],
                               accepted["optical_features"]["sha256"], accepted["optical_features"]["bytes"])
    receipt_path = _root_file(root, accepted["optical_features"]["receipt_path"],
                               file_hash(root/accepted["optical_features"]["receipt_path"]))
    features = _read_json(feature_path)
    bundle_path = root/features["input_bundle_path"]
    feature_receipt, feature_check = _verify_optical_inputs(root, feature_path, receipt_path, features)
    _require(feature_check["status"] == "PASS_LOCAL_DEVELOPMENT_FEATURE_INTEGRITY_ONLY",
             "independent feature replay failed")
    sar = _root_file(root, accepted["native_sar_execution"]["path"],
                     accepted["native_sar_execution"]["sha256"], accepted["native_sar_execution"]["bytes"])
    execution = _read_json(sar)
    safe_product, _ = _archive_members(root, execution)
    outputs = _source_output_map(execution)
    record_by_site = {x["site_id"]: x for x in features["records"]}
    source_array_checks = 0
    for group in document["groups"]:
        prefix = group["arrays"]["values_key"].rsplit("_values", 1)[0]
        emitted = arrays[group["arrays"]["values_key"]]
        valid_emitted = arrays[group["arrays"]["observed_valid_key"]]
        research_emitted = arrays[group["arrays"]["research_admissible_key"]]
        _require(emitted.shape == tuple(group["arrays"]["shape"]) and
                 valid_emitted.shape == emitted.shape and research_emitted.shape == emitted.shape,
                 f"panel values/masks shape mismatch: {group['group_id']}")
        _require(not research_emitted.any() and np.isfinite(emitted[valid_emitted]).all() and
                 np.isnan(emitted[~valid_emitted]).all(),
                 f"canonical missingness or quarantine mask changed: {group['group_id']}")
        if group["sensor"] == "SENTINEL_2_OPTICAL":
            record = record_by_site[group["site_id"]]
            _require(group["product_id"] == record["product_id"] and
                     group["source_id"] == record["source_id"] and
                     group["support_id"] == record["support_id"] and
                     group["acquisition"]["at"] == record["time_axes"]["acquisition"]["at"] and
                     group["acquisition"]["timezone_status"] == "EXPLICIT_UTC",
                     f"optical product, support, source or acquisition binding changed: {group['group_id']}")
            expected_grid = {key: record["grid"][key] for key in ("crs", "shape", "transform")}
            _require({key: group["grid"][key] for key in expected_grid} == expected_grid,
                     f"optical panel grid differs from accepted source grid: {group['group_id']}")
            _require(group["grid"]["transform_sha256"] == _sha_json(expected_grid) and
                     group["grid"]["cross_sensor_registration"] ==
                     "SHARED_GRID_ONLY_PIXEL_REGISTRATION_UNVERIFIED",
                     f"optical transform digest or registration status changed: {group['group_id']}")
            source = _asset_map(record)
            expected_lineage = sorted((item["path"], item["sha256"], item["bytes"], item["layer"])
                                      for item in group["lineage"]["decoded_assets"])
            required_layers = set(OPTICAL_LAYER_BY_CHANNEL.values()) | set(OPTICAL_VALID_LAYER_BY_CHANNEL.values()) | {
                "scl_raw_20m_classes", "scl_valid_mask", "scl_clear_4_5_6_mask"}
            actual_lineage = sorted((source[name]["path"], source[name]["sha256"],
                                     source[name]["bytes"], name) for name in required_layers)
            _require(expected_lineage == actual_lineage,
                     f"optical decoded-asset lineage differs from accepted feature record: {group['group_id']}")
            source_channels = {x["channel_id"]: x for x in record["channels"]}
            for descriptor in group["channels"]:
                channel = descriptor["channel_id"]
                if channel in {"B03", "B08", "B11"}:
                    _require(descriptor["radiometry"] == source_channels[channel]["calibration"],
                             f"optical band calibration differs from accepted record: {group['group_id']}/{channel}")
            source_valid = {}
            for channel, layer in OPTICAL_VALID_LAYER_BY_CHANNEL.items():
                valid_item = source[layer]
                mask, _, _ = _raster(root, valid_item, expected_grid=expected_grid)
                source_valid[channel] = mask.astype(bool)
            source_values = {}
            for ci, channel in enumerate(group["arrays"]["channel_axis_order"]):
                item = source[OPTICAL_LAYER_BY_CHANNEL[channel]]
                raw, _, _ = _raster(root, item, expected_grid=expected_grid)
                raw = raw.astype(np.float32)
                if channel in source_valid:
                    expected_valid = source_valid[channel] & np.isfinite(raw)
                else:
                    left_name, right_name = (("B03", "B08") if channel == "NDWI" else ("B03", "B11"))
                    left = source_values.get(left_name)
                    right = source_values.get(right_name)
                    if left is None:
                        left, _, _ = _raster(root, source[OPTICAL_LAYER_BY_CHANNEL[left_name]], expected_grid=expected_grid)
                        left = left.astype(np.float32)
                    if right is None:
                        right, _, _ = _raster(root, source[OPTICAL_LAYER_BY_CHANNEL[right_name]], expected_grid=expected_grid)
                        right = right.astype(np.float32)
                    denom = left.astype(np.float64) + right.astype(np.float64)
                    expected_valid = (source_valid[left_name] & source_valid[right_name] &
                        np.isfinite(left) & np.isfinite(right) & np.isfinite(denom) & (denom != 0))
                    replay = np.full(raw.shape, np.nan, dtype=np.float32)
                    replay[expected_valid] = ((left.astype(np.float64)[expected_valid] -
                        right.astype(np.float64)[expected_valid]) / denom[expected_valid]).astype(np.float32)
                    _require(np.allclose(replay, raw, rtol=0, atol=INDEX_ATOL, equal_nan=True),
                             f"source-array {channel} replay differs for {group['group_id']}")
                expected_values = raw.copy(); expected_values[~expected_valid] = np.nan
                _require(np.array_equal(expected_valid, valid_emitted[ci]) and
                         np.allclose(expected_values, emitted[ci], rtol=0, atol=0, equal_nan=True),
                         f"source-array replay differs for {group['group_id']}/{channel}")
                source_values[channel] = raw
                source_array_checks += 1
            for qa_name, source_layer in (("scl_classes", "scl_raw_20m_classes"),
                    ("scl_valid", "scl_valid_mask"), ("scl_clear_4_5_6", "scl_clear_4_5_6_mask")):
                qa_item = source[source_layer]
                qa_source, _, _ = _raster(root, qa_item, expected_grid=expected_grid)
                _require(np.array_equal(qa_source, arrays[group["arrays"]["qa_keys"][qa_name]]),
                         f"optical QA source-array replay differs: {group['group_id']}/{qa_name}")
        else:
            roi = execution["roi_extraction"][group["site_id"]]
            _require(group["product_id"] == safe_product, "SAR SAFE product identity changed")
            sar_product = execution["products"]["VV"]
            _require(group["acquisition"]["start_raw"] == sar_product["start_time"] and
                     group["acquisition"]["end_raw"] == sar_product["stop_time"] and
                     group["acquisition"]["timezone_status"] ==
                     "NOT_EXPLICIT_IN_RETAINED_START_STOP_STRINGS",
                     f"SAR source interval or timezone status changed: {group['group_id']}")
            expected_grid = {key: roi["target_grid"][key] for key in ("crs", "shape", "transform")}
            _require({key: group["grid"][key] for key in expected_grid} == expected_grid,
                     f"SAR panel grid differs from retained diagnostic grid: {group['group_id']}")
            _require(group["grid"]["transform_sha256"] == _sha_json(expected_grid) and
                     group["grid"]["cross_sensor_registration"] ==
                     "SHARED_GRID_ONLY_PIXEL_REGISTRATION_UNVERIFIED",
                     f"SAR transform digest or registration status changed: {group['group_id']}")
            expected_lineage = sorted((item["path"], item["sha256"], item["bytes"])
                for polarization in ("VV", "VH")
                for item in roi["diagnostic_products"][polarization]["outputs"].values())
            actual_lineage = sorted((item["path"], item["sha256"], item["bytes"])
                                    for item in group["lineage"]["diagnostic_assets"])
            _require(actual_lineage == expected_lineage,
                     f"SAR diagnostic lineage differs from retained output manifest: {group['group_id']}")
            for ci, pol in enumerate(("VV", "VH")):
                diag = roi["diagnostic_products"][pol]
                linear_item, valid_item, coverage_item, db_item = (diag["outputs"][key]
                    for key in ("linear", "valid", "coverage", "db"))
                values, _, _ = _raster(root, linear_item, expected_grid=expected_grid)
                source_valid, _, _ = _raster(root, valid_item, expected_grid=expected_grid)
                coverage, _, _ = _raster(root, coverage_item, expected_grid=expected_grid)
                db, _, _ = _raster(root, db_item, expected_grid=expected_grid)
                expected_valid = source_valid.astype(bool) & np.isfinite(values) & (coverage > 0)
                expected_values = values.astype(np.float32); expected_values[~expected_valid] = np.nan
                _require(np.array_equal(expected_valid, valid_emitted[ci]) and
                         np.allclose(expected_values, emitted[ci], rtol=0, atol=0, equal_nan=True),
                         f"SAR source-array replay differs: {group['group_id']}/{pol}")
                _require(np.array_equal(coverage.astype(np.float32), arrays[group["arrays"]["coverage_key"]][ci]),
                         f"SAR coverage source replay differs: {group['group_id']}/{pol}")
                expected_db_valid = expected_valid & np.isfinite(values) & (values > 0)
                expected_db = np.full(values.shape, np.nan, dtype=np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    expected_db[expected_db_valid] = (10.0 * np.log10(
                        values.astype(np.float64)[expected_db_valid])).astype(np.float32)
                db_source = db.astype(np.float32)
                _require(np.array_equal(np.isfinite(db_source), expected_db_valid) and
                         np.allclose(db_source, expected_db, rtol=0,
                                     atol=DB_FLOAT32_ROUNDING_ATOL, equal_nan=True),
                         f"strict positive-finite raw dB conversion replay differs: {group['group_id']}/{pol}")
                expected_db[~expected_db_valid] = np.nan
                _require(np.array_equal(expected_db_valid, arrays[group["arrays"]["raw_db_valid_key"]][ci]) and
                         np.allclose(expected_db, arrays[group["arrays"]["raw_db_alternative_key"]][ci],
                                     rtol=0, atol=0, equal_nan=True),
                         f"strict raw dB sidecar replay differs: {group['group_id']}/{pol}")
                source_array_checks += 1
    return {"status":"PASS_SOURCE_BOUND_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "research_admissible":False,"training_admitted":False,
        "verified_groups":len(document["groups"]),"verified_panels":len(document["panels"]),
        "source_array_channel_replays":source_array_checks,
        "source_memberships":len(document["inputs"]),"optical_feature_receipt_status":feature_receipt["status"],
        "native_sar_research_admissible":execution["research_admissible"],
        "plan_executable":False,"release_status":"BLOCKED",
        "input_retention":input_retention,
        "fresh_retired_native_pixel_replay":False}


class RaggedPanelDataset:
    """Raw ragged groups; loading requires explicit quarantined diagnostic mode."""
    def __init__(self, root: str | Path, panel_path: str | Path, *, mode: str = "research",
                 transform_state: Mapping[str, Any] | None = None):
        if mode != "development-diagnostic":
            raise AdmissionError("research/training loader admission is blocked for this quarantined panel")
        if transform_state is not None:
            raise AdmissionError("fitted transforms are not enabled; only raw unfit diagnostic values are exposed")
        self.root = Path(root).resolve(strict=True)
        self.panel_path = Path(panel_path)
        if not self.panel_path.is_absolute(): self.panel_path = self.root/self.panel_path
        self.document, self.arrays = _read_panel(self.root, self.panel_path.resolve(strict=True))
        verification = verify_multimodal_panel(self.root, self.panel_path)
        _require(verification["status"] == "PASS_SOURCE_BOUND_DEVELOPMENT_DIAGNOSTIC_ONLY",
                 "panel source replay did not pass before loader admission")
        self.panels = self.document["panels"]

    def __len__(self) -> int:
        return len(self.panels)

    def __getitem__(self, index: int) -> dict:
        panel = self.panels[index]
        groups = []
        for group_id in panel["group_ids"]:
            metadata = next(x for x in self.document["groups"] if x["group_id"] == group_id)
            key = metadata["arrays"]["values_key"]
            groups.append({"metadata": metadata,
                "values": self.arrays[key].copy(),
                "observed_valid": self.arrays[metadata["arrays"]["observed_valid_key"]].copy(),
                "research_admissible": self.arrays[metadata["arrays"]["research_admissible_key"]].copy(),
                "channel_ids": tuple(metadata["arrays"]["channel_axis_order"]),
                "quality": {name:self.arrays[key].copy() for name,key in metadata["arrays"].get("qa_keys", {}).items()},
                "coverage": self.arrays[metadata["arrays"]["coverage_key"]].copy()
                    if "coverage_key" in metadata["arrays"] else None,
                "representations": {"raw_db": self.arrays[metadata["arrays"]["raw_db_alternative_key"]].copy(),
                    "raw_db_valid": self.arrays[metadata["arrays"]["raw_db_valid_key"]].copy()}
                    if "raw_db_alternative_key" in metadata["arrays"] else {}})
        return {"panel_id":panel["panel_id"],"site_id":panel["site_id"],"groups":groups,
            "scope_membership":panel["scope_membership"],"sampling_graph":panel["sampling_graph"],
            "provenance_graph":panel["provenance_graph"],"site_exposure":panel["site_exposure"],
            "research_admissible":False}


def collate_ragged_groups(samples: Sequence[Mapping[str, Any]], *,
                          target_masks: Sequence[Sequence[np.ndarray | None]] | None = None) -> dict:
    """Build leak-safe context; requested targets and dependency-hidden values stay distinct."""
    _require(bool(samples), "cannot collate an empty sample batch")
    if target_masks is not None:
        _require(len(target_masks) == len(samples), "target mask sample count differs from batch")
    max_groups = max(len(sample["groups"]) for sample in samples)
    _require(max_groups > 0, "sample batch contains no acquisition groups")
    max_channels = max(group["values"].shape[0] for sample in samples for group in sample["groups"])
    max_pixels = max(group["values"].shape[1] * group["values"].shape[2]
                     for sample in samples for group in sample["groups"])
    shape = (len(samples), max_groups, max_channels, max_pixels)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    channel_mask = np.zeros((len(samples), max_groups, max_channels), dtype=bool)
    attention = np.zeros((len(samples), max_groups), dtype=bool)
    padding = np.ones(shape, dtype=bool)
    targets = np.zeros(shape, dtype=bool)
    dependency_hidden = np.zeros(shape, dtype=bool)
    target_values = np.full(shape, np.nan, dtype=np.float32)
    metadata: list[dict] = []
    for bi, sample in enumerate(samples):
        sample_meta = {key:sample.get(key) for key in
            ("panel_id", "site_id", "scope_membership", "sampling_graph", "provenance_graph")}
        group_meta = []
        masks_for_sample = target_masks[bi] if target_masks is not None else [None] * len(sample["groups"])
        _require(len(masks_for_sample) == len(sample["groups"]), "target mask group count differs from ragged sample")
        for gi, group in enumerate(sample["groups"]):
            x = np.asarray(group["values"], dtype=np.float32)
            valid = np.asarray(group["observed_valid"], dtype=bool)
            _require(x.ndim == 3 and valid.shape == x.shape and x.shape[0] == len(group["channel_ids"]),
                     "group values, validity and channel membership shapes disagree")
            _require(np.isfinite(x[valid]).all() and np.isnan(x[~valid]).all(),
                     "group missingness must be NaN with false observed-valid mask")
            c, h, w = x.shape; p = h*w
            artificial = masks_for_sample[gi]
            if artificial is None:
                artificial_mask = np.zeros(x.shape, dtype=bool)
            else:
                artificial_mask = np.asarray(artificial, dtype=bool)
                _require(artificial_mask.shape == x.shape, "artificial target mask shape differs from group")
                _require(np.all(artificial_mask <= valid), "reconstruction target includes missing/invalid observations")
            flat_x, flat_valid = x.reshape(c,p), valid.reshape(c,p)
            flat_target = artificial_mask.reshape(c,p)
            group_record = group.get("metadata", {})
            if flat_target.any():
                _require(group_record.get("sensor") in {"SENTINEL_1_IW_GRD", "SENTINEL_2_OPTICAL"},
                         "cannot form a leak-safe target context for an untyped sensor group")
            flat_padding = padding[bi,gi,:c,:p]
            flat_padding[:] = False
            attention[bi,gi] = True
            channel_mask[bi,gi,:c] = True
            observed[bi,gi,:c,:p] = flat_valid
            targets[bi,gi,:c,:p] = flat_target
            target_values[bi,gi,:c,:p][flat_target] = flat_x[flat_target]
            if group_record.get("sensor") == "SENTINEL_2_OPTICAL":
                # NDWI and MNDWI are deterministic functions of their optical
                # source bands. Hide the full observed optical channel vector
                # at every target pixel without enlarging the requested target
                # set. Separate acquisitions/modalities stay explicit context.
                target_pixels = flat_target.any(axis=0)
                closed = np.broadcast_to(target_pixels, (c, p)) & flat_valid & ~flat_target
                dependency_hidden[bi,gi,:c,:p] = closed
            visible = flat_valid & ~flat_target & ~dependency_hidden[bi,gi,:c,:p]
            values[bi,gi,:c,:p] = np.where(visible, flat_x, 0.0)
            group_meta.append({"group_id":group_record.get("group_id"),
                "channel_ids":list(group["channel_ids"]),"spatial_shape":[h,w],
                "sensor":group_record.get("sensor"),
                "channel_descriptors":group_record.get("channels", []),
                "acquisition":group_record.get("acquisition"),
                "source_id":group_record.get("source_id"),
                "product_id":group_record.get("product_id"),
                "product_uri":group_record.get("product_uri"),
                "support_id":group_record.get("support_id"),
                "grid":group_record.get("grid"),
                "site_exposure":group_record.get("site_exposure"),
                "scope_membership":group_record.get("scope_membership"),
                "sampling_graph":group_record.get("sampling_graph"),
                "provenance_graph":group_record.get("provenance_graph"),
                "lineage":group_record.get("lineage"),
                "admission":group_record.get("admission"),
                "context_dependency_policy":CONTEXT_POLICY_VERSION,
                "group_attention":True,"target_count":int(flat_target.sum())})
        sample_meta["groups"] = group_meta
        metadata.append(sample_meta)
    target_count = int(targets.sum())
    return {"values":values,"encoder_context_values":values,
        "observed_valid":observed,"context_visible":observed & ~targets & ~dependency_hidden,
        "target_mask":targets,"artificial_target_mask":targets.copy(),
        "dependency_hidden_mask":dependency_hidden,"target_values":target_values,
        "dependency_hidden_status":"OBSERVED_CONTEXT_VALUES_MASKED_BY_OPTICAL_INDEX_DEPENDENCY_CLOSURE",
        "natural_missing_status":"ORIGINAL_OBSERVATION_MASK_FALSE_WITH_NAN_VALUE",
        "channel_membership_mask":channel_mask,"attention_mask":attention,
        "padding_mask":padding,"natural_missing_mask":(~observed) & (~padding),
        "context_policy_version":CONTEXT_POLICY_VERSION,
        "context_policy":{"optical_target_rule":"At each pixel with any requested optical target, hide every observed B03/B08/B11/NDWI/MNDWI channel in that same acquisition group from encoder context.",
            "target_set":"Unchanged requested true-valid coordinates; dependency hiding never adds loss targets.",
            "other_sensor_groups":"Retained as separate asynchronous context; cross-sensor prediction is not labeled same-acquisition reconstruction."},
        "status":"TARGET_SUPPORT_AVAILABLE" if target_count else "SKIP_EMPTY_OBSERVED_TARGET_SUPPORT",
        "estimable":bool(target_count),"target_count":target_count,
        "independent_support_count":None,"independent_support_status":"UNKNOWN",
        "sample_metadata":metadata,"padding_value":0.0,
        "padding_value_semantics":"NEUTRAL_COMPUTATIONAL_VALUE_ONLY_BEHIND_FALSE_MASKS"}

