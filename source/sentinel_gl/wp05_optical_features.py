"""Source-bound, quarantined optical summaries for the WP05 development pilot.

This module reads only the already retained WP03 ROI decode. It does not admit
scientific measurements or create a temporal panel.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import rasterio

from .admission import AdmissionError, validate_schema
from .integrity import file_hash, loads_strict, verified_file
from .wp04_manifests import verify_bundle as verify_wp04_bundle


FEATURE_SCHEMA_VERSION = 1
FEATURE_BUNDLE_ID = "WP05-OPTICAL-DEVELOPMENT-FEATURES-V1"
DECODE_ATTEMPT_ID = "attempt_2026-10-14_roi_decode_005"
QA_CLASSES = (4, 5, 6)
INDEX_RTOL = 0.0
INDEX_ATOL = 2e-6
CALIBRATION_ATOL = 2e-6

LAYER_NAMES = (
    "b03_raw_20m_mean_dn.tif", "b03_reflectance_20m.tif", "b03_valid_mask.tif",
    "b03_valid_support_count.tif", "b08_raw_20m_mean_dn.tif", "b08_reflectance_20m.tif",
    "b08_valid_mask.tif", "b08_valid_support_count.tif", "b11_raw_20m_mean_dn.tif",
    "b11_reflectance_20m.tif", "b11_valid_mask.tif", "b11_valid_support_count.tif",
    "scl_raw_20m_classes.tif", "scl_valid_mask.tif", "scl_clear_4_5_6_mask.tif",
    "scl_qa_excluded_mask.tif", "analysis_valid_mask.tif", "unknown_mask.tif",
    "ndwi.tif", "mndwi.tif", "candidate_water_mask.tif",
)

RAW_ROLES = {"B03": "green", "B08": "nir", "B11": "swir16", "SCL": "scl"}
REFLECTANCE_LAYER = {"B03": "b03_reflectance_20m.tif",
                     "B08": "b08_reflectance_20m.tif",
                     "B11": "b11_reflectance_20m.tif"}
RAW_DN_LAYER = {"B03": "b03_raw_20m_mean_dn.tif",
                 "B08": "b08_raw_20m_mean_dn.tif",
                 "B11": "b11_raw_20m_mean_dn.tif"}
VALID_LAYER = {"B03": "b03_valid_mask.tif", "B08": "b08_valid_mask.tif",
               "B11": "b11_valid_mask.tif"}

MISSING_CHANNELS = (
    ("B04", "NOT_STAGED_BY_AUTHORIZED_SCOPE", "B04 source bytes were not staged."),
    ("B02", "NOT_STAGED_BY_AUTHORIZED_SCOPE", "B02 source bytes were not staged."),
    ("B12", "NOT_STAGED_BY_AUTHORIZED_SCOPE", "B12 source bytes were not staged."),
    ("SAR_VV", "NO_SAR_MEASUREMENT_BYTES", "The retained WP04 source set has no SAR measurement bytes."),
    ("SAR_VH", "NO_SAR_MEASUREMENT_BYTES", "The retained WP04 source set has no SAR measurement bytes."),
    ("WEATHER_TEMPERATURE", "NO_WEATHER_MEASUREMENT_BYTES", "No admitted weather source is bound to this product."),
    ("WEATHER_PRECIPITATION", "NO_WEATHER_MEASUREMENT_BYTES", "No admitted weather source is bound to this product."),
    ("DEM_ELEVATION", "NO_DEM_MEASUREMENT_BYTES", "No admitted DEM source is bound to this product."),
    ("TERRAIN_MASK", "NO_VERIFIED_TERRAIN_INPUT", "Terrain, shadow and layover support is unverified."),
    ("VELOCITY", "NO_VELOCITY_MEASUREMENT_BYTES", "No independent velocity measurement is retained."),
    ("COHERENCE", "NO_INTERFEROMETRIC_PHASE", "GRD does not supply interferometric phase/coherence."),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _read_json(path: Path) -> dict:
    value = loads_strict(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, f"expected JSON object: {path}")
    return value


def _sha_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    import hashlib
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_number(value, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"invalid finite number: {label}")
    return float(value)


def _load_context(root: str | Path, wp04_bundle_path: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    bundle_path = Path(wp04_bundle_path)
    if not bundle_path.is_absolute():
        bundle_path = root / bundle_path
    bundle_path = bundle_path.resolve(strict=True)
    _require(bundle_path.is_relative_to(root) and bundle_path.name == "bundle.v2.json",
             "WP04 bundle must be the retained bundle.v2.json inside the repository")
    bundle_dir = bundle_path.parent
    result = verify_wp04_bundle(root, bundle_dir)
    _require(result.get("status") == "PASS_INTEGRITY_LOCAL_DEVELOPMENT_DRAFT_ONLY" and
             result.get("research_admissible") is False and
             result.get("research_status") == "RESEARCH_NOT_READY" and
             result.get("plan_executable") is False and result.get("release_status") == "BLOCKED" and
             result.get("verified_current_decode_products") == 2 and
             result.get("verified_decoded_output_files") == 50,
             "WP04 retained draft verification did not pass at its accepted quarantined scope")
    bundle = _read_json(bundle_path)
    sources_binding = bundle["sections"]["sources"]
    sources_path = verified_file(bundle_dir, sources_binding["path"], sources_binding["sha256"])
    sources_doc = _read_json(sources_path)
    registry_binding = bundle["sections"]["registry"]
    registry_path = verified_file(bundle_dir, registry_binding["path"], registry_binding["sha256"])
    registry_doc = _read_json(registry_path)
    decode_binding = bundle["inputs"]["roi_decode_report"]
    decode_path = verified_file(root, decode_binding["path"], decode_binding["sha256"])
    decode_report = _read_json(decode_path)
    inventory_binding = bundle["inputs"]["roi_decode_inventory"]
    inventory_path = verified_file(root, inventory_binding["path"], inventory_binding["sha256"])
    decode_inventory = _read_json(inventory_path)
    return {
        "root": root,
        "bundle_path": bundle_path,
        "bundle": bundle,
        "bundle_sha256": file_hash(bundle_path),
        "bundle_verification": result,
        "sources_doc": sources_doc,
        "sources_path": sources_path,
        "registry_doc": registry_doc,
        "decode_report": decode_report,
        "decode_report_path": decode_path,
        "decode_inventory": decode_inventory,
        "decode_inventory_path": inventory_path,
    }


def _raw_asset_map(source: dict) -> dict:
    found = {}
    for asset in source.get("assets", []):
        role = asset.get("band_role")
        _require(role not in found, f"duplicate raw source asset role: {role}")
        found[role] = asset
    _require(set(RAW_ROLES.values()) <= set(found), "raw product lacks one or more required source assets")
    return found


def _binding_map(source: dict) -> dict:
    decode = source.get("current_decode", {})
    _require(decode.get("attempt_id") == DECODE_ATTEMPT_ID and
             decode.get("decode_status") == "ROI_PIXEL_DECODED_QUARANTINED",
             "feature input is not the accepted quarantined roi_decode_005 product")
    bindings = decode.get("decoded_output_bindings", [])
    by_name = {}
    for item in bindings:
        _require(item.get("site_id") == source.get("site_id") and
                 item.get("product_id") == source.get("product_id"),
                 "decoded output has a mismatched site/product join")
        name = Path(item.get("path", "")).name
        _require(name not in by_name, f"duplicate decoded output layer: {name}")
        by_name[name] = item
    _require(set(LAYER_NAMES) <= set(by_name), "accepted decode lacks required feature layers")
    return by_name


def _read_layers(context: dict, source: dict, report_site: dict) -> tuple[dict, dict, dict]:
    root = context["root"]
    site_id, product_id = source["site_id"], source["product_id"]
    pixel = report_site.get("pixel_inspection", {})
    roi = pixel.get("roi", {})
    _require(pixel.get("research_admissible") is False and roi.get("research_admissible", False) is False and
             pixel.get("origin") == "provider_download" and roi.get("origin") == "provider_download" and
             roi.get("status") == "ROI_PIXEL_DECODED_QUARANTINED" and
             roi.get("site_geometry_status") == "PI_QUERY_COORDINATE_ONLY_LAKE_GEOMETRY_UNVERIFIED",
             "decoded product origin, quarantine or unresolved geometry status changed")
    bindings = _binding_map(source)
    raw_assets = _raw_asset_map(source)
    layer_arrays, layer_meta, input_assets = {}, {}, []
    for name in LAYER_NAMES:
        binding = bindings[name]
        path = verified_file(root, binding["path"], binding["sha256"])
        _require(path.stat().st_size == binding["bytes"], f"decoded output byte count changed: {name}")
        with rasterio.open(path) as ds:
            _require(ds.count == 1, f"unexpected raster band count: {name}")
            array = ds.read(1, masked=False)
            crs = ds.crs
            _require(crs is not None and crs.to_epsg() == 32645 and crs.is_projected and
                     crs.linear_units.lower() in ("metre", "meter"),
                     f"decoded feature grid is not projected EPSG:32645 metres: {name}")
            _require(array.shape == (499, 499), f"decoded feature grid shape changed: {name}")
            affine = ds.transform
            grid = [float(affine.a), float(affine.b), float(affine.c),
                    float(affine.d), float(affine.e), float(affine.f)]
            meta = {"shape": list(array.shape), "crs": crs.to_string(), "transform": grid,
                    "nodata": None if ds.nodata is None else float(ds.nodata),
                    "dtype": ds.dtypes[0], "tags": ds.tags(), "bounds": list(ds.bounds)}
        if layer_meta:
            reference = layer_meta[LAYER_NAMES[0]]
            _require(meta["shape"] == reference["shape"] and meta["crs"] == reference["crs"] and
                     np.allclose(meta["transform"], reference["transform"], rtol=0, atol=1e-9),
                     f"decoded layers are not co-registered on the accepted ROI grid: {name}")
        tags = meta["tags"]
        _require(tags.get("origin") == "provider_download" and tags.get("site_id") == site_id,
                 f"decoded layer origin/site tags changed: {name}")
        source_joined_layers = {"scl_raw_20m_classes.tif", "scl_valid_mask.tif",
                                *[name for name in LAYER_NAMES if "_raw_20m_mean_dn" in name or
                                  "_reflectance_20m" in name or "_valid_mask" in name and name.startswith("b") ]}
        if name in source_joined_layers:
            role = tags.get("role", "")
            raw_role = ("scl" if name.startswith("scl_") else
                        {"B03": "green", "B08": "nir", "B11": "swir16"}.get(role))
            _require(raw_role is not None and
                     (name.endswith("valid_mask.tif") or
                      (tags.get("source_item_id") == product_id and
                       tags.get("product_uri") == source.get("product_uri"))),
                     f"decoded layer source identity tags changed: {name}")
            if name == "scl_valid_mask.tif":
                _require(role == "SCL validity", f"SCL validity role tag changed: {name}")
            else:
                _require(role == {"green": "B03", "nir": "B08", "swir16": "B11", "scl": "SCL"}[raw_role],
                         f"decoded band role tag changed: {name}")
            expected_raw_sha = raw_assets[raw_role]["sha256"]
            _require(tags.get("source_sha256") == expected_raw_sha,
                     f"decoded layer is not joined to the retained raw product bytes: {name}")
        layer_arrays[name] = array
        layer_meta[name] = meta
        raw_sha = tags.get("source_sha256")
        input_assets.append({"layer": name.removesuffix(".tif"), "path": binding["path"],
                             "sha256": binding["sha256"], "bytes": binding["bytes"],
                             "source_sha256": raw_sha})
    reference = layer_meta[LAYER_NAMES[0]]
    transform = reference["transform"]
    determinant = abs(transform[0] * transform[4] - transform[1] * transform[3])
    _require(math.isfinite(determinant) and determinant > 0,
             "projected affine pixel determinant is unavailable")
    expected_bounds = roi.get("output_roi_bounds")
    _require(isinstance(expected_bounds, list) and len(expected_bounds) == 4 and
             np.allclose(reference["bounds"], expected_bounds, rtol=0, atol=1e-6),
             "decoded pixel coordinates differ from the accepted WP03 ROI bounds")
    return layer_arrays, layer_meta, {"assets": input_assets, "pixel_area_m2": determinant,
                                     "grid": reference}


def _check_calibration_tags(tags: dict, expected_scale: float, expected_offset: float,
                            band: str) -> None:
    _require(tags.get("units") == "dimensionless_surface_reflectance" and
             tags.get("calibration_status") == "MATCHED_COG_STAC_AND_NATIVE_XML",
             f"{band} reflectance units/calibration status are not verified")
    for key in ("native_scale", "cog_scale"):
        _require(math.isclose(_finite_number(float(tags.get(key, "nan")), f"{band}.{key}"),
                              expected_scale, rel_tol=0, abs_tol=1e-12),
                 f"{band} {key} differs from native source calibration")
    for key in ("native_offset", "cog_offset"):
        _require(math.isclose(_finite_number(float(tags.get(key, "nan")), f"{band}.{key}"),
                              expected_offset, rel_tol=0, abs_tol=1e-12),
                 f"{band} {key} differs from native source calibration")


def _ratio(numerator: np.ndarray, denominator: np.ndarray,
           valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = (valid & np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0))
    values = np.full(numerator.shape, np.nan, dtype=np.float64)
    values[support] = (numerator[support] / denominator[support]).astype(np.float64)
    support &= np.isfinite(values)
    values[~support] = np.nan
    return values, support


def _summary(values: np.ndarray, valid: np.ndarray, denominator_count: int) -> dict:
    selected = np.asarray(values[valid], dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    count = int(selected.size)
    if count == 0:
        status = "EMPTY_POPULATION" if denominator_count == 0 else "NO_VALID_VALUES_IN_POPULATION"
        return {"status": status, "valid_count": 0, "denominator_count": int(denominator_count),
                "coverage_fraction": None if denominator_count == 0 else 0.0,
                "mean": None, "median": None, "minimum": None, "maximum": None,
                "population_std": None}
    return {"status": "AVAILABLE", "valid_count": count,
            "denominator_count": int(denominator_count),
            "coverage_fraction": count / denominator_count if denominator_count else None,
            "mean": float(np.mean(selected)), "median": float(np.median(selected)),
            "minimum": float(np.min(selected)), "maximum": float(np.max(selected)),
            "population_std": float(np.std(selected, ddof=0))}


def summarize_values(values, valid, denominator_count: int) -> dict:
    """Public arithmetic helper; callers must label supplied arrays' origin."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    _require(values.shape == valid.shape, "summary value and support arrays differ in shape")
    _require(type(denominator_count) is int and denominator_count >= 0,
             "summary denominator must be a nonnegative integer")
    return _summary(values, valid, denominator_count)


def _channel(channel_id: str, aliases: list[str], units: str, values: np.ndarray,
             value_valid: np.ndarray, qa_population: np.ndarray,
             candidate_population: np.ndarray, calibration: dict | None = None) -> dict:
    qa_count = int(qa_population.sum())
    candidate_count = int(candidate_population.sum())
    return {"channel_id": channel_id, "aliases": aliases, "status": "AVAILABLE",
            "reason_code": None, "reason": None, "units": units,
            "calibration": calibration,
            "summary": {
                "qa_supported": _summary(values, value_valid & qa_population, qa_count),
                "candidate_conditioned": _summary(values, value_valid & candidate_population,
                                                   candidate_count),
            }}


def _candidate_area(qa_count: int, candidate_count: int,
                    pixel_area_m2: float) -> tuple[float | None, str]:
    _require(type(qa_count) is int and qa_count >= 0,
             "QA population count must be a nonnegative integer")
    _require(type(candidate_count) is int and candidate_count >= 0,
             "candidate population count must be a nonnegative integer")
    _require(candidate_count <= qa_count,
             "candidate population count exceeds its QA support")
    pixel_area_m2 = _finite_number(pixel_area_m2, "pixel_area_m2")
    _require(pixel_area_m2 > 0, "pixel area must be positive")
    if qa_count == 0:
        return None, "EMPTY_POPULATION"
    return candidate_count * pixel_area_m2, "AVAILABLE_WITHIN_ROI"


def _build_record(context: dict, source: dict) -> dict:
    site_id = source["site_id"]
    report_site = context["decode_report"]["sentinel2_c1"].get(site_id)
    _require(isinstance(report_site, dict) and report_site.get("item_id") == source["product_id"] and
             report_site.get("product_uri") == source["product_uri"] and
             report_site.get("site_scope") == "DEVELOPMENT_EXPOSED",
             "decode report site/product/exposure join changed")
    pixel = report_site["pixel_inspection"]
    roi = pixel["roi"]
    arrays, metas, loaded = _read_layers(context, source, report_site)
    raw_assets = _raw_asset_map(source)
    radiometry = pixel["product_temporal_join"]["native_radiometry"]
    bands = {}
    for band in ("B03", "B08", "B11"):
        refl_name = REFLECTANCE_LAYER[band]
        raw_name = RAW_DN_LAYER[band]
        valid_name = VALID_LAYER[band]
        refl = arrays[refl_name].astype(np.float64)
        raw = arrays[raw_name].astype(np.float64)
        valid_layer = arrays[valid_name]
        _require(np.isin(valid_layer, (0, 1)).all(), f"{band} validity layer is not binary")
        nodata = metas[refl_name]["nodata"]
        valid = ((valid_layer == 1) & np.isfinite(refl) & np.isfinite(raw) &
                 (refl != nodata) & (raw != metas[raw_name]["nodata"]))
        _require(np.array_equal(valid_layer == 1, valid),
                 f"{band} validity mask differs from actual reflectance/raw support")
        expected = radiometry[band]
        scale = _finite_number(expected["expected_scale"], f"{band}.expected_scale")
        offset = _finite_number(expected["expected_offset"], f"{band}.expected_offset")
        tags = metas[refl_name]["tags"]
        _check_calibration_tags(tags, scale, offset, band)
        _require(np.allclose(refl[valid], raw[valid] * scale + offset,
                             rtol=0, atol=CALIBRATION_ATOL),
                 f"{band} calibrated reflectance does not match raw DN with one native offset")
        bands[band] = {"values": refl, "valid": valid, "scale": scale, "offset": offset}

    scl_raw = arrays["scl_raw_20m_classes.tif"]
    scl_valid_layer = arrays["scl_valid_mask.tif"]
    _require(np.isin(scl_valid_layer, (0, 1)).all(), "SCL validity layer is not binary")
    scl_nodata = metas["scl_raw_20m_classes.tif"]["nodata"]
    scl_valid = (scl_valid_layer == 1) & (scl_raw != scl_nodata)
    _require(np.array_equal(scl_valid_layer == 1, scl_valid),
             "SCL validity mask differs from retained categorical support")
    _require(np.isin(scl_raw[scl_valid], np.arange(1, 12)).all(),
             "SCL valid support contains an undocumented class code")
    scl_clear_expected = scl_valid & np.isin(scl_raw, QA_CLASSES)
    clear = arrays["scl_clear_4_5_6_mask.tif"]
    excluded = arrays["scl_qa_excluded_mask.tif"]
    _require(np.isin(clear, (0, 1)).all() and np.isin(excluded, (0, 1)).all(),
             "SCL QA layers contain invalid categorical values")
    _require(np.array_equal(clear == 1, scl_clear_expected) and
             np.array_equal(excluded == 1, scl_valid & ~np.isin(scl_raw, QA_CLASSES)),
             "retained SCL clear/excluded QA differs from raw SCL class arithmetic")

    b03, b08, b11 = (bands[k] for k in ("B03", "B08", "B11"))
    ndwi, ndwi_valid = _ratio(b03["values"] - b08["values"],
                              b03["values"] + b08["values"], b03["valid"] & b08["valid"])
    mndwi, mndwi_valid = _ratio(b03["values"] - b11["values"],
                                b03["values"] + b11["values"], b03["valid"] & b11["valid"])
    analysis_valid = ndwi_valid & mndwi_valid & scl_valid
    analysis_layer = arrays["analysis_valid_mask.tif"]
    unknown_layer = arrays["unknown_mask.tif"]
    _require(np.isin(analysis_layer, (0, 1)).all() and np.isin(unknown_layer, (0, 1)).all() and
             np.array_equal(analysis_layer == 1, analysis_valid) and
             np.array_equal(unknown_layer == 1, ~analysis_valid),
             "analysis/unknown masks differ from measured bands, denominators and SCL support")
    qa_population = analysis_valid & scl_clear_expected
    candidate_support = qa_population
    expected_candidate = candidate_support & (ndwi > 0) & (mndwi > 0)
    candidate_layer = arrays["candidate_water_mask.tif"]
    expected_encoded = np.where(candidate_support, expected_candidate.astype(np.uint8), 255).astype(np.uint8)
    _require(np.array_equal(candidate_layer, expected_encoded),
             "retained candidate mask differs from the versioned preliminary rule and source arrays")
    for name, computed, support in (("ndwi.tif", ndwi, ndwi_valid), ("mndwi.tif", mndwi, mndwi_valid)):
        cached = arrays[name].astype(np.float64)
        cached_valid = np.isfinite(cached) & (cached != metas[name]["nodata"])
        _require(np.array_equal(cached_valid, support) and
                 np.allclose(cached[support], computed[support], rtol=INDEX_RTOL, atol=INDEX_ATOL),
                 f"cached {name} differs from independent calibrated reflectance arithmetic")
    for name in ("ndwi.tif", "mndwi.tif"):
        tags = metas[name]["tags"]
        _require(tags.get("algorithm") == "development_initial_score_blind_v1" and
                 tags.get("units") == "dimensionless", f"derived index version/units changed: {name}")
    candidate_tags = metas["candidate_water_mask.tif"]["tags"]
    _require(candidate_tags.get("rule") == "NDWI>0 AND MNDWI>0 AND SCL in 4,5,6" and
             candidate_tags.get("interpretation") == "exploratory candidate pixels within ROI; not lake ground truth",
             "candidate mask rule or interpretation changed")

    roi_cells = int(analysis_valid.size)
    scl_count = int(scl_valid.sum())
    clear_count = int(scl_clear_expected.sum())
    qa_count = int(qa_population.sum())
    candidate_count = int(expected_candidate.sum())
    determinant = loaded["pixel_area_m2"]
    area, area_status = _candidate_area(qa_count, candidate_count, determinant)
    class_rows = []
    for code in range(1, 12):
        count = int(np.count_nonzero(scl_valid & (scl_raw == code)))
        class_rows.append({"class_id": code, "count": count,
                           "fraction": count / scl_count if scl_count else None,
                           "status": "AVAILABLE" if scl_count else "EMPTY_POPULATION"})
    channels = [
        _channel("B03", [], "dimensionless_surface_reflectance", b03["values"], b03["valid"],
                 qa_population, expected_candidate,
                 {"scale": b03["scale"], "offset": b03["offset"],
                  "application": "RAW_MEAN_DN_TIMES_SCALE_PLUS_OFFSET_ONCE",
                  "raw_source_sha256": raw_assets["green"]["sha256"]}),
        _channel("B08", [], "dimensionless_surface_reflectance", b08["values"], b08["valid"],
                 qa_population, expected_candidate,
                 {"scale": b08["scale"], "offset": b08["offset"],
                  "application": "RAW_MEAN_DN_TIMES_SCALE_PLUS_OFFSET_ONCE",
                  "raw_source_sha256": raw_assets["nir"]["sha256"]}),
        _channel("B11", [], "dimensionless_surface_reflectance", b11["values"], b11["valid"],
                 qa_population, expected_candidate,
                 {"scale": b11["scale"], "offset": b11["offset"],
                  "application": "RAW_MEAN_DN_TIMES_SCALE_PLUS_OFFSET_ONCE",
                  "raw_source_sha256": raw_assets["swir16"]["sha256"]}),
        _channel("NDWI", [], "dimensionless", ndwi, ndwi_valid, qa_population, expected_candidate),
        _channel("MNDWI_NDSI", ["MNDWI", "NDSI"], "dimensionless", mndwi,
                 mndwi_valid, qa_population, expected_candidate),
    ]
    channels.extend({"channel_id": name, "aliases": [], "status": "UNAVAILABLE",
                     "reason_code": code, "reason": reason, "units": None,
                     "calibration": None, "summary": None}
                    for name, code, reason in MISSING_CHANNELS)

    raw_lineage = []
    for band, raw_role in RAW_ROLES.items():
        asset = raw_assets[raw_role]
        raw_lineage.append({"band": band, "role": raw_role, "path": asset["path"],
                            "sha256": asset["sha256"], "bytes": asset["bytes"],
                            "receipt_path": asset["receipt_path"],
                            "receipt_sha256": asset["receipt_sha256"],
                            "receipt_bytes": asset["receipt_bytes"]})
    input_names = {x["layer"] for x in loaded["assets"]}
    _require(input_names == {Path(name).stem for name in LAYER_NAMES},
             "feature input lineage is incomplete or contains orphan raster layers")
    registry_sites = {x["site_id"]: x for x in context["registry_doc"]["sites"]}
    site = registry_sites[site_id]
    _require(site["development_exposure"] == "DEVELOPMENT_EXPOSED" and
             site["geometry"]["status"] == "UNAVAILABLE" and
             site["candidate_lake_id"] is None,
             "WP04 registry no longer marks this site as exposed with unresolved lake geometry")
    query_point = site["query_point"]
    time_axes = source["time_axes"]
    scopes = context["bundle_verification"]
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "record_id": f"{FEATURE_BUNDLE_ID}:{source['source_id']}",
        "source_id": source["source_id"], "site_id": site_id,
        "product_id": source["product_id"], "product_uri": source["product_uri"],
        "origin": "provider_download", "site_exposure": "DEVELOPMENT_EXPOSED",
        "support_id": "SUPPORT-" + _sha_json({"source_id": source["source_id"],
                    "grid": loaded["grid"], "assets": loaded["assets"]})[:20],
        "query_point": {"longitude": query_point["longitude"], "latitude": query_point["latitude"],
                        "status": query_point["status"], "lake_geometry_status": "UNVERIFIED"},
        "time_axes": time_axes,
        "scope_membership": {"site_exposure": "DEVELOPMENT_EXPOSED",
                              "source_membership_count": scopes["source_membership_count"],
                              "context_membership_count": scopes["context_membership_count"],
                              "scientific_source_memberships": [],
                              "status": "NO_ADMITTED_SCIENTIFIC_SOURCE_MEMBERSHIP"},
        "sampling_unit": {"physical_unit_id": None, "status": "UNRESOLVED",
                          "reason": "No verified lake geometry or independent physical sampling unit is admitted."},
        "lineage": {
            "wp04_bundle_path": context["bundle_path"].relative_to(context["root"]).as_posix(),
            "wp04_bundle_sha256": context["bundle_sha256"],
            "sources_path": context["sources_path"].relative_to(context["root"]).as_posix(),
            "sources_sha256": context["bundle"]["sections"]["sources"]["sha256"],
            "stage_report_path": context["bundle"]["inputs"]["stage_report"]["path"],
            "stage_report_sha256": context["bundle"]["inputs"]["stage_report"]["sha256"],
            "stage_inventory_path": context["bundle"]["inputs"]["output_inventory"]["path"],
            "stage_inventory_sha256": context["bundle"]["inputs"]["output_inventory"]["sha256"],
            "decode_report_path": context["decode_report_path"].relative_to(context["root"]).as_posix(),
            "decode_report_sha256": context["bundle"]["inputs"]["roi_decode_report"]["sha256"],
            "decode_inventory_path": context["decode_inventory_path"].relative_to(context["root"]).as_posix(),
            "decode_inventory_sha256": context["bundle"]["inputs"]["roi_decode_inventory"]["sha256"],
            "raw_assets": raw_lineage, "decoded_assets": loaded["assets"],
        },
        "grid": {"crs": loaded["grid"]["crs"], "shape": loaded["grid"]["shape"],
                 "transform": loaded["grid"]["transform"], "coordinate_units": "metre",
                 "pixel_area_m2": determinant,
                 "pixel_area_method": "ABS_AFFINE_DETERMINANT_AE_MINUS_BD",
                 "roi_grid_version": "WP03_ACCEPTED_ROI_DECODE_005"},
        "support": {
            "roi_cell_count": roi_cells, "analysis_valid_count": int(analysis_valid.sum()),
            "scl_valid_count": scl_count, "qa_population_id": "SCL_CLEAR_CLASSES_4_5_6",
            "qa_population_count": qa_count, "qa_denominator_id": "VALID_SCL_CELLS",
            "qa_denominator_count": scl_count,
            "qa_fraction_of_valid_scl": qa_count / scl_count if scl_count else None,
            "candidate_population_id": "PRELIMINARY_NDWI_MNDWI_SCL_CANDIDATE",
            "candidate_population_count": candidate_count,
            "candidate_denominator_id": "QA_SUPPORTED_CELLS",
            "candidate_denominator_count": qa_count,
            "candidate_fraction_of_qa_population": candidate_count / qa_count if qa_count else None,
            "candidate_area_m2": area,
            "candidate_area_status": area_status,
            "candidate_area_scope": "OBSERVED_PRELIMINARY_CANDIDATE_AREA_WITHIN_ROI_NOT_LAKE_AREA",
            "candidate_mask_unavailable_count": int(np.count_nonzero(candidate_layer == 255)),
        },
        "scl": {"valid_denominator_count": scl_count, "class_counts_and_fractions": class_rows,
                "clear_classes": list(QA_CLASSES), "clear_count": clear_count,
                "clear_fraction_of_valid_scl": clear_count / scl_count if scl_count else None,
                "status": "AVAILABLE" if scl_count else "EMPTY_POPULATION"},
        "channels": channels,
        "candidate_rule": {
            "mask_version": DECODE_ATTEMPT_ID,
            "rule": candidate_tags["rule"],
            "mask_sha256": next(x["sha256"] for x in loaded["assets"]
                                 if x["layer"] == "candidate_water_mask"),
            "interpretation": candidate_tags["interpretation"],
            "four_band_snow_aware_candidate": {
                "status": "UNAVAILABLE_MISSING_B04", "value": None,
                "required_rule": "((MNDWI>0.25) OR (NDWI>0.12)) AND B08<0.20 AND B11<0.08 AND B04<0.25",
                "reason": "B04 was not staged; no three-band substitute is evaluated.",
            },
        },
        "admission": {"status": "DEVELOPMENT_ONLY_QUARANTINED",
                      "research_admissible": False, "measurement_admission": "BLOCKED",
                      "scientific_schedule_membership": False,
                      "2023_operational_decision_status": "EXCLUDED_PRODUCT_GENERATED_AFTER_EVENT_CUTOFF"},
    }


def _verified_inputs(root: str | Path, wp04_bundle_path: str | Path) -> dict:
    context = _load_context(root, wp04_bundle_path)
    sources = context["sources_doc"].get("source_items", [])
    _require(sources and len({x.get("source_id") for x in sources}) == len(sources),
             "WP04 source membership is empty or duplicated")
    _require(len({x.get("site_id") for x in sources}) == len(sources) and
             len({x.get("product_id") for x in sources}) == len(sources),
             "WP04 product/site support was duplicated or merged")
    _require(all(x.get("site_scope") == "DEVELOPMENT_EXPOSED" and
                 x.get("admission") == "SOURCE_BYTES_ACCEPTED_FOR_LOCAL_DEVELOPMENT_PROVENANCE_ONLY_MEASUREMENTS_NOT_ADMITTED"
                 for x in sources),
             "WP04 source rows are not all quarantined development provenance")
    context["sources"] = sources
    return context


def _make_records(context: dict) -> list[dict]:
    records = []
    for source in context["sources"]:
        records.append(_build_record(context, source))
    return records


def validate_feature_bundle_document(document: dict, context: dict) -> None:
    validate_schema(document, "wp05_feature_bundle")
    records = document["records"]
    _require(document["source_count"] == len(context["sources"]) and
             document["record_count"] == len(records),
             "feature bundle counts differ from its source and record memberships")
    expected_by_id = {source["source_id"]: source for source in context["sources"]}
    actual_ids = [record["source_id"] for record in records]
    _require(len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == set(expected_by_id),
             "feature source membership is duplicated, omitted or orphaned")
    _require(len({record["support_id"] for record in records}) == len(records),
             "shared checkpoints/support IDs merged distinct geographic acquisitions")
    for record in records:
        validate_schema(record, "wp05_feature_record")
        source = expected_by_id[record["source_id"]]
        _require(record["site_id"] == source["site_id"] and
                 record["product_id"] == source["product_id"] and
                 record["product_uri"] == source["product_uri"] and
                 record["site_exposure"] == source["site_scope"] == "DEVELOPMENT_EXPOSED" and
                 record["time_axes"] == source["time_axes"],
                 "feature product/site/time/exposure join differs from the accepted source row")
        _require(record["scope_membership"]["source_membership_count"] == 0 and
                 record["scope_membership"]["context_membership_count"] == 0 and
                 not record["scope_membership"]["scientific_source_memberships"],
                 "WP04 source or context membership was incorrectly admitted")
        _require(record["sampling_unit"]["physical_unit_id"] is None and
                 record["sampling_unit"]["status"] == "UNRESOLVED",
                 "unverified geographic sampling unit was assigned")
        channel_ids = [row["channel_id"] for row in record["channels"]]
        required = {"B03", "B08", "B11", "NDWI", "MNDWI_NDSI", *(x[0] for x in MISSING_CHANNELS)}
        _require(len(channel_ids) == len(set(channel_ids)) and set(channel_ids) == required,
                 "feature channels are missing, duplicated or orphaned")
        channel_by_id = {row["channel_id"]: row for row in record["channels"]}
        for name, reason_code, _reason in MISSING_CHANNELS:
            row = channel_by_id[name]
            _require(row["status"] == "UNAVAILABLE" and row["reason_code"] == reason_code and
                     row["summary"] is None and row["units"] is None,
                     f"unsupported channel was filled or admitted: {name}")
        _require(channel_by_id["MNDWI_NDSI"]["aliases"] == ["MNDWI", "NDSI"] and
                 all(channel_by_id[name]["calibration"] is not None for name in ("B03", "B08", "B11")),
                 "channel aliases or measured calibration lineage changed")
        _require(record["admission"]["research_admissible"] is False and
                 record["admission"]["measurement_admission"] == "BLOCKED" and
                 record["admission"]["scientific_schedule_membership"] is False,
                 "development feature bundle changed research admission")
        determinant = abs(record["grid"]["transform"][0] * record["grid"]["transform"][4] -
                          record["grid"]["transform"][1] * record["grid"]["transform"][3])
        _require(math.isclose(record["grid"]["pixel_area_m2"], determinant, rel_tol=0, abs_tol=1e-12),
                 "pixel area is not the actual affine determinant")


def build_feature_bundle(root: str | Path, wp04_bundle_path: str | Path) -> dict:
    context = _verified_inputs(root, wp04_bundle_path)
    records = _make_records(context)
    document = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "bundle_id": FEATURE_BUNDLE_ID,
        "origin": "RETAINED_PROVIDER_BYTES_DEVELOPMENT_ONLY",
        "status": "LOCAL_OPTICAL_FEATURES_QUARANTINED",
        "research_status": "RESEARCH_NOT_READY", "plan_executable": False,
        "release_status": "BLOCKED", "research_admissible": False,
        "input_bundle_path": context["bundle_path"].relative_to(context["root"]).as_posix(),
        "input_bundle_sha256": context["bundle_sha256"],
        "source_count": len(context["sources"]), "record_count": len(records),
        "records": records,
    }
    validate_feature_bundle_document(document, context)
    return document


def verify_feature_bundle(root: str | Path, wp04_bundle_path: str | Path,
                          feature_document: dict) -> dict:
    context = _verified_inputs(root, wp04_bundle_path)
    validate_feature_bundle_document(feature_document, context)
    expected = _make_records(context)
    _require(feature_document["records"] == expected,
             "WP05 output values or lineage differ from independent current-source recomputation")
    _require(feature_document["input_bundle_sha256"] == context["bundle_sha256"] and
             feature_document["input_bundle_path"] == context["bundle_path"].relative_to(context["root"]).as_posix(),
             "WP05 bundle is not bound to the accepted WP04 bundle bytes")
    return {"status": "PASS_LOCAL_DEVELOPMENT_FEATURE_INTEGRITY_ONLY",
            "verified_records": len(expected), "research_admissible": False,
            "research_status": "RESEARCH_NOT_READY", "plan_executable": False,
            "release_status": "BLOCKED", "source_membership_count": 0,
            "context_membership_count": 0}


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
