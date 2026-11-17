"""Bounded, offline extraction and native sigma0 calibration for one retained SAFE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from zipfile import ZipFile

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARCHIVE_SHA256 = "85b0adb8c7057e98fe6e4b40a70777bad2618b5f757c6cec5f713b1920f7f71b"
EXPECTED_ARCHIVE_BYTES = 1737975879
DEFAULT_ARCHIVE = ROOT / "data/raw/wp03/cdse_stage_20261015_restart002/1579285a-a3dd-4b77-b2e7-4b3654f12b56.zip.partial"
DEFAULT_ROLE_EVIDENCE = ROOT / "project/architecture/wp03_safe_role_fix/actual_archive_validation_002_scoped_identity.json"
SITE_GRID_FILES = {
    "PILOT-SOUTH-LHONAK": ROOT / "data/raw/wp03/attempt_2026-10-14_roi_decode_005/c1/PILOT-SOUTH-LHONAK/roi/candidate_water_mask.tif",
    "PILOT-KHANGCHUNG-TSHO": ROOT / "data/raw/wp03/attempt_2026-10-14_roi_decode_005/c1/PILOT-KHANGCHUNG-TSHO/roi/candidate_water_mask.tif",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_member_name(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return member


def _selected_members(role_evidence: dict) -> dict[str, dict[str, str]]:
    source = role_evidence.get("zip_crc_and_safe_identity_by_polarization")
    if not isinstance(source, dict):
        raise ValueError("accepted SAFE role evidence lacks polarization role map")
    selected: dict[str, dict[str, str]] = {}
    for pol in ("VV", "VH"):
        row = source.get(pol)
        if not isinstance(row, dict) or row.get("role_mapping_status") != "SAFE_SCOPED_MEMBER_AND_ROLE_IDENTITIES_MATCHED":
            raise ValueError(f"accepted role evidence is incomplete for {pol}")
        roles = row.get("roles")
        if not isinstance(roles, dict) or set(roles) != {"measurement", "calibration", "noise", "manifest", "annotation"}:
            raise ValueError(f"accepted role list is malformed for {pol}")
        selected[pol] = {str(k): str(v) for k, v in roles.items()}
        rfi = row.get("rfi_metadata", {})
        if rfi.get("status") == "PRESENT":
            members = rfi.get("members", [])
            if len(members) != 1:
                raise ValueError(f"RFI role identity is ambiguous for {pol}")
            selected[pol]["rfi"] = str(members[0])
    if selected["VV"]["manifest"] != selected["VH"]["manifest"]:
        raise ValueError("polarization role evidence does not share one SAFE manifest")
    return selected


def _extract_roles(archive: Path, destination: Path, role_evidence: dict) -> dict:
    selected = _selected_members(role_evidence)
    member_to_role: dict[str, list[str]] = {}
    output_paths: dict[str, dict[str, Path]] = {pol: {} for pol in selected}
    for pol, roles in selected.items():
        for role, member in roles.items():
            _safe_member_name(member)
            member_to_role.setdefault(member, []).append(f"{pol}:{role}")
    with ZipFile(archive) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        missing = sorted(set(member_to_role) - set(infos))
        if missing:
            raise ValueError(f"role-bound archive members are missing: {missing}")
        required_bytes = 0
        for member in member_to_role:
            info = infos[member]
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode) or info.is_dir():
                raise ValueError(f"archive role member is a symlink or directory: {member}")
            required_bytes += info.file_size
        output_estimate = 2 * 16731 * 25851 * (4 * 5 + 2 * 1)
        available = shutil.disk_usage(destination.parent).free
        if available < required_bytes + output_estimate + 512 * 1024 * 1024:
            raise ValueError(f"insufficient free space: need at least {required_bytes + output_estimate + 512*1024*1024}, have {available}")
        files = {}
        for member in sorted(member_to_role):
            info = infos[member]
            role_labels = member_to_role[member]
            pol = role_labels[0].split(":", 1)[0]
            role = role_labels[0].split(":", 1)[1]
            suffix = Path(member).suffix or ".bin"
            target = destination / "inputs" / f"{pol}_{role}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            crc = 0
            count = 0
            with zf.open(info, "r") as src, target.open("xb") as dst:
                while True:
                    block = src.read(8 * 1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    crc = zlib.crc32(block, crc)
                    h.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            if count != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
                raise ValueError(f"extracted member length/CRC mismatch: {member}")
            for label in role_labels:
                output_pol, output_role = label.split(":", 1)
                output_paths[output_pol][output_role] = target
            files[member] = {
                "roles": role_labels, "advertised_bytes": info.file_size,
                "extracted_bytes": count, "zip_crc32": f"{crc & 0xFFFFFFFF:08x}",
                "sha256": h.hexdigest(), "relative_path": str(target.relative_to(destination)),
            }
    return {"files": files, "paths": output_paths}


def _verify_reused_inputs(archive: Path, input_dir: Path, role_evidence: dict) -> dict:
    """Bind an earlier extracted attempt to exact archive members without rewriting it."""
    selected = _selected_members(role_evidence)
    files, output_paths = {}, {pol: {} for pol in selected}
    with ZipFile(archive) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        for pol, roles in selected.items():
            for role, member in roles.items():
                info = infos.get(member)
                if info is None:
                    raise ValueError(f"role-bound archive member is missing: {member}")
                _safe_member_name(member)
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode) or info.is_dir():
                    raise ValueError(f"role-bound archive member is a symlink or directory: {member}")
                suffix = Path(member).suffix or ".bin"
                path = (output_paths["VV"]["manifest"] if pol == "VH" and role == "manifest"
                        else input_dir / f"{pol}_{role}{suffix}")
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"reused source input is absent or not a regular file: {path}")
                if path.stat().st_size != info.file_size:
                    raise ValueError(f"reused source input length differs from archive member: {member}")
                file_hash = hashlib.sha256()
                crc = 0
                count = 0
                with path.open("rb") as local:
                    for block in iter(lambda: local.read(8 * 1024 * 1024), b""):
                        file_hash.update(block)
                source_hash = hashlib.sha256()
                with zf.open(info, "r") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                        count += len(block)
                        source_hash.update(block)
                extracted_hash = file_hash.hexdigest()
                source_digest = source_hash.hexdigest()
                if count != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
                    raise ValueError(f"reused source archive member length/CRC mismatch: {member}")
                if source_digest != extracted_hash:
                    raise ValueError(f"reused extracted member SHA-256 differs from archive member: {member}")
                size_bind = role_evidence["zip_crc_and_safe_identity_by_polarization"][pol]["archive_member_sizes"].get(role)
                if size_bind is not None and size_bind != info.file_size:
                    raise ValueError(f"reused input advertised size differs from accepted role evidence: {member}")
                label = f"{pol}:{role}"
                output_paths[pol][role] = path
                if member not in files:
                    files[member] = {"roles": [], "advertised_bytes": info.file_size,
                    "extracted_bytes": path.stat().st_size, "archive_stream_bytes": count,
                    "zip_crc32": f"{crc & 0xFFFFFFFF:08x}", "sha256": source_digest,
                    "relative_path": str(path)}
                files[member]["roles"].append(label)
    # One manifest is shared by both polarization roles.
    output_paths["VH"]["manifest"] = output_paths["VV"]["manifest"]
    return {"files": files, "paths": output_paths}


def _xml_values(xml_path: Path, name: str) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [(node.text or "").strip() for node in root.iter()
            if node.tag.rsplit("}", 1)[-1] == name and node.text and node.text.strip()]


def manifest_ipf_identity(manifest_xml: bytes) -> tuple[str, int]:
    try:
        root = ET.fromstring(manifest_xml)
    except ET.ParseError as exc:
        raise ValueError(f"malformed SAFE manifest XML: {exc}") from exc
    declarations = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "software":
            continue
        name = node.attrib.get("name", "")
        if name == "Sentinel-1 IPF" or name.startswith("Sentinel-1 IPF "):
            version = node.attrib.get("version")
            if name != "Sentinel-1 IPF" or not isinstance(version, str) or not re.fullmatch(r"\d{3}\.\d{2}", version):
                raise ValueError("malformed Sentinel-1 IPF name or version declaration")
            declarations.append(version)
    if not declarations:
        raise ValueError("manifest has no Sentinel-1 IPF version declaration")
    if len(set(declarations)) != 1:
        raise ValueError("manifest has conflicting Sentinel-1 IPF version declarations")
    return declarations[0], len(declarations)


def _product_metadata(paths: dict[str, Path], manifest_path: Path) -> dict:
    from sentinel_gl.ingestion.sentinel1_calibration import parse_native_sentinel1_product

    parsed = parse_native_sentinel1_product(paths["annotation"].read_bytes(),
        paths["calibration"].read_bytes(), paths["noise"].read_bytes())
    parsed["ipf_version"], parsed["ipf_declaration_count"] = manifest_ipf_identity(manifest_path.read_bytes())
    parsed["thermal_noise_correction_applied"] = parsed["thermal_noise_correction_applied"]
    return parsed


def _write_calibration(measurement: Path, output_dir: Path, pol: str, metadata: dict) -> dict:
    from sentinel_gl.ingestion.sentinel1_calibration import calibrate_native_noise_window, strict_power_to_db

    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(measurement) as src:
        if (src.count != 1 or src.width != metadata["columns"] or src.height != metadata["rows"]
                or src.dtypes[0] not in {"uint16", "int16", "uint32"}):
            raise ValueError(f"measurement raster dimensions/dtype differ from annotation for {pol}")
        profile = {"driver": "GTiff", "width": src.width, "height": src.height,
                   "count": 1, "dtype": "float32", "nodata": float("nan"),
                   "compress": "deflate", "predictor": 3, "tiled": True,
                   "blockxsize": 512, "blockysize": 256}
        paths = {
            "sigma0_raw_linear": output_dir / f"{pol}_sigma0_raw_linear_native.tif",
            "sigma0_raw_db": output_dir / f"{pol}_sigma0_raw_db_native.tif",
            "sigma0_signed_denoised_linear": output_dir / f"{pol}_sigma0_noise_subtracted_signed_linear_native.tif",
            "sigma0_signed_denoised_db": output_dir / f"{pol}_sigma0_noise_subtracted_signed_db_native.tif",
            "nesz_linear": output_dir / f"{pol}_nesz_linear_native.tif",
            "calibration_valid": output_dir / f"{pol}_sigma0_calibration_valid_native.tif",
            "noise_valid": output_dir / f"{pol}_sigma0_noise_valid_native.tif",
        }
        for path in paths.values():
            if path.exists():
                raise ValueError(f"refusing to overwrite native calibration output: {path}")
        summary = {"polarization": pol, "rows": src.height, "columns": src.width,
                   "source_crs": src.crs.to_string() if src.crs else None,
                   "source_transform": list(src.transform)[:6], "source_dtype": src.dtypes[0],
                   "native_map_registration": "UNVERIFIED_NATIVE_GRD_PIXEL_GRID",
                   "valid_pixel_count": 0, "noise_supported_pixel_count": 0,
                   "raw_positive_db_pixel_count": 0, "signed_positive_db_pixel_count": 0,
                   "signed_nonpositive_db_missing_count": 0, "unsupported_calibration_count": 0,
                   "unsupported_noise_count": 0}
        vectors = metadata["calibration_vectors"]
        data_datasets = {key: rasterio.open(path, "w", **profile) for key, path in paths.items()
                         if key not in {"calibration_valid", "noise_valid"}}
        mask_datasets = {key: rasterio.open(path, "w", driver="GTiff", width=src.width, height=src.height,
                             count=1, dtype="uint8", compress="deflate", tiled=True,
                             blockxsize=512, blockysize=256, nodata=0)
                         for key, path in paths.items() if key in {"calibration_valid", "noise_valid"}}
        try:
            for row0 in range(0, src.height, 128):
                height = min(128, src.height - row0)
                window = Window(0, row0, src.width, height)
                dn = src.read(1, window=window)
                raw, raw_valid, nesz, signed, noise_valid = calibrate_native_noise_window(
                    dn, vectors, metadata["noise_range_vectors"], metadata["noise_azimuth_blocks"],
                    line_offset=row0, sample_offset=0,
                    thermal_noise_correction_applied=metadata["thermal_noise_correction_applied"])
                raw_db, raw_positive = strict_power_to_db(raw, raw_valid)
                signed_db, signed_positive = strict_power_to_db(signed, noise_valid)
                values = {"sigma0_raw_linear": raw, "sigma0_raw_db": raw_db,
                          "sigma0_signed_denoised_linear": signed,
                          "sigma0_signed_denoised_db": signed_db, "nesz_linear": nesz}
                for key, value in values.items():
                    data_datasets[key].write(value, 1, window=window)
                mask_datasets["calibration_valid"].write(raw_valid.astype(np.uint8), 1, window=window)
                mask_datasets["noise_valid"].write(noise_valid.astype(np.uint8), 1, window=window)
                summary["valid_pixel_count"] += int(raw_valid.sum())
                summary["noise_supported_pixel_count"] += int(noise_valid.sum())
                summary["raw_positive_db_pixel_count"] += int(raw_positive.sum())
                summary["signed_positive_db_pixel_count"] += int(signed_positive.sum())
                summary["signed_nonpositive_db_missing_count"] += int((noise_valid & ~signed_positive).sum())
                summary["unsupported_calibration_count"] += int((~raw_valid).sum())
                summary["unsupported_noise_count"] += int((~noise_valid).sum())
            for key, ds in data_datasets.items():
                ds.update_tags(ORIGIN="provider_download", RESEARCH_ADMISSIBLE="false",
                    RADIOMETRY=("NESZ_noise_power_over_sigmaNought_squared" if key == "nesz_linear" else
                        "signed_noise_subtracted_sigma0=(DN_squared-range_noise_times_azimuth_noise)/sigmaNought_squared" if "signed" in key else
                        "raw_sigma0=DN_squared/sigmaNought_squared"),
                    MAP_GEOREFERENCING="UNAVAILABLE_NATIVE_GRD_PIXEL_COORDINATES",
                    DECIBEL_RULE="10log10(linear power) only where finite and strictly positive; other cells NoData")
            for key, ds in mask_datasets.items():
                ds.update_tags(ORIGIN="provider_download", RESEARCH_ADMISSIBLE="false",
                    STATUS="1=supported; 0=unsupported",
                    SUPPORT=("calibration LUT" if key == "calibration_valid" else "calibration and noise LUT blocks"))
        finally:
            for ds in (*data_datasets.values(), *mask_datasets.values()):
                ds.close()
        summary["outputs"] = {name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                      "sha256": sha256_file(path)} for name, path in paths.items()}
        return summary


def _tiepoint_diagnostic(native_linear: Path, native_valid: Path, tiepoints: list[dict],
                         grid_path: Path, output_dir: Path, site: str, pol: str) -> dict:
    """Resample linearly by annotation tiepoints for explicitly non-research diagnostics."""
    from pyproj import Transformer
    from scipy.interpolate import LinearNDInterpolator
    from sentinel_gl.ingestion.sentinel1_calibration import bilinear_resample_with_valid_weights

    with rasterio.open(grid_path) as grid:
        if grid.crs is None or grid.crs.to_string() != "EPSG:32645" or (grid.height, grid.width) != (499, 499):
            raise ValueError(f"accepted target grid metadata is not the exact 499x499 EPSG:32645 grid for {site}")
        transform = grid.transform
        grid_meta = {"crs": grid.crs.to_string(), "shape": [grid.height, grid.width],
                     "transform": list(transform)[:6], "grid_file_sha256": sha256_file(grid_path),
                     "pixel_values_read": False}
    if len(tiepoints) < 3:
        return {"status": "UNAVAILABLE_INSUFFICIENT_GEOLOCATION_TIE_POINTS", "grid": grid_meta,
                "outputs": {}, "valid_target_pixels": 0, "total_target_pixels": 499 * 499}
    ll = np.asarray([[p["longitude"], p["latitude"]] for p in tiepoints], dtype=np.float64)
    ij = np.asarray([[p["line"], p["pixel"]] for p in tiepoints], dtype=np.float64)
    if len(np.unique(ll, axis=0)) != len(ll):
        raise ValueError("annotation geolocation grid has duplicate longitude/latitude points")
    rr, cc = np.indices((499, 499), dtype=np.float64)
    x = transform.c + transform.a * (cc + 0.5) + transform.b * (rr + 0.5)
    y = transform.f + transform.d * (cc + 0.5) + transform.e * (rr + 0.5)
    lon, lat = Transformer.from_crs(grid_meta["crs"], "EPSG:4326", always_xy=True).transform(x, y)
    query = np.column_stack((np.asarray(lon).ravel(), np.asarray(lat).ravel()))
    mapped = LinearNDInterpolator(ll, ij, fill_value=np.nan)(query).reshape((499, 499, 2))
    native_line, native_sample = mapped[..., 0], mapped[..., 1]
    with rasterio.open(native_linear) as src, rasterio.open(native_valid) as valid_src:
        rows, cols = src.height, src.width
        inside = (np.isfinite(native_line) & np.isfinite(native_sample) &
                  (native_line >= 0) & (native_line <= rows - 1) &
                  (native_sample >= 0) & (native_sample <= cols - 1))
        if not inside.any():
            return {"status": "UNAVAILABLE_TARGET_GRID_OUTSIDE_ANNOTATION_TIE_POINT_SUPPORT",
                    "grid": grid_meta, "outputs": {}, "valid_target_pixels": 0,
                    "total_target_pixels": 499 * 499,
                    "geolocation_method": "piecewise_linear_annotation_tiepoint_inverse_ellipsoid_diagnostic"}
        r0 = np.floor(native_line[inside]).astype(np.int64)
        c0 = np.floor(native_sample[inside]).astype(np.int64)
        rmin, rmax = max(0, int(r0.min())), min(rows, int(np.ceil(native_line[inside].max())) + 2)
        cmin, cmax = max(0, int(c0.min())), min(cols, int(np.ceil(native_sample[inside].max())) + 2)
        window = Window(cmin, rmin, cmax - cmin, rmax - rmin)
        source = src.read(1, window=window).astype(np.float64)
        source_valid = valid_src.read(1, window=window).astype(bool) & np.isfinite(source)
    local_r, local_c = native_line - rmin, native_sample - cmin
    sampled, coverage, valid_target = bilinear_resample_with_valid_weights(
        source, source_valid, local_r, local_c)
    mapped_power = sampled.astype(np.float32)
    valid_target &= np.isfinite(mapped_power)
    mapped_power[~valid_target] = np.nan
    db_values, db_valid = __import__("sentinel_gl.ingestion.sentinel1_calibration", fromlist=["strict_power_to_db"]).strict_power_to_db(mapped_power, valid_target)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{site}__{pol}"
    products = {
        "linear": (output_dir / f"{stem}_sigma0_raw_linear_tiepoint_diagnostic_20m.tif", mapped_power, "float32", float("nan")),
        "db": (output_dir / f"{stem}_sigma0_raw_db_tiepoint_diagnostic_20m.tif", db_values, "float32", float("nan")),
        "coverage": (output_dir / f"{stem}_linear_valid_weight_coverage_tiepoint_diagnostic_20m.tif", coverage.astype(np.float32), "float32", 0.0),
        "valid": (output_dir / f"{stem}_valid_tiepoint_diagnostic_20m.tif", valid_target.astype(np.uint8), "uint8", 0),
    }
    with rasterio.open(grid_path) as grid:
        for kind, (path, values, dtype, nodata) in products.items():
            if path.exists():
                raise ValueError(f"refusing to overwrite tiepoint diagnostic output: {path}")
            profile = {"driver":"GTiff","width":499,"height":499,"count":1,"dtype":dtype,
                       "crs":grid.crs,"transform":grid.transform,"nodata":nodata,
                       "compress":"deflate","tiled":True,"blockxsize":256,"blockysize":256}
            with rasterio.open(path,"w",**profile) as dst:
                dst.write(values,1)
                dst.update_tags(ORIGIN="provider_download", RESEARCH_ADMISSIBLE="false",
                    GEOCODING="piecewise-linear Sentinel-1 annotation geolocation tiepoints; ellipsoid diagnostic only",
                    TERRAIN_CORRECTION="NOT_PERFORMED", LAYOVER_SHADOW_QA="UNAVAILABLE",
                    TARGET_CELL_SIZE="20m; this does not assert 20m sensor resolution",
                    WATER_JOIN="NOT_PERFORMED")
    return {"status":"ELLIPSOID_TIEPOINT_DIAGNOSTIC_ONLY_NO_TERRAIN_CORRECTION",
            "grid":grid_meta,"geolocation_method":"scipy LinearNDInterpolator inverse mapping of annotation tiepoints",
            "valid_target_pixels":int(valid_target.sum()),"total_target_pixels":499*499,
            "positive_db_target_pixels":int(db_valid.sum()),
            "mean_linear_power_over_valid_diagnostic_pixels":float(np.mean(mapped_power[valid_target])) if valid_target.any() else None,
            "coverage_weight_min":float(coverage[valid_target].min()) if valid_target.any() else None,
            "coverage_weight_mean":float(coverage[valid_target].mean()) if valid_target.any() else None,
            "whole_roi_research_summary":None,"candidate_water_research_summary":None,
            "summary_status":"UNAVAILABLE_NO_TERRAIN_CORRECTED_LAKE_GEOMETRY_OR_VALIDATED_MASK_ALIGNMENT",
            "outputs":{k:{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)}
                       for k,(p,_,_,_) in products.items()}}


def regenerate_tiepoint_diagnostics(source_execution_record: Path, input_dir: Path,
                                    output_dir: Path) -> dict:
    """Regenerate only map-grid diagnostics from verified native outputs."""
    started = time.monotonic()
    if output_dir.exists():
        raise ValueError(f"attempt directory already exists; refusing overwrite: {output_dir}")
    source_record = json.loads(source_execution_record.read_text(encoding="utf-8"))
    if source_record.get("research_admissible") is not False or source_record.get("archive", {}).get("sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("source execution record is not bound to the quarantined retained SAR archive")
    output_index = {item["path"]: item for item in source_record.get("outputs", [])}
    member_index = source_record.get("extracted_members", {})
    role_evidence = json.loads(DEFAULT_ROLE_EVIDENCE.read_text(encoding="utf-8"))
    selected = _selected_members(role_evidence)
    metadata, verified_inputs, native_refs = {}, {}, {}
    for pol in ("VV", "VH"):
        paths = {}
        for role in ("annotation", "calibration", "noise", "manifest"):
            member = selected[pol][role]
            entry = member_index.get(member)
            if not isinstance(entry, dict):
                raise ValueError(f"source execution receipt lacks {pol}:{role} input binding")
            suffix = Path(member).suffix
            input_name = f"{pol}_{role}{suffix}"
            if pol == "VH" and role == "manifest":
                input_name = "VV_manifest.safe"
            path = (input_dir / input_name).resolve()
            if not path.is_file() or path.is_symlink() or path.stat().st_size != entry.get("extracted_bytes"):
                raise ValueError(f"verified prior SAR input is absent or changed: {path}")
            digest = sha256_file(path)
            if digest != entry.get("sha256"):
                raise ValueError(f"verified prior SAR input hash changed: {path}")
            verified_inputs[member] = {"sha256": digest, "bytes": path.stat().st_size}
            paths[role] = path
        metadata[pol] = _product_metadata(paths, paths["manifest"])
        if metadata[pol]["polarization"] != pol or metadata[pol]["ipf_version"] != "003.61":
            raise ValueError(f"native metadata identity changed for {pol}")
        for kind, filename in (("linear", f"{pol}_sigma0_raw_linear_native.tif"),
                               ("valid", f"{pol}_sigma0_calibration_valid_native.tif")):
            rel = f"{source_execution_record.parent.relative_to(ROOT)}/calibrated_native/{filename}"
            entry = output_index.get(rel)
            path = (source_execution_record.parent / "calibrated_native" / filename).resolve()
            if not isinstance(entry, dict) or not path.is_file() or path.stat().st_size != entry.get("bytes"):
                raise ValueError(f"prior native calibration output is absent or changed: {path}")
            digest = sha256_file(path)
            if digest != entry.get("sha256"):
                raise ValueError(f"prior native calibration output hash changed: {path}")
            native_refs[f"{pol}:{kind}"] = {"path": str(path.relative_to(ROOT)), "sha256": digest,
                                              "bytes": path.stat().st_size}
            if kind == "linear":
                paths[kind] = path
            else:
                paths[kind] = path
        metadata[pol]["native_linear_path"] = paths["linear"]
        metadata[pol]["native_valid_path"] = paths["valid"]
    output_dir.mkdir(parents=True)
    roi = {}
    for site, grid_path in SITE_GRID_FILES.items():
        site_diagnostics = {}
        with rasterio.open(grid_path) as grid:
            grid_binding = {"crs": grid.crs.to_string() if grid.crs else None,
                "shape": [grid.height, grid.width], "transform": list(grid.transform)[:6],
                "sha256": sha256_file(grid_path), "pixel_values_read": False}
        for pol in ("VV", "VH"):
            diagnostic = _tiepoint_diagnostic(metadata[pol]["native_linear_path"],
                metadata[pol]["native_valid_path"], metadata[pol]["geolocation_grid_points"],
                grid_path, output_dir, site, pol)
            site_diagnostics[pol] = diagnostic
        roi[site] = {"target_grid": grid_binding, "diagnostic_products": site_diagnostics,
            "whole_roi_research_summary": None, "candidate_water_research_summary": None,
            "water_mask_status": "UNAVAILABLE_NO_TERRAIN_CORRECTED_SAR_MASK_ALIGNMENT",
            "cross_date_note": "2023-09-01 SAR precedes 2023-09-04/21 optical masks; retrospective diagnostic only."}
    outputs = []
    for path in sorted(output_dir.glob("*.tif")):
        outputs.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    result = {"schema_version": 1, "status": "TIEPOINT_DIAGNOSTICS_REGENERATED_FROM_VERIFIED_NATIVE_OUTPUTS",
        "research_admissible": False, "native_calibration_rerun": False,
        "network_requests": 0, "source_execution_record": str(source_execution_record.relative_to(ROOT)),
        "source_execution_record_sha256": sha256_file(source_execution_record),
        "verified_metadata_inputs": verified_inputs, "verified_native_calibration_outputs": native_refs,
        "roi_diagnostics": roi, "outputs": outputs, "elapsed_seconds": time.monotonic()-started,
        "limitations":["Tiepoint geolocation is ellipsoid diagnostic only; terrain correction and layover/shadow QA unavailable.",
            "Research whole-ROI and water-conditioned summaries remain null.",
            "Optical candidate-water pixel values were not read; any later-date join is retrospective only."]}
    receipt=output_dir/"tiepoint_diagnostics_execution.json"
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    result["execution_record"] = str(receipt.relative_to(ROOT))
    result["execution_record_sha256"] = sha256_file(receipt)
    return result


def run(archive: Path, role_evidence_path: Path, output_dir: Path,
        reuse_input_dir: Path | None = None) -> dict:
    started = time.monotonic()
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES or sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("retained archive size/SHA-256 differs from the authorized SAFE")
    role_evidence = json.loads(role_evidence_path.read_text(encoding="utf-8"))
    if role_evidence.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256 or role_evidence.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("accepted role evidence is not bound to the authorized archive")
    if output_dir.exists():
        raise ValueError(f"attempt directory already exists; refusing overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    extracted = (_verify_reused_inputs(archive, reuse_input_dir, role_evidence)
                 if reuse_input_dir is not None else _extract_roles(archive, output_dir, role_evidence))
    products, calibrations, parsed_products = {}, [], {}
    for pol in ("VV", "VH"):
        paths = extracted["paths"][pol]
        metadata = _product_metadata(paths, paths["manifest"])
        if metadata["polarization"] != pol or metadata["rows"] != 16731 or metadata["columns"] != 25851:
            raise ValueError(f"native annotation identity/dimensions differ from role-bound {pol} metadata")
        parsed_products[pol] = metadata
        products[pol] = {
            "mission": metadata["mission"], "product_type": metadata["product_type"],
            "mode": metadata["mode"], "polarization": pol, "ipf_version": metadata["ipf_version"],
            "ipf_declaration_count": metadata["ipf_declaration_count"],
            "rows": metadata["rows"], "columns": metadata["columns"],
            "start_time": metadata["start_time"], "stop_time": metadata["stop_time"],
            "thermal_noise_correction_applied": metadata["thermal_noise_correction_applied"],
            "processing_flags": metadata["processing_flags"],
            "geolocation_grid_point_count": metadata["geolocation_grid_point_count"],
            "calibration_line_gap_values": sorted(set(np.diff([v[0] for v in metadata["calibration_vectors"]]).tolist())),
            "calibration_sample_support": [int(metadata["calibration_vectors"][0][1][0]), int(metadata["calibration_vectors"][0][1][-1])],
            "calibration_vector_count": len(metadata["calibration_vectors"]),
            "noise_range_vector_count": len(metadata["noise_range_vectors"]),
            "noise_range_line_gap_values": sorted(set(np.diff([v[0] for v in metadata["noise_range_vectors"]]).tolist())),
            "noise_range_per_vector_sample_support": [[int(v[1][0]), int(v[1][-1])] for v in metadata["noise_range_vectors"]],
            "noise_azimuth_blocks": [{k:v for k,v in block.items() if k not in {"lines","values"}}
                                      | {"line_vector_count":len(block["lines"])}
                                      for block in metadata["noise_azimuth_blocks"]],
            "noise_corrected_sigma0_status": "SOURCE_NATIVE_RANGE_AND_AZIMUTH_LUTS_APPLIED_WITH_DECLARED_BLOCK_SUPPORT",
            "noise_status_reason": "Range LUT interpolated within each native vector along pixel then line coordinates; azimuth LUT interpolated in line coordinates only within each declared swath/sample support block. Thermal correction flag is independently parsed and must be false.",
        }
        calibrations.append(_write_calibration(paths["measurement"], output_dir / "calibrated_native", pol, metadata))
    grids = {}
    for site, path in SITE_GRID_FILES.items():
        with rasterio.open(path) as src:
            grids[site] = {"crs": src.crs.to_string() if src.crs else None,
                           "shape": [src.height, src.width], "transform": list(src.transform)[:6],
                           "sha256": sha256_file(path),
                           "status": "TARGET_GRID_METADATA_ONLY_QUARANTINED_OPTICAL_VALUES_NOT_READ"}
    roi = {}
    for site, grid in grids.items():
        site_row = {"target_grid": grid, "diagnostic_products": {},
                    "whole_roi_summary": None, "candidate_water_summary": None,
                    "water_mask_status": "UNAVAILABLE_NO_TERRAIN_CORRECTED_SAR_MASK_ALIGNMENT",
                    "cross_date_note": "2023-09-01 SAR precedes 2023-09-04/21 optical masks; any future join is retrospective diagnostic only."}
        for pol in ("VV", "VH"):
            native_dir = output_dir / "calibrated_native"
            diagnostic = _tiepoint_diagnostic(
                native_dir / f"{pol}_sigma0_raw_linear_native.tif",
                native_dir / f"{pol}_sigma0_calibration_valid_native.tif",
                parsed_products[pol]["geolocation_grid_points"], SITE_GRID_FILES[site],
                output_dir / "roi_tiepoint_diagnostics", site, pol)
            site_row["diagnostic_products"][pol] = diagnostic
            site_row[f"{pol.lower()}_status"] = diagnostic["status"]
        roi[site] = site_row
    outputs = []
    for path in sorted(output_dir.rglob("*.tif")):
        outputs.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    result = {
        "schema_version": 1, "status": "NATIVE_RAW_AND_SIGNED_DENOISED_SIGMA0_COMPLETE_TIEPOINT_ROI_DIAGNOSTICS_ONLY",
        "research_admissible": False, "network_requests": 0,
        "archive": {"path": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
                    "sha256": sha256_file(archive), "expected_sha256": EXPECTED_ARCHIVE_SHA256},
        "role_evidence": str(role_evidence_path.relative_to(ROOT)),
        "extracted_members": extracted["files"], "products": products, "calibrations": calibrations,
        "roi_extraction": roi,
        "terrain_support": {"local_dem": "NOT_FOUND_IN_DATA_TREE_BY_FILENAME_SEARCH",
                            "precise_orbit": "NOT_FOUND_IN_DATA_TREE_BY_FILENAME_SEARCH",
                            "gdalinfo": shutil.which("gdalinfo"), "gdalwarp": shutil.which("gdalwarp"),
                            "gdal_translate": shutil.which("gdal_translate"),
                            "otbcli_OrthoRectification": shutil.which("otbcli_OrthoRectification"),
                            "status": "UNAVAILABLE_NO_LOCAL_DEM_PRECISE_ORBIT_OR_TERRAIN_CORRECTION_TOOL"},
        "outputs": outputs, "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Native GRD products have no CRS and identity transform; diagnostic grid outputs use piecewise-linear annotation geolocation only.",
                        "Tiepoint outputs are not terrain corrected; layover/shadow QA and lake measurement admission remain unavailable.",
                        "Whole-ROI and candidate-water research summaries remain null because terrain-corrected lake geometry and validated mask alignment are absent.",
                        "Optical candidate-water mask values were not read. Its later acquisition date makes any future join retrospective only.",
                        "All native calibration and tiepoint outputs are quarantined engineering transforms, not admitted research measurements."],
    }
    result_path = output_dir / "native_sar_execution.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["execution_record"] = str(result_path.relative_to(ROOT))
    result["execution_record_sha256"] = sha256_file(result_path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--role-evidence", type=Path, default=DEFAULT_ROLE_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-input-dir", type=Path)
    parser.add_argument("--diagnostics-only-from", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.diagnostics_only_from:
            if not args.reuse_input_dir:
                raise ValueError("--diagnostics-only-from requires --reuse-input-dir")
            result = regenerate_tiepoint_diagnostics(args.diagnostics_only_from.resolve(),
                args.reuse_input_dir.resolve(), args.output_dir.resolve())
        else:
            result = run(args.archive.resolve(), args.role_evidence.resolve(), args.output_dir.resolve(),
                         args.reuse_input_dir.resolve() if args.reuse_input_dir else None)
    except Exception as exc:
        print(f"WP03_NATIVE_SAR_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: result.get(k) for k in ("status", "research_admissible", "archive", "execution_record",
                                           "execution_record_sha256", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
