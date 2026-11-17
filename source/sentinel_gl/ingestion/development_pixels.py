"""Offline Sentinel-1/2 pixel decoding for explicitly sourced development data.

This adapter returns decoded arrays and provenance joins. It does not admit
observations or establish scientific readiness. Constructed inputs remain
explicitly tagged as fixtures in every result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import uuid
import zipfile

from defusedxml import ElementTree as SafeET
import numpy as np
import rasterio

from ..admission import AdmissionError, QUARANTINED
from ..integrity import file_hash, loads_strict


def _local(element):
    return element.tag.rsplit("}", 1)[-1]


def _elements(root, name):
    return [node for node in root.iter() if _local(node) == name]


def _first_text(root, name, *, required=True):
    for node in _elements(root, name):
        if node.text and node.text.strip():
            return node.text.strip()
    if required:
        raise AdmissionError(f"product metadata is missing {name}")
    return None


def _unique_number(root, name, label):
    values = [_number(node.text.strip(), label) for node in _elements(root, name)
              if node.text and node.text.strip()]
    if not values:
        raise AdmissionError(f"product metadata is missing {name}")
    if any(value != values[0] for value in values[1:]):
        raise AdmissionError(f"conflicting duplicate {label} metadata")
    return values[0]


def _asset_member_for_path(asset_path, archive_members):
    asset_path = Path(asset_path)
    matches = [member for member, path in archive_members.items()
               if Path(path) == asset_path]
    if len(matches) != 1:
        raise AdmissionError("asset path has an ambiguous archive member mapping")
    return matches[0]


def _fixture_asset_roles(root):
    roles = {}
    for node in _elements(root, "Engineering_Fixture_Asset_Role"):
        role = node.attrib.get("role")
        band = node.attrib.get("band")
        polarization = node.attrib.get("polarization")
        member = node.attrib.get("member")
        if not role or not member:
            raise AdmissionError("constructed fixture asset role is incomplete")
        key = (role, band, polarization.upper() if polarization else None)
        if key in roles:
            raise AdmissionError("duplicate constructed fixture asset role")
        roles[key] = member
    return roles


def _require_fixture_source(source, product):
    if source.get("origin") != "constructed_fixture":
        raise AdmissionError(
            f"provider {product} pixel decoding is blocked: retained local evidence "
            "does not establish product asset-role to archive-member mapping"
        )


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdmissionError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AdmissionError(f"{label} is not finite")
    return result


def _utc(value):
    if not isinstance(value, str) or not value:
        raise AdmissionError("source time is unavailable")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdmissionError("source time is malformed") from exc
    if parsed.tzinfo is None:
        raise AdmissionError("source time must include a timezone")
    return parsed.astimezone(timezone.utc)


def _source_join(source, product_name, archive_path):
    required = ("origin", "provider_product_id", "provider_record_path",
                "provider_record_sha256", "archive_sha256")
    if not isinstance(source, dict) or any(not source.get(key) for key in required):
        raise AdmissionError("complete source identity and time join is required")
    if source["origin"] not in {"constructed_fixture", "provider_download"}:
        raise AdmissionError("source origin must be explicit")
    if not isinstance(source["provider_product_id"], str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", source["provider_product_id"]):
        raise AdmissionError("provider product identifier is malformed")
    try:
        uuid.UUID(source["provider_product_id"])
    except ValueError as exc:
        raise AdmissionError("provider product identifier is malformed") from exc
    for key in ("provider_record_sha256", "archive_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", source[key]):
            raise AdmissionError(f"{key} must be a SHA-256 digest")
    record_path, archive_path = Path(source["provider_record_path"]), Path(archive_path)
    if (record_path.is_symlink() or archive_path.is_symlink() or not record_path.is_file() or
            not archive_path.is_file()):
        raise AdmissionError("regular provider record and archive files are required")
    if file_hash(record_path) != source["provider_record_sha256"] or file_hash(archive_path) != source["archive_sha256"]:
        raise AdmissionError("provider record or archive digest mismatch")
    if source["archive_sha256"] in QUARANTINED:
        raise AdmissionError("quarantined archive bytes cannot be decoded")
    try:
        record = loads_strict(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise AdmissionError("provider record is unreadable") from exc
    if not isinstance(record, dict) or not isinstance(record.get("Id"), str) or record["Id"].lower() != source["provider_product_id"].lower():
        raise AdmissionError("provider record product ID does not match source join")
    if record.get("Name") != product_name:
        raise AdmissionError("source product name does not match decoded metadata")
    dates = record.get("ContentDate", {})
    start, end = _utc(dates.get("Start", "")), _utc(dates.get("End", ""))
    if end < start:
        raise AdmissionError("source acquisition interval is reversed")
    return {"provider_product_id": source["provider_product_id"],
            "provider_product_name": record["Name"],
            "acquisition_start": start.isoformat(), "acquisition_end": end.isoformat(),
            "provider_record_sha256": source["provider_record_sha256"],
            "archive_sha256": source["archive_sha256"],
            "origin": source["origin"],
            "evidence_scope": "CONSTRUCTED_ENGINEERING_FIXTURE_ONLY" if source["origin"] == "constructed_fixture"
                             else "DEVELOPMENT_SOURCE_NOT_RESEARCH_ADMITTED",
            "research_admissible": False}


def _verify_archive_members(archive_path, members, product_name):
    if not isinstance(members, dict) or not members:
        raise AdmissionError("archive member mapping is required for source joins")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names_list = archive.namelist()
            if len(names_list) != len(set(names_list)):
                raise AdmissionError("duplicate paths in source archive")
            names = set(names_list)
            member_hashes = {}
            for member, extracted_path in members.items():
                posix = PurePosixPath(member)
                if (posix.is_absolute() or ".." in posix.parts or "\\" in member or
                        not posix.parts or posix.parts[0] != product_name or member not in names):
                    raise AdmissionError("unsafe or missing source archive member")
                info = archive.getinfo(member)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise AdmissionError("symbolic link in source archive member")
                extracted_path = Path(extracted_path)
                if extracted_path.is_symlink() or not extracted_path.is_file():
                    raise AdmissionError("regular extracted source asset required")
                member_digest = hashlib.sha256()
                with archive.open(member) as member_stream:
                    while chunk := member_stream.read(1024 * 1024):
                        member_digest.update(chunk)
                if info.file_size != extracted_path.stat().st_size or member_digest.hexdigest() != file_hash(extracted_path):
                    raise AdmissionError("extracted asset bytes do not match source archive member")
                member_hashes[member] = member_digest.hexdigest()
            return member_hashes
    except zipfile.BadZipFile as exc:
        raise AdmissionError("source archive is not a readable ZIP") from exc


def _metadata_time_join(root, join):
    def unique_time(names, label):
        values = [node.text.strip() for name in names for node in _elements(root, name)
                  if node.text and node.text.strip()]
        if not values:
            return None
        parsed = [_utc(value) for value in values]
        if any(value != parsed[0] for value in parsed[1:]):
            raise AdmissionError(f"conflicting duplicate metadata acquisition {label}")
        return parsed[0]

    metadata_start = unique_time(("PRODUCT_START_TIME", "startTime"), "start time")
    metadata_end = unique_time(("PRODUCT_STOP_TIME", "stopTime"), "stop time")
    if metadata_start is None and metadata_end is None:
        return "UNVERIFIED_METADATA_TIME_UNAVAILABLE"
    if metadata_start is None or metadata_end is None:
        return "UNVERIFIED_METADATA_TIME_INCOMPLETE"
    if metadata_end < metadata_start:
        raise AdmissionError("metadata acquisition interval is reversed")
    source_start, source_end = _utc(join["acquisition_start"]), _utc(join["acquisition_end"])
    if metadata_start < source_start or metadata_end > source_end:
        raise AdmissionError("metadata acquisition interval falls outside provider record interval")
    return "MATCHED_METADATA_WITHIN_PROVIDER_INTERVAL"


def _grid(dataset):
    authority = dataset.crs.to_authority() if dataset.crs else None
    return {"crs": dataset.crs.to_string() if dataset.crs else None,
            "crs_authority": list(authority) if authority else None,
            "transform": list(dataset.transform)[:6],
            "shape": [dataset.height, dataset.width],
            "resolution": [float(dataset.res[0]), float(dataset.res[1])],
            "bounds": [float(v) for v in dataset.bounds],
            "nodata": dataset.nodata}


def _read_raster(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AdmissionError("regular raster file required")
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise AdmissionError("single-band measurement raster required")
        values = dataset.read(1)
        mask = (dataset.read_masks(1) != 0) & np.isfinite(values)
        return values, mask, _grid(dataset), file_hash(path)


def decode_sentinel2_l2a(metadata_path, band_paths, scl_path, source, archive_path,
                         archive_members):
    """Decode B03/B08/B11 reflectance with product-stated offsets and scale."""
    metadata_path = Path(metadata_path)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise AdmissionError("regular Sentinel-2 product metadata XML required")
    try:
        root = SafeET.parse(metadata_path).getroot()
    except Exception as exc:
        raise AdmissionError("Sentinel-2 product metadata XML is unreadable") from exc
    product_name = (_first_text(root, "PRODUCT_URI", required=False) or
                    _first_text(root, "PRODUCT_ID"))
    join = _source_join(source, product_name, archive_path)
    _require_fixture_source(source, "Sentinel-2")
    member_hashes = _verify_archive_members(archive_path, archive_members, product_name)
    expected_assets = {Path(metadata_path), Path(scl_path), *(Path(p) for p in band_paths.values())}
    if {Path(p) for p in archive_members.values()} != expected_assets:
        raise AdmissionError("archive member mapping does not cover exactly the decoded Sentinel-2 assets")
    time_join = _metadata_time_join(root, join)
    quantification = _unique_number(root, "BOA_QUANTIFICATION_VALUE", "BOA quantification value")
    if quantification <= 0:
        raise AdmissionError("BOA quantification value must be positive")

    band_ids = {}
    band_id_owners = {}
    for node in _elements(root, "Spectral_Information"):
        physical = node.attrib.get("physicalBand")
        band_id = node.attrib.get("bandId")
        if physical is None and band_id is None:
            continue
        if not physical or band_id is None:
            raise AdmissionError("spectral identity metadata is incomplete")
        if physical in band_ids and band_ids[physical] != band_id:
            raise AdmissionError(f"conflicting duplicate spectral identity for {physical}")
        if band_id in band_id_owners and band_id_owners[band_id] != physical:
            raise AdmissionError(f"conflicting duplicate spectral identity for band ID {band_id}")
        band_ids[physical] = band_id
        band_id_owners[band_id] = physical
    offsets = {}
    for node in _elements(root, "BOA_ADD_OFFSET"):
        band_id = node.attrib.get("band_id")
        if band_id is None or not node.text or not node.text.strip():
            raise AdmissionError("BOA additive offset identity/value is incomplete")
        value = _number(node.text.strip(), "BOA additive offset")
        if band_id in offsets and offsets[band_id] != value:
            raise AdmissionError(f"conflicting duplicate BOA additive offset for band ID {band_id}")
        offsets[band_id] = value

    fixture_roles = _fixture_asset_roles(root)
    if set(band_paths) != {"B03", "B08", "B11"}:
        raise AdmissionError("constructed Sentinel-2 fixture must map exactly B03, B08 and B11")
    expected_role_keys = {("S2_BAND", band, None) for band in ("B03", "B08", "B11")}
    expected_role_keys.add(("S2_SCL", None, None))
    if set(fixture_roles) != expected_role_keys:
        raise AdmissionError("constructed Sentinel-2 fixture must declare each band and SCL asset role")
    for band in ("B03", "B08", "B11"):
        path = band_paths.get(band)
        if path is None:
            raise AdmissionError(f"Sentinel-2 raster is missing for {band}")
        member = _asset_member_for_path(path, archive_members)
        if fixture_roles[("S2_BAND", band, None)] != member:
            raise AdmissionError(f"Sentinel-2 {band} role does not match its archive member")
    scl_member = _asset_member_for_path(scl_path, archive_members)
    if fixture_roles[("S2_SCL", None, None)] != scl_member:
        raise AdmissionError("Sentinel-2 SCL role does not match its archive member")

    decoded = {}
    for band in ("B03", "B08", "B11"):
        band_id = band_ids.get(band)
        if band_id is None or band_id not in offsets:
            raise AdmissionError(f"Sentinel-2 offset is unresolved for {band}")
        if band not in band_paths:
            raise AdmissionError(f"Sentinel-2 raster is missing for {band}")
        dn, mask, grid, digest = _read_raster(band_paths[band])
        # PB 04+ reserves DN 0 for no-data; retain the reason as a separate mask.
        zero = dn == 0
        valid = mask & ~zero
        reason = np.zeros(dn.shape, dtype=np.uint8)
        reason[~mask] = 1  # raster nodata or provider mask
        reason[mask & zero] = 2  # DN zero is product no-data
        reflectance = (dn.astype(np.float32) + np.float32(offsets[band_id])) / np.float32(quantification)
        decoded[band] = {"reflectance": np.ma.array(reflectance, mask=~valid),
                         "valid_mask": valid, "missing_reason_code": reason,
                         "digital_number": dn, "additive_offset": offsets[band_id],
                         "quantification_value": quantification, "grid": grid,
                         "source_sha256": digest}

    scl, scl_mask, scl_grid, scl_digest = _read_raster(scl_path)
    scl_valid = scl_mask & (scl != 0)
    return {"status": "PIXELS_DECODED_DEVELOPMENT_ONLY", "source_join": join,
            "source_time_join": time_join,
            "metadata_sha256": file_hash(metadata_path),
            "archive_member_sha256": member_hashes,
            "bands": decoded,
            "scene_classification": {"class_values": scl, "valid_mask": scl_valid,
                                     "missing_reason_code": np.where(scl_mask, np.where(scl == 0, 2, 0), 1).astype(np.uint8),
                                     "grid": scl_grid, "source_sha256": scl_digest,
                                     "qa_policy_status": "RAW_CLASSES_ONLY_NO_VALIDITY_POLICY_APPLIED"},
            "grid_alignment": "PER_ASSET_GRIDS_RETAINED_NO_IMPLICIT_RESAMPLING",
            "research_admissible": False}


def _vector_nodes(root, value_name):
    vectors = []
    for node in _elements(root, "calibrationVector") if value_name == "sigmaNought" else _elements(root, "noiseRangeVector"):
        values = {}
        for child in node:
            name = _local(child)
            if name in {"line", "pixel", value_name} and child.text:
                values[name] = child.text.strip()
        if not {"line", "pixel", value_name} <= values.keys():
            raise AdmissionError(f"{value_name} vector is incomplete")
        line = int(values["line"])
        pixels = np.asarray([int(v) for v in values["pixel"].split()], dtype=np.float64)
        lut = np.asarray([_number(v, value_name) for v in values[value_name].split()], dtype=np.float64)
        if pixels.size < 2 or pixels.size != lut.size or np.any(np.diff(pixels) <= 0) or np.any(lut < 0):
            raise AdmissionError(f"{value_name} vector coordinates/values are invalid")
        vectors.append((line, pixels, lut))
    if len(vectors) < 2 or len({item[0] for item in vectors}) != len(vectors):
        raise AdmissionError(f"at least two unique {value_name} vectors are required")
    return sorted(vectors, key=lambda item: item[0])


def _interpolate_vectors(vectors, height, width):
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    line_values = np.asarray([v[0] for v in vectors], dtype=np.float64)
    range_values = np.full((len(vectors), width), np.nan, dtype=np.float64)
    for index, (_, pixels, lut) in enumerate(vectors):
        inside = (cols >= pixels[0]) & (cols <= pixels[-1])
        range_values[index, inside] = np.interp(cols[inside], pixels, lut)
    result = np.full((height, width), np.nan, dtype=np.float64)
    for col in range(width):
        finite = np.isfinite(range_values[:, col])
        if finite.sum() >= 2:
            inside = (rows >= line_values[finite][0]) & (rows <= line_values[finite][-1])
            result[inside, col] = np.interp(rows[inside], line_values[finite], range_values[finite, col])
    return result


def decode_sentinel1_grd(measurement_path, calibration_xml, noise_xml, polarization, source,
                         archive_path, archive_members, *, noise_lut_scale=None):
    """Decode GRD amplitude and Sigma0 LUTs; noise correction requires an explicit verified scale."""
    calibration_xml, noise_xml = Path(calibration_xml), Path(noise_xml)
    try:
        calibration = SafeET.parse(calibration_xml).getroot()
        noise = SafeET.parse(noise_xml).getroot()
    except Exception as exc:
        raise AdmissionError("Sentinel-1 calibration/noise XML is unreadable") from exc
    source_name = source.get("provider_product_name") if isinstance(source, dict) else None
    join = _source_join(source, source_name, archive_path)
    _require_fixture_source(source, "Sentinel-1")
    member_hashes = _verify_archive_members(archive_path, archive_members, join["provider_product_name"])
    expected_assets = {Path(measurement_path), calibration_xml, noise_xml}
    if {Path(p) for p in archive_members.values()} != expected_assets:
        raise AdmissionError("archive member mapping does not cover exactly the decoded Sentinel-1 assets")
    time_join = _metadata_time_join(calibration, join)
    requested_polarization = polarization.upper() if isinstance(polarization, str) else ""
    if requested_polarization not in {"VV", "VH", "HH", "HV"}:
        raise AdmissionError("requested measurement polarization is invalid")
    declared_pols = [n.text.strip().upper() for n in _elements(calibration, "polarisation")
                     if n.text and n.text.strip()]
    noise_pols = [n.text.strip().upper() for n in _elements(noise, "polarisation")
                  if n.text and n.text.strip()]
    if len(set(declared_pols)) != 1 or len(set(noise_pols)) != 1:
        raise AdmissionError("calibration and noise polarization metadata must each be unique")
    if declared_pols[0] != requested_polarization or noise_pols[0] != requested_polarization:
        raise AdmissionError("measurement, calibration and noise polarization do not match")
    calibration_roles = _fixture_asset_roles(calibration)
    noise_roles = _fixture_asset_roles(noise)
    if set(calibration_roles) & set(noise_roles):
        raise AdmissionError("duplicate constructed fixture role across calibration/noise metadata")
    fixture_roles = {**calibration_roles, **noise_roles}
    expected_role_keys = {("S1_MEASUREMENT", None, requested_polarization),
                          ("S1_CALIBRATION", None, requested_polarization),
                          ("S1_NOISE", None, requested_polarization)}
    if set(fixture_roles) != expected_role_keys:
        raise AdmissionError("constructed Sentinel-1 fixture must declare measurement, calibration and noise roles")
    assets = {"S1_MEASUREMENT": measurement_path,
              "S1_CALIBRATION": calibration_xml, "S1_NOISE": noise_xml}
    for role, path in assets.items():
        member = _asset_member_for_path(path, archive_members)
        if fixture_roles[(role, None, requested_polarization)] != member:
            raise AdmissionError(f"Sentinel-1 {role.lower()} role does not match its archive member")
    with rasterio.open(measurement_path) as dataset:
        if dataset.count != 1:
            raise AdmissionError("single-polarization GRD measurement raster required")
        dn = dataset.read(1).astype(np.float64)
        valid = (dataset.read_masks(1) != 0) & np.isfinite(dn) & (dn != 0)
        grid = _grid(dataset)
    sigma = _interpolate_vectors(_vector_nodes(calibration, "sigmaNought"), *dn.shape)
    valid &= np.isfinite(sigma) & (sigma > 0)
    # Sentinel-1 GRD measurement pixels are detected amplitudes; calibration
    # vectors provide the amplitude-domain sigma-nought LUT.
    sigma0 = np.divide(np.square(dn), np.square(sigma), out=np.full(dn.shape, np.nan), where=valid)
    noise_vectors = _vector_nodes(noise, "noiseRangeLut")
    noise_lut = _interpolate_vectors(noise_vectors, *dn.shape)
    noise_status = "NOT_APPLIED_NOISE_LUT_SCALE_UNVERIFIED"
    sigma0_noise_corrected = None
    if noise_lut_scale is not None:
        if source["origin"] != "constructed_fixture":
            raise AdmissionError("fixture noise scale cannot be applied to provider data before scientific review")
        if noise_lut_scale != "POWER_DN_SQUARED":
            noise_status = "UNSUPPORTED_NOISE_LUT_SCALE"
        else:
            corrected_power = np.square(dn) - noise_lut
            noise_valid = valid & np.isfinite(noise_lut) & (corrected_power > 0)
            sigma0_noise_corrected = np.ma.array(
                np.divide(corrected_power, np.square(sigma), out=np.full(dn.shape, np.nan), where=noise_valid),
                mask=~noise_valid)
            noise_status = "APPLIED_EXPLICIT_POWER_DN_SQUARED_FIXTURE_SCALE"
    noise_azimuth_present = bool(_elements(noise, "noiseAzimuthVector"))
    if noise_azimuth_present:
        noise_status = "NOT_APPLIED_AZIMUTH_NOISE_COMPONENT_UNSUPPORTED"
        sigma0_noise_corrected = None
    return {"status": "PIXELS_DECODED_DEVELOPMENT_ONLY", "source_join": join,
            "source_time_join": time_join,
            "archive_member_sha256": member_hashes,
            "measurement_sha256": file_hash(Path(measurement_path)),
            "calibration_sha256": file_hash(calibration_xml), "noise_sha256": file_hash(noise_xml),
            "polarization": polarization.upper(), "digital_number_amplitude": dn,
            "sigma_nought_lut": sigma, "sigma0": np.ma.array(sigma0, mask=~valid),
            "noise_lut": noise_lut, "sigma0_noise_corrected": sigma0_noise_corrected,
            "noise_correction_status": noise_status,
            "noise_azimuth_component_present": noise_azimuth_present,
            "grid": grid, "invalid_pixel_count": int((~valid).sum()),
            "research_admissible": False}
