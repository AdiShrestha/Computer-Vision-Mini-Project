"""Strict, support-bounded Sentinel-1 GRD radiometry primitives.

The LUT interpolation policy here is an explicit numerical contract. A product
adapter must still establish that its IPF version, flags and native XML use this
policy before observational results can be admitted.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

import numpy as np


class CalibrationError(ValueError):
    """Malformed, conflicting, unsupported or out-of-support calibration input."""


_HEX64 = re.compile(r"[a-f0-9]{64}")
_ROLES = {"measurement", "calibration", "noise", "annotation", "manifest"}
_FLAGS = {
    "input_radiometry": "RAW_DN",
    "sigma0_calibration_applied": False,
    "thermal_noise_correction_applied": False,
    "incidence_normalization_applied": False,
    "terrain_correction_applied": False,
}


def _exact_keys(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise CalibrationError(f"{label} fields differ from the strict schema")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CalibrationError(f"{label} must be a nonempty exact identifier")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CalibrationError(f"{label} must be a positive integer")
    return value


def validate_native_sar_metadata(metadata: Mapping[str, Any], *,
                                  measurement_id: str | None = None,
                                  product_id: str | None = None,
                                  artifact_manifest: Mapping[str, Any] | None = None,
                                  expected_origin: str | None = None) -> dict:
    """Validate native SAR identity, flags, role hashes and fixed dimensions.

    This only validates a source description. It deliberately does not enable
    `AdmissionContext` observational decoding.
    """
    row = _exact_keys(dict(metadata), {
        "schema_version", "product_id", "product_type", "mode", "polarization",
        "ipf_version", "subswath", "roles", "rows", "columns", "dtype",
        "line_offset", "sample_offset", "processing_flags", "noise_status",
    }, "native SAR metadata")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise CalibrationError("unsupported native SAR metadata schema version")
    native_product_id = _identifier(row["product_id"], "product_id")
    if product_id is not None and native_product_id != product_id:
        raise CalibrationError("native SAR product identity differs from measurement record")
    if measurement_id is not None:
        _identifier(measurement_id, "measurement_id")
    if row["product_type"] != "GRD" or row["mode"] != "IW":
        raise CalibrationError("only explicit Sentinel-1 IW GRD metadata is supported")
    if row["polarization"] not in {"VV", "VH"}:
        raise CalibrationError("native SAR polarization must be VV or VH")
    _identifier(row["ipf_version"], "ipf_version")
    _identifier(row["subswath"], "subswath")
    roles = _exact_keys(row["roles"], _ROLES, "native role map")
    role_ids = set()
    for role in sorted(_ROLES):
        ref = _exact_keys(roles[role], {"artifact_id", "sha256"}, role + " role reference")
        artifact_id, sha = _identifier(ref["artifact_id"], role + ".artifact_id"), ref["sha256"]
        if not isinstance(sha, str) or _HEX64.fullmatch(sha) is None:
            raise CalibrationError(role + " role SHA-256 is malformed")
        if artifact_id in role_ids:
            raise CalibrationError("native SAR roles alias the same artifact")
        role_ids.add(artifact_id)
        if artifact_manifest is not None:
            artifacts = artifact_manifest.get("artifacts", artifact_manifest)
            entry = artifacts.get(artifact_id) if isinstance(artifacts, Mapping) else None
            if (not isinstance(entry, Mapping) or entry.get("artifact_id", artifact_id) != artifact_id
                    or entry.get("sha256") != sha
                    or (expected_origin is not None and entry.get("origin") != expected_origin)):
                raise CalibrationError(role + " role digest is not bound to the artifact manifest")
    if product_id is not None and roles["measurement"]["artifact_id"] != product_id:
        raise CalibrationError("measurement role and product artifact IDs differ")
    if any(type(row[key]) is not int or row[key] <= 0 for key in ("rows", "columns")):
        raise CalibrationError("native SAR raster dimensions must be positive integers")
    if row["dtype"] not in {"uint16", "int16", "uint32", "float32", "float64"}:
        raise CalibrationError("native SAR raster dtype is unsupported")
    if any(type(row[key]) is not int or row[key] < 0 for key in ("line_offset", "sample_offset")):
        raise CalibrationError("cropped raster offsets must be nonnegative integer full-image coordinates")
    flags = _exact_keys(row["processing_flags"], set(_FLAGS), "processing flags")
    if flags != _FLAGS:
        raise CalibrationError("input must be raw DN with no prior sigma0, noise, incidence or terrain correction")
    if row["noise_status"] not in {
        "SUPPORTED_FIXTURE_POLICY", "UNAVAILABLE_MISSING", "UNAVAILABLE_UNVERIFIED_PRODUCT",
        "ALREADY_CORRECTED", "UNSUPPORTED_SCHEMA",
    }:
        raise CalibrationError("unknown noise status")
    if row["noise_status"] == "ALREADY_CORRECTED":
        raise CalibrationError("already-corrected noise status conflicts with the required raw-DN state")
    return row


@dataclass(frozen=True)
class LutSamples:
    values: np.ndarray
    valid: np.ndarray


def _vector_rows(vectors: Sequence[Mapping[str, Any]], label: str):
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence) or not vectors:
        raise CalibrationError(label + " vectors must be a nonempty sequence")
    result = {}
    for vector in vectors:
        row = _exact_keys(dict(vector), {"block_id", "line", "pixels", "values"}, label + " vector")
        block = _identifier(row["block_id"], label + ".block_id")
        line = row["line"]
        pixels, values = row["pixels"], row["values"]
        if type(line) is not int or line < 0:
            raise CalibrationError(label + " line coordinate must be a nonnegative absolute integer")
        if (not isinstance(pixels, list) or not isinstance(values, list) or len(pixels) != len(values)
                or not pixels):
            raise CalibrationError(label + " sample/value vectors must be matching nonempty lists")
        if any(type(p) is not int or p < 0 for p in pixels) or any(b <= a for a, b in zip(pixels, pixels[1:])):
            raise CalibrationError(label + " sample coordinates must be strictly increasing nonnegative integers")
        vals = np.asarray(values)
        if vals.dtype.kind not in "iuf" or not np.isfinite(vals).all():
            raise CalibrationError(label + " LUT values must be finite real numbers")
        key = block, line
        if key in result:
            raise CalibrationError(label + " has duplicate block/line vectors")
        result[key] = (np.asarray(pixels, dtype=np.float64), vals.astype(np.float64))
    return result


def _interp_samples(pixels, values, x, max_sample_gap, *, positive, label):
    if x < pixels[0] or x > pixels[-1]:
        return None
    exact = np.flatnonzero(pixels == x)
    if exact.size:
        value = float(values[int(exact[0])])
    else:
        hi = int(np.searchsorted(pixels, x))
        lo = hi - 1
        if pixels[hi] - pixels[lo] > max_sample_gap:
            return None
        weight = (x - pixels[lo]) / (pixels[hi] - pixels[lo])
        value = float(values[lo] + weight * (values[hi] - values[lo]))
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise CalibrationError(label + " LUT values violate their declared positive/nonnegative domain")
    return value


def sample_surface_lut(vectors, lines, samples, blocks, *, max_line_gap, max_sample_gap,
                       positive=True, label="LUT") -> LutSamples:
    """Sample a declared rectilinear LUT without extrapolating across support.

    Linear interpolation is along sample coordinates within each native vector,
    then along absolute image line coordinates. Interpolation never crosses a
    named sub-swath/block. The caller must bind this policy to verified product
    semantics before observational use.
    """
    if type(max_line_gap) is not int or max_line_gap <= 0 or type(max_sample_gap) is not int or max_sample_gap <= 0:
        raise CalibrationError("LUT interpolation gaps must be explicit positive integers")
    y, x, block_array = np.asarray(lines), np.asarray(samples), np.asarray(blocks)
    if y.ndim != 1 or x.ndim != 1 or block_array.shape != (len(y), len(x)):
        raise CalibrationError("absolute line/sample coordinates and block map have inconsistent shapes")
    if y.dtype.kind not in "iu" or x.dtype.kind not in "iu" or block_array.dtype.kind not in "USO":
        raise CalibrationError("absolute coordinates must be integers and blocks explicit strings")
    if (y < 0).any() or (x < 0).any() or not all(isinstance(v, str) and v for v in block_array.flat):
        raise CalibrationError("absolute coordinates or block IDs are invalid")
    row_map = _vector_rows(vectors, label)
    out = np.full(block_array.shape, np.nan, dtype=np.float64)
    valid = np.zeros(block_array.shape, dtype=bool)
    by_block = {}
    for (block, line), vector in row_map.items():
        by_block.setdefault(block, []).append((line, *vector))
    for rows in by_block.values():
        rows.sort(key=lambda item: item[0])
    for iy, line in enumerate(y.astype(np.int64)):
        for ix, sample in enumerate(x.astype(np.int64)):
            block = str(block_array[iy, ix])
            candidates = by_block.get(block, [])
            exact = [entry for entry in candidates if entry[0] == line]
            if exact:
                lo = hi = exact[0]
            else:
                lower = [entry for entry in candidates if entry[0] < line]
                upper = [entry for entry in candidates if entry[0] > line]
                if not lower or not upper:
                    continue
                lo, hi = lower[-1], upper[0]
                if hi[0] - lo[0] > max_line_gap:
                    continue
            vlo = _interp_samples(lo[1], lo[2], float(sample), max_sample_gap,
                                  positive=positive, label=label)
            if vlo is None:
                continue
            if lo[0] == hi[0]:
                value = vlo
            else:
                vhi = _interp_samples(hi[1], hi[2], float(sample), max_sample_gap,
                                      positive=positive, label=label)
                if vhi is None:
                    continue
                ratio = (int(line) - lo[0]) / (hi[0] - lo[0])
                value = vlo + ratio * (vhi - vlo)
            if math.isfinite(value):
                out[iy, ix], valid[iy, ix] = value, True
    return LutSamples(out, valid)


def sample_line_lut(vectors, lines, blocks, *, max_line_gap, nonnegative=True,
                    label="azimuth noise") -> LutSamples:
    """Interpolate line-coordinate-only LUT vectors within block support."""
    if type(max_line_gap) is not int or max_line_gap <= 0:
        raise CalibrationError("line-LUT interpolation gap must be an explicit positive integer")
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence) or not vectors:
        raise CalibrationError(label + " vectors must be a nonempty sequence")
    parsed = {}
    for vector in vectors:
        row = _exact_keys(dict(vector), {"block_id", "line", "value"}, label + " vector")
        block = _identifier(row["block_id"], label + ".block_id")
        line, value = row["line"], row["value"]
        if type(line) is not int or line < 0 or type(value) not in (int, float) or not math.isfinite(value):
            raise CalibrationError(label + " line/value is malformed")
        if value < 0 if nonnegative else value == 0:
            raise CalibrationError(label + " values violate declared noise domain")
        key = block, line
        if key in parsed:
            raise CalibrationError(label + " has duplicate block/line vectors")
        parsed[key] = float(value)
    y, block_array = np.asarray(lines), np.asarray(blocks)
    if y.ndim != 1 or block_array.shape != y.shape or y.dtype.kind not in "iu":
        raise CalibrationError("absolute line coordinates and line block IDs have inconsistent shapes")
    out = np.full(y.shape, np.nan, dtype=np.float64)
    valid = np.zeros(y.shape, dtype=bool)
    for idx, line in enumerate(y.astype(np.int64)):
        block = block_array[idx]
        if not isinstance(block, str) or not block:
            raise CalibrationError("line block ID is missing")
        candidates = sorted((line0, value) for (bid, line0), value in parsed.items() if bid == block)
        exact = [value for line0, value in candidates if line0 == line]
        if exact:
            out[idx], valid[idx] = exact[0], True
            continue
        lower = [(line0, value) for line0, value in candidates if line0 < line]
        upper = [(line0, value) for line0, value in candidates if line0 > line]
        if not lower or not upper:
            continue
        left, right = lower[-1], upper[0]
        if right[0] - left[0] > max_line_gap:
            continue
        weight = (line - left[0]) / (right[0] - left[0])
        out[idx], valid[idx] = left[1] + weight * (right[1] - left[1]), True
    return LutSamples(out, valid)


@dataclass(frozen=True)
class CalibrationResult:
    raw_sigma0: np.ndarray
    raw_valid: np.ndarray
    noise_equivalent_sigma0: np.ndarray | None
    signed_denoised_sigma0: np.ndarray | None
    denoised_valid: np.ndarray
    denoised_db: np.ndarray
    db_status: np.ndarray
    origin: str
    research_admissible: bool = False
    calibration_status: str = "CALIBRATED_RAW_SIGMA0"
    noise_status: str = "UNAVAILABLE_NOISE_LUT"


def calibrate_dn(dn, calibration_lut, *, line_offset: int, sample_offset: int,
                 block_ids, processing_flags, noise_range_lut=None, noise_azimuth_lut=None,
                 noise_support: Mapping[str, int] | None = None, origin="fixture") -> CalibrationResult:
    """Calibrate explicit DN and lookup arrays; results stay non-admissible.

    `calibration_lut` and `noise_range_lut` use `sample_surface_lut` records;
    azimuth noise uses `sample_line_lut` records. The interpolation limits are
    explicit in `noise_support`; missing noise returns raw calibration only.
    """
    if origin not in {"fixture", "simulation", "observational"}:
        raise CalibrationError("origin must be explicit")
    if type(line_offset) is not int or line_offset < 0 or type(sample_offset) is not int or sample_offset < 0:
        raise CalibrationError("absolute source window offsets are required")
    flags = _exact_keys(dict(processing_flags), set(_FLAGS), "processing flags")
    if flags != _FLAGS:
        raise CalibrationError("cannot calibrate data with prior radiometric or terrain correction")
    raw_dn = np.asarray(dn)
    if raw_dn.ndim != 2 or not raw_dn.size or raw_dn.dtype.kind not in "iuf" or not np.isfinite(raw_dn).all():
        raise CalibrationError("measurement must be a nonempty finite real raster")
    block_array = np.asarray(block_ids)
    if block_array.shape != raw_dn.shape or block_array.dtype.kind not in "USO" or not all(
        isinstance(value, str) and value for value in block_array.flat
    ):
        raise CalibrationError("each measurement pixel requires an explicit block ID")
    lines = np.arange(line_offset, line_offset + raw_dn.shape[0], dtype=np.int64)
    samples = np.arange(sample_offset, sample_offset + raw_dn.shape[1], dtype=np.int64)
    a = sample_surface_lut(calibration_lut["vectors"], lines, samples, block_array,
        max_line_gap=calibration_lut["max_line_gap"], max_sample_gap=calibration_lut["max_sample_gap"],
        positive=True, label="A_sigma calibration")
    dn64 = raw_dn.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.square(dn64)
        denominator = np.square(a.values)
        raw = squared / denominator
    raw_valid = a.valid & np.isfinite(raw)
    if not np.isfinite(squared).all() or (a.valid & ~np.isfinite(denominator)).any():
        raise CalibrationError("float64 DN square or calibration denominator overflowed")
    raw = np.where(raw_valid, raw, np.nan)
    db = np.full(raw_dn.shape, np.nan, dtype=np.float64)
    db_status = np.full(raw_dn.shape, "UNAVAILABLE_NO_CALIBRATION_SUPPORT", dtype=object)
    db_status[raw_valid] = "UNAVAILABLE_NO_NOISE_SUPPORT"
    if noise_range_lut is None or noise_azimuth_lut is None or noise_support is None:
        if any(value is not None for value in (noise_range_lut, noise_azimuth_lut, noise_support)):
            raise CalibrationError("range noise, azimuth noise, and interpolation support must be supplied together")
        db_status[raw_valid] = "UNAVAILABLE_NOISE_LUT"
        return CalibrationResult(raw, raw_valid, None, None, np.zeros(raw.shape, bool), db,
                                 db_status, origin, False,
                                 "CALIBRATED_RAW_SIGMA0" if raw_valid.any() else "NOT_ESTIMABLE_NO_CALIBRATION_SUPPORT",
                                 "UNAVAILABLE_NOISE_LUT")
    if set(noise_support) != {"max_line_gap", "max_sample_gap", "max_azimuth_line_gap"}:
        raise CalibrationError("noise support interpolation limits must be explicit and exact")
    nr = sample_surface_lut(noise_range_lut["vectors"], lines, samples, block_array,
        max_line_gap=noise_support["max_line_gap"], max_sample_gap=noise_support["max_sample_gap"],
        positive=False, label="range noise")
    nb = np.asarray([str(block_array[i, 0]) for i in range(block_array.shape[0])], dtype=object)
    if any(not np.all(block_array[i, :] == block_array[i, 0]) for i in range(block_array.shape[0])):
        raise CalibrationError("one azimuth noise block per image line is required")
    na = sample_line_lut(noise_azimuth_lut["vectors"], lines, nb,
        max_line_gap=noise_support["max_azimuth_line_gap"], nonnegative=True)
    noise_valid = raw_valid & nr.valid & na.valid
    with np.errstate(over="ignore", invalid="ignore"):
        noise_power = nr.values * na.values[:, None]
        nesz = noise_power / denominator
        signed = (squared - noise_power) / denominator
    if (noise_valid & (~np.isfinite(noise_power) | ~np.isfinite(nesz) | ~np.isfinite(signed))).any():
        raise CalibrationError("noise or calibrated power overflowed")
    noise_power = np.where(noise_valid, noise_power, np.nan)
    nesz = np.where(noise_valid, nesz, np.nan)
    signed = np.where(noise_valid, signed, np.nan)
    positive_power = noise_valid & (signed > 0)
    db[positive_power] = 10.0 * np.log10(signed[positive_power])
    db_status[noise_valid] = "UNAVAILABLE_NONPOSITIVE_POWER"
    db_status[positive_power] = "ESTIMATED"
    if raw_valid.any() and np.array_equal(noise_valid, raw_valid):
        noise_status = "CONSTRUCTED_LUT_SUPPORTED"
    elif noise_valid.any():
        noise_status = "CONSTRUCTED_LUT_PARTIAL_SUPPORT"
    else:
        noise_status = "CONSTRUCTED_LUT_NO_SUPPORT"
    return CalibrationResult(raw, raw_valid, nesz, signed, noise_valid, db, db_status, origin, False,
        "CALIBRATED_RAW_SIGMA0" if raw_valid.any() else "NOT_ESTIMABLE_NO_CALIBRATION_SUPPORT",
        noise_status)


def calibrate_native_sar_fixture(metadata, dn, calibration_lut, *, block_ids,
                                 artifact_manifest, measurement_id=None,
                                 noise_range_lut=None, noise_azimuth_lut=None,
                                 noise_support=None, expected_origin="fixture"):
    """Consume the strict v2 metadata for constructed numerical fixtures only.

    Native XML decoding and source-product interpretation remain outside this
    adapter. The input LUT arrays must be explicitly constructed fixture values;
    this function never makes an observational result admissible.
    """
    if expected_origin not in {"fixture", "simulation"}:
        raise CalibrationError("native product calibration is blocked outside explicit fixture/simulation origin")
    row = validate_native_sar_metadata(metadata, measurement_id=measurement_id,
        product_id=metadata.get("product_id") if isinstance(metadata, Mapping) else None,
        artifact_manifest=artifact_manifest, expected_origin=expected_origin)
    values = np.asarray(dn)
    if values.shape != (row["rows"], row["columns"]) or values.dtype.name != row["dtype"]:
        raise CalibrationError("constructed DN array dimensions/dtype differ from bound native metadata")
    declared_noise = row["noise_status"]
    if declared_noise == "SUPPORTED_FIXTURE_POLICY":
        if any(value is None for value in (noise_range_lut, noise_azimuth_lut, noise_support)):
            raise CalibrationError("supported fixture noise status requires both LUTs and declared interpolation support")
        result = calibrate_dn(values, calibration_lut, line_offset=row["line_offset"],
            sample_offset=row["sample_offset"], block_ids=block_ids,
            processing_flags=row["processing_flags"], noise_range_lut=noise_range_lut,
            noise_azimuth_lut=noise_azimuth_lut, noise_support=noise_support, origin=expected_origin)
        return replace(result, noise_status="SUPPORTED_FIXTURE_POLICY")
    if any(value is not None for value in (noise_range_lut, noise_azimuth_lut, noise_support)):
        raise CalibrationError("LUT inputs cannot override a declared missing or unsupported noise status")
    result = calibrate_dn(values, calibration_lut, line_offset=row["line_offset"],
        sample_offset=row["sample_offset"], block_ids=block_ids,
        processing_flags=row["processing_flags"], origin=expected_origin)
    status = {
        "UNAVAILABLE_MISSING": "UNAVAILABLE_NOISE_MISSING",
        "UNAVAILABLE_UNVERIFIED_PRODUCT": "UNAVAILABLE_UNVERIFIED_PRODUCT_NOISE_POLICY",
        "UNSUPPORTED_SCHEMA": "UNSUPPORTED_NOISE_SCHEMA",
    }[declared_noise]
    db_status = result.db_status.copy()
    db_status[result.raw_valid] = status
    return replace(result, db_status=db_status, noise_status=declared_noise)


def _xml_local(element):
    return element.tag.rsplit("}", 1)[-1]


def _xml_child_text(parent, name, *, required=True):
    values = [node.text.strip() for node in list(parent)
              if _xml_local(node) == name and node.text and node.text.strip()]
    if not values and not required:
        return None
    if len(values) != 1:
        raise CalibrationError(f"native XML field {name} is absent or ambiguous")
    return values[0]


def _native_vector(xml_bytes, vector_tag, coordinate_tag, value_tag, *, positive=True, common_grid=True):
    """Decode a product-declared one-dimensional calibration surface."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise CalibrationError(f"malformed native {vector_tag} XML: {exc}") from exc
    vectors = []
    for node in root.iter():
        if _xml_local(node) != vector_tag:
            continue
        try:
            line_text = _xml_child_text(node, "line")
            coord_node = next(x for x in list(node) if _xml_local(x) == coordinate_tag)
            values_node = next(x for x in list(node) if _xml_local(x) == value_tag)
            coordinates = [int(value) for value in coord_node.text.split()]
            values = [float(value) for value in values_node.text.split()]
            coord_count, value_count = int(coord_node.attrib["count"]), int(values_node.attrib["count"])
            line = int(line_text)
        except (StopIteration, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed fields in native {vector_tag}") from exc
        if (coord_count != len(coordinates) or value_count != len(values) or
                len(coordinates) != len(values) or not coordinates or
                any(b <= a for a, b in zip(coordinates, coordinates[1:])) or
                any(not math.isfinite(v) or (v <= 0 if positive else v < 0) for v in values) or line < 0):
            raise CalibrationError(f"native {vector_tag} count, coordinate, or value validation failed")
        vectors.append((line, np.asarray(coordinates, dtype=np.int64),
                        np.asarray(values, dtype=np.float64)))
    if not vectors:
        raise CalibrationError(f"native XML has no {vector_tag} records")
    vectors.sort(key=lambda item: item[0])
    if len({row[0] for row in vectors}) != len(vectors):
        raise CalibrationError(f"native {vector_tag} has duplicate line records")
    reference_pixels = vectors[0][1]
    if common_grid and any(not np.array_equal(v[1], reference_pixels) for v in vectors[1:]):
        raise CalibrationError(f"native {vector_tag} pixel grids differ between lines")
    return vectors


def parse_native_sentinel1_product(annotation_xml, calibration_xml, noise_xml):
    """Parse source-bound Sentinel-1 GRD metadata without asserting admission.

    Calibration uses only the product's declared sigmaNought surface. The
    product's noise arrays and correction flag are retained for explicit
    status reporting; this adapter does not claim a verified noise-product
    interpolation policy.
    """
    try:
        annotation = ET.fromstring(annotation_xml)
        calibration_root = ET.fromstring(calibration_xml)
        noise_root = ET.fromstring(noise_xml)
    except ET.ParseError as exc:
        raise CalibrationError(f"malformed Sentinel-1 native XML: {exc}") from exc
    def identity(root):
        result = {}
        for key in ("missionId", "productType", "mode", "polarisation"):
            values = [node.text.strip() for node in root.iter()
                      if _xml_local(node) == key and node.text and node.text.strip()]
            if len(values) != 1:
                raise CalibrationError(f"native XML role has absent or conflicting {key}")
            result[key] = values[0]
        return result
    annotation_identity = identity(annotation)
    if identity(calibration_root) != annotation_identity or identity(noise_root) != annotation_identity:
        raise CalibrationError("annotation, calibration and noise XML identities differ")
    fields = {}
    for name in ("missionId", "productType", "mode", "polarisation", "numberOfLines", "numberOfSamples",
                 "startTime", "stopTime"):
        matches = [node.text.strip() for node in annotation.iter()
                   if _xml_local(node) == name and node.text and node.text.strip()]
        if len(matches) != 1:
            raise CalibrationError(f"annotation {name} is absent or conflicting")
        fields[name] = matches[0]
    if (fields["missionId"] not in {"S1A", "S1B", "S1C"} or fields["productType"] != "GRD"
            or fields["mode"] != "IW" or fields["polarisation"] not in {"VV", "VH"}):
        raise CalibrationError("unsupported Sentinel-1 product identity")
    processing = [node for node in annotation.iter() if _xml_local(node) == "processingInformation"]
    if len(processing) != 1:
        raise CalibrationError("annotation processingInformation is absent or ambiguous")
    proc = {}
    for node in processing[0]:
        name = _xml_local(node)
        if name.endswith("Applied") or name.endswith("Performed") or name.endswith("Used"):
            proc[name] = (node.text or "").strip()
    thermal = [node.text.strip().lower() for node in annotation.iter()
               if _xml_local(node) == "thermalNoiseCorrectionPerformed" and node.text]
    if len(thermal) != 1 or thermal[0] not in {"true", "false"}:
        raise CalibrationError("thermalNoiseCorrectionPerformed is absent or invalid")
    calibration = _native_vector(calibration_xml, "calibrationVector", "pixel", "sigmaNought")
    noise_range = _native_vector(noise_xml, "noiseRangeVector", "pixel", "noiseRangeLut",
                                 positive=False, common_grid=False)
    noise_azimuth = []
    for node in noise_root.iter():
        if _xml_local(node) != "noiseAzimuthVector":
            continue
        try:
            swath = _xml_child_text(node, "swath")
            first_line, last_line = int(_xml_child_text(node, "firstAzimuthLine")), int(_xml_child_text(node, "lastAzimuthLine"))
            first_sample, last_sample = int(_xml_child_text(node, "firstRangeSample")), int(_xml_child_text(node, "lastRangeSample"))
            line_node = next(x for x in list(node) if _xml_local(x) == "line")
            value_node = next(x for x in list(node) if _xml_local(x) == "noiseAzimuthLut")
            lines = [int(v) for v in line_node.text.split()]
            values = [float(v) for v in value_node.text.split()]
            if int(line_node.attrib["count"]) != len(lines) or int(value_node.attrib["count"]) != len(values):
                raise ValueError("count mismatch")
        except (StopIteration, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("malformed native noiseAzimuthVector") from exc
        if (first_line < 0 or last_line < first_line or first_sample < 0 or last_sample < first_sample
                or len(lines) != len(values) or not lines or any(b <= a for a,b in zip(lines,lines[1:]))
                or any(not math.isfinite(v) or v < 0 for v in values)):
            raise CalibrationError("noise azimuth coordinates or values violate native support")
        noise_azimuth.append({"swath":swath,"first_line":first_line,"last_line":last_line,
                              "first_sample":first_sample,"last_sample":last_sample,
                              "lines":lines,"values":values})
    geolocation_points = sum(1 for node in annotation.iter() if _xml_local(node) == "geolocationGridPoint")
    geolocation = []
    for node in annotation.iter():
        if _xml_local(node) != "geolocationGridPoint":
            continue
        try:
            point = {key: float(_xml_child_text(node, key)) for key in ("line", "pixel", "latitude", "longitude", "height")}
        except ValueError as exc:
            raise CalibrationError("malformed geolocationGridPoint coordinates") from exc
        if not all(math.isfinite(value) for value in point.values()):
            raise CalibrationError("geolocationGridPoint contains non-finite coordinates")
        if not -90 <= point["latitude"] <= 90 or not -180 <= point["longitude"] <= 180:
            raise CalibrationError("geolocationGridPoint latitude/longitude is outside valid bounds")
        geolocation.append(point)
    return {
        "mission": fields["missionId"], "product_type": fields["productType"], "mode": fields["mode"],
        "polarization": fields["polarisation"], "rows": int(fields["numberOfLines"]),
        "columns": int(fields["numberOfSamples"]), "thermal_noise_correction_applied": thermal[0] == "true",
        "start_time": fields["startTime"], "stop_time": fields["stopTime"],
        "processing_flags": proc, "geolocation_grid_point_count": geolocation_points,
        "geolocation_grid_points": geolocation,
        "calibration_vectors": calibration, "noise_range_vectors": noise_range,
        "noise_azimuth_blocks": noise_azimuth,
    }


def calibrate_native_sigma0_window(dn, calibration_vectors, *, line_offset, sample_offset):
    """Apply source sigmaNought to a bounded native GRD window, without extrapolation."""
    raw = np.asarray(dn)
    if raw.ndim != 2 or raw.dtype.kind not in "iu" or not raw.size:
        raise CalibrationError("native measurement window must be a nonempty integer DN array")
    if type(line_offset) is not int or line_offset < 0 or type(sample_offset) is not int or sample_offset < 0:
        raise CalibrationError("native window requires absolute nonnegative line/sample offsets")
    rows, cols = raw.shape
    sample_coords = np.arange(sample_offset, sample_offset + cols, dtype=np.float64)
    vector_lines = np.asarray([v[0] for v in calibration_vectors], dtype=np.int64)
    lut_pixels = calibration_vectors[0][1]
    if any(not np.array_equal(v[1], lut_pixels) for v in calibration_vectors):
        raise CalibrationError("calibration pixel coordinates are inconsistent")
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    valid = np.zeros(raw.shape, dtype=bool)
    dn64 = raw.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.square(dn64)
    if not np.isfinite(squared).all():
        raise CalibrationError("float64 DN square overflowed")
    for local_row in range(rows):
        line = line_offset + local_row
        hi = int(np.searchsorted(vector_lines, line, side="left"))
        if hi < len(vector_lines) and vector_lines[hi] == line:
            lo = hi
        elif hi == 0 or hi == len(vector_lines):
            continue
        else:
            lo = hi - 1
        if lo == hi:
            weights = np.interp(sample_coords, lut_pixels, calibration_vectors[lo][2])
            sample_ok = (sample_coords >= lut_pixels[0]) & (sample_coords <= lut_pixels[-1])
        else:
            left, right = calibration_vectors[lo], calibration_vectors[hi]
            if right[0] <= left[0]:
                raise CalibrationError("calibration line coordinates are not strictly increasing")
            ratio = (line - left[0]) / (right[0] - left[0])
            left_values = np.interp(sample_coords, lut_pixels, left[2])
            right_values = np.interp(sample_coords, lut_pixels, right[2])
            weights = left_values + ratio * (right_values - left_values)
            sample_ok = (sample_coords >= lut_pixels[0]) & (sample_coords <= lut_pixels[-1])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sigma0 = squared[local_row] / np.square(weights)
        valid[local_row] = sample_ok & np.isfinite(sigma0) & (sigma0 >= 0)
        out[local_row, valid[local_row]] = sigma0[valid[local_row]].astype(np.float32)
    return out, valid


def sample_native_surface_window(vectors, *, line_offset, sample_offset, shape, positive):
    """Interpolate native XML vectors along their declared sample then line axes."""
    rows, cols = shape
    line_coords = np.arange(line_offset, line_offset + rows, dtype=np.int64)
    sample_coords = np.arange(sample_offset, sample_offset + cols, dtype=np.float64)
    vector_lines = np.asarray([v[0] for v in vectors], dtype=np.int64)
    result = np.full((rows, cols), np.nan, dtype=np.float64)
    valid = np.zeros((rows, cols), dtype=bool)
    if not vectors or np.any(np.diff(vector_lines) <= 0):
        raise CalibrationError("native LUT lines must be nonempty and strictly increasing")
    for r, line in enumerate(line_coords):
        hi = int(np.searchsorted(vector_lines, line, side="left"))
        if hi < len(vector_lines) and vector_lines[hi] == line:
            lo = hi
        elif hi == 0 or hi == len(vector_lines):
            continue
        else:
            lo = hi - 1
        def sample(entry):
            pixels, vals = entry[1], entry[2]
            values = np.interp(sample_coords, pixels, vals)
            ok = (sample_coords >= pixels[0]) & (sample_coords <= pixels[-1])
            return values, ok
        left, left_ok = sample(vectors[lo])
        if lo == hi:
            values, supported = left, left_ok
        else:
            right, right_ok = sample(vectors[hi])
            weight = (line - vector_lines[lo]) / (vector_lines[hi] - vector_lines[lo])
            values = left + weight * (right - left)
            supported = left_ok & right_ok
        supported &= np.isfinite(values)
        supported &= values > 0 if positive else values >= 0
        result[r, supported] = values[supported]
        valid[r] = supported
    return result, valid


def sample_native_azimuth_noise_window(blocks, *, line_offset, sample_offset, shape):
    """Interpolate per-swath azimuth noise only inside each declared block."""
    rows, cols = shape
    lines = np.arange(line_offset, line_offset + rows, dtype=np.int64)
    samples = np.arange(sample_offset, sample_offset + cols, dtype=np.int64)
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    owners = np.zeros((rows, cols), dtype=np.uint8)
    for block in blocks:
        block_samples = (samples >= block["first_sample"]) & (samples <= block["last_sample"])
        if not block_samples.any():
            continue
        block_lines = np.asarray(block["lines"], dtype=np.int64)
        block_values = np.asarray(block["values"], dtype=np.float64)
        for r, line in enumerate(lines):
            if line < block["first_line"] or line > block["last_line"]:
                continue
            hi = int(np.searchsorted(block_lines, line, side="left"))
            if hi < len(block_lines) and block_lines[hi] == line:
                value = block_values[hi]
            elif hi == 0 or hi == len(block_lines):
                continue
            else:
                lo = hi - 1
                value = block_values[lo] + ((line - block_lines[lo]) /
                    (block_lines[hi] - block_lines[lo])) * (block_values[hi] - block_values[lo])
            owners[r, block_samples] += 1
            out[r, block_samples] = value
    if (owners > 1).any():
        raise CalibrationError("native noise azimuth blocks overlap in line/sample support")
    return out, owners == 1


def calibrate_native_noise_window(dn, calibration_vectors, noise_range_vectors,
                                  noise_azimuth_blocks, *, line_offset, sample_offset,
                                  thermal_noise_correction_applied):
    """Return raw sigma0, signed noise-subtracted sigma0, and NESZ on native support."""
    if thermal_noise_correction_applied is not False:
        raise CalibrationError("thermal noise already corrected or correction flag not explicitly false")
    raw_dn = np.asarray(dn)
    if raw_dn.ndim != 2 or raw_dn.dtype.kind not in "iu" or not raw_dn.size:
        raise CalibrationError("native measurement window must be a nonempty integer DN array")
    shape = raw_dn.shape
    a, a_valid = sample_native_surface_window(calibration_vectors, line_offset=line_offset,
        sample_offset=sample_offset, shape=shape, positive=True)
    nr, nr_valid = sample_native_surface_window(noise_range_vectors, line_offset=line_offset,
        sample_offset=sample_offset, shape=shape, positive=False)
    na, na_valid = sample_native_azimuth_noise_window(noise_azimuth_blocks,
        line_offset=line_offset, sample_offset=sample_offset, shape=shape)
    dn64 = raw_dn.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.square(dn64)
        denominator = np.square(a)
        noise = nr * na
        raw = power / denominator
        nesz = noise / denominator
        signed = (power - noise) / denominator
    raw_valid = a_valid & np.isfinite(raw)
    noise_valid = raw_valid & nr_valid & na_valid & np.isfinite(noise) & np.isfinite(nesz) & np.isfinite(signed)
    raw_out = np.full(shape, np.nan, dtype=np.float32)
    nesz_out = np.full(shape, np.nan, dtype=np.float32)
    signed_out = np.full(shape, np.nan, dtype=np.float32)
    raw_out[raw_valid] = raw[raw_valid].astype(np.float32)
    nesz_out[noise_valid] = nesz[noise_valid].astype(np.float32)
    signed_out[noise_valid] = signed[noise_valid].astype(np.float32)
    return raw_out, raw_valid, nesz_out, signed_out, noise_valid


def strict_power_to_db(power, support):
    """Return dB only for finite, strictly positive linear power."""
    values, valid = np.asarray(power, dtype=np.float64), np.asarray(support, dtype=bool)
    if values.shape != valid.shape:
        raise CalibrationError("power and support arrays have different shapes")
    db = np.full(values.shape, np.nan, dtype=np.float32)
    positive = valid & np.isfinite(values) & (values > 0)
    db[positive] = (10.0 * np.log10(values[positive])).astype(np.float32)
    return db, positive


def bilinear_resample_with_valid_weights(source, source_valid, source_rows, source_samples):
    """Resample linear values without allowing missing neighbors to poison sums.

    Coverage is the sum of bilinear weights from finite, explicitly valid source
    cells. The returned value is the weighted mean over those contributors.
    Coordinates outside the source pixel-center extent remain missing; no
    extrapolation or clamping is performed.
    """
    values = np.asarray(source, dtype=np.float64)
    valid_source = np.asarray(source_valid, dtype=bool)
    rows, samples = np.asarray(source_rows, dtype=np.float64), np.asarray(source_samples, dtype=np.float64)
    if values.ndim != 2 or valid_source.shape != values.shape or values.size == 0:
        raise CalibrationError("source values and validity must be matching nonempty 2D arrays")
    if rows.shape != samples.shape:
        raise CalibrationError("resampling row and sample coordinate shapes differ")
    finite_coordinates = np.isfinite(rows) & np.isfinite(samples)
    inside = (finite_coordinates & (rows >= 0) & (rows <= values.shape[0] - 1)
              & (samples >= 0) & (samples <= values.shape[1] - 1))
    safe_rows = np.where(inside, rows, 0.0)
    safe_samples = np.where(inside, samples, 0.0)
    base_rows, base_samples = np.floor(safe_rows).astype(np.int64), np.floor(safe_samples).astype(np.int64)
    row_fraction, sample_fraction = safe_rows - base_rows, safe_samples - base_samples
    weighted = np.zeros(rows.shape, dtype=np.float64)
    coverage = np.zeros(rows.shape, dtype=np.float64)
    for row_step, sample_step, weight in (
        (0, 0, (1 - row_fraction) * (1 - sample_fraction)),
        (0, 1, (1 - row_fraction) * sample_fraction),
        (1, 0, row_fraction * (1 - sample_fraction)),
        (1, 1, row_fraction * sample_fraction),
    ):
        source_row, source_sample = base_rows + row_step, base_samples + sample_step
        within = (inside & (source_row >= 0) & (source_row < values.shape[0])
                  & (source_sample >= 0) & (source_sample < values.shape[1]))
        bounded_row = np.clip(source_row, 0, values.shape[0] - 1)
        bounded_sample = np.clip(source_sample, 0, values.shape[1] - 1)
        neighbor = values[bounded_row, bounded_sample]
        eligible = within & valid_source[bounded_row, bounded_sample] & np.isfinite(neighbor) & (weight > 0)
        effective_weight = np.where(eligible, weight, 0.0)
        safe_neighbor = np.where(eligible, neighbor, 0.0)
        weighted += effective_weight * safe_neighbor
        coverage += effective_weight
    # The diagnostic products are stored as float32. Apply that representable
    # range here so the support bit cannot claim a value that becomes Inf when
    # the caller writes the product.
    output = np.full(rows.shape, np.nan, dtype=np.float32)
    possible = inside & (coverage > 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = weighted[possible] / coverage[possible]
    finite_output = (np.isfinite(normalized) &
                     (np.abs(normalized) <= np.finfo(np.float32).max))
    positions = np.flatnonzero(possible)
    flat_output = output.reshape(-1)
    flat_output[positions[finite_output]] = normalized[finite_output].astype(np.float32)
    output_valid = np.isfinite(output) & inside & (coverage > 0)
    return output, coverage, output_valid

