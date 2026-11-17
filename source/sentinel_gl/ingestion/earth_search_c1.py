"""Bounded anonymous Earth Search C1 COG staging and source-joined inspection.

These are derived COG assets, not original Sentinel-2 SAFE/JP2 archives. The
adapter retains provider origin and never turns COG metadata into lake labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import html
import json
import math
import os
import platform
from pathlib import Path
import re
import struct
import zlib
import math
import urllib.error
import urllib.parse
import urllib.request

from defusedxml import ElementTree as SafeET
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as transform_coordinates

from ..admission import AdmissionError

BUCKET_HOST = "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"
ASSET_ROLES = {"green": "B03", "nir": "B08", "swir16": "B11", "scl": "SCL"}
EXPECTED_GSD = {"green": 10, "nir": 10, "swir16": 20, "scl": 20}
FLOAT_NODATA = -9999.0
CENTROIDS = {
    "PILOT-SOUTH-LHONAK": (27.915, 88.204),  # latitude, longitude (PI manifest)
    "PILOT-KHANGCHUNG-TSHO": (27.985, 88.752),  # latitude, longitude (PI manifest)
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _runtime_identity():
    import sys
    return {"implementation": platform.python_implementation(),
            "python_version": platform.python_version(), "executable": sys.executable,
            "adapter": "sentinel_gl.ingestion.earth_search_c1"}


def _safe_literal_error(exc):
    message = str(exc)
    message = re.sub(r"(https?://)[^/@\s:]+:[^/@\s]+@", r"\1[REDACTED]@", message)
    message = re.sub(r"(https?://[^\s?#]+)\?[^\s#]*", r"\1?[REDACTED]", message)
    message = re.sub(r"(?i)(authorization|cookie|token|password|credential)(\s*[:=]\s*)[^\s,;]+",
                     r"\1\2[REDACTED]", message)
    return message


def _write_receipt(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AdmissionError("Earth Search timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _inside_declared_window(value, acquisition_start, acquisition_end_exclusive, cutoff, label):
    instant = _utc(value)
    start, end, exclusive_cutoff = (_utc(acquisition_start), _utc(acquisition_end_exclusive),
                                    _utc(cutoff))
    if not start < end or not start <= instant < end or instant >= exclusive_cutoff:
        raise AdmissionError(f"{label} is outside the pinned acquisition window or pre-event cutoff")
    return instant


def _expected_tile_code(properties):
    try:
        zone = int(properties["mgrs:utm_zone"])
        latitude_band = properties["mgrs:latitude_band"]
        grid_square = properties["mgrs:grid_square"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AdmissionError("C1 item lacks complete MGRS tile identity") from exc
    if (not 1 <= zone <= 60 or not isinstance(latitude_band, str) or
            not re.fullmatch(r"[C-HJ-NP-X]", latitude_band) or
            not isinstance(grid_square, str) or not re.fullmatch(r"[A-HJ-NP-Z]{2}", grid_square)):
        raise AdmissionError("C1 item MGRS tile identity is malformed")
    return f"T{zone:02d}{latitude_band}{grid_square}"


def _grid_extent(shape, transform):
    height, width = shape
    a, b, c, d, e, f = transform
    corners = ((c, f), (c + a * width, f + d * width),
               (c + b * height, f + e * height),
               (c + a * width + b * height, f + d * width + e * height))
    return (min(x for x, _ in corners), min(y for _, y in corners),
            max(x for x, _ in corners), max(y for _, y in corners))


def _validate_declared_grids(assets, properties):
    epsg = properties.get("proj:epsg")
    if type(epsg) is not int or epsg <= 0:
        raise AdmissionError("C1 item lacks a positive projected CRS authority code")
    grids = {}
    for role, band in ASSET_ROLES.items():
        asset = assets[role]
        shape, transform = asset.get("proj:shape"), asset.get("proj:transform")
        gsd = asset.get("gsd")
        raster_bands = asset.get("raster:bands", [])
        if (not isinstance(shape, list) or len(shape) != 2 or
                any(type(dimension) is not int or dimension <= 0 for dimension in shape) or
                not isinstance(transform, list) or len(transform) < 6 or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in transform[:6]) or
                type(gsd) not in (int, float) or not math.isfinite(gsd) or gsd <= 0 or
                len(raster_bands) != 1):
            raise AdmissionError(f"C1 {role} lacks complete native grid/type metadata")
        raster = raster_bands[0]
        if (int(gsd) != EXPECTED_GSD[role] or float(raster.get("spatial_resolution", -1)) != float(gsd) or
                not raster.get("data_type") or raster.get("nodata") is None):
            raise AdmissionError(f"C1 {role} resolution/type/nodata metadata is incomplete or unexpected")
        if role != "scl":
            for key in ("scale", "offset"):
                value = raster.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise AdmissionError(f"C1 {role} {key} encoding metadata is missing")
        resolution = float(gsd)
        a, b, _, d, e, _ = map(float, transform[:6])
        if (abs(abs(a) - resolution) > 1e-8 or abs(b) > 1e-8 or
                abs(d) > 1e-8 or abs(abs(e) - resolution) > 1e-8):
            raise AdmissionError(f"C1 {role} affine pixel size disagrees with its native resolution")
        grids[role] = {"epsg": epsg, "shape": shape, "transform": list(map(float, transform[:6])),
                       "gsd": resolution, "extent": _grid_extent(shape, transform[:6])}
    reference = grids["green"]
    for role, grid in grids.items():
        if grid["epsg"] != reference["epsg"]:
            raise AdmissionError(f"C1 {role} CRS differs from the other native-resolution assets")
        if any(abs(a - b) > 1e-6 for a, b in zip(grid["extent"], reference["extent"])):
            raise AdmissionError(f"C1 {role} grid footprint differs from the declared item footprint")
        ratio = max(reference["gsd"], grid["gsd"]) / min(reference["gsd"], grid["gsd"])
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
            raise AdmissionError(f"C1 {role} grid is not an integer native-resolution relation")
    for left, right in (("green", "nir"), ("swir16", "scl")):
        a, b = grids[left], grids[right]
        if a["shape"] != b["shape"] or a["transform"] != b["transform"]:
            raise AdmissionError(f"C1 {left}/{right} same-resolution native grids differ")
    return grids


def _local(element):
    return element.tag.rsplit("}", 1)[-1]


def _single_text(root, name):
    values = [node.text.strip() for node in root.iter()
              if _local(node) == name and node.text and node.text.strip()]
    if not values or len(set(values)) != 1:
        raise AdmissionError(f"{name} is absent or conflicting in native product metadata")
    return values[0]


def _safe_asset_url(href):
    parts = urllib.parse.urlsplit(href)
    if (parts.scheme != "https" or parts.hostname != BUCKET_HOST or parts.port not in (None, 443)
            or parts.username or parts.password or parts.query or parts.fragment):
        raise AdmissionError("Earth Search COG href is outside the approved anonymous host")
    if not parts.path.startswith("/sentinel-2-c1-l2a/"):
        raise AdmissionError("Earth Search COG href is outside the declared collection")
    return parts


def validate_item(item, site_id, cutoff, *, acquisition_start, acquisition_end_exclusive,
                  selection_record=None):
    if site_id not in CENTROIDS or not isinstance(item, dict):
        raise AdmissionError("declared development site and C1 item are required")
    properties = item.get("properties", {})
    item_id = item.get("id", "")
    if properties.get("s2:processing_baseline") != "05.11":
        raise AdmissionError("selected C1 item identity or processing baseline differs")
    item_time = _inside_declared_window(properties.get("datetime", ""), acquisition_start,
                                        acquisition_end_exclusive, cutoff, "Earth Search item datetime")
    tile_code = _expected_tile_code(properties)
    if not re.fullmatch(rf"S2[A-Z]_{tile_code}_\d{{8}}T\d{{6}}_L2A", item_id):
        raise AdmissionError("C1 item ID does not bind to its MGRS tile identity")
    product_uri = properties.get("s2:product_uri")
    generation_time = properties.get("s2:generation_time")
    tile_id = properties.get("s2:tile_id")
    if not isinstance(product_uri, str) or not isinstance(generation_time, str) or not isinstance(tile_id, str):
        raise AdmissionError("C1 product URI and generation time are required")
    product_match = re.fullmatch(
        rf"S2[A-Z]_MSIL2A_(\d{{8}}T\d{{6}})_N(\d{{4}})_R\d{{3}}_({tile_code})_(\d{{8}}T\d{{6}})\.SAFE",
        product_uri)
    if (not product_match or product_match.group(2) != "0511" or
            not tile_id.endswith(f"_{tile_code}_N05.11")):
        raise AdmissionError("C1 product URI, processing baseline, and tile ID disagree")
    product_sensing_time = datetime.strptime(product_match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc)
    _inside_declared_window(product_sensing_time.isoformat(), acquisition_start,
                            acquisition_end_exclusive, cutoff, "native product URI sensing timestamp")
    assets = item.get("assets", {})
    if set(ASSET_ROLES) - set(assets):
        raise AdmissionError("selected C1 item lacks required native-resolution assets")
    for role, band_name in ASSET_ROLES.items():
        asset = assets[role]
        parts = _safe_asset_url(asset.get("href", ""))
        expected_leaf = "SCL.tif" if role == "scl" else f"{band_name}.tif"
        path_parts = parts.path.strip("/").split("/")
        expected_date = item_time.strftime("%Y")
        expected_month = str(item_time.month)
        expected_directory = ["sentinel-2-c1-l2a", str(properties["mgrs:utm_zone"]),
                              properties["mgrs:latitude_band"], properties["mgrs:grid_square"],
                              expected_date, expected_month, item_id, expected_leaf]
        if path_parts != expected_directory:
            raise AdmissionError(f"C1 {role} asset path does not match the selected item/tile directory")
        if role != "scl":
            eo_bands = asset.get("eo:bands", [])
            raster_bands = asset.get("raster:bands", [])
            if (len(eo_bands) != 1 or eo_bands[0].get("name") != band_name or
                    len(raster_bands) != 1):
                raise AdmissionError(f"C1 {role} spectral identity metadata is incomplete")
        elif Path(parts.path).name != expected_leaf:
            raise AdmissionError(f"C1 {role} role does not match its provider asset path")
        if (type(asset.get("file:size")) is not int or asset["file:size"] <= 0 or
                not re.fullmatch(r"1220[0-9a-f]{64}", str(asset.get("file:checksum", "")))):
            raise AdmissionError(f"C1 {role} size/SHA-256 multihash metadata is malformed")
    _validate_declared_grids(assets, properties)
    if selection_record is not None:
        if (selection_record.get("item_id") != item_id or
                selection_record.get("product_uri") != product_uri or
                selection_record.get("baseline") != properties["s2:processing_baseline"] or
                _utc(selection_record.get("acquisition_datetime", "")) != item_time or
                selection_record.get("generation_time") != generation_time):
            raise AdmissionError("selected C1 record does not match retained STAC product identity/time")
        for role, asset in ((role, assets[role]) for role in ASSET_ROLES):
            selected = selection_record.get("assets", {}).get(role, {})
            if (selected.get("href") != asset.get("href") or
                    selected.get("file_checksum_multihash") != asset.get("file:checksum") or
                    selected.get("file_size") != asset.get("file:size") or
                    selected.get("band") != ASSET_ROLES[role]):
                raise AdmissionError(f"selected C1 {role} asset differs from retained STAC identity")
    return item


def validate_native_metadata(item, product_xml, granule_xml, *, acquisition_start,
                             acquisition_end_exclusive, cutoff):
    try:
        product_root = SafeET.parse(product_xml).getroot()
        granule_root = SafeET.parse(granule_xml).getroot()
    except Exception as exc:
        raise AdmissionError("retained native Sentinel-2 XML is unreadable") from exc
    properties = item["properties"]
    if _single_text(product_root, "PRODUCT_URI") != properties["s2:product_uri"]:
        raise AdmissionError("C1 item and native product URI differ")
    if _single_text(product_root, "PROCESSING_BASELINE") != properties["s2:processing_baseline"]:
        raise AdmissionError("C1 item and native processing baseline differ")
    product_start, product_stop = (_utc(_single_text(product_root, key))
                                   for key in ("PRODUCT_START_TIME", "PRODUCT_STOP_TIME"))
    if product_stop < product_start:
        raise AdmissionError("native Sentinel-2 product interval is reversed")
    for label, instant in (("native product start", product_start), ("native product stop", product_stop)):
        _inside_declared_window(instant.isoformat(), acquisition_start, acquisition_end_exclusive,
                                cutoff, label)
    item_time = _inside_declared_window(properties["datetime"], acquisition_start,
                                        acquisition_end_exclusive, cutoff, "Earth Search item datetime")
    expected_tile = properties.get("s2:tile_id")
    tile_ids = [node.text.strip() for node in granule_root.iter()
                if _local(node) == "TILE_ID" and node.text and node.text.strip()]
    if not expected_tile or len(tile_ids) != 1 or tile_ids[0] != expected_tile:
        raise AdmissionError("C1 item and native granule tile identities are absent or unequal")
    tile_code = _expected_tile_code(properties)
    if not expected_tile.endswith(f"_{tile_code}_N05.11"):
        raise AdmissionError("native tile identity does not match selected MGRS tile/baseline")
    sensing_values = [node.text.strip() for node in granule_root.iter()
                      if _local(node) == "SENSING_TIME" and node.text and node.text.strip()]
    if len(sensing_values) != 1:
        raise AdmissionError("native tile sensing time is absent or ambiguous")
    tile_sensing = _inside_declared_window(sensing_values[0], acquisition_start,
                                           acquisition_end_exclusive, cutoff,
                                           "native tile sensing time")
    if abs((item_time - tile_sensing).total_seconds()) > 2:
        raise AdmissionError("C1 item datetime and native tile sensing time contradict")
    native_epsg_values = [node.text.strip() for node in granule_root.iter()
                          if _local(node) == "HORIZONTAL_CS_CODE" and node.text and node.text.strip()]
    epsg = properties.get("proj:epsg")
    if len(set(native_epsg_values)) != 1 or native_epsg_values[0] != f"EPSG:{epsg}":
        raise AdmissionError("native granule CRS does not match C1 item projected CRS")
    temporal_join = ("PRODUCT_INTERVAL_AND_TILE_SENSING_DISTINCT_BOTH_WITHIN_PINNED_PRE_EVENT_WINDOW"
                     if product_start != tile_sensing or product_stop != tile_sensing else
                     "PRODUCT_AND_TILE_TIME_SUPPORTS_MATCH_WITHIN_PINNED_PRE_EVENT_WINDOW")
    offsets = {}
    identities = {}
    for node in product_root.iter():
        if _local(node) == "Spectral_Information":
            physical, band_id = node.attrib.get("physicalBand"), node.attrib.get("bandId")
            if physical and band_id is not None:
                canonical = {"B3": "B03", "B8": "B08"}.get(physical, physical)
                if canonical in identities and identities[canonical] != band_id:
                    raise AdmissionError("conflicting native band identity")
                identities[canonical] = band_id
        elif _local(node) == "BOA_ADD_OFFSET":
            band_id = node.attrib.get("band_id")
            value = float((node.text or "").strip())
            if band_id is None or not np.isfinite(value):
                raise AdmissionError("native BOA offset identity/value is invalid")
            if band_id in offsets and offsets[band_id] != value:
                raise AdmissionError("conflicting native BOA offsets")
            offsets[band_id] = value
    quantification = float(_single_text(product_root, "BOA_QUANTIFICATION_VALUE"))
    if not np.isfinite(quantification) or quantification <= 0:
        raise AdmissionError("native BOA quantification is invalid")
    radiometry = {}
    for band in ("B03", "B08", "B11"):
        band_id = identities.get(band)
        if band_id is None or band_id not in offsets:
            raise AdmissionError(f"native BOA correction is unresolved for {band}")
        radiometry[band] = {"band_id": band_id, "quantification": quantification,
                            "native_additive_offset": offsets[band_id],
                            "expected_scale": 1.0 / quantification,
                            "expected_offset": offsets[band_id] / quantification}
    image_files = [node.text.strip() for node in product_root.iter()
                   if _local(node) == "IMAGE_FILE" and node.text and node.text.strip()]
    required_native = {"B03": {"10m", "20m", "60m"}, "B08": {"10m"},
                       "B11": {"20m", "60m"}, "SCL": {"20m", "60m"}}
    observed_native = {band: set() for band in required_native}
    for path in image_files:
        match = re.search(r"_(B03|B08|B11|SCL)_(10m|20m|60m)$", path)
        if match:
            band, resolution = match.groups()
            if tile_code not in path:
                raise AdmissionError("native IMAGE_FILE role belongs to a different MGRS tile")
            observed_native[band].add(resolution)
    for band, resolutions in required_native.items():
        if observed_native[band] != resolutions:
            raise AdmissionError(f"native product metadata has absent/ambiguous {band} image roles")
    tile_join = "ITEM_PRODUCT_AND_GRANULE_TILE_IDENTITY_EXACT_MATCH"
    return {"product_uri": properties["s2:product_uri"],
            "processing_baseline": properties["s2:processing_baseline"],
            "product_start": product_start.isoformat(), "product_stop": product_stop.isoformat(),
            "item_datetime": item_time.isoformat(), "tile_sensing_time": tile_sensing.isoformat(),
            "generation_time": properties["s2:generation_time"],
            "time_join_status": temporal_join, "tile_join_status": tile_join,
            "native_tile_ids": tile_ids, "native_radiometry": radiometry,
            "native_image_file_roles": {k: sorted(v) for k, v in observed_native.items()},
            "native_crs": native_epsg_values[0],
            "historical_operational_availability": "UNVERIFIED_REPROCESSED_ARCHIVE_GENERATED_2025"}


def _multihash_sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"1220[0-9a-f]{64}", value):
        raise AdmissionError("C1 file:checksum is not a SHA-256 multihash")
    return value[4:]


def stage_asset(asset, head_record, destination, *, origin="provider_download", context=None):
    """Download one anonymous asset; preserve partial bytes and never resume."""
    if origin not in {"provider_download", "constructed_fixture"}:
        raise AdmissionError("C1 transfer source origin must be explicit")
    href = asset["href"]
    parts = _safe_asset_url(href)
    if (head_record.get("method") != "HEAD" or head_record.get("http_status") != 200 or
            head_record.get("url") != href):
        raise AdmissionError("matching successful retained HEAD evidence is required")
    expected_size = asset["file:size"]
    expected_head_size = int(head_record["response_headers"]["Content-Length"])
    if expected_size != expected_head_size:
        raise AdmissionError("retained HEAD length differs from C1 asset metadata")
    expected_etag = head_record.get("response_headers", {}).get("ETag")
    target = Path(destination)
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_name(target.name + ".receipt.json")
    if target.exists() or partial.exists() or receipt_path.exists():
        raise AdmissionError("existing asset attempts are immutable; select a new attempt directory")
    receipt = {"origin": origin, "provider": "Earth Search C1", "href": href,
               "representation": "individual_native_resolution_COG", "expected_bytes": expected_size,
               "head_content_length": expected_head_size, "head_etag_opaque": expected_etag,
               "expected_sha256": _multihash_sha256(asset["file:checksum"]),
               "resume_supported": False, "status": "REQUEST_PENDING",
               "request_started_utc": _now(), "runtime": _runtime_identity(),
               "selection_id": (context or {}).get("selection_id"),
               "selection_sha256": (context or {}).get("selection_sha256"),
               "product_id": (context or {}).get("product_id"),
               "site_id": (context or {}).get("site_id"), "asset_role": (context or {}).get("asset_role")}
    digest = hashlib.sha256()
    received = 0
    response_length = None
    response_etag = None
    headers = {"User-Agent": "Sentinel-GL-WP03/1.0", "Accept-Encoding": "identity"}
    try:
        request = urllib.request.Request(href, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            final = urllib.parse.urlsplit(response.geturl())
            if response.status != 200 or final.scheme != "https" or final.hostname != parts.hostname or final.path != parts.path:
                raise AdmissionError("C1 response status or final URL differs from approved asset")
            response_length = response.headers.get("Content-Length")
            response_etag = response.headers.get("ETag")
            if response_length is not None and int(response_length) != expected_size:
                raise AdmissionError("C1 GET Content-Length differs from retained asset size")
            with partial.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    if received + len(chunk) > expected_size or received + len(chunk) > 2 * 1024**3:
                        raise AdmissionError("C1 response exceeds its declared product byte bound")
                    stream.write(chunk)
                    received += len(chunk)
                    digest.update(chunk)
                stream.flush()
            if expected_etag is not None and response_etag != expected_etag:
                receipt.update({"status": "GET_ETAG_DIFFERS_FROM_RETAINED_HEAD",
                                "received_bytes": received, "partial_sha256": digest.hexdigest(),
                                "get_etag_opaque": response_etag})
                raise AdmissionError("C1 GET ETag differs from retained HEAD identity")
            actual_sha = digest.hexdigest()
            if received != expected_size or actual_sha != receipt["expected_sha256"]:
                receipt.update({"status": "LENGTH_OR_SHA256_MISMATCH", "received_bytes": received,
                                "partial_sha256": actual_sha, "get_content_length": response_length,
                                "get_etag_opaque": response_etag})
                raise AdmissionError("C1 transfer length or SHA-256 multihash mismatch")
            partial.replace(target)
            receipt.update({"status": "COMPLETE_HASH_VERIFIED", "received_bytes": received,
                            "sha256": actual_sha, "get_content_length": response_length,
                            "get_etag_opaque": response_etag,
                            "etag_scope": "OPAQUE_IDENTITY_ONLY_NOT_INTERPRETED_AS_CHECKSUM",
                            "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED"})
    except urllib.error.HTTPError as exc:
        partial_size = partial.stat().st_size if partial.is_file() else 0
        receipt.update({"status": "HTTP_ERROR", "http_status": exc.code,
                        "literal_error": _safe_literal_error(exc), "received_bytes": partial_size,
                        "request_finished_utc": _now(),
                        "partial_path": partial.name if partial.exists() else None,
                        "partial_sha256": _sha(partial) if partial.exists() else None,
                        "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED"})
        _write_receipt(receipt_path, receipt)
        raise
    except Exception as exc:
        partial_size = partial.stat().st_size if partial.is_file() else 0
        if receipt.get("status") == "REQUEST_PENDING":
            receipt.update({"status": "INTERRUPTED_OR_INCOMPLETE_RETAINED",
                            "error_type": type(exc).__name__,
                            "literal_error": _safe_literal_error(exc),
                            "received_bytes": partial_size, "request_finished_utc": _now(),
                            "partial_path": partial.name if partial.exists() else None,
                            "partial_sha256": _sha(partial) if partial.exists() else None,
                            "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED"})
        elif not receipt.get("request_finished_utc"):
            receipt["request_finished_utc"] = _now()
        receipt.setdefault("get_content_length", response_length)
        receipt.setdefault("get_etag_opaque", response_etag)
        if not receipt_path.exists():
            _write_receipt(receipt_path, receipt)
        raise
    receipt["output_sha256"] = receipt.get("sha256")
    receipt["output_bytes"] = receipt.get("received_bytes") if receipt.get("status") == "COMPLETE_HASH_VERIFIED" else None
    receipt.setdefault("request_finished_utc", _now())
    _write_receipt(receipt_path, receipt)
    return receipt


def _write_geotiff(path, array, *, crs, transform, nodata, description, tags):
    values = np.asarray(array)
    if values.ndim == 2:
        count, height, width = 1, *values.shape
    elif values.ndim == 3:
        count, height, width = values.shape
    else:
        raise AdmissionError("ROI output must be a single grid or band-first image")
    profile = {"driver": "GTiff", "width": int(width), "height": int(height),
               "count": int(count), "dtype": values.dtype, "crs": crs,
               "transform": transform, "compress": "deflate"}
    if nodata is not None:
        profile["nodata"] = nodata
    if width >= 256 and height >= 256:
        profile.update({"tiled": True, "blockxsize": 256, "blockysize": 256})
    if np.issubdtype(values.dtype, np.floating):
        profile["predictor"] = 3
    elif values.dtype != np.uint8:
        profile["predictor"] = 2
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1) if values.ndim == 2 else dataset.write(values)
        if values.ndim == 2:
            dataset.set_band_description(1, description)
        else:
            for index, band_name in enumerate(description, start=1):
                dataset.set_band_description(index, band_name)
        dataset.update_tags(**{key: str(value) for key, value in tags.items()})
    return path


def _png_chunk(kind, payload):
    encoded_kind = kind.encode("ascii")
    body = encoded_kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def _write_rgb_png(path, rgb, *, metadata=None):
    values = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if values.ndim != 3 or values.shape[2] != 3:
        raise AdmissionError("overlay preview must be an RGB image")
    height, width, _ = values.shape
    rows = b"".join(b"\x00" + values[row].tobytes() for row in range(height))
    text_chunks = b"".join(
        _png_chunk("tEXt", key.encode("latin-1") + b"\x00" + str(value).encode("latin-1"))
        for key, value in (metadata or {}).items())
    payload = (b"\x89PNG\r\n\x1a\n" +
               _png_chunk("IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
               text_chunks + _png_chunk("IDAT", zlib.compress(rows, level=6)) +
               _png_chunk("IEND", b""))
    Path(path).write_bytes(payload)
    return path


def _decode_c1_roi(item, site_id, asset_paths, metadata, source_sha256s, *,
                   latitude, longitude, origin, output_dir, halfwidth_m):
    if origin not in {"provider_download", "constructed_fixture"}:
        raise AdmissionError("C1 decoder source origin must be explicit")
    if (not math.isfinite(float(halfwidth_m)) or halfwidth_m <= 0 or halfwidth_m > 5000):
        raise AdmissionError("ROI halfwidth must be positive and no greater than 5000 metres")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    reference_path = Path(asset_paths["swir16"])
    with rasterio.open(reference_path) as reference:
        if not reference.crs or reference.crs.to_epsg() != 32645:
            raise AdmissionError("optical ROI reference grid is not EPSG:32645")
        rx, ry = abs(float(reference.transform.a)), abs(float(reference.transform.e))
        if (abs(rx - 20) > 1e-8 or abs(ry - 20) > 1e-8 or
                abs(float(reference.transform.b)) > 1e-8 or abs(float(reference.transform.d)) > 1e-8):
            raise AdmissionError("B11 reference grid is not north-up native 20 metre sampling")
        center_x, center_y = transform_coordinates(
            "EPSG:4326", reference.crs, [float(longitude)], [float(latitude)])
        cx, cy = float(center_x[0]), float(center_y[0])
        roi_bounds = (cx - halfwidth_m, cy - halfwidth_m,
                      cx + halfwidth_m, cy + halfwidth_m)
        left, bottom, right, top = roi_bounds
        c, f = float(reference.transform.c), float(reference.transform.f)
        col_start = math.ceil((left - c) / rx - 1e-10)
        col_stop = math.floor((right - c) / rx + 1e-10)
        row_start = math.ceil((f - top) / ry - 1e-10)
        row_stop = math.floor((f - bottom) / ry + 1e-10)
        if col_stop <= col_start or row_stop <= row_start:
            raise AdmissionError("declared ROI contains no complete native 20 metre cells")
        out_width, out_height = col_stop - col_start, row_stop - row_start
        reference_window = Window(col_start, row_start, out_width, out_height)
        output_transform = reference.window_transform(reference_window)
        output_bounds = (float(output_transform.c),
                         float(output_transform.f + output_transform.e * out_height),
                         float(output_transform.c + output_transform.a * out_width),
                         float(output_transform.f))
        center_row, center_col = reference.index(cx, cy)
        if not (0 <= center_col < reference.width and 0 <= center_row < reference.height):
            raise AdmissionError("PI query coordinate is outside the native C1 B11 tile")
        ref_crs = reference.crs

    west_trim, south_trim, east_trim, north_trim = (
        output_bounds[0] - left, output_bounds[1] - bottom,
        right - output_bounds[2], top - output_bounds[3])
    if any(value < -1e-7 or value >= 20 + 1e-7
           for value in (west_trim, south_trim, east_trim, north_trim)):
        raise AdmissionError("native-grid ROI edge trimming exceeds the declared 20 metre bound")
    role_arrays = {}
    source_windows = {}
    source_coverage = {}
    invalid_reason_counts = {}
    scl_raw = None
    scl_valid = None
    scl_counts = {}
    for role, band in ASSET_ROLES.items():
        path = Path(asset_paths[role])
        asset = item["assets"][role]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != asset["file:size"]:
            raise AdmissionError(f"staged C1 {role} bytes changed before ROI decode")
        actual_hash = _sha(path)
        expected_hash = _multihash_sha256(asset["file:checksum"])
        if actual_hash != expected_hash or actual_hash != source_sha256s[role]:
            raise AdmissionError(f"staged C1 {role} hash changed before ROI decode")
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            intersect = (max(left, bounds.left), max(bottom, bounds.bottom),
                         min(right, bounds.right), min(top, bounds.top))
            intersect_area = max(0.0, intersect[2] - intersect[0]) * max(0.0, intersect[3] - intersect[1])
            source_coverage[role] = {
                "full_requested_roi_covered": bool(bounds.left <= left and bounds.bottom <= bottom and
                                                    bounds.right >= right and bounds.top >= top),
                "source_bounds": [float(bounds.left), float(bounds.bottom),
                                  float(bounds.right), float(bounds.top)],
                "roi_intersection_bounds": list(map(float, intersect)),
                "roi_area_fraction_covered": float(intersect_area / ((right-left) * (top-bottom)))
            }
            gsd = float(EXPECTED_GSD[role])
            factor_float = 20.0 / gsd
            factor = int(round(factor_float))
            if factor < 1 or abs(factor - factor_float) > 1e-10:
                raise AdmissionError(f"C1 {role} grid cannot register to the native 20 metre ROI grid")
            x_offset = (float(output_transform.c) - float(dataset.transform.c)) / float(dataset.transform.a)
            y_offset = (float(output_transform.f) - float(dataset.transform.f)) / float(dataset.transform.e)
            if (abs(x_offset - round(x_offset)) > 1e-8 or
                    abs(y_offset - round(y_offset)) > 1e-8 or
                    abs(float(dataset.transform.a) * factor - float(output_transform.a)) > 1e-8 or
                    abs(float(dataset.transform.e) * factor - float(output_transform.e)) > 1e-8):
                raise AdmissionError(f"C1 {role} source pixels are not registered to B11 20 metre cells")
            col0, row0 = int(round(x_offset)), int(round(y_offset))
            width, height = out_width * factor, out_height * factor
            if width > 1024 or height > 1024:
                raise AdmissionError("bounded ROI source window exceeds the 1024-pixel side limit")
            window = Window(col0, row0, width, height)
            fill = dataset.nodata if dataset.nodata is not None else 0
            raw = dataset.read(1, window=window, boundless=True, fill_value=fill)
            source_mask = dataset.read_masks(1, window=window, boundless=True) > 0
            source_valid = source_mask & np.isfinite(raw)
            if dataset.nodata is not None:
                source_valid &= raw != dataset.nodata
            if raw.shape != (height, width) or source_valid.shape != raw.shape:
                raise AdmissionError(f"C1 {role} bounded source window has unexpected shape")
            source_windows[role] = {"row": row0, "column": col0,
                                    "height": height, "width": width,
                                    "grid_factor": factor}
            if role == "scl":
                scl_raw = raw.astype(np.uint8, copy=False)
                scl_valid = source_valid
                values, counts = np.unique(scl_raw, return_counts=True)
                scl_counts = {str(int(value)): int(count)
                              for value, count in zip(values, counts)}
                scl_support = scl_valid.astype(np.uint8)
                _write_geotiff(destination / "scl_raw_20m_classes.tif", scl_raw,
                    crs=ref_crs, transform=output_transform, nodata=dataset.nodata,
                    description="native Sentinel-2 Scene Classification Layer class",
                    tags={"origin": origin, "site_id": site_id,
                          "source_item_id": item["id"], "product_uri": metadata["product_uri"],
                          "role": "SCL", "resampling": "NONE_CATEGORICAL_NATIVE_20M",
                          "source_sha256": actual_hash})
                _write_geotiff(destination / "scl_valid_mask.tif", scl_support,
                    crs=ref_crs, transform=output_transform, nodata=255,
                    description="1=valid native SCL source support; 0=invalid or missing",
                    tags={"origin": origin, "site_id": site_id,
                          "source_item_id": item["id"], "product_uri": metadata["product_uri"],
                          "role": "SCL validity", "source_sha256": actual_hash})
                continue
            scale = float(dataset.scales[0]) if dataset.scales else 1.0
            offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
            native = metadata["native_radiometry"][band]
            declared = asset.get("raster:bands", [{}])[0]
            calibration_ok = (
                abs(scale - float(native["expected_scale"])) <= 1e-12 and
                abs(offset - float(native["expected_offset"])) <= 1e-9 and
                abs(scale - float(declared["scale"])) <= 1e-12 and
                abs(offset - float(declared["offset"])) <= 1e-9)
            if factor == 2:
                blocks = raw.reshape(out_height, 2, out_width, 2)
                valid_blocks = source_valid.reshape(out_height, 2, out_width, 2)
                support = valid_blocks.sum(axis=(1, 3)).astype(np.uint8)
                full_support = support == 4
                raw_mean = np.full((out_height, out_width), FLOAT_NODATA, dtype=np.float32)
                reflectance = np.full((out_height, out_width), np.nan, dtype=np.float64)
                if calibration_ok:
                    calibrated_blocks = blocks.astype(np.float64) * scale + offset
                    complete_mean = calibrated_blocks.mean(axis=(1, 3))
                    raw_values = blocks.astype(np.float64).mean(axis=(1, 3))
                    raw_mean[full_support] = raw_values[full_support].astype(np.float32)
                    reflectance[full_support] = complete_mean[full_support]
                reason_key = f"{band}_10M_CONTRIBUTOR_SUPPORT_LT_4"
                invalid_reason_counts[reason_key] = int(np.count_nonzero(~full_support))
            else:
                support = source_valid.astype(np.uint8)
                full_support = source_valid
                raw_mean = np.full(raw.shape, FLOAT_NODATA, dtype=np.float32)
                raw_mean[full_support] = raw[full_support].astype(np.float32)
                reflectance = np.full(raw.shape, np.nan, dtype=np.float64)
                if calibration_ok:
                    reflectance[full_support] = raw[full_support].astype(np.float64) * scale + offset
                reason_key = f"{band}_20M_SOURCE_INVALID"
                invalid_reason_counts[reason_key] = int(np.count_nonzero(~full_support))
            if not calibration_ok:
                invalid_reason_counts[f"{band}_COG_STAC_NATIVE_ENCODING_MISMATCH"] = int(full_support.size)
                full_support = np.zeros(full_support.shape, dtype=bool)
                reflectance[:] = np.nan
                raw_mean[:] = FLOAT_NODATA
            role_arrays[role] = {"band": band, "raw": raw, "source_valid": source_valid,
                                 "raw_mean": raw_mean, "valid": full_support,
                                 "reflectance": reflectance, "scale": scale,
                                 "offset": offset, "calibration_ok": calibration_ok,
                                 "support_count": support, "window": source_windows[role],
                                 "hash": actual_hash, "dtype": raw.dtype,
                                 "nodata": dataset.nodata}
            role_tags = {"origin": origin, "site_id": site_id,
                         "source_item_id": item["id"], "product_uri": metadata["product_uri"],
                         "role": band, "source_sha256": actual_hash,
                         "native_scale": native["expected_scale"],
                         "native_offset": native["expected_offset"],
                         "cog_scale": scale, "cog_offset": offset,
                         "calibration_status": "MATCHED_COG_STAC_AND_NATIVE_XML" if calibration_ok
                                              else "UNRESOLVED_ENCODING_NO_REFLECTANCE",
                         "aggregation": "REGISTERED_2X2_MEAN_INVALID_SUPPORT_PROPAGATED"
                                       if factor == 2 else "NATIVE_20M_NO_RESAMPLING"}
            _write_geotiff(destination / f"{band.lower()}_raw_20m_mean_dn.tif", raw_mean,
                crs=ref_crs, transform=output_transform, nodata=FLOAT_NODATA,
                description=f"{band} raw digital number mean on the native 20 metre grid",
                tags=role_tags)
            _write_geotiff(destination / f"{band.lower()}_reflectance_20m.tif",
                np.where(full_support, reflectance, FLOAT_NODATA).astype(np.float32),
                crs=ref_crs, transform=output_transform, nodata=FLOAT_NODATA,
                description=f"{band} BOA reflectance; scale and offset applied once",
                tags={**role_tags, "units": "dimensionless_surface_reflectance"})
            _write_geotiff(destination / f"{band.lower()}_valid_support_count.tif", support,
                crs=ref_crs, transform=output_transform, nodata=None,
                description=("valid contributing 10 metre samples (0-4)" if factor == 2
                             else "valid native 20 metre source sample (0-1)"),
                tags={"origin": origin, "site_id": site_id,
                      "role": band, "source_sha256": actual_hash})
            _write_geotiff(destination / f"{band.lower()}_valid_mask.tif", full_support.astype(np.uint8),
                crs=ref_crs, transform=output_transform, nodata=255,
                description="1=valid calibrated support; 0=invalid support",
                tags={"origin": origin, "site_id": site_id,
                      "role": band, "source_sha256": actual_hash})
            if full_support.any():
                row, column = np.argwhere(full_support)[0]
                if factor == 2:
                    raw_sample = blocks[row, :, column, :].astype(np.float64)
                    direct = float(np.mean(raw_sample * float(native["expected_scale"]) +
                                           float(native["expected_offset"])))
                    sample_raw = raw_sample.reshape(-1).tolist()
                    method = "independent_mean_of_four_raw_to_BOA_samples"
                else:
                    raw_sample = float(raw[row, column])
                    direct = raw_sample * float(native["expected_scale"]) + float(native["expected_offset"])
                    sample_raw = [raw_sample]
                    method = "independent_single_native_sample_raw_to_BOA"
                recorded = float(reflectance[row, column])
                if not math.isclose(direct, recorded, rel_tol=0, abs_tol=1e-7):
                    raise AdmissionError(f"{band} independent raw-to-BOA sample recomputation differs")
                role_arrays[role]["sample_check"] = {
                    "roi_row_column": [int(row), int(column)], "source_raw_values": sample_raw,
                    "scale": float(native["expected_scale"]),
                    "offset": float(native["expected_offset"]),
                    "independent_value": direct, "decoded_value": recorded,
                    "absolute_difference": abs(direct-recorded), "method": method,
                    "status": "PASS"}
            else:
                role_arrays[role]["sample_check"] = {"status": "UNAVAILABLE_NO_VALID_SUPPORT"}

    if scl_raw is None or scl_valid is None:
        raise AdmissionError("native SCL source could not be read for ROI QA")
    scl_counts_valid = {}
    for value, count in zip(*np.unique(scl_raw[scl_valid], return_counts=True)):
        scl_counts_valid[str(int(value))] = int(count)
    scl_clear = scl_valid & np.isin(scl_raw, (4, 5, 6))
    scl_excluded = scl_valid & ~np.isin(scl_raw, (4, 5, 6))
    _write_geotiff(destination / "scl_clear_4_5_6_mask.tif", scl_clear.astype(np.uint8),
        crs=ref_crs, transform=output_transform, nodata=255,
        description="1=SCL 4/5/6; 0=valid non-clear class; 255=invalid SCL",
        tags={"origin": origin, "site_id": site_id,
              "qa_policy": "SCL classes 4,5,6 admitted; others excluded, not labelled dry"})
    _write_geotiff(destination / "scl_qa_excluded_mask.tif", scl_excluded.astype(np.uint8),
        crs=ref_crs, transform=output_transform, nodata=255,
        description="1=valid SCL outside 4/5/6; 0=clear class; 255=invalid SCL",
        tags={"origin": origin, "site_id": site_id})
    invalid_reason_counts["SCL_INVALID_OR_NODATA"] = int(np.count_nonzero(~scl_valid))

    green = role_arrays["green"]
    nir = role_arrays["nir"]
    swir = role_arrays["swir16"]
    green_values, nir_values, swir_values = (green["reflectance"], nir["reflectance"],
                                              swir["reflectance"])
    ndwi_denominator = green_values + nir_values
    mndwi_denominator = green_values + swir_values
    ndwi_valid = (green["valid"] & nir["valid"] & np.isfinite(ndwi_denominator) &
                  (ndwi_denominator != 0))
    mndwi_valid = (green["valid"] & swir["valid"] & np.isfinite(mndwi_denominator) &
                   (mndwi_denominator != 0))
    invalid_reason_counts["NDWI_ZERO_OR_INVALID_DENOMINATOR"] = int(np.count_nonzero(
        green["valid"] & nir["valid"] & (~np.isfinite(ndwi_denominator) | (ndwi_denominator == 0))))
    invalid_reason_counts["MNDWI_ZERO_OR_INVALID_DENOMINATOR"] = int(np.count_nonzero(
        green["valid"] & swir["valid"] & (~np.isfinite(mndwi_denominator) | (mndwi_denominator == 0))))
    ndwi = np.full((out_height, out_width), np.nan, dtype=np.float64)
    mndwi = np.full((out_height, out_width), np.nan, dtype=np.float64)
    ndwi[ndwi_valid] = ((green_values[ndwi_valid] - nir_values[ndwi_valid]) /
                        ndwi_denominator[ndwi_valid])
    mndwi[mndwi_valid] = ((green_values[mndwi_valid] - swir_values[mndwi_valid]) /
                          mndwi_denominator[mndwi_valid])
    analysis_valid = (ndwi_valid & mndwi_valid & scl_valid)
    unknown = ~analysis_valid
    candidate_support = analysis_valid & scl_clear
    candidate = candidate_support & (ndwi > 0) & (mndwi > 0)
    candidate_encoded = np.full((out_height, out_width), 255, dtype=np.uint8)
    candidate_encoded[candidate_support] = candidate[candidate_support].astype(np.uint8)
    _write_geotiff(destination / "ndwi.tif", np.where(ndwi_valid, ndwi, FLOAT_NODATA).astype(np.float32),
        crs=ref_crs, transform=output_transform, nodata=FLOAT_NODATA,
        description="NDWI=(B03-B08)/(B03+B08); undefined values are nodata",
        tags={"origin": origin, "site_id": site_id,
              "algorithm": "development_initial_score_blind_v1", "units": "dimensionless"})
    _write_geotiff(destination / "mndwi.tif", np.where(mndwi_valid, mndwi, FLOAT_NODATA).astype(np.float32),
        crs=ref_crs, transform=output_transform, nodata=FLOAT_NODATA,
        description="MNDWI=(B03-B11)/(B03+B11); undefined values are nodata",
        tags={"origin": origin, "site_id": site_id,
              "algorithm": "development_initial_score_blind_v1", "units": "dimensionless"})
    _write_geotiff(destination / "analysis_valid_mask.tif", analysis_valid.astype(np.uint8),
        crs=ref_crs, transform=output_transform, nodata=255,
        description="1=valid bands and denominators plus valid SCL; 0=unknown support",
        tags={"origin": origin, "site_id": site_id})
    _write_geotiff(destination / "unknown_mask.tif", unknown.astype(np.uint8),
        crs=ref_crs, transform=output_transform, nodata=255,
        description="1=missing/invalid source support or undefined index denominator",
        tags={"origin": origin, "site_id": site_id,
              "unknown_is_not_dry_land": "true"})
    _write_geotiff(destination / "candidate_water_mask.tif", candidate_encoded,
        crs=ref_crs, transform=output_transform, nodata=255,
        description="1=candidate water; 0=no candidate under declared rule; 255=unsupported or QA-excluded",
        tags={"origin": origin, "site_id": site_id,
              "rule": "NDWI>0 AND MNDWI>0 AND SCL in 4,5,6",
              "interpretation": "exploratory candidate pixels within ROI; not lake ground truth",
              "nodata_semantics": "raw invalid/undefined or SCL QA-excluded; neither is labelled dry",
              "water_mask_unavailable_pixel_count": int((~candidate_support).sum()),
              "water_mask_unavailable_fraction": float((~candidate_support).mean())})

    roi_area = out_width * out_height * abs(float(output_transform.a * output_transform.e))
    if candidate_support.any():
        mask_area_status = "ROI_CANDIDATE_AREA_COMPUTED_ON_VALID_QA_SUPPORT"
        candidate_area = int(candidate.sum()) * abs(float(output_transform.a * output_transform.e))
    else:
        mask_area_status = "NO_VALID_QA_SUPPORT_AREA_UNAVAILABLE"
        candidate_area = None
    candidate_count = int(candidate.sum()) if candidate_support.any() else None
    row_grid, column_grid = np.indices((out_height, out_width))
    rgb = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    display_low, display_high = -0.1, 0.6
    for channel, source in enumerate((swir_values, nir_values, green_values)):
        valid = np.isfinite(source)
        channel_values = np.zeros(source.shape, dtype=np.float64)
        channel_values[valid] = np.clip((source[valid] - display_low) /
                                        (display_high - display_low), 0, 1) * 255
        rgb[..., channel] = channel_values.astype(np.uint8)
    overlay = rgb.astype(np.float32)
    qa_hatch = scl_excluded & (((row_grid + column_grid) % 9) < 3)
    unknown_hatch = unknown & (((row_grid - column_grid) % 9) < 3)
    overlay[candidate_support & candidate] = (255, 0, 255)
    overlay[qa_hatch] = (255, 176, 0)
    overlay[unknown_hatch] = (175, 175, 175)
    query_col = int(math.floor((cx - output_transform.c) / output_transform.a))
    query_row = int(math.floor((output_transform.f - cy) / abs(output_transform.e)))
    if 0 <= query_row < out_height and 0 <= query_col < out_width:
        for delta in range(-6, 7):
            for rr, cc in ((query_row, query_col + delta), (query_row + delta, query_col)):
                if 0 <= rr < out_height and 0 <= cc < out_width:
                    overlay[rr, cc] = (255, 255, 255) if abs(delta) > 1 else (0, 0, 0)
    scale_pixels = max(1, int(round(1000 / abs(output_transform.a))))
    bar_row = out_height - 12
    if out_width >= scale_pixels + 20 and bar_row >= 2:
        overlay[bar_row-1:bar_row+2, 10:10+scale_pixels] = (0, 0, 0)
        overlay[bar_row:bar_row+1, 10:10+scale_pixels] = (255, 255, 255)
    overlay_rgb = np.clip(overlay, 0, 255).astype(np.uint8)
    false_color_path = destination / "false_colour_B11_B08_B03.tif"
    _write_geotiff(false_color_path, rgb.transpose(2, 0, 1), crs=ref_crs,
        transform=output_transform, nodata=None,
        description=("R=B11", "G=B08", "B=B03"),
        tags={"origin": origin, "site_id": site_id,
              "display_stretch": "fixed reflectance -0.1 to 0.6; visualization only"})
    _write_geotiff(destination / "diagnostic_overlay.tif", overlay_rgb.transpose(2, 0, 1),
        crs=ref_crs, transform=output_transform, nodata=None,
        description=("red", "green", "blue"),
        tags={"origin": origin, "site_id": site_id,
              "map_overlay": "candidate water magenta; SCL-excluded amber hatch; unknown gray hatch; query cross white/black",
              "georeferencing": "EPSG:32645 native 20 metre grid",
              "water_mask_unavailable_pixel_count": int((~candidate_support).sum()),
              "water_mask_unavailable_fraction": float((~candidate_support).mean())})
    png_path = destination / "diagnostic_overlay.png"
    _write_rgb_png(png_path, overlay_rgb, metadata={
        "origin": origin, "site_id": site_id, "source_item_id": item["id"],
        "product_uri": metadata["product_uri"], "research_admitted": "false",
        "visual_product": "diagnostic_overlay",
        "water_mask_unavailable_pixel_count": int((~candidate_support).sum()),
        "water_mask_unavailable_fraction": float((~candidate_support).mean())})
    image64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    site_name = "South Lhonak" if site_id == "PILOT-SOUTH-LHONAK" else "Khangchung Tsho"
    acquisition = item["properties"].get("datetime", "UNAVAILABLE")
    svg_lines = [
        f"{site_name} | acquisition {acquisition}",
        "False colour: R=B11, G=B08, B=B03; fixed reflectance stretch [-0.1, 0.6]",
        "Magenta: NDWI>0 AND MNDWI>0 with SCL 4/5/6; amber hatch: valid SCL excluded by QA",
        "Gray hatch: raw invalid/undefined support; amber QA exclusions are also water-mask nodata",
        "White/black cross: PI query coordinate; excluded cells are not labelled dry",
        "1 km scale bar; EPSG:32645; display is ROI-clipped, not verified lake geometry",
        f"Item: {item['id']}", f"Product: {metadata['product_uri']}",
    ]
    svg_text = "".join(
        f'<text x="{out_width + 36}" y="{36 + index*30}" font-family="sans-serif" font-size="15">'
        f'{html.escape(line)}</text>' for index, line in enumerate(svg_lines))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_width + 600}" '
           f'height="{max(out_height + 24, 250)}" viewBox="0 0 {out_width + 600} '
           f'{max(out_height + 24, 250)}"><title>{html.escape(site_name)} {html.escape(origin)} ROI overlay; quarantined</title>'
           f'<metadata>CRS=EPSG:32645; bounds={output_bounds}; requested_bounds={roi_bounds}; '
           f'pixel_size=20m; origin={html.escape(origin)}; research_admitted=false; '
           f'water_mask_unavailable_pixels={int((~candidate_support).sum())}; '
           f'water_mask_unavailable_fraction={float((~candidate_support).mean())}</metadata>'
           f'<rect width="100%" height="100%" fill="#f7f7f7"/>'
           f'<image x="0" y="0" width="{out_width}" height="{out_height}" '
           f'href="data:image/png;base64,{image64}"/>{svg_text}</svg>')
    (destination / "diagnostic_overlay.svg").write_text(svg, encoding="utf-8")

    return {
        "status": "ROI_PIXEL_DECODED_QUARANTINED",
        "origin": origin,
        "provider": "Earth Search C1" if origin == "provider_download" else "constructed C1 engineering fixture",
        "site_id": site_id,
        "site_scope": "DEVELOPMENT_EXPOSED", "site_geometry_status": "PI_QUERY_COORDINATE_ONLY_LAKE_GEOMETRY_UNVERIFIED",
        "site_item_id": item["id"], "product_uri": metadata["product_uri"],
        "acquisition_datetime": acquisition, "processing_generation_time": item["properties"].get("s2:generation_time"),
        "centroid_wgs84_lon_lat": [float(longitude), float(latitude)],
        "centroid_epsg32645": [cx, cy], "roi_crs": "EPSG:32645",
        "requested_roi_bounds": list(roi_bounds), "output_roi_bounds": list(output_bounds),
        "requested_halfwidth_m": float(halfwidth_m), "output_grid_resolution_m": 20,
        "output_shape_rows_columns": [out_height, out_width],
        "requested_roi_area_m2": float((right-left) * (top-bottom)),
        "grid_edge_trim_m_west_south_east_north": [west_trim, south_trim, east_trim, north_trim],
        "source_alignment_halo_in_output_m": 0,
        "source_coverage": source_coverage,
        "source_windows_row_col_height_width_and_aggregation_factor": source_windows,
        "resampling": "registered 10m B03/B08 arithmetic mean of exact 2x2 source pixels; any invalid contributor invalidates 20m output; B11/SCL native20m; SCL categorical and never resampled",
        "calibration": {role_arrays[role]["band"]: {
            "scale": role_arrays[role]["scale"], "offset": role_arrays[role]["offset"],
            "status": "MATCHED_COG_STAC_AND_NATIVE_XML" if role_arrays[role]["calibration_ok"]
                      else "UNRESOLVED_ENCODING_NO_REFLECTANCE",
            "raw_to_reflectance_sample_recomputation": role_arrays[role].get("sample_check")}
            for role in role_arrays},
        "scl_class_counts_all_roi_values_including_nodata": scl_counts,
        "scl_class_counts_valid_source_support": scl_counts_valid,
        "scl_clear_classes": [4, 5, 6], "scl_clear_pixel_count": int(scl_clear.sum()),
        "scl_qa_excluded_pixel_count": int(scl_excluded.sum()),
        "analysis_valid_pixel_count": int(analysis_valid.sum()),
        "unknown_pixel_count": int(unknown.sum()),
        "raw_support_or_denominator_unknown_pixel_count": int(unknown.sum()),
        "unknown_is_dry_land": False,
        "unknown_pixel_definition": "invalid band/SCL support or undefined/nonfinite index denominator; SCL QA exclusions are reported separately",
        "unknown_invalidity_reason_counts_nonexclusive": invalid_reason_counts,
        "candidate_support_pixel_count": int(candidate_support.sum()),
        "water_mask_unavailable_pixel_count": int((~candidate_support).sum()),
        "water_mask_unavailable_fraction": float((~candidate_support).mean()),
        "water_mask_unavailable_reason_counts_nonexclusive": {
            "raw_support_or_denominator_unknown": int(unknown.sum()),
            "valid_scl_qa_excluded": int(scl_excluded.sum())},
        "water_mask_observability_status": (
            "PARTIAL_QA_OR_RAW_SUPPORT_NODATA" if np.any(~candidate_support)
            else "COMPLETE_WITHIN_VALID_QA_SUPPORT"),
        "candidate_water_pixel_count": candidate_count,
        "candidate_water_area_status": mask_area_status,
        "candidate_water_area_m2": candidate_area,
        "candidate_water_area_km2": None if candidate_area is None else candidate_area / 1_000_000,
        "total_roi_grid_area_m2": roi_area,
        "mask_rule": "NDWI>0 AND MNDWI>0 AND SCL in {4,5,6}; exploratory, not ground truth",
        "NDWI_valid_pixel_count": int(ndwi_valid.sum()), "MNDWI_valid_pixel_count": int(mndwi_valid.sum()),
        "artifacts": [
            "b03_raw_20m_mean_dn.tif", "b08_raw_20m_mean_dn.tif", "b11_raw_20m_mean_dn.tif",
            "b03_reflectance_20m.tif", "b08_reflectance_20m.tif", "b11_reflectance_20m.tif",
            "scl_raw_20m_classes.tif", "scl_valid_mask.tif", "scl_clear_4_5_6_mask.tif",
            "scl_qa_excluded_mask.tif", "ndwi.tif", "mndwi.tif", "analysis_valid_mask.tif",
            "unknown_mask.tif", "candidate_water_mask.tif", "false_colour_B11_B08_B03.tif",
            "diagnostic_overlay.tif", "diagnostic_overlay.png", "diagnostic_overlay.svg"],
        "research_admissible": False,
        "2023_operational_availability": "UNVERIFIED_REPROCESSED_IN_2025",
    }


def decode_c1_bundle(item, site_id, product_xml, granule_xml, asset_paths, cutoff, *,
                     acquisition_start, acquisition_end_exclusive,
                     selection_record=None, origin="provider_download",
                     roi_output_dir=None, roi_halfwidth_m=5000.0):
    """Inspect real COG rasters and small centroid windows without lake claims."""
    if origin not in {"provider_download", "constructed_fixture"}:
        raise AdmissionError("C1 decoder source origin must be explicit")
    validate_item(item, site_id, cutoff, acquisition_start=acquisition_start,
                  acquisition_end_exclusive=acquisition_end_exclusive,
                  selection_record=selection_record)
    metadata = validate_native_metadata(item, product_xml, granule_xml,
        acquisition_start=acquisition_start, acquisition_end_exclusive=acquisition_end_exclusive,
        cutoff=cutoff)
    lat, lon = CENTROIDS[site_id]
    outputs = {}
    actual_grids = {}
    for role, band in ASSET_ROLES.items():
        asset = item["assets"][role]
        path = Path(asset_paths[role])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != asset["file:size"]:
            raise AdmissionError(f"staged C1 {role} bytes are unavailable or changed")
        expected_hash = _multihash_sha256(asset["file:checksum"])
        actual_hash = _sha(path)
        if actual_hash != expected_hash:
            raise AdmissionError(f"staged C1 {role} SHA-256 differs from source multihash")
        declared = asset.get("raster:bands", [{}])[0]
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise AdmissionError(f"C1 {role} asset is not single-band")
            authority = dataset.crs.to_authority() if dataset.crs else None
            actual_shape = [dataset.height, dataset.width]
            actual_transform = list(dataset.transform)[:6]
            actual_resolution = [abs(float(dataset.res[0])), abs(float(dataset.res[1]))]
            actual_dtype = dataset.dtypes[0]
            actual_nodata = dataset.nodata
            expected_epsg = item["properties"]["proj:epsg"]
            if not dataset.crs or dataset.crs.to_epsg() != expected_epsg:
                raise AdmissionError(f"C1 {role} raster CRS differs from native item/grid identity")
            expected_shape = asset.get("proj:shape")
            expected_transform = asset.get("proj:transform")
            if list(expected_shape) != actual_shape:
                raise AdmissionError(f"C1 {role} raster shape differs from retained STAC metadata")
            if any(abs(float(a)-float(b)) > 1e-8 for a,b in zip(actual_transform, expected_transform[:6])):
                raise AdmissionError(f"C1 {role} raster transform differs from retained STAC metadata")
            if any(abs(actual-resolution) > 1e-8 for actual, resolution in
                   zip(actual_resolution, (EXPECTED_GSD[role], EXPECTED_GSD[role]))):
                raise AdmissionError(f"C1 {role} raster native resolution differs from role contract")
            if declared.get("data_type") and declared["data_type"] != actual_dtype:
                raise AdmissionError(f"C1 {role} raster type differs from retained STAC metadata")
            if declared.get("nodata") is not None and actual_nodata != declared["nodata"]:
                raise AdmissionError(f"C1 {role} nodata differs from retained STAC metadata")
            actual_grids[role] = {"crs_epsg": dataset.crs.to_epsg(), "shape": actual_shape,
                                  "transform": actual_transform, "resolution": actual_resolution,
                                  "extent": _grid_extent(actual_shape, actual_transform)}
            x, y = transform_coordinates("EPSG:4326", dataset.crs, [lon], [lat])
            row, col = dataset.index(x[0], y[0])
            if not 0 <= row < dataset.height or not 0 <= col < dataset.width:
                raise AdmissionError(f"query centroid is outside the selected C1 {role} tile")
            window = Window(col - 1, row - 1, 3, 3)
            raw = dataset.read(1, window=window, boundless=True, fill_value=actual_nodata or 0)
            masks = dataset.read_masks(1, window=window, boundless=True) > 0
            valid = masks & np.isfinite(raw)
            if actual_nodata is not None:
                valid &= raw != actual_nodata
            scale = float(dataset.scales[0]) if dataset.scales else 1.0
            offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
            if role != "scl":
                native = metadata["native_radiometry"][band]
                scale_offset_status = ("MATCHED_COG_TAGS_STAC_AND_NATIVE_XML" if
                                       abs(scale-native["expected_scale"]) <= 1e-12 and
                                       abs(offset-native["expected_offset"]) <= 1e-9 and
                                       abs(scale-float(declared["scale"])) <= 1e-12 and
                                       abs(offset-float(declared["offset"])) <= 1e-9 else
                                       "UNRESOLVED_COG_CONVERSION_ENCODING")
                if scale_offset_status == "MATCHED_COG_TAGS_STAC_AND_NATIVE_XML":
                    values = raw.astype(np.float64) * scale + offset
                    reflectance = [[float(values[i, j]) if bool(valid[i, j]) else None
                                    for j in range(values.shape[1])]
                                   for i in range(values.shape[0])]
                    if not valid.any():
                        reflectance_status = "NO_VALID_PIXELS"
                    elif not valid.all():
                        reflectance_status = "PARTIAL_INVALID_PIXELS_EXPLICITLY_NULL"
                    else:
                        reflectance_status = "VALID_PIXELS_ENCODING_VERIFIED"
                else:
                    reflectance, reflectance_status = None, "UNRESOLVED_COG_CONVERSION_ENCODING"
            else:
                scale_offset_status = "CATEGORICAL_SCL_RAW_CLASSES_NO_QA_POLICY"
                reflectance, reflectance_status = None, "NOT_APPLICABLE_CATEGORICAL_SCL"
            outputs[role] = {"band_identity": band, "source_href": asset["href"],
                "source_sha256": actual_hash, "bytes": path.stat().st_size,
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "crs_authority": list(authority) if authority else None,
                "shape": actual_shape, "transform": actual_transform,
                "resolution": actual_resolution, "dtype": actual_dtype, "nodata": actual_nodata,
                "scale": scale, "offset": offset, "scale_offset_status": scale_offset_status,
                "window_row_col": [row, col], "window_shape": list(raw.shape),
                "raw_values": raw.tolist(), "valid_mask": valid.tolist(),
                "reflectance_values": reflectance,
                "reflectance_status": reflectance_status,
                "qa_policy_status": "NOT_APPLIED" if role == "scl" else None}
    reference = actual_grids["green"]
    for role, grid in actual_grids.items():
        if grid["crs_epsg"] != reference["crs_epsg"]:
            raise AdmissionError(f"actual C1 {role} CRS differs from native bundle CRS")
        if any(abs(a-b) > 1e-6 for a, b in zip(grid["extent"], reference["extent"])):
            raise AdmissionError(f"actual C1 {role} pixel footprint differs across selected bundle")
    for left, right in (("green", "nir"), ("swir16", "scl")):
        if (actual_grids[left]["shape"] != actual_grids[right]["shape"] or
                actual_grids[left]["transform"] != actual_grids[right]["transform"]):
            raise AdmissionError(f"actual C1 {left}/{right} same-resolution grids differ")
    roi = None
    if roi_output_dir is not None:
        roi = _decode_c1_roi(item, site_id, asset_paths, metadata,
            {role: outputs[role]["source_sha256"] for role in outputs},
            latitude=lat, longitude=lon, origin=origin, output_dir=roi_output_dir,
            halfwidth_m=roi_halfwidth_m)
    return {"status": "AUTHENTIC_PROVIDER_COGS_INSPECTED_NOT_RESEARCH_ADMITTED"
            if origin == "provider_download" else "CONSTRUCTED_FIXTURE_INSPECTED_NOT_RESEARCH_ADMITTED",
            "origin": origin, "provider": "Earth Search sentinel-2-c1-l2a" if origin == "provider_download"
            else "constructed C1 engineering fixture",
            "site_id": site_id, "site_scope": "DEVELOPMENT_EXPOSED",
            "query_centroid_wgs84": [lon, lat], "lake_geometry_status": "NOT_VERIFIED_CENTROID_ONLY",
            "site_item_id": item["id"], "product_uri": metadata["product_uri"],
            "product_temporal_join": metadata, "assets": outputs,
            "collection_packaging": "DERIVED_COG_BUNDLE_NOT_ORIGINAL_SAFE_OR_JP2",
            "roi": roi,
            "research_admissible": False, "2023_operational_availability": "UNVERIFIED"}
