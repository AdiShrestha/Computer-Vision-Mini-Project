"""Guarded, quarantined staging for one explicitly selected CDSE S1 granule."""
from __future__ import annotations

import hashlib
import base64
import binascii
from pathlib import Path
import re
import stat
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request
import zipfile

from ..errors import AdmissionError

CATALOGUE_HOST = "catalogue.dataspace.copernicus.eu"
DOWNLOAD_HOST = "download.dataspace.copernicus.eu"
ALLOWED_HOSTS = frozenset({CATALOGUE_HOST, DOWNLOAD_HOST})
PRODUCT_NAME = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F.SAFE"
PRODUCT_UUID = "1579285a-a3dd-4b77-b2e7-4b3654f12b56"
SITES = {
    "PILOT-SOUTH-LHONAK": (88.204, 27.915),
    "PILOT-KHANGCHUNG-TSHO": (88.752, 27.985),
}


class StreamIntegrityError(AdmissionError):
    def __init__(self, message, *, received_bytes, written_bytes, sha256, local_md5,
                 received_sha256=None):
        super().__init__(message)
        self.received_bytes = received_bytes
        self.written_bytes = written_bytes
        self.sha256 = sha256
        self.local_md5 = local_md5
        self.received_sha256 = received_sha256 or sha256


def validate_cdse_url(url: str, expected_host: str | None = None) -> str:
    """Fail closed on scheme, host, userinfo, or nonstandard port."""
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or
            parts.port not in (None, 443) or parts.username or parts.password or
            parts.fragment or (expected_host is not None and parts.hostname != expected_host)):
        raise AdmissionError("CDSE URL is outside the exact HTTPS host allowlist")
    return parts.hostname


class NoRedirect(HTTPRedirectHandler):
    """Surface redirects as HTTP responses; never contact the Location target."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_for(url: str, method: str, *, token: str | None = None, extra_headers=None):
    host = validate_cdse_url(url)
    if method not in {"GET", "HEAD"}:
        raise AdmissionError("unsupported CDSE request method")
    headers = {"Accept": "application/json" if host == CATALOGUE_HOST else "application/octet-stream"}
    if token is not None:
        if not token:
            raise AdmissionError("empty credential cannot be attached")
        if host != DOWNLOAD_HOST:
            raise AdmissionError("credentials may only be attached to the approved download host")
        headers["Authorization"] = "Bearer " + token
    extra_headers = extra_headers or {}
    if extra_headers and not (host == DOWNLOAD_HOST and method == "GET" and
                              extra_headers == {"Range": "bytes=0-0"}):
        raise AdmissionError("only the exact one-byte download range header is supported")
    headers.update(extra_headers)
    return Request(url, method=method, headers=headers)


def point_in_ring(point, ring) -> bool:
    """Ray-casting point test for a GeoJSON linear ring."""
    x, y = point
    inside = False
    if not isinstance(ring, list) or len(ring) < 4:
        return False
    for index, current in enumerate(ring):
        following = ring[(index + 1) % len(ring)]
        x1, y1 = current[:2]
        x2, y2 = following[:2]
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def footprint_contains(footprint, point) -> bool:
    if not isinstance(footprint, dict) or footprint.get("type") != "Polygon":
        return False
    rings = footprint.get("coordinates")
    if not isinstance(rings, list) or not rings or not point_in_ring(point, rings[0]):
        return False
    return not any(point_in_ring(point, hole) for hole in rings[1:])


def validate_product_record(row: dict) -> dict:
    """Validate exact S1 identity and report scene-footprint point coverage only."""
    if not isinstance(row, dict):
        raise AdmissionError("CDSE metadata row must be an object")
    if row.get("Id") != PRODUCT_UUID or row.get("Name") != PRODUCT_NAME:
        raise AdmissionError("fresh CDSE record does not match the selected product identity")
    start = row.get("ContentDate", {}).get("Start")
    end = row.get("ContentDate", {}).get("End")
    if not isinstance(start, str) or not start.startswith("2023-09-") or not isinstance(end, str) or not end.startswith("2023-09-"):
        raise AdmissionError("selected product is outside the September 2023 acquisition interval")
    if row.get("Online") is not True:
        raise AdmissionError("selected product is not currently reported online")
    size = row.get("ContentLength")
    if type(size) is not int or size <= 0:
        raise AdmissionError("catalogue native ContentLength is absent or invalid")
    attrs = row.get("Attributes")
    if not isinstance(attrs, list):
        raise AdmissionError("selected product attributes are absent")
    values = {item.get("Name"): item.get("Value") for item in attrs if isinstance(item, dict)}
    required = {"platformSerialIdentifier": "A", "operationalMode": "IW",
                "productType": "IW_GRDH_1S", "polarisationChannels": "VV&VH",
                "orbitNumber": 50134}
    if any(values.get(key) != value for key, value in required.items()):
        raise AdmissionError("selected product mission, mode, polarization, type, or orbit differs")
    coverage = {site: footprint_contains(row.get("GeoFootprint"), point)
                for site, point in SITES.items()}
    checksums = row.get("Checksum")
    if not isinstance(checksums, list) or not checksums:
        raise AdmissionError("selected product provider checksum metadata is absent")
    return {
        "uuid": PRODUCT_UUID, "name": PRODUCT_NAME,
        "product_type": values["productType"], "platform": "S1A",
        "mode": "IW", "polarizations": values["polarisationChannels"],
        "orbit_number": values["orbitNumber"], "orbit_direction": values.get("orbitDirection"),
        "processor_version": values.get("processorVersion"),
        "content_date": row["ContentDate"],
        "online": row["Online"],
        "catalogue_content_length_bytes": size,
        "provider_checksums": checksums,
        "footprint_wgs84": row.get("GeoFootprint"),
        "scene_footprint_contains_query_centroids": coverage,
        "footprint_scope": "METADATA_SCENE_COVERAGE_ONLY_NOT_LAKE_OR_PIXEL_QA",
    }


def validate_download_head(status: int, headers, remaining_budget: int) -> dict:
    """Require a successful bounded HEAD and reserve the exact HTTP representation length."""
    if status != 200:
        raise AdmissionError(f"download authorization HEAD returned HTTP {status}")
    raw_length = headers.get("Content-Length")
    if raw_length is None or not re.fullmatch(r"[0-9]+", raw_length.strip()):
        raise AdmissionError("download HEAD lacks a valid representation Content-Length")
    length = int(raw_length)
    if length <= 0 or length > remaining_budget:
        raise AdmissionError("download representation exceeds the remaining byte budget")
    return {"download_representation_content_length_bytes": length,
            "download_representation_content_type": headers.get("Content-Type"),
            "catalogue_length_comparison": "RECORDED_SEPARATELY_NOT_ASSUMED_EQUAL"}


def validate_range_probe(status: int, headers, received_body_bytes: int, remaining_budget: int) -> dict:
    """Accept only a one-byte 206 that declares the complete representation size."""
    if status != 206:
        raise AdmissionError(f"download range probe did not return HTTP 206 (received {status})")
    content_range = headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes 0-0/([0-9]+)", content_range.strip())
    if not match:
        raise AdmissionError("download range probe Content-Range is absent or does not describe byte zero")
    content_length = headers.get("Content-Length")
    if content_length != "1" or received_body_bytes != 1:
        raise AdmissionError("download range probe did not return exactly one byte")
    total = int(match.group(1))
    if total <= 0 or total + 1 > remaining_budget:
        raise AdmissionError("download representation plus EOF guard exceeds remaining byte budget")
    return {"download_representation_content_length_bytes": total,
            "range_probe_bytes": 1,
            "range_probe_status": "ONE_BYTE_206_TOTAL_LENGTH_VERIFIED",
            "content_type": headers.get("Content-Type")}


def safe_archive_check(path: Path, expected_name: str) -> dict:
    """Verify ZIP CRC and SAFE member identity without decoding pixels."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or sum(entry.file_size for entry in entries) > 20 * 1024**3:
            raise AdmissionError("downloaded product archive is empty or exceeds the structure bound")
        names = []
        for entry in entries:
            parts = entry.filename.split("/")
            if (entry.filename.startswith("/") or ".." in parts or "\\" in entry.filename or
                    not parts or parts[0] != expected_name or stat.S_ISLNK(entry.external_attr >> 16)):
                raise AdmissionError("downloaded archive has an unsafe or mismatched SAFE member")
            names.append(entry.filename)
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise AdmissionError("downloaded SAFE has duplicate members or failed CRC validation")
        if expected_name + "/manifest.safe" not in names:
            raise AdmissionError("downloaded Sentinel-1 SAFE lacks manifest.safe")
        measurements = [name for name in names if "/measurement/" in name and name.lower().endswith((".tif", ".tiff"))]
        if not measurements:
            raise AdmissionError("downloaded Sentinel-1 SAFE lacks measurement rasters")
    return {"archive_identity": "MATCHED_SELECTED_SAFE_NAME",
            "archive_crc": "ALL_MEMBERS_PASSED",
            "measurement_members": measurements,
            "pixel_calibration_or_quality": "NOT_VERIFIED",
            "research_admissible": False}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stream_response_to_partial(response, path: Path, expected_length: int | None,
                               byte_reservation: int):
    """Stream one GET under a hard cap, verifying declared length when available and EOF."""
    if type(byte_reservation) is not int or byte_reservation < 2:
        raise AdmissionError("byte reservation must include payload capacity and one EOF guard byte")
    max_payload = byte_reservation - 1
    if expected_length is not None and (type(expected_length) is not int or expected_length <= 0):
        raise AdmissionError("declared download representation length is invalid")
    if path.exists():
        raise AdmissionError("staging target already exists; preserve prior attempt")
    response_length = response.headers.get("Content-Length")
    if response_length is not None:
        if not re.fullmatch(r"[0-9]+", response_length.strip()):
            raise AdmissionError("download response Content-Length is malformed")
        header_length = int(response_length)
        if header_length <= 0 or header_length > max_payload:
            raise AdmissionError("download representation exceeds the reserved payload bound")
        if expected_length is not None and header_length != expected_length:
            raise AdmissionError("GET representation length differs from the expected response length")
        expected_length = header_length
    elif expected_length is not None:
        raise AdmissionError("expected response length was supplied but Content-Length is absent")
    if expected_length is not None and expected_length > max_payload:
        raise AdmissionError("download representation exceeds the reserved payload bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    received_digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    consumed = 0
    written = 0
    extra = b""
    try:
        with path.open("xb") as stream:
            while expected_length is None or written < expected_length:
                remaining = max_payload - written
                if expected_length is not None:
                    remaining = min(remaining, expected_length - written)
                if remaining <= 0:
                    break
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                consumed += len(chunk)
                received_digest.update(chunk)
                if consumed > max_payload:
                    raise AdmissionError("stream exceeded reserved payload bytes")
                stream.write(chunk)
                written += len(chunk)
                digest.update(chunk)
                md5.update(chunk)
            extra = response.read(1)
            if extra:
                consumed += len(extra)
                received_digest.update(extra)
                stream.write(extra)
                written += len(extra)
                digest.update(extra)
                md5.update(extra)
                message = ("download body exceeded declared Content-Length" if expected_length is not None
                           else "download body exceeded reserved payload bound")
                raise AdmissionError(message)
    except Exception as exc:
        if isinstance(exc, StreamIntegrityError):
            raise
        raise StreamIntegrityError(
            str(exc), received_bytes=consumed, written_bytes=written,
            sha256=digest.hexdigest(), local_md5=md5.hexdigest(),
            received_sha256=received_digest.hexdigest()) from exc
    eof_confirmed = (expected_length is None or written == expected_length) and not extra
    if not eof_confirmed:
        raise StreamIntegrityError(
            "download body was truncated before declared Content-Length and EOF",
            received_bytes=consumed, written_bytes=written,
            sha256=digest.hexdigest(), local_md5=md5.hexdigest(),
            received_sha256=received_digest.hexdigest())
    result = {
        "received_payload_bytes": consumed,
        "written_payload_bytes": written,
        "eof_confirmed": eof_confirmed,
        "expected_content_length_bytes": expected_length,
        "content_length_header_status": "PRESENT" if response_length is not None else "ABSENT",
        "sha256": digest.hexdigest(),
        "local_md5": md5.hexdigest(),
        "content_md5_status": "NOT_PRESENT",
    }
    content_md5 = response.headers.get("Content-MD5")
    if content_md5 is not None:
        try:
            expected_md5 = base64.b64decode(content_md5.strip(), validate=True).hex()
        except (ValueError, binascii.Error) as exc:
            raise StreamIntegrityError(
                "HTTP Content-MD5 header is malformed", received_bytes=consumed,
                written_bytes=written, sha256=digest.hexdigest(), local_md5=md5.hexdigest(),
                received_sha256=received_digest.hexdigest()) from exc
        if expected_md5 != result["local_md5"]:
            raise StreamIntegrityError(
                "download bytes do not match HTTP Content-MD5", received_bytes=consumed,
                written_bytes=written, sha256=digest.hexdigest(), local_md5=md5.hexdigest(),
                received_sha256=received_digest.hexdigest())
        result["content_md5_status"] = "MATCHED_HTTP_CONTENT_MD5"
    return result
