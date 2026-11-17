"""Bounded development catalogue discovery and original-product staging.

Catalogue metadata and verified archives do not establish calibrated pixels,
lake identity, historical availability, usable QA, or research readiness.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

from ..admission import AdmissionError, QUARANTINED
from ..integrity import file_hash, loads_strict

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TERMS = "https://dataspace.copernicus.eu/terms-and-conditions"
PRODUCTS = {"SENTINEL-1_GRD": ("SENTINEL-1", "_GRD"),
            "SENTINEL-2_MSI_L2A": ("SENTINEL-2", "_MSIL2A_")}


def utc(value):
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if time.tzinfo is None:
        raise AdmissionError("timestamp timezone is required")
    return time.astimezone(timezone.utc)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def replace_json(path, value):
    """Update a small in-attempt receipt without hiding the attempt history."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_bound(path, expected):
    if not re.fullmatch("[0-9a-f]{64}", expected) or path.is_symlink():
        raise AdmissionError("regular byte-bound input required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or expected in QUARANTINED:
        raise AdmissionError("input digest mismatch or quarantined bytes")
    return loads_strict(raw.decode())


def validate_manifest(manifest):
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("purpose") != "development_acquisition_only":
        raise AdmissionError("development manifest required")
    if manifest.get("research_executable") is not False or manifest.get("crs") != "EPSG:4326":
        raise AdmissionError("only nonresearch WGS84 acquisition is supported")
    start, end = utc(manifest["acquisition_start"]), utc(manifest["acquisition_end_exclusive"])
    if not start < end or (end - start).days > 62:
        raise AdmissionError("pilot acquisition interval must be positive and at most 62 days")
    utc(manifest["pre_event_cutoff_exclusive"])
    sites = manifest["sites"]
    if not isinstance(sites, list) or not 1 <= len(sites) <= 2:
        raise AdmissionError("one or two development sites required")
    identifiers = set()
    for site in sites:
        if not re.fullmatch("[A-Z0-9-]+", site["site_id"]) or site["site_id"] in identifiers:
            raise AdmissionError("unique safe site identity required")
        identifiers.add(site["site_id"])
        if site["scope"] != "DEVELOPMENT_EXPOSED":
            raise AdmissionError("pilot sites must remain development-exposed")
        for key, limit in (("latitude", 90), ("longitude", 180)):
            value = site[key]
            if type(value) not in (int, float) or not -limit <= value <= limit:
                raise AdmissionError("invalid WGS84 query centroid")
    if manifest["products"] != list(PRODUCTS):
        raise AdmissionError("declared Sentinel-1 GRD and Sentinel-2 L2A products required")
    return manifest


class ProviderRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if (target.scheme != "https" or target.username or target.password or
                target.hostname not in {"catalogue.dataspace.copernicus.eu",
                                        "download.dataspace.copernicus.eu",
                                        "zipper.dataspace.copernicus.eu"}):
            raise AdmissionError("unapproved provider redirect; credentials not forwarded")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def request(url, token=None):
    headers = {"User-Agent": "Sentinel-GL-Development/1.0"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return urllib.request.build_opener(ProviderRedirect()).open(
        urllib.request.Request(url, headers=headers), timeout=45)


def query_url(manifest, site, product):
    collection, marker = PRODUCTS[product]
    point = f"POINT({site['longitude']} {site['latitude']})"
    expression = (f"Collection/Name eq '{collection}' and contains(Name,'{marker}') and "
                  f"OData.CSC.Intersects(area=geography'SRID=4326;{point}') and "
                  f"ContentDate/Start ge {manifest['acquisition_start']} and "
                  f"ContentDate/Start lt {manifest['acquisition_end_exclusive']}")
    return CATALOGUE + "?" + urllib.parse.urlencode({
        "$filter": expression, "$top": 100, "$orderby": "ContentDate/Start asc",
        "$expand": "Attributes"})


def validate_product(row, manifest, product):
    identifier = str(uuid.UUID(row["Id"]))
    if identifier != row["Id"].lower():
        raise AdmissionError("noncanonical provider identifier")
    name = row["Name"]
    if (not re.fullmatch("[A-Za-z0-9_.-]+", name) or
            PRODUCTS[product][1] not in name or not name.endswith(".SAFE")):
        raise AdmissionError("provider product type/name mismatch")
    if (product == "SENTINEL-1_GRD" and not name.startswith("S1")) or (
            product == "SENTINEL-2_MSI_L2A" and not name.startswith("S2")):
        raise AdmissionError("provider mission mismatch")
    start, end = utc(row["ContentDate"]["Start"]), utc(row["ContentDate"]["End"])
    if not utc(manifest["acquisition_start"]) <= start < utc(manifest["acquisition_end_exclusive"]) or end < start:
        raise AdmissionError("provider acquisition is outside the requested interval")
    return identifier


def discover(manifest, output):
    """Retain exact response bytes, complete pagination and site/product joins."""
    products = {}
    queries = []
    (output / "responses").mkdir()
    for site in manifest["sites"]:
        for kind in manifest["products"]:
            url = query_url(manifest, site, kind)
            seen = set()
            pages = []
            while url:
                parsed = urllib.parse.urlsplit(url)
                if (parsed.scheme != "https" or parsed.netloc != "catalogue.dataspace.copernicus.eu" or
                        parsed.path != "/odata/v1/Products" or url in seen or len(seen) >= 100):
                    raise AdmissionError("unsafe, repeated or excessive catalogue pagination; discovery incomplete")
                seen.add(url)
                with request(url) as response:
                    raw = response.read(32 * 1024 * 1024 + 1)
                    if len(raw) > 32 * 1024 * 1024:
                        raise AdmissionError("catalogue response exceeds bound")
                path = output / "responses" / f"{site['site_id']}-{kind}-{len(seen):03d}.json"
                path.write_bytes(raw)
                page = loads_strict(raw.decode())
                if not isinstance(page.get("value"), list):
                    raise AdmissionError("provider catalogue omitted product rows")
                pages.append({"url": url, "path": str(path.relative_to(output)),
                              "sha256": file_hash(path), "retrieved_at": now()})
                for row in page["value"]:
                    identifier = validate_product(row, manifest, kind)
                    if identifier in products and products[identifier]["provider_record"] != row:
                        raise AdmissionError("conflicting duplicate provider product identity")
                    cutoff = utc(manifest["pre_event_cutoff_exclusive"])
                    item = products.setdefault(identifier, {
                        "product": kind, "provider_record": row, "sites": [],
                        "temporal_support": "STRICTLY_PRE_EVENT" if utc(row["ContentDate"]["End"]) < cutoff
                                            else "EVENT_OR_POST_EVENT_NOT_FOR_PRE_EVENT_DECISIONS"})
                    if site["site_id"] not in item["sites"]:
                        item["sites"].append(site["site_id"])
                url = page.get("@odata.nextLink")
                if url is not None and not isinstance(url, str):
                    raise AdmissionError("malformed next-page link")
            queries.append({"site_id": site["site_id"], "product": kind, "pages": pages,
                            "status": "CATALOGUE_QUERY_COMPLETE",
                            "measurement_status": "NO_PIXELS_ACQUIRED"})
    return {"status": "CATALOGUE_ONLY", "research_admissible": False,
            "manifest": manifest, "queries": queries, "products": products,
            "rights_source": TERMS, "roi_qa": None, "historical_availability": None}


def inspect_archive(path, name, product):
    """Check archive identity/CRC/structure, explicitly without pixel decoding."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or sum(i.file_size for i in entries) > 20 * 1024**3:
            raise AdmissionError("empty or excessive product archive")
        paths = []
        for entry in entries:
            p = PurePosixPath(entry.filename)
            if (p.is_absolute() or ".." in p.parts or "\\" in entry.filename or
                    not p.parts or p.parts[0] != name or
                    stat.S_ISLNK(entry.external_attr >> 16)):
                raise AdmissionError("unsafe or mismatched SAFE archive member")
            paths.append(entry.filename)
        if len(paths) != len(set(paths)) or archive.testzip() is not None:
            raise AdmissionError("duplicate or corrupt product archive")
        if product == "SENTINEL-2_MSI_L2A":
            required = ("MTD_MSIL2A.xml", "_B03_", "_B08_", "_B11_", "_SCL_")
            if not all(any(marker in p for p in paths) for marker in required):
                raise AdmissionError("L2A metadata/bands/SCL missing from archive")
            pixels = [p for p in paths if p.lower().endswith(".jp2")]
        else:
            if not any(p.endswith("/manifest.safe") for p in paths):
                raise AdmissionError("GRD manifest missing")
            pixels = [p for p in paths if "/measurement/" in p and p.lower().endswith((".tif", ".tiff"))]
        if not pixels:
            raise AdmissionError("product archive contains no raster assets")
    return {"archive_verification": "SAFE_IDENTITY_CRC_STRUCTURE_ONLY", "raster_assets": pixels,
            "pixel_verification": "NOT_DECODED", "units_qa_geometry": "PENDING_INDEPENDENT_DECODE"}


def provider_checksum_metadata(row):
    """Preserve catalogue checksums without assuming they bind to ``$value``."""
    declared = row.get("Checksum")
    if declared is None:
        return {}
    if not isinstance(declared, list):
        raise AdmissionError("provider checksum metadata is malformed")
    values = {}
    for item in declared:
        if not isinstance(item, dict):
            raise AdmissionError("provider checksum metadata is malformed")
        algorithm = item.get("Algorithm")
        value = item.get("Value")
        if not isinstance(algorithm, str) or not algorithm.strip() or not isinstance(value, str) or not value.strip():
            raise AdmissionError("provider checksum metadata is incomplete")
        algorithm = algorithm.strip().upper().replace("-", "")
        value = value.strip()
        if algorithm in {"MD5", "SHA256"} and not re.fullmatch(r"[0-9a-fA-F]+", value):
            raise AdmissionError("provider checksum metadata is malformed")
        if algorithm == "MD5" and len(value) != 32:
            raise AdmissionError("provider MD5 metadata is malformed")
        if algorithm == "SHA256" and len(value) != 64:
            raise AdmissionError("provider SHA-256 metadata is malformed")
        normalized = value.lower() if algorithm in {"MD5", "SHA256"} else value
        if algorithm in values and values[algorithm]["value"] != normalized:
            raise AdmissionError("provider returned conflicting checksums")
        values[algorithm] = {"value": normalized,
                             "checksum_date": item.get("ChecksumDate"),
                             "verification_status": "UNSUPPORTED_ALGORITHM" if algorithm not in {"MD5", "SHA256"}
                                                   else "NOT_BOUND_TO_DOWNLOAD_REPRESENTATION"}
    return values


def http_checksum_metadata(headers):
    """Decode only checksums carried by the exact HTTP representation response."""
    values = {}
    unsupported = []
    content_md5 = headers.get("Content-MD5")
    if content_md5 is not None:
        try:
            raw = base64.b64decode(content_md5.strip(), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise AdmissionError("HTTP Content-MD5 header is malformed") from exc
        if len(raw) != 16:
            raise AdmissionError("HTTP Content-MD5 header is malformed")
        values["MD5"] = raw.hex()
    digest = headers.get("Digest")
    if digest is not None:
        for component in digest.split(","):
            if "=" not in component:
                raise AdmissionError("HTTP Digest header is malformed")
            name, encoded = (part.strip() for part in component.split("=", 1))
            algorithm = name.upper().replace("-", "")
            if algorithm not in {"MD5", "SHA256"}:
                unsupported.append(name)
                continue
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise AdmissionError("HTTP Digest header is malformed") from exc
            if len(raw) != (16 if algorithm == "MD5" else 32):
                raise AdmissionError("HTTP Digest header is malformed")
            if algorithm in values and values[algorithm] != raw.hex():
                raise AdmissionError("HTTP response returned conflicting checksums")
            values[algorithm] = raw.hex()
    if unsupported:
        values["_unsupported_algorithms"] = unsupported
    if headers.get("Content-Digest") is not None:
        values.setdefault("_unsupported_algorithms", []).append("Content-Digest:UNPARSED")
    return values


def compare_provider_identity(current, discovered):
    """Compare stable identity and preserve optional provider metadata absence."""
    for key in ("Id", "Name", "ContentDate"):
        if key not in current or current[key] != discovered.get(key):
            raise AdmissionError("provider product changed since catalogue discovery")
    optional = {}
    for key in ("ContentLength", "Checksum"):
        if key not in discovered or discovered[key] is None:
            optional[key] = ("ABSENT_IN_DISCOVERY_CURRENT_PRESENT" if current.get(key) is not None
                             else "ABSENT_IN_DISCOVERY_AND_CURRENT")
        elif key not in current or current[key] is None:
            optional[key] = "ABSENT_IN_CURRENT_RECORD"
        elif current[key] != discovered[key]:
            raise AdmissionError("provider product changed since catalogue discovery")
        else:
            optional[key] = "PRESENT_BUT_REPRESENTATION_SCOPE_UNBOUND"
    return optional


def redacted_error(error, token=None):
    detail = str(error)
    if token:
        detail = detail.replace(token, "[REDACTED]")
    return detail


def partial_state(path, product_id, catalogue_bytes, checksums, status, error=None,
                  actual_bytes=None, sha256=None, md5=None, token=None,
                  catalogue_length_status="UNKNOWN", optional_identity=None):
    if path.is_file():
        actual_bytes = path.stat().st_size
        sha256 = file_hash(path)
        partial_md5 = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                partial_md5.update(chunk)
        md5 = partial_md5.hexdigest()
    state = {"product_id": product_id, "status": status,
             "catalogue_content_length": catalogue_bytes,
             "partial_bytes": actual_bytes or 0,
             "partial_sha256": sha256,
             "partial_md5": md5,
             "provider_checksums": checksums,
             "optional_identity_metadata": optional_identity or {},
             "catalogue_content_length_status": catalogue_length_status,
             "provider_checksum_status": "NOT_BOUND_TO_DOWNLOAD_REPRESENTATION",
             "resume_supported": False,
             "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED"}
    if error is not None:
        state["error_type"] = type(error).__name__
        state["error"] = redacted_error(error, token)
    return state


def stage(catalogue, manifest, output, identifiers, token, max_bytes):
    if not token:
        raise AdmissionError("CDSE_ACCESS_TOKEN is required; no credentials were recorded")
    if catalogue["manifest"] != manifest or catalogue.get("status") != "CATALOGUE_ONLY":
        raise AdmissionError("catalogue does not match the pinned development manifest")
    if not identifiers or len(identifiers) > 4 or len(set(identifiers)) != len(identifiers):
        raise AdmissionError("select one to four unique granules explicitly")
    staged = []
    for identifier in identifiers:
        item = catalogue["products"][identifier]
        row = item["provider_record"]
        validate_product(row, manifest, item["product"])
        checksums = provider_checksum_metadata(row)
        catalogue_bytes = row.get("ContentLength")
        if catalogue_bytes is not None and (type(catalogue_bytes) is not int or catalogue_bytes <= 0):
            raise AdmissionError("provider content length is malformed")
        catalogue_length_status = ("PRESENT_SCOPE_UNBOUND_TO_VALUE_RESPONSE" if catalogue_bytes is not None
                                   else "OPTIONAL_METADATA_ABSENT")
        # Authenticate the current provider identity independently of caller files.
        with request(CATALOGUE + "(" + str(uuid.UUID(identifier)) + ")") as response:
            current = loads_strict(response.read(32 * 1024 * 1024 + 1).decode())
        optional_identity = compare_provider_identity(current, row)
        path = output / (identifier + ".zip.partial")
        state_path = output / (identifier + ".download.json")
        replace_json(state_path, partial_state(path, identifier, catalogue_bytes, checksums,
                                                "DOWNLOAD_PENDING", token=token,
                                                catalogue_length_status=catalogue_length_status,
                                                optional_identity=optional_identity))
        sha_digest = hashlib.sha256()
        md5_digest = hashlib.md5(usedforsecurity=False)
        size = 0
        declared = None
        response_checksums = {}
        try:
            with request(DOWNLOAD + "(" + identifier + ")/$value", token) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared = int(declared)
                    except (TypeError, ValueError) as exc:
                        raise AdmissionError("provider response content length is malformed") from exc
                    if declared <= 0 or declared > max_bytes:
                        raise AdmissionError("provider response exceeds explicit byte bound")
                response_checksums = http_checksum_metadata(response.headers)
                replace_json(state_path, partial_state(path, identifier, catalogue_bytes, checksums,
                                                        "DOWNLOADING", token=token,
                                                        catalogue_length_status=catalogue_length_status,
                                                        optional_identity=optional_identity))
                with path.open("xb") as stream:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise AdmissionError("granule download exceeds explicit byte bound")
                        stream.write(chunk)
                        sha_digest.update(chunk)
                        md5_digest.update(chunk)
            if size == 0:
                raise AdmissionError("empty granule download")
            if declared is not None and size != declared:
                raise AdmissionError("download body is truncated relative to response content length")
        except (Exception, KeyboardInterrupt) as exc:
            replace_json(state_path, partial_state(path, identifier, catalogue_bytes, checksums,
                                                    "INTERRUPTED_OR_INCOMPLETE_RETAINED", exc,
                                                    size, sha_digest.hexdigest(), md5_digest.hexdigest(), token,
                                                    catalogue_length_status, optional_identity))
            raise

        sha = sha_digest.hexdigest()
        md5 = md5_digest.hexdigest()
        if sha in QUARANTINED:
            raise AdmissionError("quarantined product bytes")
        checksum_status = "HTTP_CONTENT_CHECKSUM_ABSENT"
        for algorithm, expected in response_checksums.items():
            if algorithm.startswith("_"):
                continue
            actual = md5 if algorithm == "MD5" else sha
            if actual != expected:
                replace_json(state_path, partial_state(path, identifier, catalogue_bytes, checksums,
                                                        "HTTP_CONTENT_CHECKSUM_MISMATCH", None,
                                                        size, sha, md5, token, catalogue_length_status,
                                                        optional_identity))
                raise AdmissionError("download bytes do not match HTTP response checksum")
            checksum_status = "MATCHED_HTTP_CONTENT_CHECKSUM"
        try:
            inspection = inspect_archive(path, row["Name"], item["product"])
        except (AdmissionError, zipfile.BadZipFile) as exc:
            replace_json(state_path, partial_state(path, identifier, catalogue_bytes, checksums,
                                                    "ARCHIVE_STRUCTURE_REJECTED", exc,
                                                    size, sha, md5, token, catalogue_length_status,
                                                    optional_identity))
            raise
        final = output / (identifier + ".zip")
        path.rename(final)
        replace_json(state_path, {"product_id": identifier,
                                  "status": "STAGED_ARCHIVE_STRUCTURE_ONLY",
                                  "staged_path": final.name, "bytes": size,
                                  "sha256": sha, "local_md5": md5,
                                  "provider_checksums": checksums,
                                  "provider_checksum_status": "NOT_BOUND_TO_DOWNLOAD_REPRESENTATION",
                                  "http_checksum_status": checksum_status,
                                  "http_content_length_status": "MATCHED_BODY_LENGTH" if declared is not None else "HEADER_ABSENT",
                                  "http_checksum_unsupported_algorithms": response_checksums.get("_unsupported_algorithms", []),
                                  "catalogue_content_length": catalogue_bytes,
                                  "catalogue_content_length_status": catalogue_length_status,
                                  "optional_identity_metadata": optional_identity,
                                  "resume_supported": False,
                                  "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED",
                                  "pixel_verification": "NOT_DECODED"})
        staged.append({"product_id": identifier, "name": row["Name"], "sites": item["sites"],
                       "origin": "provider_download", "path": final.name, "sha256": sha,
                       "bytes": size, "retrieved_at": now(), **inspection,
                       "local_md5": md5, "provider_checksums": checksums,
                       "provider_checksum_status": "NOT_BOUND_TO_DOWNLOAD_REPRESENTATION",
                       "http_checksum_status": checksum_status,
                       "http_checksum_unsupported_algorithms": response_checksums.get("_unsupported_algorithms", []),
                       "catalogue_content_length": catalogue_bytes,
                       "catalogue_content_length_status": catalogue_length_status,
                       "optional_identity_metadata": optional_identity,
                       "resume_supported": False,
                       "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED",
                       "research_admissible": False})
    return {"status": "ARCHIVES_STAGED_PIXEL_DECODE_PENDING", "staged": staged,
            "research_admissible": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("catalogue", "stage"), default="catalogue")
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--catalogue-sha256")
    parser.add_argument("--product-id", action="append", default=[])
    parser.add_argument("--max-product-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args(argv)
    started = now()
    created_output = False
    try:
        manifest = validate_manifest(read_bound(args.manifest, args.manifest_sha256))
        if not 1 <= args.max_product_bytes <= 4 * 1024**3:
            raise AdmissionError("product bound must be positive and at most 4 GiB")
        if args.mode == "stage" and (not args.catalogue or not args.catalogue_sha256):
            raise AdmissionError("staging requires a byte-bound catalogue")
        catalogue = read_bound(args.catalogue, args.catalogue_sha256) if args.mode == "stage" else None
        out = args.output_dir.absolute()
        if any(p.is_symlink() for p in (out, *out.parents)) or out.exists():
            raise AdmissionError("new nonsymlink output directory required; preserve previous attempts")
        out.mkdir(parents=True)
        created_output = True
        write(out / "input_manifest.json", manifest)
        result = discover(manifest, out) if args.mode == "catalogue" else stage(
            catalogue, manifest, out, args.product_id, os.environ.get("CDSE_ACCESS_TOKEN"), args.max_product_bytes)
        if file_hash(args.manifest) != args.manifest_sha256:
            raise AdmissionError("input manifest changed during execution")
        if args.mode == "stage" and file_hash(args.catalogue) != args.catalogue_sha256:
            raise AdmissionError("catalogue changed during execution")
        write(out / "result.json", result)
        write(out / "execution.json", {"started_at": started, "finished_at": now(), "exit": 0,
              "manifest_sha256": args.manifest_sha256, "catalogue_sha256": args.catalogue_sha256,
              "result_sha256": file_hash(out / "result.json"), "scope": "development acquisition only",
              "research_admissible": False, "cpu_memory": None,
              "runtime": {"python": platform.python_version(), "executable": sys.executable,
                          "executable_sha256": file_hash(Path(sys.executable).resolve()),
                          "adapter_sha256": file_hash(Path(__file__)),
                          "dependency_scope": "source/requirements.lock; development pixel decoder imports Rasterio, NumPy and defusedxml",
                          "seed": None, "seed_reason": "NO_STOCHASTIC_OPERATION"}})
        print(json.dumps({"status": result["status"], "output_dir": str(out), "research_admissible": False}))
        return 0
    except (AdmissionError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        token = os.environ.get("CDSE_ACCESS_TOKEN", "")
        detail = redacted_error(exc, token)
        failure = {"status": "ACQUISITION_FAILED", "error_type": type(exc).__name__,
                   "error": detail, "started_at": started, "finished_at": now(),
                   "research_admissible": False}
        if created_output and not (out / "failure.json").exists():
            write(out / "failure.json", failure)
        print(json.dumps(failure))
        return 2
