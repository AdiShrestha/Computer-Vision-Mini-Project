"""Bounded ASF SearchAPI product staging using an existing Earthdata .netrc.

No credential value is emitted, no archive is resumed, and authentication is
restricted to the Earthdata Login host named by the local .netrc.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import http.cookiejar
import json
import netrc
import os
import platform
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from defusedxml import ElementTree as SafeET

from ..errors import AdmissionError

ASF_HOST = "datapool.asf.alaska.edu"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
ALLOWED_REDIRECT_HOSTS = {ASF_HOST, EARTHDATA_HOST}
MAX_PRODUCT_BYTES = 2 * 1024**3


def _now():
    return datetime.now(timezone.utc).isoformat()


def _runtime_identity():
    import sys
    return {"implementation": platform.python_implementation(),
            "python_version": platform.python_version(), "executable": sys.executable,
            "adapter": "sentinel_gl.ingestion.asf_grd"}


def _safe_literal_error(exc):
    message = str(exc)
    message = re.sub(r"(https?://)[^/@\s:]+:[^/@\s]+@", r"\1[REDACTED]@", message)
    message = re.sub(r"(https?://[^\s?#]+)\?[^\s#]*", r"\1?[REDACTED]", message)
    message = re.sub(r"(?i)(authorization|cookie|token|password|credential)(\s*[:=]\s*)[^\s,;]+",
                     r"\1\2[REDACTED]", message)
    return message


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _sha_md5(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    size = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            sha.update(chunk)
            md5.update(chunk)
    return size, sha.hexdigest(), md5.hexdigest()


def _earthdata_opener(netrc_path=None):
    path = Path(netrc_path) if netrc_path else Path.home() / ".netrc"
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise AdmissionError("existing Earthdata .netrc must be a regular mode-0600 file")
        auth_file = netrc.netrc(str(path))
        auth = auth_file.authenticators(EARTHDATA_HOST)
    except AdmissionError:
        raise
    except Exception as exc:
        raise AdmissionError("existing Earthdata .netrc could not be read safely") from exc
    if not auth or not auth[0] or not auth[2]:
        raise AdmissionError("existing .netrc lacks an Earthdata Login host entry")
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, f"https://{EARTHDATA_HOST}", auth[0], auth[2])
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        urllib.request.HTTPBasicAuthHandler(manager),
        _ASFSafeRedirect())


class _ASFSafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_REDIRECT_HOSTS or
                parsed.username or parsed.password):
            raise AdmissionError("ASF/Earthdata redirect host is outside the approved authentication flow")
        # urllib's redirect handler removes Authorization on a cross-host hop;
        # the password manager can answer only for urs.earthdata.nasa.gov.
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _validate_feature(feature):
    props = feature.get("properties", {})
    url = props.get("url", "")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != ASF_HOST or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise AdmissionError("ASF product URL is outside the approved HTTPS host")
    if Path(parsed.path).name != props.get("fileName") or not str(props.get("fileName", "")).endswith(".zip"):
        raise AdmissionError("ASF URL path does not match retained product filename")
    if props.get("sceneName") + "-GRD_HD" != props.get("fileID"):
        raise AdmissionError("ASF scene/file identity does not match retained feature")
    size = props.get("bytes")
    checksum = props.get("md5sum")
    if type(size) is not int or not 0 < size <= MAX_PRODUCT_BYTES:
        raise AdmissionError("ASF product exceeds the declared 2-GiB per-product bound")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", checksum):
        raise AdmissionError("ASF catalogue MD5 is missing or malformed")
    start = datetime.fromisoformat(props["startTime"].replace("Z", "+00:00"))
    stop = datetime.fromisoformat(props["stopTime"].replace("Z", "+00:00"))
    if start.tzinfo is None or stop.tzinfo is None or stop < start:
        raise AdmissionError("ASF acquisition time support is missing or reversed")
    return props


def stage_feature(feature, site_joins, destination, *, netrc_path=None, context=None):
    """Stage one retained ASF feature; an auth denial is explicit, not retried."""
    props = _validate_feature(feature)
    if not site_joins or not set(site_joins) <= {"PILOT-SOUTH-LHONAK", "PILOT-KHANGCHUNG-TSHO"}:
        raise AdmissionError("ASF site joins must use the two declared development sites")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    receipt_path = target.with_name(target.name + ".download.json")
    if target.exists() or partial.exists() or receipt_path.exists():
        raise AdmissionError("existing ASF attempts are immutable; select a new attempt directory")
    receipt = {"origin": "provider_download", "provider": "Alaska Satellite Facility SearchAPI",
               "fileID": props["fileID"], "sceneName": props["sceneName"],
               "site_joins": sorted(set(site_joins)), "url": props["url"],
               "expected_bytes": props["bytes"], "catalogue_md5": props["md5sum"].lower(),
               "catalogue_checksum_scope": "ASF feature md5sum for named product ZIP; verify against received ZIP bytes",
               "polarization": props.get("polarization"), "startTime": props["startTime"],
               "stopTime": props["stopTime"], "resume_supported": False,
               "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED", "status": "REQUEST_PENDING",
               "request_started_utc": _now(), "runtime": _runtime_identity(),
               "selection_id": (context or {}).get("selection_id"),
               "selection_sha256": (context or {}).get("selection_sha256"),
               "site_scope_id": (context or {}).get("site_scope_id")}
    try:
        opener = _earthdata_opener(netrc_path)
    except AdmissionError as exc:
        receipt.update({"status": "AUTH_CONFIGURATION_UNAVAILABLE", "prerequisite": str(exc)})
        receipt["request_finished_utc"] = _now()
        _write_json(receipt_path, receipt)
        return receipt
    request = urllib.request.Request(props["url"], headers={
        "User-Agent": "Sentinel-GL/1.0", "Accept-Encoding": "identity"})
    received, sha, md5 = 0, hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    response_length = response_etag = None
    try:
        with opener.open(request, timeout=90) as response:
            final = urllib.parse.urlsplit(response.geturl())
            if (response.status != 200 or final.scheme != "https" or
                    final.hostname not in ALLOWED_REDIRECT_HOSTS or final.hostname != ASF_HOST):
                raise AdmissionError("ASF final response is not the expected HTTPS product host")
            response_length = response.headers.get("Content-Length")
            response_etag = response.headers.get("ETag")
            if response_length is not None and int(response_length) != props["bytes"]:
                raise AdmissionError("ASF GET Content-Length differs from retained product bytes")
            with partial.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    if received + len(chunk) > props["bytes"] or received + len(chunk) > MAX_PRODUCT_BYTES:
                        raise AdmissionError("ASF response exceeds its retained product byte bound")
                    stream.write(chunk)
                    received += len(chunk)
                    sha.update(chunk)
                    md5.update(chunk)
                stream.flush()
        actual_sha, actual_md5 = sha.hexdigest(), md5.hexdigest()
        if received != props["bytes"] or actual_md5 != props["md5sum"].lower():
            partial_size, partial_sha, partial_md5 = _sha_md5(partial)
            receipt.update({"status": "LENGTH_OR_CATALOGUE_MD5_MISMATCH", "received_bytes": partial_size,
                            "partial_sha256": partial_sha, "computed_md5": partial_md5,
                            "response_content_length": response_length,
                            "response_etag_opaque": response_etag,
                            "request_finished_utc": _now()})
            _write_json(receipt_path, receipt)
            raise AdmissionError("ASF ZIP length or feature MD5 differs from received bytes")
        partial.replace(target)
        receipt.update({"status": "COMPLETE_ARCHIVE_BYTES_VERIFIED", "received_bytes": received,
                        "sha256": actual_sha, "computed_md5": actual_md5,
                        "output_bytes": received, "output_sha256": actual_sha,
                        "response_content_length": response_length, "response_etag_opaque": response_etag,
                        "etag_scope": "OPAQUE_IDENTITY_ONLY_NOT_INTERPRETED_AS_CHECKSUM",
                        "resume_status": "UNSUPPORTED_NEW_ATTEMPT_REQUIRED",
                        "request_finished_utc": _now()})
        _write_json(receipt_path, receipt)
        return receipt
    except urllib.error.HTTPError as exc:
        partial_size, partial_sha, partial_md5 = _sha_md5(partial) if partial.exists() else (0, None, None)
        receipt.update({"status": "AUTHENTICATION_OR_AUTHORIZATION_DENIED" if exc.code in (401, 403) else "HTTP_ERROR",
                        "http_status": exc.code, "literal_error": _safe_literal_error(exc),
                        "received_bytes": partial_size, "partial_sha256": partial_sha,
                        "partial_md5": partial_md5, "request_finished_utc": _now()})
        _write_json(receipt_path, receipt)
        if exc.code in (401, 403):
            return receipt
        raise
    except Exception as exc:
        partial_size, partial_sha, partial_md5 = _sha_md5(partial) if partial.exists() else (0, None, None)
        if receipt.get("status") == "REQUEST_PENDING":
            receipt.update({"status": "INTERRUPTED_OR_INCOMPLETE_RETAINED",
                            "error_type": type(exc).__name__, "literal_error": _safe_literal_error(exc),
                            "received_bytes": partial_size, "partial_sha256": partial_sha,
                            "partial_md5": partial_md5, "request_finished_utc": _now(),
                            "response_content_length": response_length,
                            "response_etag_opaque": response_etag})
            _write_json(receipt_path, receipt)
        else:
            receipt.setdefault("request_finished_utc", _now())
        raise


def _direct_text(root, name, label):
    values = [node.text.strip() for node in root
              if node.tag.rsplit("}", 1)[-1] == name and node.text and node.text.strip()]
    if len(values) != 1:
        raise AdmissionError(f"{label} is absent or ambiguous in scoped SAFE metadata")
    return values[0]


def _metadata_object_xml_data(root, object_id):
    objects = [node for node in root.iter()
               if node.tag.rsplit("}", 1)[-1] == "metadataObject" and
               node.attrib.get("ID") == object_id]
    if len(objects) != 1:
        raise AdmissionError(f"SAFE manifest metadata object {object_id} is absent or ambiguous")
    wraps = [node for node in objects[0]
             if node.tag.rsplit("}", 1)[-1] == "metadataWrap"]
    if len(wraps) != 1:
        raise AdmissionError(f"SAFE manifest metadata object {object_id} has no unique metadataWrap")
    xml_data = [node for node in wraps[0]
                if node.tag.rsplit("}", 1)[-1] == "xmlData"]
    if len(xml_data) != 1:
        raise AdmissionError(f"SAFE manifest metadata object {object_id} has no unique xmlData")
    return xml_data[0]


def _product_scene_parts(scene_name):
    parts = scene_name.split("_")
    if (len(parts) != 9 or not re.fullmatch(r"S1[A-Z]", parts[0]) or
            not re.fullmatch(r"[A-Z]{2}", parts[1]) or
            not re.fullmatch(r"GRD[HMF]", parts[2]) or
            parts[3] not in {"1SDV", "1SDH", "1SSV", "1SSH"} or
            not re.fullmatch(r"\d{8}T\d{6}", parts[4]) or
            not re.fullmatch(r"\d{8}T\d{6}", parts[5]) or
            not re.fullmatch(r"\d{6}", parts[6]) or
            not re.fullmatch(r"\d{6}", parts[7]) or
            not re.fullmatch(r"[0-9A-F]{4}", parts[8])):
        raise AdmissionError("selected SAFE scene name lacks supported Sentinel-1 identity fields")
    return parts


def _scene_second_token(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise AdmissionError("SAFE acquisition time is malformed") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y%m%dT%H%M%S")


def _measurement_stem_matches(member, prefix):
    stem = PurePosixPath(member).stem.lower()
    return bool(re.fullmatch(re.escape(prefix.lower()) + r"-\d{3}", stem))


def _role_stem_matches(member, marker, measurement_stem):
    stem = PurePosixPath(member).stem.lower()
    return stem.removeprefix(marker.lower()) == measurement_stem.lower()


def _xml_single_value(root, name, label):
    values = {node.text.strip() for node in root.iter()
              if node.tag.rsplit("}", 1)[-1].lower() == name.lower() and
              node.text and node.text.strip()}
    if len(values) != 1:
        raise AdmissionError(f"ASF {label} {name} is absent or conflicting")
    return next(iter(values))


def _validate_ads_header(root, *, mission, product_type, mode, polarization,
                         start_token, stop_token, role):
    if (_xml_single_value(root, "missionId", role) != mission or
            _xml_single_value(root, "productType", role).upper() != product_type or
            _xml_single_value(root, "mode", role).upper() != mode or
            _xml_single_value(root, "polarisation", role).upper() != polarization.upper() or
            _scene_second_token(_xml_single_value(root, "startTime", role)) != start_token or
            _scene_second_token(_xml_single_value(root, "stopTime", role)) != stop_token):
        raise AdmissionError(f"ASF {role} mission/product/time/mode/polarization differs from selected scene")


def _scoped_manifest_identity(manifest_root, scene_name):
    parts = _product_scene_parts(scene_name)
    mission, mode, product_code, pol_code, start_token, stop_token, orbit, datatake, _ = parts
    product_type = product_code[:3]
    product_polarizations = {
        "1SDV": {"VV", "VH"}, "1SDH": {"HH", "HV"},
        "1SSV": {"VV"}, "1SSH": {"HH"},
    }[pol_code]

    platform_data = _metadata_object_xml_data(manifest_root, "platform")
    platform_nodes = [node for node in platform_data
                      if node.tag.rsplit("}", 1)[-1] == "platform"]
    if len(platform_nodes) != 1:
        raise AdmissionError("SAFE manifest platform scope is absent or ambiguous")
    platform = platform_nodes[0]
    platform_family = _direct_text(platform, "familyName", "platform familyName")
    platform_number = _direct_text(platform, "number", "platform number")
    instruments = [node for node in platform if node.tag.rsplit("}", 1)[-1] == "instrument"]
    if len(instruments) != 1:
        raise AdmissionError("SAFE manifest instrument scope is absent or ambiguous")
    instrument = instruments[0]
    instrument_family = _direct_text(instrument, "familyName", "instrument familyName")
    mode_nodes = [node for node in instrument.iter()
                  if node.tag.rsplit("}", 1)[-1] == "instrumentMode"]
    if len(mode_nodes) != 1:
        raise AdmissionError("SAFE manifest instrumentMode scope is absent or ambiguous")
    manifest_mode = _direct_text(mode_nodes[0], "mode", "instrument mode")

    product_data = _metadata_object_xml_data(manifest_root, "generalProductInformation")
    standalone = [node for node in product_data
                  if node.tag.rsplit("}", 1)[-1] == "standAloneProductInformation"]
    if len(standalone) != 1:
        raise AdmissionError("SAFE manifest standAloneProductInformation scope is absent or ambiguous")
    product = standalone[0]
    manifest_product_type = _direct_text(product, "productType", "product type").upper()
    polarization_values = {node.text.strip().upper() for node in product
                           if node.tag.rsplit("}", 1)[-1] == "transmitterReceiverPolarisation" and
                           node.text and node.text.strip()}
    if polarization_values != product_polarizations:
        raise AdmissionError("SAFE manifest polarization scope differs from selected product code")

    acquisition_data = _metadata_object_xml_data(manifest_root, "acquisitionPeriod")
    periods = [node for node in acquisition_data
               if node.tag.rsplit("}", 1)[-1] == "acquisitionPeriod"]
    if len(periods) != 1:
        raise AdmissionError("SAFE manifest acquisitionPeriod scope is absent or ambiguous")
    manifest_start = _direct_text(periods[0], "startTime", "acquisition start time")
    manifest_stop = _direct_text(periods[0], "stopTime", "acquisition stop time")

    orbit_data = _metadata_object_xml_data(manifest_root, "measurementOrbitReference")
    orbit_scopes = [node for node in orbit_data
                    if node.tag.rsplit("}", 1)[-1] == "orbitReference"]
    if len(orbit_scopes) != 1:
        raise AdmissionError("SAFE manifest orbitReference scope is absent or ambiguous")
    orbit_values = {node.text.strip() for node in orbit_scopes[0]
                    if node.tag.rsplit("}", 1)[-1] == "orbitNumber" and node.text and node.text.strip()}
    if (platform_family.upper() != "SENTINEL-1" or platform_number != mission[2:] or
            instrument_family.upper() != "SYNTHETIC APERTURE RADAR" or
            manifest_mode.upper() != mode or manifest_product_type != product_type or
            _scene_second_token(manifest_start) != start_token or
            _scene_second_token(manifest_stop) != stop_token or
            orbit_values != {str(int(orbit))}):
        raise AdmissionError("SAFE scoped manifest mission/product/orbit/time identity differs from selected scene")
    return {"mission": mission, "product_type": product_type, "mode": mode,
            "polarizations": sorted(polarization_values), "start_time_scene_second": start_token,
            "stop_time_scene_second": stop_token, "absolute_orbit": str(int(orbit)),
            "datatake_id_from_bound_member_stems": datatake,
            "manifest_platform_scope": "SENTINEL-1/PLATFORM-" + platform_number,
            "manifest_instrument_scope": instrument_family,
            "identity_status": "SCOPED_MISSION_PRODUCT_ORBIT_TIME_MATCHED"}


def inspect_safe_roles(archive_path, expected_scene_name, polarization="VV", *,
                       origin="provider_download"):
    """Validate SAFE member identity and native role names without decoding pixels."""
    if origin not in {"provider_download", "constructed_fixture"}:
        raise AdmissionError("SAFE archive source origin must be explicit")
    scene_parts = _product_scene_parts(expected_scene_name)
    mission, mode, product_code, pol_code, start_token, stop_token, orbit, datatake, _ = scene_parts
    product_type = product_code[:3]
    pol = polarization.upper()
    if pol not in {"VV", "VH", "HH", "HV"}:
        raise AdmissionError("selected SAFE polarization is unsupported")
    product_polarizations = {"1SDV": {"VV", "VH"}, "1SDH": {"HH", "HV"},
                             "1SSV": {"VV"}, "1SSH": {"HH"}}[pol_code]
    if pol not in product_polarizations:
        raise AdmissionError("selected SAFE polarization is not declared by product identity")
    member_prefix = (f"{mission}-{mode}-{product_type}-{pol}-{start_token}-{stop_token}-"
                     f"{orbit}-{datatake}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if not names or len(names) != len(set(names)):
                raise AdmissionError("ASF SAFE archive is empty or has duplicate member paths")
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise AdmissionError("ASF SAFE member CRC validation failed")
            root_name = expected_scene_name + ".SAFE"
            for name in names:
                part = PurePosixPath(name)
                if (part.is_absolute() or ".." in part.parts or "\\" in name or
                        not part.parts or part.parts[0] != root_name):
                    raise AdmissionError("ASF SAFE archive contains an unsafe or mismatched member")
            manifests = [n for n in names if n.endswith("/manifest.safe")]
            if len(manifests) != 1:
                raise AdmissionError("ASF SAFE must contain exactly one manifest.safe")
            measurements = [n for n in names if "/measurement/" in n.lower() and
                            n.lower().endswith((".tif", ".tiff")) and
                            re.search(rf"(?:^|[-_]){pol.lower()}(?:[-_.]|$)", PurePosixPath(n).name.lower())]
            calibrations = [n for n in names if "/annotation/calibration/" in n.lower() and
                            "/calibration-" in n.lower() and n.lower().endswith(".xml") and
                            re.search(rf"(?:^|[-_]){pol.lower()}(?:[-_.]|$)", PurePosixPath(n).name.lower())]
            noises = [n for n in names if "/annotation/calibration/" in n.lower() and
                      "/noise-" in n.lower() and n.lower().endswith(".xml") and
                      re.search(rf"(?:^|[-_]){pol.lower()}(?:[-_.]|$)", PurePosixPath(n).name.lower())]
            annotation_root = PurePosixPath(root_name) / "annotation"
            annotations = [n for n in names if PurePosixPath(n).parent == annotation_root and
                           n.lower().endswith(".xml") and
                           re.search(rf"(?:^|[-_]){pol.lower()}(?:[-_.]|$)", PurePosixPath(n).name.lower())]
            rfi_annotations = [n for n in names if len(PurePosixPath(n).parts) == 4 and
                               PurePosixPath(n).parts[:3] == (*annotation_root.parts, "rfi") and
                               n.lower().endswith(".xml") and
                               re.search(rf"(?:^|[-_]){pol.lower()}(?:[-_.]|$)", PurePosixPath(n).name.lower())]
            if not all(len(x) == 1 for x in (measurements, calibrations, noises)):
                raise AdmissionError("ASF SAFE measurement/calibration/noise role is absent or ambiguous")
            if len(annotations) != 1:
                raise AdmissionError("ASF SAFE product annotation role is absent or ambiguous for selected polarization")
            if len(rfi_annotations) > 1:
                raise AdmissionError("ASF SAFE RFI metadata role is ambiguous for selected polarization")
            measurement_stem = PurePosixPath(measurements[0]).stem.lower()
            annotation_stem = PurePosixPath(annotations[0]).stem.lower()
            if (not _measurement_stem_matches(measurements[0], member_prefix) or
                    annotation_stem != measurement_stem or
                    not _role_stem_matches(calibrations[0], "calibration-", measurement_stem) or
                    not _role_stem_matches(noises[0], "noise-", measurement_stem) or
                    (rfi_annotations and not _role_stem_matches(
                        rfi_annotations[0], "rfi-", measurement_stem))):
                raise AdmissionError("ASF SAFE measurement/annotation/calibration/noise identity stems do not join selected scene")
            roles = {"measurement": measurements[0], "calibration": calibrations[0],
                     "noise": noises[0], "manifest": manifests[0], "annotation": annotations[0]}
            roots = {}
            for role in ("annotation", "calibration", "noise"):
                roots[role] = SafeET.fromstring(archive.read(roles[role]))
            pols = [n.text.strip().upper() for n in roots["calibration"].iter()
                    if n.tag.rsplit("}", 1)[-1] == "polarisation" and n.text and n.text.strip()]
            if len(set(pols)) != 1 or pols[0] != pol:
                raise AdmissionError("ASF calibration polarization does not match selected measurement role")
            for role in ("annotation", "calibration", "noise"):
                _validate_ads_header(roots[role], mission=mission, product_type=product_type,
                    mode=mode, polarization=pol, start_token=start_token, stop_token=stop_token,
                    role=role)
            if rfi_annotations:
                rfi_root = SafeET.fromstring(archive.read(rfi_annotations[0]))
                _validate_ads_header(rfi_root, mission=mission, product_type=product_type,
                    mode=mode, polarization=pol, start_token=start_token, stop_token=stop_token,
                    role="RFI metadata")
            manifest_root = SafeET.fromstring(archive.read(roles["manifest"]))
            manifest_identity = _scoped_manifest_identity(manifest_root, expected_scene_name)
            referenced_members = []
            for node in manifest_root.iter():
                if node.tag.rsplit("}", 1)[-1].lower() != "filelocation":
                    continue
                href = node.attrib.get("href")
                if not href:
                    raise AdmissionError("ASF manifest fileLocation lacks a member href")
                relative = PurePosixPath(href.removeprefix("./"))
                if relative.is_absolute() or ".." in relative.parts or "\\" in href:
                    raise AdmissionError("ASF manifest contains an unsafe member href")
                candidates = (relative.as_posix(), (PurePosixPath(root_name) / relative).as_posix())
                resolved = next((candidate for candidate in candidates if candidate in names), None)
                if resolved is None:
                    raise AdmissionError("ASF manifest member href does not resolve in SAFE archive")
                referenced_members.append(resolved)
            for role, member in roles.items():
                if role != "manifest" and member not in referenced_members:
                    raise AdmissionError(f"ASF manifest does not reference its selected {role} member")
            if any(member not in referenced_members for member in rfi_annotations):
                raise AdmissionError("ASF manifest does not reference retained RFI metadata")
            annotation_status = "MISSION_PRODUCT_TIME_POLARIZATION_MATCHED"
            return {"roles": roles, "archive_member_sizes": {r: archive.getinfo(n).file_size
                    for r, n in roles.items()}, "declared_polarization": pols[0],
                    "rfi_metadata": {"status": "PRESENT" if rfi_annotations else "UNAVAILABLE",
                                     "members": sorted(rfi_annotations)},
                    "noise_azimuth_present": any(n.tag.rsplit("}", 1)[-1] == "noiseAzimuthVector"
                                                   for n in roots["noise"].iter()),
                    "role_mapping_status": "SAFE_SCOPED_MEMBER_AND_ROLE_IDENTITIES_MATCHED",
                    "pixel_decoding_status": "NOT_DECODED", "origin": origin,
                    "archive_crc_status": "ALL_ZIP_MEMBER_CRC_CHECKS_PASSED",
                    "manifest_identity_status": "SCOPED_MISSION_PRODUCT_ORBIT_TIME_POLARIZATION_AND_MEMBER_JOIN_MATCHED",
                    "manifest_identity": manifest_identity,
                    "manifest_referenced_members": sorted(set(referenced_members)),
                    "annotation_identity_status": annotation_status,
                    "research_admitted": False}
    except zipfile.BadZipFile as exc:
        raise AdmissionError("ASF product is not a readable SAFE ZIP") from exc


def extract_safe_roles(archive_path, role_map, output_dir):
    """Extract only named SAFE role members into a new, source-hashed attempt dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    with zipfile.ZipFile(archive_path) as archive:
        for role, member in role_map["roles"].items():
            destination = output_dir / (role + Path(member).suffix.lower())
            digest = hashlib.sha256()
            size = 0
            with archive.open(member) as source, destination.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    target.write(chunk)
                    digest.update(chunk)
            if size != role_map["archive_member_sizes"][role]:
                raise AdmissionError("extracted SAFE role length differs from ZIP member")
            result[role] = {"archive_member": member, "path": str(destination),
                            "bytes": size, "sha256": digest.hexdigest()}
    return result
