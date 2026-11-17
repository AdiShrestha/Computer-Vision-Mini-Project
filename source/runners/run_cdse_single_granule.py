"""Authenticated, bounded staging for one retained CDSE Sentinel-1 product."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from urllib.request import build_opener

from sentinel_gl.admission import AdmissionError
from sentinel_gl.ingestion import asf_grd
from sentinel_gl.ingestion.cdse_single_granule import (
    CATALOGUE_HOST, DOWNLOAD_HOST, PRODUCT_NAME, PRODUCT_UUID, NoRedirect,
    StreamIntegrityError, request_for, safe_archive_check, sha256_file,
    stream_response_to_partial, validate_product_record,
)

ROOT = Path.cwd()
CATALOGUE_HOSTED_METADATA = ROOT / "data/raw/wp03/cdse_stage_20261014_restart002"
CATALOGUE_URL = (
    f"https://{CATALOGUE_HOST}/odata/v1/Products?" +
    urlencode({"$filter": f"Name eq '{PRODUCT_NAME}'", "$top": "1", "$expand": "Attributes"})
)
DOWNLOAD_URL = f"https://{DOWNLOAD_HOST}/odata/v1/Products({PRODUCT_UUID})/$value"
METADATA_RESPONSE_CAP = 1_000_000
ERROR_BODY_CAP = 1_000_000
PRIOR_LEDGER_ID = "project/wp03_cdse_budget_ledger.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sanitized_location(value):
    if not value:
        return None
    parts = urlsplit(value)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def attempt_path(value, root=ROOT):
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    required_parent = (root / "data/raw/wp03").resolve()
    if resolved.parent != required_parent:
        raise AdmissionError("attempt output must be a single fresh child of data/raw/wp03")
    if resolved.exists():
        raise AdmissionError("attempt output already exists; preserve prior attempt")
    return resolved


def load_and_verify_retained_metadata(root=ROOT):
    raw = CATALOGUE_HOSTED_METADATA
    receipt_path = raw / "catalogue_request_receipt.json"
    response_path = raw / "catalogue_response.bin"
    metadata_path = raw / "fresh_product_metadata.json"
    summary_path = raw / "fresh_product_summary.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = response_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parts = urlsplit(receipt.get("request_url", ""))
    query = parse_qs(parts.query)
    if (parts.scheme != "https" or parts.hostname != CATALOGUE_HOST or
            query.get("$expand") != ["Attributes"] or
            receipt.get("http_status") != 200 or
            receipt.get("received_body_bytes") != len(body) or
            receipt.get("body_sha256") != sha256_bytes(body)):
        raise AdmissionError("retained expanded catalogue receipt or response binding is invalid")
    rows = json.loads(body).get("value", [])
    matches = [row for row in rows if row.get("Id") == PRODUCT_UUID and row.get("Name") == PRODUCT_NAME]
    if len(matches) != 1 or matches[0] != metadata:
        raise AdmissionError("retained catalogue body does not contain the exact selected metadata row")
    recomputed = validate_product_record(metadata)
    if recomputed != summary:
        raise AdmissionError("retained product summary does not match independent metadata validation")
    return {
        "metadata": metadata,
        "summary": recomputed,
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": sha256_file(receipt_path),
        "response_path": response_path.relative_to(root).as_posix(),
        "response_sha256": sha256_file(response_path),
        "metadata_path": metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
    }


def _receipt_payload_bytes(receipt):
    for key in ("received_payload_bytes", "received_body_bytes"):
        value = receipt.get(key)
        if type(value) is int and value >= 0:
            return value
    return 0


def reconcile_ledger(root=ROOT):
    """Verify every previously recorded request and append omitted error attempts."""
    path = root / PRIOR_LEDGER_ID
    ledger = json.loads(path.read_text(encoding="utf-8"))
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise AdmissionError("shared byte ledger entries are absent")
    verified_total = 0
    recorded_receipts = set()
    for entry in entries:
        receipt_ref = entry.get("receipt")
        if receipt_ref:
            recorded_receipts.add(receipt_ref)
            receipt = json.loads((root / receipt_ref).read_text(encoding="utf-8"))
            actual = _receipt_payload_bytes(receipt)
            if actual != entry.get("network_payload_bytes_this_entry"):
                raise AdmissionError(f"ledger byte count differs from retained receipt {receipt_ref}")
            if entry.get("http_status") != receipt.get("http_status"):
                raise AdmissionError(f"ledger HTTP status differs from retained receipt {receipt_ref}")
            receipt_digest = receipt.get("body_sha256", receipt.get("received_payload_sha256"))
            if entry.get("body_sha256") and entry["body_sha256"] != receipt_digest:
                raise AdmissionError(f"ledger body hash differs from retained receipt {receipt_ref}")
            response_ref = (receipt.get("response_path") or receipt.get("staged_path") or
                            receipt.get("partial_path") or receipt.get("error_body_path"))
            if actual and response_ref:
                body_path = root / response_ref
                if not body_path.is_file():
                    raise AdmissionError(f"retained response payload is missing for receipt {receipt_ref}")
                body = body_path.read_bytes()
                if len(body) != actual or sha256_bytes(body) != receipt_digest:
                    raise AdmissionError(f"retained response bytes differ from receipt {receipt_ref}")
            elif actual:
                raise AdmissionError(f"retained payload path is absent for receipt {receipt_ref}")
        else:
            actual = entry.get("network_payload_bytes_this_entry", 0)
        verified_total += actual

    # Reconcile every retained CDSE request receipt, including zero-byte failures.
    added = []
    for receipt_path in sorted((root / "data/raw/wp03").rglob("*receipt*.json")):
        receipt_ref = receipt_path.relative_to(root).as_posix()
        if receipt_ref in recorded_receipts:
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        attempted = (
            receipt.get("request_attempted") is True or
            receipt.get("network_request_sent") is True or
            (receipt.get("evidence_kind") != "RETAINED_PUBLIC_CATALOGUE_RESPONSE_NOT_A_NEW_REQUEST" and
             receipt.get("request_method") in {"GET", "HEAD"} and
             bool(receipt.get("request_url")) and
             bool(receipt.get("started_utc") or receipt.get("request_started_utc")))
        )
        if not attempted:
            continue
        payload_bytes = _receipt_payload_bytes(receipt)
        body_path_ref = receipt.get("response_path")
        if body_path_ref:
            body_path = root / body_path_ref
            if body_path.exists() and body_path.stat().st_size != payload_bytes:
                raise AdmissionError(f"retained DNS-attempt body size differs from its receipt: {body_path_ref}")
            if payload_bytes and not body_path.exists():
                raise AdmissionError(f"retained response bytes are missing for receipt {receipt_ref}")
        body_digest = (receipt.get("body_sha256", receipt.get("received_payload_sha256",
                       receipt.get("response_sha256"))) if payload_bytes else None)
        entry = {
            "entry_id": "reconciled_" + receipt_path.parent.name + "_" + receipt_path.stem,
            "operation": receipt.get("request_method", "UNKNOWN") + "_RETAINED_UNLEDGERED_ATTEMPT",
            "receipt": receipt_ref,
            "http_status": receipt.get("http_status"),
            "network_payload_bytes_this_entry": payload_bytes,
            "body_sha256": body_digest,
            "error": receipt.get("error", receipt.get("literal_error")),
            "processing_units": 0,
        }
        entries.append(entry)
        recorded_receipts.add(receipt_ref)
        verified_total += payload_bytes
        added.append(entry)

    if verified_total != ledger.get("cumulative_new_payload_bytes_under_current_2026-10-14_authorization"):
        raise AdmissionError("shared ledger cumulative byte count differs from reconciled receipts")
    ledger["status"] = "RECONCILED_ALL_RETAINED_CDSE_REQUEST_RECEIPTS_BEFORE_AUTHENTICATED_GET"
    ledger["reconciliation_utc"] = now()
    ledger["reconciled_payload_bytes"] = verified_total
    ledger["reconciliation_added_zero_byte_failures"] = len(added)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger, verified_total


def inspect_safe(path):
    base_name = PRODUCT_NAME.removesuffix(".SAFE")
    structural = safe_archive_check(path, PRODUCT_NAME)
    roles = {pol: asf_grd.inspect_safe_roles(
        path, base_name, pol, origin="provider_download") for pol in ("VV", "VH")}
    return {"structure": structural, "polarization_identity": roles,
            "calibration_or_quality_admission": "NOT_PERFORMED",
            "research_admissible": False}


def download_authenticated(metadata_summary, token, used_before, cap, raw, root=ROOT):
    remaining = cap - used_before
    available_disk = shutil.disk_usage(raw).free
    reservation = min(remaining, available_disk)
    if reservation < 2:
        raise AdmissionError("no aggregate byte and EOF-guard reservation is available")
    request = request_for(DOWNLOAD_URL, "GET", token=token)
    receipt = {
        "request_url": DOWNLOAD_URL,
        "request_host": DOWNLOAD_HOST,
        "request_method": "GET",
        "request_attempted": False,
        "http_response_received": False,
        "automatic_redirects": False,
        "redirect_followed": False,
        "credential_value_read_for_authorized_header": True,
        "credential_attached_to_request": True,
        "credential_logging_or_persistence": False,
        "credential_presence_checked_utc": now(),
        "started_utc": now(),
        "reserved_aggregate_payload_bytes_including_eof_guard": reservation,
        "hard_payload_cap_bytes": reservation - 1,
        "catalogue_native_content_length_bytes": metadata_summary["catalogue_content_length_bytes"],
        "download_representation_content_length_bytes": None,
        "download_length_relation": "UNKNOWN_UNTIL_AUTHENTICATED_GET_RESPONSE",
        "research_admissible": False,
        "response_is_observation": False,
    }
    partial = raw / f"{PRODUCT_UUID}.zip.partial"
    final = raw / f"{PRODUCT_UUID}.zip"
    if partial.exists() or final.exists():
        raise AdmissionError("staging output already exists; preserve prior attempt")
    response_headers = {}
    stream = None
    body_bytes = 0
    body_sha = None
    error = None
    status = None
    opener = build_opener(NoRedirect())
    receipt["request_attempted"] = True
    try:
        with opener.open(request, timeout=120) as response:
            receipt["http_response_received"] = True
            status = response.status
            response_headers = response.headers
            if status != 200:
                raise AdmissionError(f"product GET returned HTTP {status}")
            raw_length = response_headers.get("Content-Length")
            if raw_length is not None:
                receipt["download_representation_content_length_bytes"] = raw_length
                if not raw_length.isdigit() or int(raw_length) <= 0:
                    raise AdmissionError("GET Content-Length is malformed or nonpositive")
                if int(raw_length) > reservation - 1:
                    raise AdmissionError("GET Content-Length exceeds remaining hard stream reservation")
                receipt["download_length_relation"] = (
                    "EQUAL" if int(raw_length) == metadata_summary["catalogue_content_length_bytes"]
                    else "DIFFERENT_SEPARATE_REPRESENTATIONS")
            else:
                receipt["download_length_relation"] = "HEADER_ABSENT_NATIVE_SIZE_NOT_ASSUMED"
            try:
                stream = stream_response_to_partial(response, partial, None, reservation)
            except StreamIntegrityError as exc:
                body_bytes = exc.received_bytes
                body_sha = exc.received_sha256
                raise
            body_bytes = stream["received_payload_bytes"]
            body_sha = stream["sha256"]
    except HTTPError as exc:
        receipt["http_response_received"] = True
        status = exc.code
        response_headers = exc.headers or {}
        if 300 <= status < 400:
            receipt["redirect_location_sanitized"] = sanitized_location(
                response_headers.get("Location"))
        try:
            error_body = exc.read(min(ERROR_BODY_CAP, max(0, reservation - 1)))
        except Exception:
            error_body = b""
        body_bytes = len(error_body)
        body_sha = sha256_bytes(error_body)
        if error_body:
            error_body_path = raw / "download_error_response.bin"
            error_body_path.write_bytes(error_body)
            receipt["error_body_path"] = error_body_path.relative_to(root).as_posix()
        receipt["error_body_cap_bytes"] = ERROR_BODY_CAP
        receipt["error_body_maybe_truncated"] = len(error_body) == ERROR_BODY_CAP
        error = f"HTTPError: HTTP {status}"
    except StreamIntegrityError as exc:
        error = f"StreamIntegrityError: {exc}"
        receipt.update({"partial_written_bytes": exc.written_bytes,
                        "partial_sha256": exc.sha256,
                        "partial_local_md5": exc.local_md5,
                        "received_payload_sha256": exc.received_sha256,
                        "partial_path": partial.relative_to(root).as_posix() if partial.exists() else None,
                        "eof_confirmed": False})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if token:
            error = error.replace(token, "[REDACTED]")

    receipt.update({
        "finished_utc": now(),
        "http_status": status,
        "response_content_length_header": response_headers.get("Content-Length") if response_headers else None,
        "response_content_type": response_headers.get("Content-Type") if response_headers else None,
        "response_content_md5": response_headers.get("Content-MD5") if response_headers else None,
        "received_payload_bytes": body_bytes,
        "received_payload_sha256": body_sha,
        "written_payload_bytes": stream.get("written_payload_bytes", 0) if stream else 0,
        "eof_confirmed": stream.get("eof_confirmed", False) if stream else False,
        "content_length_header_status": stream.get("content_length_header_status") if stream else (
            "PRESENT" if response_headers.get("Content-Length") is not None else "ABSENT"),
        "aggregate_payload_bytes_before_transfer": used_before,
        "aggregate_payload_bytes_after_transfer": used_before + body_bytes,
        "error": error,
    })
    if error:
        receipt["status"] = "AUTHENTICATED_GET_FAILED_OR_INCOMPLETE_QUARANTINED"
        return 2, receipt, None
    if status != 200 or not stream or not stream["eof_confirmed"]:
        receipt["status"] = "GET_RESPONSE_NOT_COMPLETE_QUARANTINED"
        receipt["error"] = "HTTP response or complete EOF evidence is unavailable"
        return 2, receipt, None
    if stream["written_payload_bytes"] != body_bytes:
        receipt["status"] = "RECEIVED_AND_WRITTEN_LENGTH_MISMATCH_QUARANTINED"
        receipt["error"] = "stream written bytes differ from measured received bytes"
        return 3, receipt, None
    try:
        safe = inspect_safe(partial)
        expected_md5 = next((item.get("Value", "").lower() for item in
                             json.loads((raw / "fresh_product_metadata.json").read_text()).get("Checksum", [])
                             if item.get("Algorithm", "").upper() == "MD5"), None)
        receipt["catalogue_md5_comparison"] = (
            "BYTE_MATCH_SCOPE_UNVERIFIED" if expected_md5 and stream["local_md5"] == expected_md5
            else "BYTE_MISMATCH_SCOPE_UNVERIFIED" if expected_md5
            else "NO_CATALOGUE_MD5")
        receipt["archive_evidence"] = safe
    except Exception as exc:
        receipt["status"] = "COMPLETE_BYTES_ARCHIVE_OR_CHECKSUM_REJECTED_QUARANTINED"
        receipt["archive_validation_error"] = f"{type(exc).__name__}: {exc}"
        receipt["partial_path"] = partial.relative_to(root).as_posix()
        return 3, receipt, None
    partial.rename(final)
    inventory = {
        "schema_version": 1,
        "origin": "provider_download_quarantined",
        "product_uuid": PRODUCT_UUID,
        "product_name": PRODUCT_NAME,
        "metadata_source": {
            "path": (CATALOGUE_HOSTED_METADATA / "fresh_product_metadata.json").relative_to(ROOT).as_posix(),
            "sha256": sha256_file(CATALOGUE_HOSTED_METADATA / "fresh_product_metadata.json"),
            "retrieval_status": "RETAINED_CATALOGUE_GET_2026_10_14_EXPANDED_ATTRIBUTES_NOT_REFRESHED_ON_2026_10_15",
        },
        "payload": {"path": final.relative_to(root).as_posix(),
                    "bytes": final.stat().st_size,
                    "sha256": sha256_file(final),
                    "md5": stream["local_md5"]},
        "archive": safe,
        "scientific_admission": False,
    }
    write_new(raw / "raw_inventory.json", inventory)
    receipt.update({
        "status": "COMPLETE_ARCHIVE_BYTES_SHA256_MD5_AND_SAFE_CRC_VERIFIED_QUARANTINED",
        "staged_path": final.relative_to(root).as_posix(),
        "staged_bytes": final.stat().st_size,
        "staged_sha256": sha256_file(final),
        "local_md5": stream["local_md5"],
        "archive_evidence": safe,
        "raw_inventory_path": (raw / "raw_inventory.json").relative_to(root).as_posix(),
        "raw_inventory_sha256": sha256_file(raw / "raw_inventory.json"),
        "research_admissible": False,
    })
    return 0, receipt, inventory


def update_ledger(ledger, raw, receipt, used_before, root=ROOT):
    payload_bytes = receipt.get("received_payload_bytes", 0)
    entry = {
        "entry_id": raw.name + "_authenticated_get",
        "operation": "AUTHENTICATED_DOWNLOAD_GET",
        "receipt": (raw / "download_response_receipt.json").relative_to(root).as_posix(),
        "http_status": receipt.get("http_status"),
        "network_payload_bytes_this_entry": payload_bytes,
        "body_sha256": receipt.get("received_payload_sha256"),
        "processing_units": 0,
        "status": receipt.get("status"),
    }
    ledger["entries"].append(entry)
    cumulative = used_before + payload_bytes
    ledger["cumulative_new_payload_bytes_under_current_2026-10-14_authorization"] = cumulative
    ledger["remaining_authorized_ceiling_bytes"] = ledger["cap_bytes"] - cumulative
    ledger["status"] = "RECONCILED_AUTHENTICATED_GET_BYTES; SCIENTIFIC_ADMISSION_FALSE"
    path = root / PRIOR_LEDGER_ID
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="fresh single-child path below data/raw/wp03")
    args = parser.parse_args(argv)
    raw = attempt_path(args.output_dir)
    auth = json.loads((ROOT / "project/wp03_cdse_expansion_authorization.json").read_text())
    manifest = json.loads((ROOT / "project/wp03_cdse_acquisition_manifest.json").read_text())
    if set(auth["network"]["allowed_exact_hosts"]) != {CATALOGUE_HOST, DOWNLOAD_HOST}:
        raise AdmissionError("authorization host set differs from the two-host restart")
    if manifest.get("target_staging", {}).get("product_uuid") != PRODUCT_UUID or manifest.get(
            "target_staging", {}).get("product_name") != PRODUCT_NAME:
        raise AdmissionError("manifest selected product identity differs")
    metadata_evidence = load_and_verify_retained_metadata()
    ledger, used = reconcile_ledger()
    cap = auth["budget"]["aggregate_received_payload_bytes_ceiling"]
    if ledger.get("cap_bytes") != cap or used >= cap:
        raise AdmissionError("shared aggregate byte budget is invalid or exhausted")
    if used >= auth["budget"]["initial_probe_received_payload_bytes_ceiling"]:
        # Initial metadata probing is complete; the PI-authorized staging GET is a bounded transfer.
        if ledger.get("status") != "RECONCILED_ALL_RETAINED_CDSE_REQUEST_RECEIPTS_BEFORE_AUTHENTICATED_GET":
            raise AdmissionError("prior request receipts were not reconciled before staging")
    raw.mkdir(parents=True, exist_ok=False)
    write_new(raw / "source_metadata_binding.json", metadata_evidence)
    write_new(raw / "fresh_product_metadata.json", metadata_evidence["metadata"])
    # Read the existing token only immediately before constructing this authorized request.
    credential_checked = now()
    token = os.environ.get("CDSE_ACCESS_TOKEN")
    if not token:
        receipt = {"status": "BLOCKED_MISSING_LOCAL_ACCESS_TOKEN",
                   "credential_presence_checked_utc": credential_checked,
                   "credential_available": False,
                   "credential_value_read_for_authorized_header": False,
                   "credential_logging_or_persistence": False,
                   "request_attempted": False, "http_response_received": False,
                   "received_payload_bytes": 0, "research_admissible": False}
        write_new(raw / "download_response_receipt.json", receipt)
        update_ledger(ledger, raw, receipt, used)
        write_new(raw / "run_status.json", {"status": receipt["status"], "research_admissible": False})
        print(json.dumps({"status": receipt["status"], "credential_available": False}))
        return 4
    transfer_result, receipt, inventory = download_authenticated(
        metadata_evidence["summary"], token, used, cap, raw)
    receipt["credential_presence_checked_utc"] = credential_checked
    receipt["credential_value_read_for_authorized_header"] = True
    receipt["credential_attached_to_request"] = True
    receipt["credential_logging_or_persistence"] = False
    write_new(raw / "download_response_receipt.json", receipt)
    update_ledger(ledger, raw, receipt, used)
    status = {"status": receipt.get("status"),
              "product_uuid": PRODUCT_UUID,
              "metadata_refresh": "NOT_REQUESTED_RETAINED_EXPANDED_ATTRIBUTES_REVALIDATED",
              "download": receipt,
              "aggregate_payload_bytes": receipt.get("aggregate_payload_bytes_after_transfer", used),
              "research_admissible": False}
    write_new(raw / "run_status.json", status)
    print(json.dumps({"status": receipt.get("status"),
                      "http_status": receipt.get("http_status"),
                      "received_payload_bytes": receipt.get("received_payload_bytes"),
                      "staged_bytes": receipt.get("staged_bytes"),
                      "staged_sha256": receipt.get("staged_sha256"),
                      "error": receipt.get("error") or receipt.get("archive_validation_error"),
                      "research_admissible": False}, sort_keys=True))
    return transfer_result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Never echo request headers or environment values in diagnostics.
        print(json.dumps({"status": "RUNNER_GUARD_OR_RUNTIME_ERROR",
                          "error": f"{type(exc).__name__}: {exc}",
                          "research_admissible": False}, sort_keys=True))
        raise SystemExit(10)
