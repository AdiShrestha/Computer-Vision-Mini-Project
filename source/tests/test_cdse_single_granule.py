"""Constructed offline guards for one quarantined CDSE SAR product."""
from __future__ import annotations

import hashlib
import io
import base64
import json
from pathlib import Path
import zipfile
from urllib.parse import parse_qs, urlsplit

import pytest

from sentinel_gl.admission import AdmissionError
from sentinel_gl.ingestion import cdse_single_granule as cdse
from runners import run_cdse_single_granule as runner


def _row():
    return {
        "Id": cdse.PRODUCT_UUID,
        "Name": cdse.PRODUCT_NAME,
        "ContentDate": {"Start": "2023-09-01T12:14:08.715352Z",
                        "End": "2023-09-01T12:14:33.714796Z"},
        "Online": True,
        "ContentLength": 1_737_975_879,
        "GeoFootprint": {"type": "Polygon", "coordinates": [[
            [86.0, 26.0], [89.5, 26.0], [89.5, 29.0], [86.0, 29.0], [86.0, 26.0]
        ]]},
        "Checksum": [{"Algorithm": "BLAKE3", "Value": "b" * 64}],
        "Attributes": [
            {"Name": "platformSerialIdentifier", "Value": "A"},
            {"Name": "operationalMode", "Value": "IW"},
            {"Name": "productType", "Value": "IW_GRDH_1S"},
            {"Name": "polarisationChannels", "Value": "VV&VH"},
            {"Name": "orbitNumber", "Value": 50134},
            {"Name": "orbitDirection", "Value": "ASCENDING"},
            {"Name": "processorVersion", "Value": "003.61"},
        ],
    }


class _BytesResponse(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(body)
        self.headers = headers


class _ContextResponse(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _FakeOpener:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_exact_cdse_host_guard_and_credentials_are_download_only():
    assert cdse.validate_cdse_url("https://catalogue.dataspace.copernicus.eu/odata/v1/Products") == cdse.CATALOGUE_HOST
    request = cdse.request_for("https://download.dataspace.copernicus.eu/odata/v1/Products(x)/$value",
                               "HEAD", token="constructed-fixture-token")
    assert request.get_header("Authorization") == "Bearer constructed-fixture-token"
    with pytest.raises(AdmissionError, match="exact HTTPS host allowlist"):
        cdse.validate_cdse_url("https://zipper.dataspace.copernicus.eu/product")
    with pytest.raises(AdmissionError, match="exact HTTPS host allowlist"):
        cdse.validate_cdse_url("http://catalogue.dataspace.copernicus.eu/odata")
    with pytest.raises(AdmissionError, match="exact HTTPS host allowlist"):
        cdse.validate_cdse_url("https://catalogue.dataspace.copernicus.eu:444/odata")
    with pytest.raises(AdmissionError, match="credentials may only"):
        cdse.request_for("https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
                         "GET", token="constructed-fixture-token")
    assert cdse.NoRedirect().redirect_request(request, None, 302, "Found", {},
                                               "https://identity.dataspace.copernicus.eu/token") is None


def test_authenticated_get_receipt_separates_attempt_and_secret_logging(tmp_path, monkeypatch):
    import urllib.error
    token = "constructed-private-fixture-token"
    fake = _FakeOpener(urllib.error.URLError("fixture DNS failure"))
    monkeypatch.setattr(runner, "build_opener", lambda handler: fake)
    raw = tmp_path / "data/raw/wp03/restart"
    raw.mkdir(parents=True)
    _, receipt, _ = runner.download_authenticated(
        cdse.validate_product_record(_row()), token, 6_413, 100_000, raw,
        root=tmp_path)
    assert receipt["request_attempted"] is True
    assert receipt["http_response_received"] is False
    assert receipt["credential_value_read_for_authorized_header"] is True
    assert receipt["credential_attached_to_request"] is True
    assert receipt["credential_logging_or_persistence"] is False
    assert len(fake.calls) == 1
    request, _ = fake.calls[0]
    assert request.get_method() == "GET"
    assert token not in repr(receipt)


def test_authenticated_get_receipt_counts_bounded_http_error_body(tmp_path, monkeypatch):
    import urllib.error
    body = b"unauthorized-fixture"
    response = urllib.error.HTTPError(
        "https://download.dataspace.copernicus.eu/odata/v1/Products(x)/$value",
        401, "Unauthorized", {"Content-Length": str(len(body))}, io.BytesIO(body))
    fake = _FakeOpener(response)
    monkeypatch.setattr(runner, "build_opener", lambda handler: fake)
    raw = tmp_path / "data/raw/wp03/restart"
    raw.mkdir(parents=True)
    _, receipt, _ = runner.download_authenticated(
        cdse.validate_product_record(_row()), "constructed-private-fixture-token",
        6_413, 100_000, raw, root=tmp_path)
    assert receipt["request_attempted"] is True
    assert receipt["http_response_received"] is True
    assert receipt["http_status"] == 401
    assert receipt["received_payload_bytes"] == len(body)
    assert (raw / "download_error_response.bin").read_bytes() == body
    assert receipt["credential_logging_or_persistence"] is False


def test_exact_metadata_query_requests_required_product_attributes():
    query = parse_qs(urlsplit(runner.CATALOGUE_URL).query)
    assert query["$filter"] == [f"Name eq '{cdse.PRODUCT_NAME}'"]
    assert query["$top"] == ["1"]
    assert query["$expand"] == ["Attributes"]


def test_receipt_reconciliation_counts_legacy_zero_byte_dns_attempt(tmp_path):
    raw = tmp_path / "data/raw/wp03/cdse_metadata_refresh_legacy"
    raw.mkdir(parents=True)
    empty = raw / "response.bin"
    empty.write_bytes(b"")
    receipt = {
        "request_method": "GET",
        "request_url": "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$top=1",
        "started_utc": "2026-10-14T15:49:16.801804+00:00",
        "http_status": None,
        "received_payload_bytes": 0,
        "response_path": empty.relative_to(tmp_path).as_posix(),
        "error": "URLError: <urlopen error fixture DNS failure>",
    }
    (raw / "receipt.json").write_text(json.dumps(receipt))
    cached = raw / "cached_metadata_receipt.json"
    cached.write_text(json.dumps({
        "evidence_kind": "RETAINED_PUBLIC_CATALOGUE_RESPONSE_NOT_A_NEW_REQUEST",
        "network_used_in_this_continuation": False,
        "request_method": "GET",
        "sanitized_request_url": receipt["request_url"],
    }))
    ledger_path = tmp_path / runner.PRIOR_LEDGER_ID
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({
        "entries": [], "cap_bytes": 10_000,
        "cumulative_new_payload_bytes_under_current_2026-10-14_authorization": 0,
    }))

    ledger, total = runner.reconcile_ledger(tmp_path)

    assert total == 0
    assert ledger["reconciliation_added_zero_byte_failures"] == 1
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["receipt"].endswith("receipt.json")
    assert ledger["entries"][0]["error"] == receipt["error"]


def test_fresh_product_metadata_and_both_query_centroid_footprints():
    result = cdse.validate_product_record(_row())
    assert result["online"] is True
    assert result["catalogue_content_length_bytes"] == 1_737_975_879
    assert result["scene_footprint_contains_query_centroids"] == {
        "PILOT-SOUTH-LHONAK": True,
        "PILOT-KHANGCHUNG-TSHO": True,
    }
    assert result["footprint_scope"] == "METADATA_SCENE_COVERAGE_ONLY_NOT_LAKE_OR_PIXEL_QA"


@pytest.mark.parametrize("mutation, message", [
    (lambda row: row.update(Online=False), "not currently reported online"),
    (lambda row: row.update(Name="S1B_wrong_product.SAFE"), "does not match the selected product identity"),
    (lambda row: row.update(ContentLength=0), "ContentLength is absent or invalid"),
    (lambda row: row["Attributes"].__setitem__(3, {"Name": "polarisationChannels", "Value": "VV"}),
     "mission, mode, polarization, type, or orbit differs"),
])
def test_product_identity_admission_rejects_unexpected_metadata(mutation, message):
    row = _row()
    mutation(row)
    with pytest.raises(AdmissionError, match=message):
        cdse.validate_product_record(row)


def test_scene_footprint_is_metadata_only_and_does_not_infer_lake_coverage():
    row = _row()
    row["GeoFootprint"] = {"type": "Polygon", "coordinates": [[
        [87.0, 27.0], [88.0, 27.0], [88.0, 28.0], [87.0, 28.0], [87.0, 27.0]
    ]]}
    result = cdse.validate_product_record(row)
    assert result["scene_footprint_contains_query_centroids"] == {
        "PILOT-SOUTH-LHONAK": False,
        "PILOT-KHANGCHUNG-TSHO": False,
    }


def test_authenticated_head_requires_actual_representation_length():
    with pytest.raises(AdmissionError, match="lacks a valid representation Content-Length"):
        cdse.validate_download_head(200, {}, 10_000)
    with pytest.raises(AdmissionError, match="exceeds the remaining byte budget"):
        cdse.validate_download_head(200, {"Content-Length": "101"}, 100)
    with pytest.raises(AdmissionError, match="HTTP 302"):
        cdse.validate_download_head(302, {"Content-Length": "100"}, 1000)
    result = cdse.validate_download_head(200, {"Content-Length": "100"}, 1000)
    assert result == {
        "download_representation_content_length_bytes": 100,
        "download_representation_content_type": None,
        "catalogue_length_comparison": "RECORDED_SEPARATELY_NOT_ASSUMED_EQUAL",
    }


def test_stream_hashes_complete_body_and_requires_eof(tmp_path):
    body = b"complete-fixture-body"
    path = tmp_path / "fixture.zip.partial"
    result = cdse.stream_response_to_partial(
        _BytesResponse(body, {"Content-Length": str(len(body))}), path,
        len(body), len(body) + 1)
    assert path.read_bytes() == body
    assert result["written_payload_bytes"] == len(body)
    assert result["received_payload_bytes"] == len(body)
    assert result["eof_confirmed"] is True
    assert result["sha256"] == hashlib.sha256(body).hexdigest()


def test_stream_checks_http_checksum_only_for_its_exact_representation(tmp_path):
    body = b"checksum-bound-representation"
    content_md5 = base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode()
    result = cdse.stream_response_to_partial(
        _BytesResponse(body, {"Content-Length": str(len(body)), "Content-MD5": content_md5}),
        tmp_path / "checksum.zip.partial", len(body), len(body) + 1)
    assert result["content_md5_status"] == "MATCHED_HTTP_CONTENT_MD5"

    with pytest.raises(AdmissionError, match="do not match HTTP Content-MD5"):
        cdse.stream_response_to_partial(
            _BytesResponse(body, {"Content-Length": str(len(body)), "Content-MD5": base64.b64encode(b"0" * 16).decode()}),
            tmp_path / "wrong-checksum.zip.partial", len(body), len(body) + 1)


def test_stream_retains_truncation_and_counts_unexpected_extra_byte(tmp_path):
    expected = b"expected"
    short_path = tmp_path / "short.zip.partial"
    with pytest.raises(cdse.StreamIntegrityError, match="truncated") as short_error:
        cdse.stream_response_to_partial(
            _BytesResponse(expected[:-1], {"Content-Length": str(len(expected))}),
            short_path, len(expected), len(expected) + 1)
    assert short_path.read_bytes() == expected[:-1]
    assert short_error.value.written_bytes == len(expected) - 1

    long_path = tmp_path / "long.zip.partial"
    with pytest.raises(cdse.StreamIntegrityError, match="exceeded declared") as long_error:
        cdse.stream_response_to_partial(
            _BytesResponse(expected + b"x", {"Content-Length": str(len(expected))}),
            long_path, len(expected), len(expected) + 1)
    assert long_path.read_bytes() == expected + b"x"
    assert long_error.value.received_bytes == len(expected) + 1
    assert long_error.value.written_bytes == len(expected) + 1
    assert long_error.value.received_sha256 == hashlib.sha256(expected + b"x").hexdigest()


def test_stream_accepts_absent_content_length_only_after_bounded_eof(tmp_path):
    body = b"unknown-length-complete-body"
    result = cdse.stream_response_to_partial(
        _BytesResponse(body, {}), tmp_path / "unknown.zip.partial", None, len(body) + 1)
    assert result["written_payload_bytes"] == len(body)
    assert result["received_payload_bytes"] == len(body)
    assert result["expected_content_length_bytes"] is None
    assert result["content_length_header_status"] == "ABSENT"
    assert result["eof_confirmed"] is True


def test_unknown_length_stream_cannot_consume_past_hard_payload_cap(tmp_path):
    body = b"123456789"
    with pytest.raises(cdse.StreamIntegrityError, match="reserved payload bound") as error:
        cdse.stream_response_to_partial(
            _BytesResponse(body, {}), tmp_path / "over-cap.zip.partial", None, 9)
    assert error.value.received_bytes == 9
    assert error.value.written_bytes == 9
    assert error.value.received_sha256 == hashlib.sha256(body).hexdigest()
    assert (tmp_path / "over-cap.zip.partial").read_bytes() == body


def test_authenticated_get_stages_complete_quarantined_archive_without_head(tmp_path, monkeypatch):
    body_stream = io.BytesIO()
    with zipfile.ZipFile(body_stream, "w") as archive:
        archive.writestr(cdse.PRODUCT_NAME + "/manifest.safe", b"fixture manifest")
        archive.writestr(cdse.PRODUCT_NAME + "/measurement/fixture-vv.tiff", b"vv fixture")
        archive.writestr(cdse.PRODUCT_NAME + "/measurement/fixture-vh.tiff", b"vh fixture")
    body = body_stream.getvalue()
    raw = tmp_path / "data/raw/wp03/restart"
    raw.mkdir(parents=True)
    row = _row()
    row["Checksum"] = [{"Algorithm": "MD5",
                         "Value": hashlib.md5(body, usedforsecurity=False).hexdigest()}]
    (raw / "fresh_product_metadata.json").write_text(__import__("json").dumps(row))
    fake = _FakeOpener(_ContextResponse(body, 200, {}))
    monkeypatch.setattr(runner, "build_opener", lambda handler: fake)
    monkeypatch.setattr(runner.asf_grd, "inspect_safe_roles",
                        lambda *args, **kwargs: {"fixture": True, "research_admitted": False})

    status, receipt, inventory = runner.download_authenticated(
        cdse.validate_product_record(row), "constructed-private-fixture-token", 6_413,
        100_000, raw, root=tmp_path)

    assert status == 0
    assert len(fake.calls) == 1
    request, _ = fake.calls[0]
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer constructed-private-fixture-token"
    assert receipt["content_length_header_status"] == "ABSENT"
    assert receipt["received_payload_bytes"] == len(body)
    assert receipt["eof_confirmed"] is True
    assert receipt["research_admissible"] is False
    assert inventory["scientific_admission"] is False
    assert (raw / f"{cdse.PRODUCT_UUID}.zip").read_bytes() == body
    assert not (raw / f"{cdse.PRODUCT_UUID}.zip.partial").exists()


def test_archive_integrity_is_quarantine_only_and_binds_safe_identity(tmp_path):
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(cdse.PRODUCT_NAME + "/manifest.safe", b"fixture manifest")
        archive.writestr(cdse.PRODUCT_NAME + "/measurement/fixture-vv.tiff", b"fixture raster")
    result = cdse.safe_archive_check(path, cdse.PRODUCT_NAME)
    assert result["archive_identity"] == "MATCHED_SELECTED_SAFE_NAME"
    assert result["archive_crc"] == "ALL_MEMBERS_PASSED"
    assert result["research_admissible"] is False
    assert result["pixel_calibration_or_quality"] == "NOT_VERIFIED"

    wrong = tmp_path / "wrong.zip"
    with zipfile.ZipFile(wrong, "w") as archive:
        archive.writestr("other.SAFE/manifest.safe", b"fixture")
    with pytest.raises(AdmissionError, match="unsafe or mismatched SAFE member"):
        cdse.safe_archive_check(wrong, cdse.PRODUCT_NAME)
