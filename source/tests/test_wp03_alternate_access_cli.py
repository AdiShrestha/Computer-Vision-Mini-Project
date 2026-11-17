"""Offline CLI and receipt checks with explicitly constructed source fixtures."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

from runners import run_wp03_alternate_access as runner
from sentinel_gl.ingestion import asf_grd, earth_search_c1
from test_alternate_provider_integrity import _c1_fixture
from safe_fixtures import role_members, scoped_manifest


PINNED_CUTOFF = "2023-10-03T00:00:00+05:30"
PINNED_START = "2023-09-01T00:00:00Z"
PINNED_END = "2023-11-01T00:00:00Z"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture_inputs(tmp_path: Path, *, post_event_native=False, shifted_grid=False,
                    asf_bytes=128, asf_md5="b" * 32, per_product_limit=2**31) -> dict:
    repository = tmp_path / "repository"
    repository.mkdir()
    source_root = repository / "retained"
    source_root.mkdir()
    manifest_path = repository / "pilot.json"
    selection_path = repository / "selection.json"
    evidence_path = repository / "asset-summary.json"
    inventory_path = repository / "source-inventory.json"
    manifest = {
        "schema_version": 1,
        "purpose": "development_acquisition_only",
        "acquisition_start": PINNED_START,
        "acquisition_end_exclusive": PINNED_END,
        "pre_event_cutoff_exclusive": PINNED_CUTOFF,
        "research_executable": False,
        "sites": [
            {"site_id": "PILOT-SOUTH-LHONAK", "scope": "DEVELOPMENT_EXPOSED"},
            {"site_id": "PILOT-KHANGCHUNG-TSHO", "scope": "DEVELOPMENT_EXPOSED"},
        ],
    }
    _write_json(manifest_path, manifest)

    source_paths = []
    evidence = {"scope": "CONSTRUCTED_FIXTURE_ONLY_NO_PROVIDER_OBSERVATIONS", "requests": []}
    selected_c1 = {}
    c1_fixtures = {}
    site_specs = (
        ("PILOT-SOUTH-LHONAK", "T45RXL", "XL", "20230904",
         "2023-09-04T05:01:01.646000Z", "2023-09-04T04:47:01Z"),
        ("PILOT-KHANGCHUNG-TSHO", "T45RXM", "XM", "20230921",
         "2023-09-21T04:50:49.264000Z", "2023-09-21T04:37:01Z"),
    )
    for site_id, tile, square, date, item_time, product_time in site_specs:
        item, base_selection, product_xml, granule_xml = _c1_fixture(
            tile, square, date, item_time=item_time, product_time=product_time,
            tile_time=item_time)
        if post_event_native:
            product_xml = product_xml.replace(product_time.encode(), b"2023-10-04T04:47:01Z")
        if shifted_grid and site_id == "PILOT-SOUTH-LHONAK":
            item["assets"]["swir16"]["proj:transform"][2] += 10

        item_rel = f"catalog/{site_id}_c1.json"
        product_rel = f"native/{site_id}_product.xml"
        granule_rel = f"native/{site_id}_granule.xml"
        _write_json(source_root / item_rel, {"type": "FeatureCollection", "features": [item]})
        (source_root / product_rel).parent.mkdir(parents=True, exist_ok=True)
        (source_root / product_rel).write_bytes(product_xml)
        (source_root / granule_rel).write_bytes(granule_xml)
        source_paths.extend((item_rel, product_rel, granule_rel))

        assets = {}
        for role, asset in item["assets"].items():
            head = {
                "site_id": site_id,
                "item_id": item["id"],
                "role": role,
                "url": asset["href"],
                "method": "HEAD",
                "http_status": 200,
                "response_headers": {
                    "Content-Length": str(asset["file:size"]),
                    "ETag": f'"fixture-{site_id}-{role}"',
                },
                "pixel_bytes_downloaded": 0,
            }
            evidence["requests"].append(head)
            assets[role] = {
                "band": earth_search_c1.ASSET_ROLES[role],
                "href": asset["href"],
                "file_checksum_multihash": asset["file:checksum"],
                "file_size": asset["file:size"],
                "head": head,
                "raster_metadata": asset["raster:bands"][0],
            }
        for role, relative in (("product_metadata", product_rel), ("granule_metadata", granule_rel)):
            path = source_root / relative
            evidence["requests"].append({
                "site_id": site_id, "item_id": item["id"], "role": role,
                "method": "GET", "http_status": 200, "path": relative,
                "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes()),
            })
        selected_c1[site_id] = {
            **{key: base_selection[key] for key in (
                "item_id", "product_uri", "baseline", "acquisition_datetime", "generation_time")},
            "item_path": item_rel,
            "product_metadata_path": product_rel,
            "granule_metadata_path": granule_rel,
            "earthsearch_boa_offset_applied": None,
            "site_scope": "DEVELOPMENT_EXPOSED",
            "site_join": [site_id],
            "assets": assets,
        }
        c1_fixtures[site_id] = item

    scene = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F"
    asf_feature = {"type": "Feature", "properties": {
        "sceneName": scene, "fileID": scene + "-GRD_HD", "fileName": scene + ".zip",
        "url": f"https://datapool.asf.alaska.edu/GRD_HD/SA/{scene}.zip",
        "bytes": asf_bytes, "md5sum": asf_md5,
        "startTime": "2023-09-01T12:14:08Z", "stopTime": "2023-09-01T12:14:33Z",
        "polarization": "VV+VH", "flightDirection": "ASCENDING",
        "processingLevel": "GRD_HD",
    }}
    query_files = {}
    for site_id, *_ in site_specs:
        relative = f"catalog/{site_id}_asf.json"
        _write_json(source_root / relative, {"features": [asf_feature]})
        query_files[site_id] = relative
        source_paths.append(relative)

    selection = {
        "attempt": "CONSTRUCTED_FIXTURE_WP03_CLI",
        "manifest_path": "pilot.json",
        "manifest_sha256": _sha(manifest_path.read_bytes()),
        "cutoff_exclusive": PINNED_CUTOFF,
        "logical_product_count": 3,
        "product_limit": 4,
        "per_product_byte_limit": per_product_limit,
        "sentinel1_asf_grd": {
            **asf_feature["properties"],
            "site_scope": "DEVELOPMENT_EXPOSED",
            "site_join": [site for site, *_ in site_specs],
            "query_files": query_files,
        },
        "sentinel2_c1": selected_c1,
    }
    _write_json(selection_path, selection)
    _write_json(evidence_path, evidence)

    inventory = {
        "schema_version": 1,
        "source_root": "retained",
        "bound_inputs": {
            "manifest": {"path": "pilot.json", "sha256": _sha(manifest_path.read_bytes())},
            "selection": {"path": "selection.json", "sha256": _sha(selection_path.read_bytes())},
            "evidence": {"path": "asset-summary.json", "sha256": _sha(evidence_path.read_bytes())},
        },
        "source_files": [
            {"path": relative, "sha256": _sha((source_root / relative).read_bytes())}
            for relative in sorted(set(source_paths))
        ],
    }
    _write_json(inventory_path, inventory)
    return {
        "repository": repository,
        "manifest": manifest_path,
        "selection": selection_path,
        "evidence": evidence_path,
        "inventory": inventory_path,
        "output": repository / "attempt-output",
        "source_root": source_root,
        "c1_fixtures": c1_fixtures,
    }


def _argv(inputs: dict, mode="validate-only") -> list[str]:
    return [
        "--repository-root", str(inputs["repository"]),
        "--manifest", str(inputs["manifest"]),
        "--selection", str(inputs["selection"]),
        "--evidence", str(inputs["evidence"]),
        "--source-inventory", str(inputs["inventory"]),
        "--output-dir", str(inputs["output"]),
        "--mode", mode,
    ]


def _report(inputs: dict) -> dict:
    return json.loads((inputs["output"] / "stage_report.json").read_text(encoding="utf-8"))


def _rebind_manifest_fixture(inputs: dict) -> None:
    selection = json.loads(inputs["selection"].read_text(encoding="utf-8"))
    selection["manifest_sha256"] = _sha(inputs["manifest"].read_bytes())
    _write_json(inputs["selection"], selection)
    inventory = json.loads(inputs["inventory"].read_text(encoding="utf-8"))
    inventory["bound_inputs"]["manifest"]["sha256"] = _sha(inputs["manifest"].read_bytes())
    inventory["bound_inputs"]["selection"]["sha256"] = _sha(inputs["selection"].read_bytes())
    _write_json(inputs["inventory"], inventory)


def test_cli_validates_complete_constructed_input_joins_offline(tmp_path, monkeypatch):
    inputs = _fixture_inputs(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("validate-only invoked a provider adapter")

    monkeypatch.setattr(runner, "stage_feature", forbidden)
    monkeypatch.setattr(runner, "stage_asset", forbidden)

    assert runner.main(_argv(inputs)) == 0
    report = _report(inputs)
    assert calls == []
    assert report["status"] == "VALIDATED_INPUTS_ONLY_NO_TRANSFER"
    assert report["network_call_status"] == "NOT_INVOKED"
    assert report["logical_product_count"] == 3
    assert report["research_admitted"] is False
    assert report["site_native_time_and_join_status"]["PILOT-SOUTH-LHONAK"]["tile_join_status"] == (
        "ITEM_PRODUCT_AND_GRANULE_TILE_IDENTITY_EXACT_MATCH")


def test_cli_preflight_rejects_changed_selection_without_provider_call(tmp_path, monkeypatch):
    inputs = _fixture_inputs(tmp_path)
    value = json.loads(inputs["selection"].read_text(encoding="utf-8"))
    value["logical_product_count"] = 1
    _write_json(inputs["selection"], value)
    called = []
    monkeypatch.setattr(runner, "stage_feature", lambda *args, **kwargs: called.append("asf"))
    monkeypatch.setattr(runner, "stage_asset", lambda *args, **kwargs: called.append("c1"))

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert "selection SHA-256" in report["literal_error"]
    assert report["network_call_status"] == "NOT_INVOKED"
    assert called == []


def test_cli_strict_json_rejects_hash_bound_duplicate_manifest_keys(tmp_path):
    inputs = _fixture_inputs(tmp_path)
    text = inputs["manifest"].read_text(encoding="utf-8")
    text = text.replace('"research_executable": false,',
                        '"research_executable": true,\n  "research_executable": false,')
    inputs["manifest"].write_text(text, encoding="utf-8")
    _rebind_manifest_fixture(inputs)

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert report["literal_error"] == "duplicate JSON key: research_executable"
    assert report["network_call_status"] == "NOT_INVOKED"


@pytest.mark.parametrize("literal", ["NaN", "1e999"])
def test_cli_strict_json_rejects_nested_nonfinite_manifest_values(tmp_path, literal):
    inputs = _fixture_inputs(tmp_path)
    text = inputs["manifest"].read_text(encoding="utf-8")
    body, closing = text.rsplit("\n}", 1)
    assert not closing.strip()
    body += f',\n  "nested_fixture_value": {{"nonfinite": {literal}}}'
    inputs["manifest"].write_text(body + "\n}\n", encoding="utf-8")
    _rebind_manifest_fixture(inputs)

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert "nonfinite JSON" in report["literal_error"]
    assert report["network_call_status"] == "NOT_INVOKED"


def test_source_inventory_boolean_version_is_not_integer_one(tmp_path):
    inputs = _fixture_inputs(tmp_path)
    text = inputs["inventory"].read_text(encoding="utf-8").replace(
        '"schema_version": 1', '"schema_version": true', 1)
    inputs["inventory"].write_text(text, encoding="utf-8")

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert report["literal_error"] == "private source inventory schema version is unsupported"
    assert report["network_call_status"] == "NOT_INVOKED"


def test_runner_receipt_parser_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "constructed-receipt.json"
    path.write_text('{"status":"COMPLETE","status":"REJECTED"}\n', encoding="utf-8")

    assert runner._read_adapter_receipt(path) is None


def test_cli_rejects_native_product_interval_after_cutoff(tmp_path):
    inputs = _fixture_inputs(tmp_path, post_event_native=True)

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert "native product start is outside the pinned acquisition window" in report["literal_error"]
    assert report["network_call_status"] == "NOT_INVOKED"


class _MemoryResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str, expected_size: int, etag: str):
        super().__init__(payload)
        self.status = 200
        self.headers = {"Content-Length": str(expected_size), "ETag": etag}
        self._url = url

    def geturl(self):
        return self._url


@pytest.mark.parametrize("failure_kind", ["checksum", "etag"])
def test_cli_integrity_failures_count_toward_two_failure_breaker(tmp_path, monkeypatch, failure_kind):
    inputs = _fixture_inputs(tmp_path)
    requests = []

    def local_asf_fixture(*args, **kwargs):
        return {"origin": "constructed_fixture", "status": "NOT_ATTEMPTED_NETWORK_PAUSED"}

    def local_corrupt_response(request, timeout):
        url = request.full_url
        requests.append(url)
        site, item = next((site, item) for site, item in inputs["c1_fixtures"].items()
                          if any(a["href"] == url for a in item["assets"].values()))
        role, asset = next((role, asset) for role, asset in item["assets"].items()
                           if asset["href"] == url)
        selected_site = json.loads(inputs["selection"].read_text())["sentinel2_c1"][site]
        expected_head = selected_site["assets"][role]["head"]
        if failure_kind == "checksum":
            payload = b"X" * (asset["file:size"] - 1)
            etag = expected_head["response_headers"]["ETag"]
        else:
            payload = (role + "-constructed-fixture").encode()
            etag = '"different-fixture-etag"'
        return _MemoryResponse(payload, url, asset["file:size"], etag)

    monkeypatch.setattr(runner, "stage_feature", local_asf_fixture)
    monkeypatch.setattr(earth_search_c1.urllib.request, "urlopen", local_corrupt_response)

    assert runner.main(_argv(inputs, mode="stage")) == 2
    report = _report(inputs)
    attempts = [failure for failure in report["failures"] if failure["branch"] == "C1_ASSET"]
    assert report["status"] == "STOPPED_AFTER_TWO_CONSECUTIVE_C1_FAILURES"
    assert len(attempts) == len(requests) == 2
    expected_status = ("LENGTH_OR_SHA256_MISMATCH" if failure_kind == "checksum"
                       else "GET_ETAG_DIFFERS_FROM_RETAINED_HEAD")
    expected_error = ("SHA-256 multihash mismatch" if failure_kind == "checksum"
                      else "C1 GET ETag differs from retained HEAD identity")
    assert all(expected_error in item["literal_error"] for item in attempts)
    role_receipts = list((inputs["output"] / "c1").rglob("*.receipt.json"))
    partials = list((inputs["output"] / "c1").rglob("*.partial"))
    assert len(role_receipts) == len(partials) == 2
    for partial, receipt_path in zip(sorted(partials), sorted(role_receipts)):
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == expected_status
        assert receipt["received_bytes"] == partial.stat().st_size
        assert receipt["partial_sha256"] == _sha(partial.read_bytes())
        assert receipt["request_started_utc"] < receipt["request_finished_utc"]


def test_cli_reports_optical_only_stage_as_nonzero_partial_modality(tmp_path, monkeypatch):
    inputs = _fixture_inputs(tmp_path)
    requests = []

    def local_asf_fixture(*args, **kwargs):
        return {"origin": "constructed_fixture", "status": "NETWORK_PAUSED"}

    def local_asset_response(request, timeout):
        url = request.full_url
        requests.append(url)
        for item in inputs["c1_fixtures"].values():
            for role, asset in item["assets"].items():
                if asset["href"] == url:
                    selection = json.loads(inputs["selection"].read_text(encoding="utf-8"))
                    site_id = next(site_id for site_id, selected_item in inputs["c1_fixtures"].items()
                                   if selected_item is item)
                    etag = selection["sentinel2_c1"][site_id]["assets"][role]["head"][
                        "response_headers"]["ETag"]
                    return _MemoryResponse((role + "-constructed-fixture").encode(), url,
                                           asset["file:size"], etag)
        raise AssertionError("unknown constructed fixture URL")

    def fixture_inspection(*args, **kwargs):
        return {"origin": "constructed_fixture",
                "status": "CONSTRUCTED_FIXTURE_INSPECTED_NOT_RESEARCH_ADMITTED",
                "research_admissible": False}

    monkeypatch.setattr(runner, "stage_feature", local_asf_fixture)
    monkeypatch.setattr(earth_search_c1.urllib.request, "urlopen", local_asset_response)
    monkeypatch.setattr(runner, "decode_c1_bundle", fixture_inspection)

    assert runner.main(_argv(inputs, mode="stage")) == 2
    report = _report(inputs)
    assert len(requests) == 8
    assert report["status"] == "PARTIAL_MODALITY_OR_STAGE"
    assert report["modality_status"] == {
        "sentinel1_asf_grd": "INCOMPLETE_OR_UNAVAILABLE",
        "sentinel2_c1": "COMPLETE",
    }
    assert report["sentinel1_asf_grd"]["status"] == "NETWORK_PAUSED"
    assert report["research_admitted"] is False
    output_manifest = json.loads((inputs["output"] / "outputs_sha256.json").read_text())
    assert output_manifest["file_count"] >= 16
    assert report["output_manifest_sha256"] == _sha(
        (inputs["output"] / "outputs_sha256.json").read_bytes())
    for entry in output_manifest["files"]:
        path = inputs["output"] / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert _sha(path.read_bytes()) == entry["sha256"]


def test_cli_rejects_shifted_native_grid_before_stage(tmp_path, monkeypatch):
    inputs = _fixture_inputs(tmp_path, shifted_grid=True)
    called = []
    monkeypatch.setattr(runner, "stage_feature", lambda *args, **kwargs: called.append("asf"))
    monkeypatch.setattr(runner, "stage_asset", lambda *args, **kwargs: called.append("c1"))

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert "grid footprint differs" in report["literal_error"]
    assert called == []


def test_cli_enforces_grouped_bytes_per_logical_product(tmp_path):
    inputs = _fixture_inputs(tmp_path, asf_bytes=16, per_product_limit=32)

    assert runner.main(_argv(inputs)) == 2
    report = _report(inputs)
    assert report["status"] == "PREFLIGHT_REJECTED"
    assert "grouped logical product bytes exceed" in report["literal_error"]
    assert report["network_call_status"] == "NOT_INVOKED"


def test_safe_stage_completion_requires_manifest_annotation_and_role_identity():
    receipt = {"status": "COMPLETE_ARCHIVE_BYTES_VERIFIED",
        "verified_output_bytes": 128,
        "verified_output_sha256": "a" * 64,
        "verified_output_path": "asf/constructed.zip",
        "output_bytes": 128, "output_sha256": "a" * 64,
        "safe_roles": {
        "archive_crc_status": "ALL_ZIP_MEMBER_CRC_CHECKS_PASSED",
        "manifest_identity_status": "SCOPED_MISSION_PRODUCT_ORBIT_TIME_POLARIZATION_AND_MEMBER_JOIN_MATCHED",
        "annotation_identity_status": "MISSION_PRODUCT_TIME_POLARIZATION_MATCHED",
        "role_mapping_status": "SAFE_SCOPED_MEMBER_AND_ROLE_IDENTITIES_MATCHED",
        "research_admitted": False,
    }}

    assert runner._asf_complete(receipt)
    receipt["safe_roles"]["annotation_identity_status"] = "UNAVAILABLE"
    assert not runner._asf_complete(receipt)


def _constructed_safe_payload(scene: str) -> bytes:
    members = role_members(scene, "VV")
    manifest = scoped_manifest(scene, members)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(scene + ".SAFE/manifest.safe", manifest)
        for name, payload in members.items():
            archive.writestr(scene + ".SAFE/" + name, payload)
    return buffer.getvalue()


def _rewrite_safe_measurement(payload: bytes, replacement: bytes) -> bytes:
    source = io.BytesIO(payload)
    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    measurement = next(name for name in members
                       if "/measurement/" in name and name.lower().endswith((".tif", ".tiff")))
    members[measurement] = replacement
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, member in members.items():
            archive.writestr(name, member)
    return output.getvalue()


@pytest.mark.parametrize("mutation", ["missing_before_safe", "changed_after_safe"])
def test_asf_complete_receipt_cannot_outlive_missing_or_changed_archive(
        tmp_path, monkeypatch, mutation):
    scene = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F"
    original = _constructed_safe_payload(scene)
    original_md5 = hashlib.md5(original, usedforsecurity=False).hexdigest()
    inputs = _fixture_inputs(tmp_path, asf_bytes=len(original), asf_md5=original_md5)
    observed_safe = []

    def constructed_stage_feature(feature, site_joins, destination, *, context=None, **kwargs):
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        receipt = {
            "origin": "constructed_fixture",
            "status": "COMPLETE_ARCHIVE_BYTES_VERIFIED",
            "fileID": feature["properties"]["fileID"],
            "site_joins": sorted(site_joins),
            "expected_bytes": len(original),
            "output_bytes": len(original),
            "output_sha256": _sha(original),
            "sha256": _sha(original),
            "computed_md5": original_md5,
            "request_started_utc": runner._now(),
            "request_finished_utc": runner._now(),
            "runtime": runner._runtime(),
            "selection_sha256": context["selection_sha256"],
            "research_admitted": False,
        }
        runner._write_json(target.with_name(target.name + ".download.json"), receipt)
        if mutation == "missing_before_safe":
            target.unlink()
        return receipt

    def inspect_then_maybe_change(path, expected_scene, polarization):
        safe = asf_grd.inspect_safe_roles(
            path, expected_scene, polarization, origin="constructed_fixture")
        observed_safe.append(safe)
        if mutation == "changed_after_safe":
            Path(path).write_bytes(_rewrite_safe_measurement(
                Path(path).read_bytes(), b"changed after receipt; valid rebuilt ZIP CRC"))
        return safe

    def constructed_c1_not_attempted(*args, **kwargs):
        return {"origin": "constructed_fixture", "status": "NETWORK_PAUSED"}

    monkeypatch.setattr(runner, "stage_feature", constructed_stage_feature)
    monkeypatch.setattr(runner, "inspect_safe_roles", inspect_then_maybe_change)
    monkeypatch.setattr(runner, "stage_asset", constructed_c1_not_attempted)

    assert runner.main(_argv(inputs, mode="stage")) == 2
    report = _report(inputs)
    sar = report["sentinel1_asf_grd"]
    assert sar["status"] == "OUTPUT_BINDING_FAILED"
    assert sar["transfer_receipt_status"] == "COMPLETE_ARCHIVE_BYTES_VERIFIED"
    assert sar["expected_output_bytes"] == len(original)
    assert sar["expected_output_sha256"] == _sha(original)
    assert report["modality_status"]["sentinel1_asf_grd"] == "INCOMPLETE_OR_UNAVAILABLE"
    assert not runner._asf_complete(sar)
    assert any(failure["branch"] == "ASF_GRD" and
               failure["cause_status"] == "OUTPUT_BYTES_DO_NOT_MATCH_RETAINED_TRANSFER_RECEIPT"
               for failure in report["failures"])

    output = inputs["output"] / "asf" / (scene + ".zip")
    if mutation == "missing_before_safe":
        assert not output.exists()
        assert sar["actual_output_bytes"] is None
        assert sar["actual_output_sha256"] is None
        assert observed_safe == []
    else:
        assert output.is_file()
        assert sar["actual_output_bytes"] == output.stat().st_size
        assert sar["actual_output_sha256"] == _sha(output.read_bytes())
        assert sar["actual_output_sha256"] != sar["expected_output_sha256"]
        assert observed_safe[0]["archive_crc_status"] == "ALL_ZIP_MEMBER_CRC_CHECKS_PASSED"
        assert observed_safe[0]["manifest_identity_status"] == (
            "SCOPED_MISSION_PRODUCT_ORBIT_TIME_POLARIZATION_AND_MEMBER_JOIN_MATCHED")
        after_change = asf_grd.inspect_safe_roles(
            output, scene, "VV", origin="constructed_fixture")
        assert after_change["archive_crc_status"] == "ALL_ZIP_MEMBER_CRC_CHECKS_PASSED"
        assert after_change["manifest_identity_status"] == (
            "SCOPED_MISSION_PRODUCT_ORBIT_TIME_POLARIZATION_AND_MEMBER_JOIN_MATCHED")

    retained_receipt = json.loads(output.with_name(output.name + ".download.json").read_text())
    assert retained_receipt["status"] == "COMPLETE_ARCHIVE_BYTES_VERIFIED"
    assert retained_receipt["output_sha256"] == _sha(original)
