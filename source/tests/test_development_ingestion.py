"""CONSTRUCTED FIXTURES ONLY: no payload here establishes provider observations."""
import copy
import base64
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sentinel_gl.admission import AdmissionError
from sentinel_gl.ingestion import development as d
from sentinel_gl.ingestion import development_pixels as pixels


def manifest():
    return {"schema_version": 1, "purpose": "development_acquisition_only",
            "research_executable": False, "crs": "EPSG:4326",
            "acquisition_start": "2023-09-01T00:00:00Z",
            "acquisition_end_exclusive": "2023-11-01T00:00:00Z",
            "pre_event_cutoff_exclusive": "2023-10-03T00:00:00+05:30",
            "sites": [{"site_id": "FIXTURE-SITE", "scope": "DEVELOPMENT_EXPOSED",
                       "latitude": 0.0, "longitude": 0.0}],
            "products": list(d.PRODUCTS)}


def product(identifier="00000000-0000-4000-8000-000000000001", end="2023-10-02T18:29:59Z"):
    return {"Id": identifier, "Name": "S2A_MSIL2A_FIXTURE.SAFE",
            "ContentDate": {"Start": "2023-10-02T10:00:00Z", "End": end}}


class Response(io.BytesIO):
    def __init__(self, payload, declared=None, headers=None):
        super().__init__(payload)
        self.headers = dict(headers or {})
        if declared is not None:
            self.headers["Content-Length"] = str(declared)


def archive_bytes(name="S2A_MSIL2A_FIXTURE.SAFE", extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name_tail in ("MTD_MSIL2A.xml", "granule_B03_10m.jp2", "granule_B08_10m.jp2",
                          "granule_B11_20m.jp2", "granule_SCL_20m.jp2"):
            z.writestr(name + "/" + name_tail, "fixture bytes; not a decoded image")
        if extra:
            z.writestr(extra, "fixture malicious path")
    return buffer.getvalue()


def test_manifest_scope_and_centroid_never_prove_lake_or_control():
    m = manifest()
    assert d.validate_manifest(m)["sites"][0]["scope"] == "DEVELOPMENT_EXPOSED"
    for changes in ({"schema_version": True}, {"research_executable": True},
                    {"acquisition_end_exclusive": "2023-12-01T00:00:00Z"}):
        with pytest.raises(AdmissionError):
            d.validate_manifest({**m, **changes})
    bad = copy.deepcopy(m); bad["sites"][0]["latitude"] = float("nan")
    with pytest.raises(AdmissionError):
        d.validate_manifest(bad)


def test_pre_event_cutoff_uses_timezone_and_source_end_not_start(tmp_path, monkeypatch):
    rows = [product(), product("00000000-0000-4000-8000-000000000002", "2023-10-02T18:30:00Z")]
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(
        json.dumps({"value": rows if "SENTINEL-2" in url else []}).encode()))
    result = d.discover(manifest(), tmp_path)
    assert result["products"][rows[0]["Id"]]["temporal_support"] == "STRICTLY_PRE_EVENT"
    assert result["products"][rows[1]["Id"]]["temporal_support"].startswith("EVENT_OR_POST")
    assert result["roi_qa"] is result["historical_availability"] is None
    assert result["research_admissible"] is False


def test_full_pagination_shared_product_dedup_and_conflict(tmp_path, monkeypatch):
    m = manifest(); m["products"] = ["SENTINEL-2_MSI_L2A"]
    m["sites"].append({**m["sites"][0], "site_id": "FIXTURE-OTHER"})
    second = d.CATALOGUE + "?second-page"
    calls = []
    def fetch(url, token=None):
        calls.append(url)
        return Response(json.dumps({"value": [product()],
                                    **({"@odata.nextLink": second} if url != second else {})}).encode())
    monkeypatch.setattr(d, "request", fetch)
    result = d.discover(m, tmp_path)
    assert len(calls) == 4 and len(result["products"]) == 1
    assert result["products"][product()["Id"]]["sites"] == ["FIXTURE-SITE", "FIXTURE-OTHER"]
    assert all(len(q["pages"]) == 2 for q in result["queries"])


def test_foreign_or_cyclic_next_page_rejects_incomplete_catalogue(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(
        json.dumps({"value": [], "@odata.nextLink": "https://example.org/private"}).encode()))
    with pytest.raises(AdmissionError, match="pagination"):
        d.discover(manifest(), tmp_path)
    assert list((tmp_path / "responses").iterdir())


def test_invalid_product_type_or_future_acquisition_is_rejected():
    for row in ({**product(), "Name": "S2A_MSIL1C_FIXTURE.SAFE"},
                {**product(), "ContentDate": {"Start": "2023-11-01T00:00:00Z", "End": "2023-11-01T01:00:00Z"}}):
        with pytest.raises(AdmissionError):
            d.validate_product(row, manifest(), "SENTINEL-2_MSI_L2A")


def test_archive_structure_is_never_pixel_or_scientific_verification(tmp_path):
    path = tmp_path / "fixture.zip"; path.write_bytes(archive_bytes())
    checked = d.inspect_archive(path, product()["Name"], "SENTINEL-2_MSI_L2A")
    assert checked["pixel_verification"] == "NOT_DECODED"
    path.write_bytes(archive_bytes(extra="../escape"))
    with pytest.raises(AdmissionError, match="unsafe"):
        d.inspect_archive(path, product()["Name"], "SENTINEL-2_MSI_L2A")


def test_stage_rechecks_provider_identity_and_rejects_truncation(tmp_path, monkeypatch):
    row = product(); m = manifest()
    catalogue = {"manifest": m, "status": "CATALOGUE_ONLY", "products": {
        row["Id"]: {"provider_record": row, "product": "SENTINEL-2_MSI_L2A", "sites": ["FIXTURE-SITE"]}}}
    payload = archive_bytes()
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(payload, len(payload) + 1)
                        if token else Response(json.dumps(row).encode()))
    with pytest.raises(AdmissionError, match="truncated"):
        d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)
    assert list(tmp_path.glob("*.partial")) and not list(tmp_path.glob("*.zip"))


def test_redirect_never_forwards_credentials_to_foreign_host():
    request = urllib.request.Request(d.DOWNLOAD, headers={"Authorization": "Bearer fixture-secret"})
    with pytest.raises(AdmissionError, match="credentials"):
        d.ProviderRedirect().redirect_request(request, None, 302, "redirect", {}, "https://example.org/upload")


def test_main_preserves_existing_attempt_and_digest_failure_writes_nothing(tmp_path):
    path = tmp_path / "input.json"; path.write_text(json.dumps(manifest()))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "prior"; out.mkdir(); (out / "history.txt").write_text("retain")
    argv = ["--manifest", str(path), "--manifest-sha256", sha, "--output-dir", str(out)]
    assert d.main(argv) == 2 and sorted(p.name for p in out.iterdir()) == ["history.txt"]
    assert d.main([*argv[:3], "0" * 64, *argv[4:]]) == 2
    assert not (out / "failure.json").exists()


def stage_inputs(row=None):
    row = row or product()
    m = manifest()
    catalogue = {"manifest": m, "status": "CATALOGUE_ONLY", "products": {
        row["Id"]: {"provider_record": row, "product": "SENTINEL-2_MSI_L2A",
                    "sites": ["FIXTURE-SITE"]}}}
    return row, m, catalogue


def test_catalogue_length_and_checksum_are_not_compared_to_unbound_download_representation(tmp_path, monkeypatch):
    payload = archive_bytes()
    row = {**product(), "ContentLength": len(payload) + 10,
           "Checksum": [{"Algorithm": "MD5", "Value": "0" * 32}]}
    row, m, catalogue = stage_inputs(row)
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(
        payload if token else json.dumps(row).encode(), len(payload) if token else None))
    result = d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)
    item = result["staged"][0]
    receipt = json.loads((tmp_path / (row["Id"] + ".download.json")).read_text())
    assert item["bytes"] == len(payload)
    assert receipt["catalogue_content_length_status"] == "PRESENT_SCOPE_UNBOUND_TO_VALUE_RESPONSE"
    assert receipt["provider_checksum_status"] == "NOT_BOUND_TO_DOWNLOAD_REPRESENTATION"
    assert receipt["http_content_length_status"] == "MATCHED_BODY_LENGTH"


def test_optional_provider_metadata_absence_is_preserved(tmp_path, monkeypatch):
    row, m, catalogue = stage_inputs()
    payload = archive_bytes()
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(
        payload if token else json.dumps(row).encode(), len(payload) if token else None))
    result = d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)
    item = result["staged"][0]
    assert item["catalogue_content_length_status"] == "OPTIONAL_METADATA_ABSENT"
    assert item["provider_checksums"] == {}
    assert item["optional_identity_metadata"]["Checksum"] == "ABSENT_IN_DISCOVERY_AND_CURRENT"


def test_provider_identity_change_is_rejected_without_indexing_optional_fields(tmp_path, monkeypatch):
    row, m, catalogue = stage_inputs()
    changed = {**row, "Name": "S2A_MSIL2A_CHANGED.SAFE"}
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(json.dumps(changed).encode()))
    with pytest.raises(AdmissionError, match="changed"):
        d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)


@pytest.mark.parametrize("matches", [True, False])
def test_http_content_checksum_match_and_mismatch_are_checked(tmp_path, monkeypatch, matches):
    row, m, catalogue = stage_inputs()
    payload = archive_bytes()
    digest = hashlib.md5(payload, usedforsecurity=False).digest()
    if not matches:
        digest = bytes([digest[0] ^ 1]) + digest[1:]
    headers = {"Content-MD5": base64.b64encode(digest).decode("ascii")}
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(
        payload if token else json.dumps(row).encode(), len(payload) if token else None,
        headers if token else None))
    if matches:
        result = d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)
        assert result["staged"][0]["http_checksum_status"] == "MATCHED_HTTP_CONTENT_CHECKSUM"
    else:
        with pytest.raises(AdmissionError, match="HTTP response checksum"):
            d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)
        state = json.loads((tmp_path / (row["Id"] + ".download.json")).read_text())
        assert state["status"] == "HTTP_CONTENT_CHECKSUM_MISMATCH"
        assert not list(tmp_path.glob("*.zip"))


def test_unsupported_catalogue_checksum_algorithm_is_retained_as_unsupported(tmp_path, monkeypatch):
    row = {**product(), "Checksum": [{"Algorithm": "BLAKE3", "Value": "fixture-digest"}]}
    row, m, catalogue = stage_inputs(row)
    payload = archive_bytes()
    monkeypatch.setattr(d, "request", lambda url, token=None: Response(
        payload if token else json.dumps(row).encode(), len(payload) if token else None))
    result = d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-token", 100000)
    checksum = result["staged"][0]["provider_checksums"]["BLAKE3"]
    assert checksum["value"] == "fixture-digest"
    assert checksum["verification_status"] == "UNSUPPORTED_ALGORITHM"


class InterruptedResponse(Response):
    def __init__(self, payload, message):
        super().__init__(payload)
        self.message = message
        self.read_count = 0

    def read(self, size=-1):
        self.read_count += 1
        if self.read_count == 1:
            return self.getvalue()[:17]
        raise OSError(self.message)


def test_interrupted_partial_bytes_hash_and_credential_redaction_are_preserved(tmp_path, monkeypatch):
    row, m, catalogue = stage_inputs()
    payload = archive_bytes()
    def fetch(url, token=None):
        if token:
            return InterruptedResponse(payload, "request failed fixture-secret")
        return Response(json.dumps(row).encode())
    monkeypatch.setattr(d, "request", fetch)
    with pytest.raises(OSError):
        d.stage(catalogue, m, tmp_path, [row["Id"]], "fixture-secret", 100000)
    partial = tmp_path / (row["Id"] + ".zip.partial")
    state = json.loads((tmp_path / (row["Id"] + ".download.json")).read_text())
    assert partial.read_bytes() == payload[:17]
    assert state["partial_bytes"] == 17
    assert state["partial_sha256"] == hashlib.sha256(payload[:17]).hexdigest()
    assert state["status"] == "INTERRUPTED_OR_INCOMPLETE_RETAINED"
    assert state["resume_supported"] is False
    assert state["resume_status"] == "UNSUPPORTED_NEW_ATTEMPT_REQUIRED"
    assert "fixture-secret" not in json.dumps(state)
    assert "[REDACTED]" in state["error"]


def source_fixture(tmp_path, product_name, product_id="00000000-0000-4000-8000-000000000001"):
    record_path = tmp_path / "provider-record.json"
    record_path.write_text(json.dumps({"Id": product_id, "Name": product_name,
                                       "ContentDate": {"Start": "2023-10-02T10:00:00Z",
                                                       "End": "2023-10-02T10:15:00Z"}}))
    archive_path = tmp_path / "fixture-source.zip"
    archive_path.write_bytes(b"constructed source archive bytes")
    return {"origin": "constructed_fixture", "provider_product_id": product_id,
            "provider_product_name": product_name, "provider_record_path": str(record_path),
            "provider_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}, archive_path


def bind_archive(source, archive_path, members):
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member, path in members.items():
            archive.write(path, member)
    source["archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return {member: path for member, path in members.items()}


def write_tiff(path, values, resolution):
    array = np.asarray(values, dtype=np.uint16)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype, crs="EPSG:32645",
                       transform=from_origin(500000, 5100000, resolution, resolution)) as dataset:
        dataset.write(array, 1)


def sentinel2_fixture(tmp_path, metadata_additions=""):
    name = "S2A_MSIL2A_FIXTURE.SAFE"
    source, archive = source_fixture(tmp_path, name)
    metadata = tmp_path / "MTD_MSIL2A.xml"
    metadata.write_text(f"""<Product>
      <PRODUCT_URI>{name}</PRODUCT_URI>
      <PRODUCT_START_TIME>2023-10-02T10:00:00Z</PRODUCT_START_TIME>
      <PRODUCT_STOP_TIME>2023-10-02T10:15:00Z</PRODUCT_STOP_TIME>
      <BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>
      <Spectral_Information physicalBand="B03" bandId="2"/>
      <Spectral_Information physicalBand="B08" bandId="7"/>
      <Spectral_Information physicalBand="B11" bandId="10"/>
      <BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET>
      <BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>
      <BOA_ADD_OFFSET band_id="10">-1000</BOA_ADD_OFFSET>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B03" member="{name}/B03.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B08" member="{name}/B08.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B11" member="{name}/B11.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_SCL" member="{name}/SCL.tif"/>
      {metadata_additions}
    </Product>""")
    bands = {}
    for band, resolution, base in (("B03", 10, 11000), ("B08", 10, 21000), ("B11", 20, 31000)):
        bands[band] = tmp_path / f"{band}.tif"
        write_tiff(bands[band], [[base, base + 1000], [0, base + 3000]], resolution)
    scl = tmp_path / "SCL.tif"
    write_tiff(scl, [[4, 6], [0, 11]], 20)
    members = {f"{name}/MTD_MSIL2A.xml": metadata,
               **{f"{name}/{band}.tif": path for band, path in bands.items()},
               f"{name}/SCL.tif": scl}
    bind_archive(source, archive, members)
    return source, archive, metadata, bands, scl, members


def sentinel1_fixture(tmp_path, *, noise_role_member=None, noise_polarization="VV"):
    name = "S1A_IW_GRDH_1SDV_FIXTURE.SAFE"
    source, archive = source_fixture(tmp_path, name)
    record_path = Path(source["provider_record_path"])
    record = json.loads(record_path.read_text())
    record["Name"] = name
    record_path.write_text(json.dumps(record))
    source["provider_record_sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.xml"
    calibration.write_text(f"""<calibration><adsHeader><polarisation>VV</polarisation>
      <startTime>2023-10-02T10:00:00Z</startTime><stopTime>2023-10-02T10:15:00Z</stopTime></adsHeader>
      <Engineering_Fixture_Asset_Role role="S1_CALIBRATION" polarization="VV" member="{name}/calibration.xml"/>
      <Engineering_Fixture_Asset_Role role="S1_MEASUREMENT" polarization="VV" member="{name}/measurement.tif"/>
      <calibrationVector><line>0</line><pixel>0 1</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
      <calibrationVector><line>1</line><pixel>0 1</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
      </calibration>""")
    noise_member = noise_role_member or f"{name}/noise.xml"
    noise = tmp_path / "noise.xml"
    noise.write_text(f"""<noise><polarisation>{noise_polarization}</polarisation>
      <Engineering_Fixture_Asset_Role role="S1_NOISE" polarization="{noise_polarization}" member="{noise_member}"/>
      <noiseRangeVector><line>0</line><pixel>0 1</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseRangeVector>
      <noiseRangeVector><line>1</line><pixel>0 1</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseRangeVector>
      </noise>""")
    measurement = tmp_path / "measurement.tif"
    write_tiff(measurement, [[4, 5], [6, 7]], 10)
    members = {f"{name}/calibration.xml": calibration, f"{name}/noise.xml": noise,
               f"{name}/measurement.tif": measurement}
    bind_archive(source, archive, members)
    return source, archive, calibration, noise, measurement, members


def test_sentinel2_constructed_raster_decodes_metadata_offsets_masks_and_source_join(tmp_path):
    name = "S2A_MSIL2A_FIXTURE.SAFE"
    source, archive = source_fixture(tmp_path, name)
    metadata = tmp_path / "MTD_MSIL2A.xml"
    metadata.write_text(f"""<Product>
      <PRODUCT_URI>{name}</PRODUCT_URI>
      <PRODUCT_START_TIME>2023-10-02T10:00:00Z</PRODUCT_START_TIME>
      <PRODUCT_STOP_TIME>2023-10-02T10:15:00Z</PRODUCT_STOP_TIME>
      <BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>
      <Spectral_Information_List>
        <Spectral_Information physicalBand="B03" bandId="2"/>
        <Spectral_Information physicalBand="B08" bandId="7"/>
        <Spectral_Information physicalBand="B11" bandId="10"/>
      </Spectral_Information_List>
      <BOA_ADD_OFFSET_VALUES_LIST>
        <BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="10">-1000</BOA_ADD_OFFSET>
      </BOA_ADD_OFFSET_VALUES_LIST>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B03" member="{name}/B03.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B08" member="{name}/B08.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B11" member="{name}/B11.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_SCL" member="{name}/SCL.tif"/>
    </Product>""")
    bands = {}
    for band, resolution, base in (("B03", 10, 11000), ("B08", 10, 21000), ("B11", 20, 31000)):
        bands[band] = tmp_path / f"{band}.tif"
        write_tiff(bands[band], [[base, base + 1000], [0, base + 3000]], resolution)
    scl = tmp_path / "SCL.tif"
    write_tiff(scl, [[4, 6], [0, 11]], 20)
    archive_members = {f"{name}/MTD_MSIL2A.xml": metadata,
                       **{f"{name}/{band}.tif": path for band, path in bands.items()},
                       f"{name}/SCL.tif": scl}
    bind_archive(source, archive, archive_members)
    result = pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive,
                                         {f"{name}/MTD_MSIL2A.xml": metadata,
                                          **{f"{name}/{band}.tif": path for band, path in bands.items()},
                                          f"{name}/SCL.tif": scl})
    assert result["status"] == "PIXELS_DECODED_DEVELOPMENT_ONLY"
    assert result["research_admissible"] is False
    assert result["source_join"]["evidence_scope"] == "CONSTRUCTED_ENGINEERING_FIXTURE_ONLY"
    assert result["source_time_join"] == "MATCHED_METADATA_WITHIN_PROVIDER_INTERVAL"
    assert result["bands"]["B03"]["reflectance"][0, 0] == pytest.approx(1.0)
    assert result["bands"]["B08"]["reflectance"][0, 0] == pytest.approx(2.0)
    assert result["bands"]["B11"]["reflectance"][0, 0] == pytest.approx(3.0)
    assert bool(result["bands"]["B03"]["reflectance"].mask[1, 0])
    assert result["bands"]["B03"]["missing_reason_code"][1, 0] == 2
    assert result["bands"]["B03"]["grid"]["crs_authority"] == ["EPSG", "32645"]
    assert result["bands"]["B03"]["grid"]["resolution"] != result["bands"]["B11"]["grid"]["resolution"]
    assert result["grid_alignment"] == "PER_ASSET_GRIDS_RETAINED_NO_IMPLICIT_RESAMPLING"
    assert result["scene_classification"]["qa_policy_status"] == "RAW_CLASSES_ONLY_NO_VALIDITY_POLICY_APPLIED"


def test_sentinel2_missing_offset_or_changed_archive_fails_closed(tmp_path):
    name = "S2A_MSIL2A_FIXTURE.SAFE"
    source, archive = source_fixture(tmp_path, name)
    metadata = tmp_path / "MTD_MSIL2A.xml"
    metadata.write_text(f"""<Product><PRODUCT_URI>{name}</PRODUCT_URI><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B03" member="{name}/B03.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B08" member="{name}/B08.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_BAND" band="B11" member="{name}/B11.tif"/>
      <Engineering_Fixture_Asset_Role role="S2_SCL" member="{name}/SCL.tif"/>
      </Product>""")
    bands = {}
    for band in ("B03", "B08", "B11"):
        bands[band] = tmp_path / f"{band}.tif"
        write_tiff(bands[band], [[1, 2], [3, 4]], 10)
    scl = tmp_path / "SCL.tif"; write_tiff(scl, [[4, 4], [4, 4]], 20)
    members = {f"{name}/MTD_MSIL2A.xml": metadata,
               **{f"{name}/{band}.tif": path for band, path in bands.items()},
               f"{name}/SCL.tif": scl}
    bind_archive(source, archive, members)
    with pytest.raises(AdmissionError, match="offset is unresolved"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)
    archive.write_bytes(b"tampered")
    with pytest.raises(AdmissionError, match="digest mismatch"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel2_distinct_band_fixture_rejects_caller_path_swap(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    swapped = {**bands, "B03": bands["B08"], "B08": bands["B03"]}
    with pytest.raises(AdmissionError, match="B03 role does not match"):
        pixels.decode_sentinel2_l2a(metadata, swapped, scl, source, archive, members)


def test_sentinel2_rejects_conflicting_duplicate_band_offset(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    metadata.write_text(metadata.read_text().replace(
        "</Product>", "<BOA_ADD_OFFSET band_id=\"2\">-2000</BOA_ADD_OFFSET></Product>"))
    bind_archive(source, archive, members)
    with pytest.raises(AdmissionError, match="conflicting duplicate BOA additive offset"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel2_rejects_conflicting_duplicate_band_identity(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    metadata.write_text(metadata.read_text().replace(
        "</Product>", "<Spectral_Information physicalBand=\"B03\" bandId=\"3\"/></Product>"))
    bind_archive(source, archive, members)
    with pytest.raises(AdmissionError, match="conflicting duplicate spectral identity"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel2_rejects_conflicting_duplicate_metadata_time(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    metadata.write_text(metadata.read_text().replace(
        "</Product>", "<PRODUCT_START_TIME>2023-10-02T10:01:00Z</PRODUCT_START_TIME></Product>"))
    bind_archive(source, archive, members)
    with pytest.raises(AdmissionError, match="conflicting duplicate metadata acquisition start time"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel2_rejects_conflicting_duplicate_quantification(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    metadata.write_text(metadata.read_text().replace(
        "</Product>", "<BOA_QUANTIFICATION_VALUE>20000</BOA_QUANTIFICATION_VALUE></Product>"))
    bind_archive(source, archive, members)
    with pytest.raises(AdmissionError, match="conflicting duplicate BOA quantification value"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel2_rejects_reversed_metadata_interval_before_join(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    metadata.write_text(metadata.read_text().replace("2023-10-02T10:00:00Z", "2023-10-02T10:10:00Z")
                        .replace("2023-10-02T10:15:00Z", "2023-10-02T10:05:00Z"))
    bind_archive(source, archive, members)
    with pytest.raises(AdmissionError, match="metadata acquisition interval is reversed"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel2_rejects_ambiguous_archive_member_role_mapping(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    ambiguous = {**members, "S2A_MSIL2A_FIXTURE.SAFE/alias_B03.tif": bands["B03"]}
    bind_archive(source, archive, ambiguous)
    with pytest.raises(AdmissionError, match="ambiguous archive member mapping"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, ambiguous)


def test_provider_sentinel2_decoding_is_blocked_without_verified_role_mapping(tmp_path):
    source, archive, metadata, bands, scl, members = sentinel2_fixture(tmp_path)
    source["origin"] = "provider_download"
    with pytest.raises(AdmissionError, match="provider Sentinel-2 pixel decoding is blocked"):
        pixels.decode_sentinel2_l2a(metadata, bands, scl, source, archive, members)


def test_sentinel1_constructed_calibration_noise_grid_and_missingness(tmp_path):
    name = "S1A_IW_GRDH_1SDV_FIXTURE.SAFE"
    source, archive = source_fixture(tmp_path, name)
    metadata_record = json.loads(Path(source["provider_record_path"]).read_text())
    metadata_record["Name"] = name
    metadata_path = Path(source["provider_record_path"])
    metadata_path.write_text(json.dumps(metadata_record))
    source["provider_record_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.xml"
    calibration.write_text("""<calibration><adsHeader><polarisation>VV</polarisation>
      <startTime>2023-10-02T10:00:00Z</startTime><stopTime>2023-10-02T10:15:00Z</stopTime></adsHeader>
      <Engineering_Fixture_Asset_Role role="S1_CALIBRATION" polarization="VV" member="S1A_IW_GRDH_1SDV_FIXTURE.SAFE/calibration.xml"/>
      <Engineering_Fixture_Asset_Role role="S1_MEASUREMENT" polarization="VV" member="S1A_IW_GRDH_1SDV_FIXTURE.SAFE/measurement.tif"/>
      <calibrationVectorList>
        <calibrationVector><line>0</line><pixel>0 1</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
        <calibrationVector><line>1</line><pixel>0 1</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
      </calibrationVectorList></calibration>""")
    noise = tmp_path / "noise.xml"
    noise.write_text("""<noise><polarisation>VV</polarisation>
      <Engineering_Fixture_Asset_Role role="S1_NOISE" polarization="VV" member="S1A_IW_GRDH_1SDV_FIXTURE.SAFE/noise.xml"/>
      <noiseRangeVectorList>
      <noiseRangeVector><line>0</line><pixel>0 1</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseRangeVector>
      <noiseRangeVector><line>1</line><pixel>0 1</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseRangeVector>
    </noiseRangeVectorList></noise>""")
    measurement = tmp_path / "measurement.tif"
    write_tiff(measurement, [[4, 0], [6, 8]], 10)
    members = {f"{name}/calibration.xml": calibration, f"{name}/noise.xml": noise,
               f"{name}/measurement.tif": measurement}
    bind_archive(source, archive, members)
    result = pixels.decode_sentinel1_grd(measurement, calibration, noise, "VV", source, archive,
                                         members, noise_lut_scale="POWER_DN_SQUARED")
    assert result["sigma0"][0, 0] == pytest.approx(4.0)
    assert bool(result["sigma0"].mask[0, 1])
    assert result["sigma0_noise_corrected"][0, 0] == pytest.approx(3.75)
    assert bool(result["sigma0_noise_corrected"].mask[0, 1])
    assert result["noise_correction_status"] == "APPLIED_EXPLICIT_POWER_DN_SQUARED_FIXTURE_SCALE"
    assert result["source_time_join"] == "MATCHED_METADATA_WITHIN_PROVIDER_INTERVAL"
    assert result["grid"]["crs_authority"] == ["EPSG", "32645"]
    assert result["research_admissible"] is False


def test_sentinel1_noise_without_verified_scale_remains_null_and_polarization_mismatch_fails(tmp_path):
    name = "S1A_IW_GRDH_1SDV_FIXTURE.SAFE"
    source, archive = source_fixture(tmp_path, name)
    record_path = Path(source["provider_record_path"])
    record = json.loads(record_path.read_text()); record["Name"] = name
    record_path.write_text(json.dumps(record))
    source["provider_record_sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.xml"
    calibration.write_text("""<calibration><adsHeader><polarisation>VV</polarisation></adsHeader>
      <Engineering_Fixture_Asset_Role role="S1_CALIBRATION" polarization="VV" member="S1A_IW_GRDH_1SDV_FIXTURE.SAFE/calibration.xml"/>
      <Engineering_Fixture_Asset_Role role="S1_MEASUREMENT" polarization="VV" member="S1A_IW_GRDH_1SDV_FIXTURE.SAFE/measurement.tif"/>
      <calibrationVector><line>0</line><pixel>0 1</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
      <calibrationVector><line>1</line><pixel>0 1</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>
      </calibration>""")
    noise = tmp_path / "noise.xml"
    noise.write_text("""<noise><polarisation>VV</polarisation>
      <Engineering_Fixture_Asset_Role role="S1_NOISE" polarization="VV" member="S1A_IW_GRDH_1SDV_FIXTURE.SAFE/noise.xml"/>
      <noiseRangeVector><line>0</line><pixel>0 1</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseRangeVector>
      <noiseRangeVector><line>1</line><pixel>0 1</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseVector>
      </noise>""".replace("</noiseVector>", "</noiseRangeVector>"))
    measurement = tmp_path / "measurement.tif"; write_tiff(measurement, [[4, 5], [6, 7]], 10)
    members = {f"{name}/calibration.xml": calibration, f"{name}/noise.xml": noise,
               f"{name}/measurement.tif": measurement}
    bind_archive(source, archive, members)
    result = pixels.decode_sentinel1_grd(measurement, calibration, noise, "VV", source, archive, members)
    assert result["sigma0_noise_corrected"] is None
    assert result["noise_correction_status"] == "NOT_APPLIED_NOISE_LUT_SCALE_UNVERIFIED"
    with pytest.raises(AdmissionError, match="polarization"):
        pixels.decode_sentinel1_grd(measurement, calibration, noise, "VH", source, archive, members)


def test_sentinel1_rejects_crossed_noise_member_and_polarization_roles(tmp_path):
    name = "S1A_IW_GRDH_1SDV_FIXTURE.SAFE"
    source, archive, calibration, noise, measurement, members = sentinel1_fixture(
        tmp_path, noise_role_member=f"{name}/calibration.xml")
    with pytest.raises(AdmissionError, match="noise role does not match"):
        pixels.decode_sentinel1_grd(measurement, calibration, noise, "VV", source, archive, members)


def test_sentinel1_rejects_noise_polarization_disagreement(tmp_path):
    source, archive, calibration, noise, measurement, members = sentinel1_fixture(
        tmp_path, noise_polarization="VH")
    with pytest.raises(AdmissionError, match="polarization do not match"):
        pixels.decode_sentinel1_grd(measurement, calibration, noise, "VV", source, archive, members)
