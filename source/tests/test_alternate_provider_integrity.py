"""Offline constructed-fixture regressions for alternate provider adapters.

No fixture created here is a provider observation or research measurement.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coordinates

from sentinel_gl.admission import AdmissionError
from sentinel_gl.ingestion import asf_grd, earth_search_c1
from safe_fixtures import role_identity_xml, role_members, scoped_manifest

START = "2023-09-01T00:00:00Z"
END = "2023-11-01T00:00:00Z"
CUTOFF = "2023-10-03T00:00:00+05:30"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _c1_fixture(tile="T45RXL", square="XL", date="20230904", *,
                item_time="2023-09-04T05:01:01.646000Z", product_time="2023-09-04T04:47:01Z",
                tile_time="2023-09-04T05:01:01.646994Z"):
    item_id = f"S2A_{tile}_{date}T044704_L2A"
    product_uri = f"S2A_MSIL2A_{date}T044701_N0511_R076_{tile}_20250807T205334.SAFE"
    tile_id = f"S2A_OPER_MSI_L2A_TL_S2RS_20250807T205334_A042830_{tile}_N05.11"
    properties = {"s2:processing_baseline": "05.11", "datetime": item_time,
                  "s2:product_uri": product_uri, "s2:generation_time": "2025-08-07T20:53:34Z",
                  "s2:tile_id": tile_id, "proj:epsg": 32645,
                  "mgrs:utm_zone": 45, "mgrs:latitude_band": "R", "mgrs:grid_square": square}
    assets = {}
    checksums = {}
    for role, band, leaf, gsd, shape, transform in (
        ("green", "B03", "B03.tif", 10, [2, 2], [10, 0, 600000, 0, -10, 3100020]),
        ("nir", "B08", "B08.tif", 10, [2, 2], [10, 0, 600000, 0, -10, 3100020]),
        ("swir16", "B11", "B11.tif", 20, [1, 1], [20, 0, 600000, 0, -20, 3100020]),
        ("scl", "SCL", "SCL.tif", 20, [1, 1], [20, 0, 600000, 0, -20, 3100020]),
    ):
        href = ("https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/"
                f"sentinel-2-c1-l2a/45/R/{square}/2023/9/{item_id}/{leaf}")
        raw = (role + "-constructed-fixture").encode()
        asset = {"href": href, "file:size": len(raw), "file:checksum": "1220" + _sha(raw),
                 "gsd": gsd, "proj:shape": shape, "proj:transform": transform,
                 "raster:bands": [{"nodata": 0, "data_type": "uint8" if role == "scl" else "uint16",
                                   "spatial_resolution": gsd}]}
        if role != "scl":
            asset["raster:bands"][0].update({"scale": 0.0001, "offset": -0.1})
            asset["eo:bands"] = [{"name": band}]
        assets[role] = asset
        checksums[role] = {"href": href, "file_checksum_multihash": asset["file:checksum"],
                           "file_size": asset["file:size"], "band": band}
    item = {"id": item_id, "properties": properties, "assets": assets}
    selection = {"item_id": item_id, "product_uri": product_uri, "baseline": "05.11",
                 "acquisition_datetime": item_time, "generation_time": properties["s2:generation_time"],
                 "assets": checksums}
    image_files = []
    role_resolutions = {"B03": ("10m", "20m", "60m"), "B08": ("10m",),
                        "B11": ("20m", "60m"), "SCL": ("20m", "60m")}
    for band, resolutions in role_resolutions.items():
        for resolution in resolutions:
            image_files.append(f"GRANULE/L2A_{tile}_A042830_{date}T044704/IMG_DATA/R{resolution}/"
                               f"{tile}_{date}T044701_{band}_{resolution}")
    product_tags = "".join(f'<Spectral_Information physicalBand="{band}" bandId="{bid}"/>'
                            for band, bid in (("B3", 2), ("B8", 7), ("B11", 11)))
    offsets = "".join(f'<BOA_ADD_OFFSET band_id="{bid}">-1000</BOA_ADD_OFFSET>' for bid in (2, 7, 11))
    product = (f"<Product><PRODUCT_URI>{product_uri}</PRODUCT_URI><PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>"
               f"<PRODUCT_START_TIME>{product_time}</PRODUCT_START_TIME><PRODUCT_STOP_TIME>{product_time}</PRODUCT_STOP_TIME>"
               f"<BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>{product_tags}{offsets}"
               + "".join(f"<IMAGE_FILE>{path}</IMAGE_FILE>" for path in image_files) + "</Product>").encode()
    granule = (f"<Granule><TILE_ID>{tile_id}</TILE_ID><SENSING_TIME>{tile_time}</SENSING_TIME>"
               f"<HORIZONTAL_CS_CODE>EPSG:32645</HORIZONTAL_CS_CODE></Granule>").encode()
    return item, selection, product, granule


class _Response(io.BytesIO):
    def __init__(self, payload, url, etag='"opaque-head"'):
        super().__init__(payload)
        self.status = 200
        self.headers = {"Content-Length": str(len(payload)), "ETag": etag}
        self._url = url

    def geturl(self):
        return self._url


def _asset_download_evidence(payload=b"constructed_fixture_tiff_bytes", etag='"opaque-head"'):
    asset = {"href": ("https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/"
                      "sentinel-2-c1-l2a/45/R/XL/2023/9/S2A_T45RXL_20230904T044704_L2A/B03.tif"),
             "file:size": len(payload), "file:checksum": "1220" + _sha(payload)}
    head = {"method": "HEAD", "http_status": 200, "url": asset["href"],
            "response_headers": {"Content-Length": str(len(payload)), "ETag": etag}}
    return asset, head


def test_c1_declared_native_10m_20m_grids_are_integer_related():
    item, _, _, _ = _c1_fixture()

    grids = earth_search_c1._validate_declared_grids(item["assets"], item["properties"])

    assert grids["green"]["gsd"] == 10
    assert grids["swir16"]["gsd"] == 20
    assert max(grids["green"]["gsd"], grids["swir16"]["gsd"]) / min(
        grids["green"]["gsd"], grids["swir16"]["gsd"]
    ) == 2


def test_c1_declared_shifted_20m_grid_is_rejected():
    item, _, _, _ = _c1_fixture()
    item["assets"]["swir16"]["proj:transform"][2] += 10

    with pytest.raises(AdmissionError, match="grid footprint differs"):
        earth_search_c1._validate_declared_grids(item["assets"], item["properties"])


def test_c1_non_integer_native_resolution_relation_is_rejected(monkeypatch):
    item, _, _, _ = _c1_fixture()
    monkeypatch.setitem(earth_search_c1.EXPECTED_GSD, "swir16", 15)
    monkeypatch.setitem(earth_search_c1.EXPECTED_GSD, "scl", 15)
    for role in ("green", "nir"):
        item["assets"][role]["proj:shape"] = [3, 3]
        item["assets"][role]["proj:transform"] = [10, 0, 600000, 0, -10, 3100020]
    for role in ("swir16", "scl"):
        asset = item["assets"][role]
        asset["gsd"] = 15
        asset["proj:shape"] = [2, 2]
        asset["proj:transform"] = [15, 0, 600000, 0, -15, 3100020]
        asset["raster:bands"][0]["spatial_resolution"] = 15

    with pytest.raises(AdmissionError, match="integer native-resolution relation"):
        earth_search_c1._validate_declared_grids(item["assets"], item["properties"])


def test_c1_asset_download_receipt_binds_selection_runtime_and_hash(tmp_path, monkeypatch):
    payload = b"constructed_fixture_tiff_bytes"
    asset, head = _asset_download_evidence(payload)
    context = {"selection_id": "constructed-fixture", "selection_sha256": "a" * 64,
               "product_id": "fixture-item", "site_id": "PILOT-SOUTH-LHONAK", "asset_role": "green"}
    monkeypatch.setattr(earth_search_c1.urllib.request, "urlopen",
                        lambda request, timeout: _Response(payload, asset["href"]))
    target = tmp_path / "constructed-fixture.tif"

    receipt = earth_search_c1.stage_asset(asset, head, target, origin="constructed_fixture", context=context)

    assert target.read_bytes() == payload
    assert receipt["status"] == "COMPLETE_HASH_VERIFIED"
    assert receipt["origin"] == "constructed_fixture"
    assert receipt["selection_sha256"] == "a" * 64
    assert receipt["runtime"]["python_version"]
    assert receipt["request_started_utc"] < receipt["request_finished_utc"]
    assert receipt["output_bytes"] == len(payload)
    assert receipt["output_sha256"] == _sha(payload)


def test_c1_etag_rejection_retains_exact_written_partial_and_receipt(tmp_path, monkeypatch):
    payload = b"constructed_fixture_tiff_bytes"
    asset, head = _asset_download_evidence(payload)
    monkeypatch.setattr(earth_search_c1.urllib.request, "urlopen",
                        lambda request, timeout: _Response(payload, asset["href"], '"different-etag"'))
    target = tmp_path / "constructed-fixture.tif"

    with pytest.raises(AdmissionError, match="ETag differs"):
        earth_search_c1.stage_asset(asset, head, target, origin="constructed_fixture")

    receipt = json.loads(target.with_name(target.name + ".receipt.json").read_text())
    partial = target.with_name(target.name + ".partial")
    assert receipt["status"] == "GET_ETAG_DIFFERS_FROM_RETAINED_HEAD"
    assert receipt["received_bytes"] == partial.stat().st_size == len(payload)
    assert receipt["partial_sha256"] == _sha(partial.read_bytes())
    assert receipt["request_started_utc"] < receipt["request_finished_utc"]
    assert not target.exists()


def test_native_product_post_event_support_is_rejected():
    item, _, product, granule = _c1_fixture()
    product = product.replace(b"2023-09-04T04:47:01Z", b"2023-10-04T04:47:01Z")

    with pytest.raises(AdmissionError, match="outside the pinned acquisition window"):
        earth_search_c1.validate_native_metadata(item, io.BytesIO(product), io.BytesIO(granule),
            acquisition_start=START, acquisition_end_exclusive=END, cutoff=CUTOFF)


def test_native_tile_sensing_post_event_is_rejected_even_when_item_time_is_pre_event():
    item, _, product, granule = _c1_fixture(tile_time="2023-10-04T05:01:01.646994Z")

    with pytest.raises(AdmissionError, match="native tile sensing time is outside"):
        earth_search_c1.validate_native_metadata(item, io.BytesIO(product), io.BytesIO(granule),
            acquisition_start=START, acquisition_end_exclusive=END, cutoff=CUTOFF)


def test_foreign_site_asset_directory_is_rejected_before_calibration():
    south, south_selection, _, _ = _c1_fixture("T45RXL", "XL")
    khang, _, _, _ = _c1_fixture("T45RXM", "XM")
    south["assets"]["green"] = khang["assets"]["green"]

    with pytest.raises(AdmissionError, match="asset path does not match"):
        earth_search_c1.validate_item(south, "PILOT-SOUTH-LHONAK", CUTOFF,
            acquisition_start=START, acquisition_end_exclusive=END,
            selection_record=south_selection)


def test_native_tile_id_and_crs_must_match_item():
    item, _, product, granule = _c1_fixture()
    granule = granule.replace(b"EPSG:32645", b"EPSG:4326")

    with pytest.raises(AdmissionError, match="native granule CRS"):
        earth_search_c1.validate_native_metadata(item, io.BytesIO(product), io.BytesIO(granule),
            acquisition_start=START, acquisition_end_exclusive=END, cutoff=CUTOFF)


def _write_cogs(tmp_path, item, *, scale_mismatch=False, values_override=None):
    manifest_path = Path(__file__).resolve().parents[2] / "project" / "wp03_pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    site = next(row for row in manifest["sites"] if row["site_id"] == "PILOT-SOUTH-LHONAK")
    lon, lat = float(site["longitude"]), float(site["latitude"])
    x, y = transform_coordinates("EPSG:4326", "EPSG:32645", [lon], [lat])
    result = {}
    values = {"green": np.array([[1000, 0], [2000, 3000]], dtype="uint16"),
              "nir": np.array([[1000, 1500], [2000, 3000]], dtype="uint16"),
              "swir16": np.array([[2500]], dtype="uint16"),
              "scl": np.array([[6]], dtype="uint8")}
    if values_override:
        values.update(values_override)
    for role, array in values.items():
        asset = item["assets"][role]
        gsd = earth_search_c1.EXPECTED_GSD[role]
        transform = from_origin(x[0] - 10, y[0] + 10, gsd, gsd)
        profile = {"driver": "GTiff", "height": array.shape[0], "width": array.shape[1],
                   "count": 1, "dtype": array.dtype, "crs": "EPSG:32645", "transform": transform,
                   "nodata": 0}
        path = tmp_path / f"{role}.tif"
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(array, 1)
            if role != "scl":
                dataset.scales = [1.0 if (scale_mismatch and role == "green") else 0.0001]
                dataset.offsets = [0.0 if (scale_mismatch and role == "green") else -0.1]
        payload = path.read_bytes()
        asset["file:size"] = len(payload)
        asset["file:checksum"] = "1220" + _sha(payload)
        asset["proj:shape"] = list(array.shape)
        asset["proj:transform"] = list(transform)[:6]
        result[role] = path
    return result


def test_decoder_centroid_order_matches_named_manifest_coordinates_and_grid_bounds(tmp_path, monkeypatch):
    manifest_path = Path(__file__).resolve().parents[2] / "project" / "wp03_pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    site = next(row for row in manifest["sites"] if row["site_id"] == "PILOT-SOUTH-LHONAK")
    longitude, latitude = float(site["longitude"]), float(site["latitude"])
    item, _, product, granule = _c1_fixture()
    paths = _write_cogs(tmp_path, item)

    correct_x, correct_y = transform_coordinates("EPSG:4326", "EPSG:32645", [longitude], [latitude])
    reversed_x, reversed_y = transform_coordinates("EPSG:4326", "EPSG:32645", [latitude], [longitude])
    with rasterio.open(paths["green"]) as dataset:
        bounds = dataset.bounds
        assert bounds.left <= correct_x[0] < bounds.right
        assert bounds.bottom < correct_y[0] <= bounds.top
        assert not (bounds.left <= reversed_x[0] < bounds.right and
                    bounds.bottom < reversed_y[0] <= bounds.top)

    decoded = earth_search_c1.decode_c1_bundle(
        item, site["site_id"], io.BytesIO(product), io.BytesIO(granule), paths, CUTOFF,
        acquisition_start=START, acquisition_end_exclusive=END, origin="constructed_fixture")
    assert earth_search_c1.CENTROIDS[site["site_id"]] == (latitude, longitude)
    assert decoded["query_centroid_wgs84"] == [longitude, latitude]

    monkeypatch.setitem(earth_search_c1.CENTROIDS, site["site_id"], (longitude, latitude))
    with pytest.raises(AdmissionError, match="query centroid is outside the selected C1 green tile"):
        earth_search_c1.decode_c1_bundle(
            item, site["site_id"], io.BytesIO(product), io.BytesIO(granule), paths, CUTOFF,
            acquisition_start=START, acquisition_end_exclusive=END, origin="constructed_fixture")


def test_decode_masks_invalid_reflectance_to_null_without_double_offset(tmp_path):
    item, _, product, granule = _c1_fixture()
    paths = _write_cogs(tmp_path, item)

    output = earth_search_c1.decode_c1_bundle(item, "PILOT-SOUTH-LHONAK",
        io.BytesIO(product), io.BytesIO(granule), paths, CUTOFF,
        acquisition_start=START, acquisition_end_exclusive=END, origin="constructed_fixture")

    green = output["assets"]["green"]
    assert output["origin"] == "constructed_fixture"
    assert output["research_admissible"] is False
    assert green["scale_offset_status"] == "MATCHED_COG_TAGS_STAC_AND_NATIVE_XML"
    assert green["reflectance_status"] == "PARTIAL_INVALID_PIXELS_EXPLICITLY_NULL"
    assert green["reflectance_values"][0][0] == pytest.approx(0.0)
    assert green["reflectance_values"][0][1] is None
    assert green["reflectance_values"][1][1] == pytest.approx(0.2)
    assert output["assets"]["scl"]["reflectance_values"] is None


def test_decode_unmatched_cog_encoding_emits_null_reflectance(tmp_path):
    item, _, product, granule = _c1_fixture()
    paths = _write_cogs(tmp_path, item, scale_mismatch=True)

    output = earth_search_c1.decode_c1_bundle(item, "PILOT-SOUTH-LHONAK",
        io.BytesIO(product), io.BytesIO(granule), paths, CUTOFF,
        acquisition_start=START, acquisition_end_exclusive=END, origin="constructed_fixture")

    green = output["assets"]["green"]
    assert green["reflectance_values"] is None
    assert green["reflectance_status"] == "UNRESOLVED_COG_CONVERSION_ENCODING"


def _decode_constructed_roi(tmp_path, *, values_override=None, roi_halfwidth_m=10):
    tmp_path.mkdir(parents=True, exist_ok=True)
    item, _, product, granule = _c1_fixture()
    paths = _write_cogs(tmp_path, item, values_override=values_override)
    output = earth_search_c1.decode_c1_bundle(
        item, "PILOT-SOUTH-LHONAK", io.BytesIO(product), io.BytesIO(granule), paths, CUTOFF,
        acquisition_start=START, acquisition_end_exclusive=END, origin="constructed_fixture",
        roi_output_dir=tmp_path / "roi-output", roi_halfwidth_m=roi_halfwidth_m)
    return output["roi"], tmp_path / "roi-output"


def _png_text_chunks(path):
    payload = Path(path).read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    result = {}
    offset = 8
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4:offset + 8]
        chunk = payload[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"tEXt":
            key, value = chunk.split(b"\x00", 1)
            result[key.decode("latin-1")] = value.decode("latin-1")
        if kind == b"IEND":
            break
    return result


def test_constructed_roi_origin_reaches_report_rasters_and_visual_metadata(tmp_path):
    roi, output_dir = _decode_constructed_roi(tmp_path)

    assert roi["origin"] == "constructed_fixture"
    assert roi["provider"] == "constructed C1 engineering fixture"
    for path in output_dir.glob("*.tif"):
        with rasterio.open(path) as dataset:
            assert dataset.tags().get("origin") == "constructed_fixture", path.name
    svg = (output_dir / "diagnostic_overlay.svg").read_text(encoding="utf-8")
    assert "origin=constructed_fixture" in svg
    assert "research_admitted=false" in svg
    png_metadata = _png_text_chunks(output_dir / "diagnostic_overlay.png")
    assert png_metadata["origin"] == "constructed_fixture"
    assert png_metadata["source_item_id"] == roi["site_item_id"]
    assert png_metadata["research_admitted"] == "false"


def test_unsupported_roi_origin_is_rejected_before_output_creation(tmp_path):
    item, selection, product, granule = _c1_fixture()
    paths = _write_cogs(tmp_path, item)
    output_dir = tmp_path / "unsupported-origin-output"

    with pytest.raises(AdmissionError, match="source origin must be explicit"):
        earth_search_c1.decode_c1_bundle(
            item, "PILOT-SOUTH-LHONAK", io.BytesIO(product), io.BytesIO(granule), paths, CUTOFF,
            acquisition_start=START, acquisition_end_exclusive=END,
            selection_record=selection, origin="generated_fallback",
            roi_output_dir=output_dir)

    assert not output_dir.exists()


def test_provider_roi_origin_reaches_artifacts_from_retained_provider_cogs(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    stage_root = repository_root / "data/raw/wp03/attempt_2026-10-14_stage_001"
    site_id = "PILOT-SOUTH-LHONAK"
    staged_site = stage_root / "c1" / site_id
    if not all((staged_site / f"{role}.tif").is_file()
               for role in ("green", "nir", "swir16", "scl")):
        pytest.skip("retained provider COGs are unavailable; no network fallback is permitted")

    manifest = json.loads((repository_root / "project/wp03_pilot_manifest.json").read_text())
    selection = json.loads((repository_root / "project/architecture/wp03_2026-10-13/attempt_13_provider_pixels/selection.json").read_text())
    inventory = json.loads((repository_root / "project/.wp03_attempt13_source_inventory.json").read_text())
    source_root = repository_root / inventory["source_root"]
    selected = selection["sentinel2_c1"][site_id]
    item_page = json.loads((source_root / selected["item_path"]).read_text())
    item = next(feature for feature in item_page["features"] if feature["id"] == selected["item_id"])
    site = next(row for row in manifest["sites"] if row["site_id"] == site_id)
    asset_paths = {role: staged_site / f"{role}.tif"
                   for role in ("green", "nir", "swir16", "scl")}
    product_xml = source_root / "asset_access" / f"{site_id}_product_metadata.xml"
    granule_xml = source_root / "asset_access" / f"{site_id}_granule_metadata.xml"
    output_dir = tmp_path / "genuine-provider-roi"

    output = earth_search_c1.decode_c1_bundle(
        item, site_id, product_xml, granule_xml, asset_paths,
        manifest["pre_event_cutoff_exclusive"],
        acquisition_start=manifest["acquisition_start"],
        acquisition_end_exclusive=manifest["acquisition_end_exclusive"],
        selection_record=selected, origin="provider_download",
        roi_output_dir=output_dir, roi_halfwidth_m=50)

    assert output["origin"] == "provider_download"
    assert output["roi"]["origin"] == "provider_download"
    assert output["roi"]["provider"] == "Earth Search C1"
    assert output["roi"]["site_id"] == site["site_id"]
    tiffs = list(output_dir.glob("*.tif"))
    assert len(tiffs) >= 15
    for path in tiffs:
        with rasterio.open(path) as dataset:
            assert dataset.tags().get("origin") == "provider_download", path.name
    svg = (output_dir / "diagnostic_overlay.svg").read_text(encoding="utf-8")
    assert "origin=provider_download" in svg
    assert "research_admitted=false" in svg
    png_metadata = _png_text_chunks(output_dir / "diagnostic_overlay.png")
    assert png_metadata["origin"] == "provider_download"
    assert png_metadata["source_item_id"] == selected["item_id"]
    assert png_metadata["product_uri"] == selected["product_uri"]
    assert png_metadata["research_admitted"] == "false"


def test_retained_khangchung_product_identity_is_exact():
    repository_root = Path(__file__).resolve().parents[2]
    selection = json.loads((repository_root / "project/architecture/wp03_2026-10-13/attempt_13_provider_pixels/selection.json").read_text())
    assert selection["sentinel2_c1"]["PILOT-KHANGCHUNG-TSHO"]["item_id"] == \
        "S2A_T45RXM_20230921T043839_L2A"


def test_roi_decoder_aggregates_registered_pixels_calibrates_once_and_derives_candidate_mask(tmp_path):
    roi, output_dir = _decode_constructed_roi(tmp_path, values_override={
        "green": np.array([[3000, 3100], [3200, 3300]], dtype="uint16"),
        "nir": np.array([[2000, 2100], [2200, 2300]], dtype="uint16"),
        "swir16": np.array([[2000]], dtype="uint16"),
        "scl": np.array([[6]], dtype="uint8"),
    })

    assert roi["status"] == "ROI_PIXEL_DECODED_QUARANTINED"
    assert roi["output_shape_rows_columns"] == [1, 1]
    assert roi["source_alignment_halo_in_output_m"] == 0
    assert all(roi["source_coverage"][role]["full_requested_roi_covered"]
               for role in ("green", "nir", "swir16", "scl"))
    assert roi["calibration"]["B03"]["raw_to_reflectance_sample_recomputation"]["status"] == "PASS"
    assert roi["calibration"]["B08"]["raw_to_reflectance_sample_recomputation"]["status"] == "PASS"
    assert roi["calibration"]["B11"]["raw_to_reflectance_sample_recomputation"]["status"] == "PASS"
    with rasterio.open(output_dir / "b03_reflectance_20m.tif") as dataset:
        assert dataset.read(1)[0, 0] == pytest.approx(0.215)
        assert dataset.crs.to_epsg() == 32645
    with rasterio.open(output_dir / "b08_reflectance_20m.tif") as dataset:
        assert dataset.read(1)[0, 0] == pytest.approx(0.115)
    with rasterio.open(output_dir / "b11_reflectance_20m.tif") as dataset:
        assert dataset.read(1)[0, 0] == pytest.approx(0.1)
    with rasterio.open(output_dir / "ndwi.tif") as dataset:
        assert dataset.read(1)[0, 0] > 0
    with rasterio.open(output_dir / "mndwi.tif") as dataset:
        assert dataset.read(1)[0, 0] > 0
    with rasterio.open(output_dir / "candidate_water_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 1
    assert roi["candidate_water_pixel_count"] == 1
    assert roi["candidate_water_area_m2"] == pytest.approx(400)
    assert roi["candidate_water_area_status"] == "ROI_CANDIDATE_AREA_COMPUTED_ON_VALID_QA_SUPPORT"
    assert roi["water_mask_unavailable_pixel_count"] == 0
    assert roi["water_mask_unavailable_fraction"] == 0


def test_roi_invalid_10m_contributor_stays_unknown_and_area_is_null(tmp_path):
    roi, output_dir = _decode_constructed_roi(tmp_path, values_override={
        "green": np.array([[3000, 0], [3200, 3300]], dtype="uint16"),
        "nir": np.array([[2000, 2100], [2200, 2300]], dtype="uint16"),
        "swir16": np.array([[2000]], dtype="uint16"),
        "scl": np.array([[6]], dtype="uint8"),
    })

    assert roi["unknown_pixel_count"] == 1
    assert roi["raw_support_or_denominator_unknown_pixel_count"] == 1
    assert roi["water_mask_unavailable_pixel_count"] == 1
    assert roi["water_mask_unavailable_fraction"] == 1
    assert roi["candidate_support_pixel_count"] == 0
    assert roi["candidate_water_pixel_count"] is None
    assert roi["candidate_water_area_m2"] is None
    with rasterio.open(output_dir / "b03_valid_support_count.tif") as dataset:
        assert dataset.read(1)[0, 0] == 3
    with rasterio.open(output_dir / "unknown_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 1
    with rasterio.open(output_dir / "candidate_water_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 255


def test_roi_qa_excluded_pixels_remain_distinct_from_unknown_and_not_candidate(tmp_path):
    roi, output_dir = _decode_constructed_roi(tmp_path, values_override={
        "green": np.full((2, 2), 3000, dtype="uint16"),
        "nir": np.full((2, 2), 2000, dtype="uint16"),
        "swir16": np.array([[2000]], dtype="uint16"),
        "scl": np.array([[11]], dtype="uint8"),
    })

    assert roi["analysis_valid_pixel_count"] == 1
    assert roi["unknown_pixel_count"] == 0
    assert roi["scl_qa_excluded_pixel_count"] == 1
    assert roi["raw_support_or_denominator_unknown_pixel_count"] == 0
    assert roi["water_mask_unavailable_pixel_count"] == 1
    assert roi["water_mask_unavailable_fraction"] == 1
    assert roi["water_mask_unavailable_reason_counts_nonexclusive"] == {
        "raw_support_or_denominator_unknown": 0,
        "valid_scl_qa_excluded": 1}
    assert roi["candidate_support_pixel_count"] == 0
    assert roi["candidate_water_area_m2"] is None
    with rasterio.open(output_dir / "scl_qa_excluded_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 1
    with rasterio.open(output_dir / "unknown_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 0
    with rasterio.open(output_dir / "candidate_water_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 255


def test_roi_preserves_negative_reflectance_and_zero_denominator_as_unknown(tmp_path):
    roi, output_dir = _decode_constructed_roi(tmp_path, values_override={
        "green": np.full((2, 2), 500, dtype="uint16"),
        "nir": np.full((2, 2), 2000, dtype="uint16"),
        "swir16": np.array([[2000]], dtype="uint16"),
        "scl": np.array([[4]], dtype="uint8"),
    })

    with rasterio.open(output_dir / "b03_reflectance_20m.tif") as dataset:
        assert dataset.read(1)[0, 0] == pytest.approx(-0.05)
    assert roi["candidate_water_area_m2"] == pytest.approx(0)

    other = tmp_path / "zero-denominator"
    zero_roi, zero_dir = _decode_constructed_roi(other, values_override={
        "green": np.full((2, 2), 1000, dtype="uint16"),
        "nir": np.full((2, 2), 1000, dtype="uint16"),
        "swir16": np.array([[2000]], dtype="uint16"),
        "scl": np.array([[4]], dtype="uint8"),
    })
    assert zero_roi["candidate_water_area_m2"] is None
    assert zero_roi["raw_support_or_denominator_unknown_pixel_count"] == 1
    assert zero_roi["water_mask_unavailable_pixel_count"] == 1
    assert zero_roi["unknown_invalidity_reason_counts_nonexclusive"]["NDWI_ZERO_OR_INVALID_DENOMINATOR"] == 1
    with rasterio.open(zero_dir / "ndwi.tif") as dataset:
        assert dataset.read(1)[0, 0] == dataset.nodata
    with rasterio.open(zero_dir / "unknown_mask.tif") as dataset:
        assert dataset.read(1)[0, 0] == 1


def test_safe_roles_check_manifest_refs_crc_and_annotation_identity(tmp_path):
    scene = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F"
    root = scene + ".SAFE/"
    archive_path = tmp_path / "constructed-fixture.zip"
    member_data = role_members(scene, "VV")
    manifest = scoped_manifest(scene, member_data)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(root + "manifest.safe", manifest)
        for name, payload in member_data.items():
            archive.writestr(root + name, payload)

    result = asf_grd.inspect_safe_roles(archive_path, scene, "VV", origin="constructed_fixture")

    assert result["origin"] == "constructed_fixture"
    assert result["research_admitted"] is False
    assert result["archive_crc_status"] == "ALL_ZIP_MEMBER_CRC_CHECKS_PASSED"
    assert result["manifest_identity_status"] == (
        "SCOPED_MISSION_PRODUCT_ORBIT_TIME_POLARIZATION_AND_MEMBER_JOIN_MATCHED")
    assert result["annotation_identity_status"] == "MISSION_PRODUCT_TIME_POLARIZATION_MATCHED"
    assert result["roles"]["annotation"].endswith("/annotation/" +
        "s1a-iw-grd-vv-20230901t121408-20230901t121433-050134-060898-001.xml")
    assert result["rfi_metadata"] == {
        "status": "PRESENT",
        "members": [scene + ".SAFE/annotation/rfi/rfi-s1a-iw-grd-vv-20230901t121408-20230901t121433-050134-060898-001.xml"],
    }
    assert result["manifest_identity"]["absolute_orbit"] == "50134"
    assert result["manifest_identity"]["polarizations"] == ["VH", "VV"]
    assert result["pixel_decoding_status"] == "NOT_DECODED"


@pytest.mark.parametrize(
    ("product_annotations", "expected_error"),
    [
        (["scene-vv.xml", "second-scene-vv.xml"], "product annotation role is absent or ambiguous"),
        ([], "product annotation role is absent or ambiguous"),
        (["other-vv.xml"], "measurement/annotation/calibration/noise identity stems do not join"),
    ],
)
def test_safe_product_annotation_must_be_unique_and_bind_to_measurement(
        tmp_path, product_annotations, expected_error):
    scene = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F"
    root = scene + ".SAFE/"
    members = role_members(scene, "VV")
    members = {name: value for name, value in members.items()
               if not (name.startswith("annotation/") and
                       "/calibration/" not in name and "/rfi/" not in name)}
    stem = next(name.rsplit("/", 1)[-1][:-5] for name in members
                if name.startswith("measurement/"))
    members.update({f"annotation/{name}": b"<annotation/>" for name in product_annotations})
    if product_annotations == ["other-vv.xml"]:
        members = {name: value for name, value in members.items()
                   if not name.endswith("/other-vv.xml")}
        members["annotation/other-vv.xml"] = role_members(scene, "VV")[
            f"annotation/{stem}.xml"]
    manifest = scoped_manifest(scene, members)
    archive_path = tmp_path / "constructed-annotation-layout.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(root + "manifest.safe", manifest)
        for name, payload in members.items():
            archive.writestr(root + name, payload)

    with pytest.raises(AdmissionError, match=expected_error):
        asf_grd.inspect_safe_roles(archive_path, scene, "VV", origin="constructed_fixture")


@pytest.mark.parametrize(
    ("manifest_overrides", "expected_error"),
    [
        ({"platform_number": "B"}, "scoped manifest mission/product/orbit/time identity"),
        ({"orbit": "50135"}, "scoped manifest mission/product/orbit/time identity"),
        ({"start": "2023-09-01T12:14:09.000000"},
         "scoped manifest mission/product/orbit/time identity"),
        ({"product_type": "SLC"}, "scoped manifest mission/product/orbit/time identity"),
        ({"polarizations": ("VV",)}, "manifest polarization scope differs"),
    ],
)
def test_safe_rejects_manifest_identity_mismatch(tmp_path, manifest_overrides, expected_error):
    scene = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F"
    archive_path = tmp_path / "constructed-fixture.zip"
    members = role_members(scene, "VV")
    manifest = scoped_manifest(scene, members, **manifest_overrides)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(scene + ".SAFE/manifest.safe", manifest)
        for name, payload in members.items():
            archive.writestr(scene + ".SAFE/" + name, payload)
    with pytest.raises(AdmissionError, match=expected_error):
        asf_grd.inspect_safe_roles(archive_path, scene, "VV", origin="constructed_fixture")


@pytest.mark.parametrize("role", ["annotation", "calibration", "noise", "rfi"])
def test_safe_rejects_role_xml_identity_mismatch(tmp_path, role):
    scene = "S1A_IW_GRDH_1SDV_20230901T121408_20230901T121433_050134_060898_363F"
    root = scene + ".SAFE/"
    members = role_members(scene, "VV")
    stem = next(name.rsplit("/", 1)[-1][:-5] for name in members
                if name.startswith("measurement/"))
    role_path = {
        "annotation": f"annotation/{stem}.xml",
        "calibration": f"annotation/calibration/calibration-{stem}.xml",
        "noise": f"annotation/calibration/noise-{stem}.xml",
        "rfi": f"annotation/rfi/rfi-{stem}.xml",
    }[role]
    members[role_path] = role_identity_xml(role, scene, "VV", mission="S1B")
    archive_path = tmp_path / f"constructed-{role}-identity-mismatch.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(root + "manifest.safe", scoped_manifest(scene, members))
        for name, payload in members.items():
            archive.writestr(root + name, payload)
    with pytest.raises(AdmissionError, match=f"ASF {role if role != 'rfi' else 'RFI metadata'} mission/product/time/mode/polarization differs"):
        asf_grd.inspect_safe_roles(archive_path, scene, "VV", origin="constructed_fixture")


def test_asf_feature_rejects_nonapproved_download_host():
    with pytest.raises(AdmissionError, match="approved HTTPS host"):
        asf_grd._validate_feature({"properties": {"url": "https://example.invalid/product.zip"}})
