"""Ingestion integrity and provenance verification tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Any mock responses or synthetic payloads in this file exist strictly to verify
parser logic, error handling, and serialization integrity in offline unit tests.
They do not represent real observational records and must never be used to support
scientific or research findings.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from sentinel_gl.ingestion.provenance import (
    AcquisitionRecord,
    compute_bytes_sha256,
    compute_file_sha256,
)
from sentinel_gl.ingestion.sentinel2 import Sentinel2Ingestor
from sentinel_gl.ingestion.sentinel1 import Sentinel1Ingestor
from sentinel_gl.ingestion.era5 import ERA5Ingestor
from sentinel_gl.ingestion.pipeline import PilotIngestionPipeline


# Mock payloads explicitly marked for offline parser testing
MOCK_S2_ODATA_RESPONSE = {
    "value": [
        {
            "Id": "00010203-0405-0607-0809-0a0b0c0d0e0f",
            "Name": "S2A_MSIL2A_20231001T043701_N0509_R033_T45RVP_20231001T080829.SAFE",
            "ContentDate": {
                "Start": "2023-10-01T04:37:01.024Z",
                "End": "2023-10-01T04:37:01.024Z",
            },
            "Attributes": [
                {"Name": "cloudCover", "Value": 12.5},
                {"Name": "processingBaseline", "Value": "05.09"},
            ],
        }
    ]
}

MOCK_S1_ODATA_RESPONSE = {
    "value": [
        {
            "Id": "11112222-3333-4444-5555-666677778888",
            "Name": "S1A_IW_GRDH_1SDV_20230925T121409_20230925T121434_050484_0614A7_45D2.SAFE",
            "ContentDate": {
                "Start": "2023-09-25T12:14:09.123Z",
                "End": "2023-09-25T12:14:34.456Z",
            },
            "Attributes": [
                {"Name": "orbitDirection", "Value": "DESCENDING"},
                {"Name": "polarisationChannels", "Value": "VV&VH"},
            ],
        }
    ]
}

MOCK_ERA5_RESPONSE = {
    "latitude": 27.91,
    "longitude": 88.20,
    "elevation": 5200.0,
    "hourly": {
        "time": [
            "2023-09-01T00:00",
            "2023-09-01T06:00",
            "2023-09-01T12:00",
            "2023-09-01T18:00",
            "2023-09-02T00:00",
            "2023-09-02T12:00",
        ],
        "temperature_2m": [-2.0, -1.0, 3.5, 0.5, -3.0, 2.0],
        "precipitation": [0.0, 0.2, 1.0, 0.4, 0.0, 0.5],
    },
}


def test_hash_utilities(tmp_path: Path):
    sample_bytes = b"sentinel-gl-provenance-test-data"
    expected_sha = hashlib.sha256(sample_bytes).hexdigest()
    assert compute_bytes_sha256(sample_bytes) == expected_sha

    test_file = tmp_path / "test_blob.bin"
    test_file.write_bytes(sample_bytes)
    assert compute_file_sha256(test_file) == expected_sha


def test_acquisition_record_dataclass():
    rec = AcquisitionRecord(
        record_id="REC-001",
        lake_id="SGL-001",
        modality="optical",
        provider_name="Copernicus Data Space Ecosystem",
        product_identifier="S2A_MSIL2A_TEST",
        acquisition_timestamp="2023-10-01T04:37:01Z",
        query_timestamp="2023-10-05T12:00:00Z",
        roi_geometry={"type": "Point", "coordinates": [88.20, 27.91]},
        status="CATALOGUED",
        sha256="abc123def456",
        storage_bytes=1024,
        metadata={"cloud_cover": 5.0},
    )
    d = rec.to_dict()
    assert d["record_id"] == "REC-001"
    assert d["lake_id"] == "SGL-001"
    assert d["modality"] == "optical"
    assert d["storage_bytes"] == 1024
    assert d["metadata"]["cloud_cover"] == 5.0


def test_sentinel2_ingestor_parsing():
    ingestor = Sentinel2Ingestor()

    # Successful mock response
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(MOCK_S2_ODATA_RESPONSE).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        records = ingestor.query_catalog(
            lake_id="SGL-001",
            centroid_lat=27.91,
            centroid_lon=88.20,
            start_date="2023-10-01",
            end_date="2023-10-01",
        )
    assert len(records) == 1
    rec = records[0]
    assert rec.status == "CATALOGUED"
    assert rec.lake_id == "SGL-001"
    assert rec.modality == "optical"
    assert rec.metadata["cloud_cover_percentage"] == 12.5
    assert rec.sha256 is not None
    assert rec.storage_bytes > 0

    # Empty mock response
    empty_resp = MagicMock()
    empty_resp.read.return_value = json.dumps({"value": []}).encode("utf-8")
    empty_resp.__enter__.return_value = empty_resp

    with patch("urllib.request.urlopen", return_value=empty_resp):
        empty_records = ingestor.query_catalog(
            lake_id="SGL-001",
            centroid_lat=27.91,
            centroid_lon=88.20,
            start_date="2023-10-01",
            end_date="2023-10-01",
        )
    assert len(empty_records) == 1
    assert empty_records[0].status == "NO_DATA"
    assert empty_records[0].missingness_reason == "NO_ACQUISITIONS_IN_WINDOW"

    # Network / HTTP error handling
    with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
        err_records = ingestor.query_catalog(
            lake_id="SGL-001",
            centroid_lat=27.91,
            centroid_lon=88.20,
            start_date="2023-10-01",
            end_date="2023-10-01",
        )
    assert len(err_records) == 1
    assert err_records[0].status == "FAILED"
    assert "PROVIDER_QUERY_ERROR" in (err_records[0].missingness_reason or "")


def test_sentinel1_ingestor_parsing():
    ingestor = Sentinel1Ingestor()

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(MOCK_S1_ODATA_RESPONSE).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        records = ingestor.query_catalog(
            lake_id="SGL-001",
            centroid_lat=27.91,
            centroid_lon=88.20,
            start_date="2023-09-25",
            end_date="2023-09-25",
        )
    assert len(records) == 1
    rec = records[0]
    assert rec.status == "CATALOGUED"
    assert rec.lake_id == "SGL-001"
    assert rec.modality == "sar"
    assert rec.metadata["orbit_direction"] == "DESCENDING"
    assert rec.metadata["polarization"] == "VV&VH"
    assert rec.sha256 is not None


def test_era5_ingestor_parsing():
    ingestor = ERA5Ingestor()

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(MOCK_ERA5_RESPONSE).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        records = ingestor.query_series(
            lake_id="SGL-001",
            centroid_lat=27.91,
            centroid_lon=88.20,
            start_date="2023-09-01",
            end_date="2023-09-02",
        )
    # Should produce 2 daily summary records: 2023-09-01 and 2023-09-02
    assert len(records) == 2
    r1 = records[0]
    assert r1.status == "CATALOGUED"
    assert r1.modality == "weather"
    assert r1.record_id == "ERA5-SGL-001-2023-09-01"
    assert r1.acquisition_timestamp is None  # No invented acquisition hour for a daily summary.
    assert r1.metadata["elevation_m"] == 5200.0
    # Mean of [-2.0, -1.0, 3.5, 0.5] = 1.0 / 4 = 0.25
    assert abs(r1.metadata["t2m_mean_c"] - 0.25) < 1e-5
    # Total precip = 0.0 + 0.2 + 1.0 + 0.4 = 1.6
    assert abs(r1.metadata["precipitation_total_mm"] - 1.6) < 1e-5


def test_pilot_ingestion_pipeline_with_mock_ingestors(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from sentinel_gl.ingestion.pipeline import PilotIngestionPipeline
    with pytest.raises(ValueError, match="BLOCKED"):
        PilotIngestionPipeline(tmp_path).run_pilot()
    assert not list(tmp_path.iterdir())



def test_live_registries_and_dossier_integrity():
    from sentinel_gl.admission import QUARANTINED
    from sentinel_gl.integrity import file_hash
    root=Path(__file__).resolve().parents[2]
    for rel in ('data/lake_registry.csv','data/event_registry.csv','data/pilot_dossier.json'):
        assert file_hash(root/rel) in QUARANTINED
