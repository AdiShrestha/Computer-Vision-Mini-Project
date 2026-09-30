"""Contract tests for strict provenance and dataset-manifest schemas."""

import copy
import hashlib

import pytest

from source.data.schemas import (
    AcquisitionProvenanceRecord,
    DatasetManifest,
    EvidenceClass,
    SchemaValidationError,
)


def digest(payload: bytes = b"") -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def valid_provenance_dict():
    return {
        "source": "COPERNICUS/S1_GRD",
        "api_endpoint": "https://earthengine.googleapis.com/v1/projects/demo",
        "authentication_method": "OAuth2",
        "query_parameters": {
            "date_range": ["2020-01-01", "2020-12-31"],
            "spatial_filter": {"type": "Point", "coordinates": [88.0, 27.5]},
            "band_selection": ["VV", "VH"],
        },
        "total_api_calls": 2,
        "total_scenes_returned": 3,
        "first_scene_date": "2020-01-03T00:00:00Z",
        "last_scene_date": "2020-12-27T00:00:00+00:00",
        "http_status_codes": [200, 200],
        "response_payload_hash": digest(b"provider-response"),
        "downloaded_file_manifest": [
            {
                "filename": "data/raw/sentinel1/LK_01/scenes.json",
                "size_bytes": 128,
                "checksum": digest(b"file-content"),
            }
        ],
        "execution_environment": {
            "network_reachable": True,
            "sandbox_bypass": False,
            "python_version": "3.12.8",
            "timestamp_utc": "2026-08-27T10:00:00Z",
        },
    }


def valid_manifest_dict():
    file_path = "data/raw/sentinel1/LK_01/scenes.json"
    return {
        "manifest_version": "2.0",
        "created_at": "2026-08-27T10:05:00Z",
        "lake_entries": {
            "LK_01": {
                "temporal_start": "2020-01-03T00:00:00Z",
                "temporal_end": "2020-12-27T00:00:00Z",
                "missingness_rate": 0.25,
                "scene_count": 3,
                "file_paths": [file_path],
                "checksums": {file_path: digest(b"file-content")},
            }
        },
        "channel_provenance": {
            "vv_db": {
                "source_sensor": "Sentinel-1 GRD",
                "raw_product": "COPERNICUS/S1_GRD",
                "transformation_pipeline": [
                    "provider calibration",
                    "region aggregation",
                ],
                "version": "1.0",
            }
        },
        "gap_statistics": {
            "observed_missingness_percentage": {"sentinel1": 25.0},
            "declared_expected_gaps": {"sentinel1": 30.0},
        },
        "evidence_class": "AUTHENTICATED",
    }


def test_complete_provenance_record_validates():
    record = AcquisitionProvenanceRecord.from_dict(valid_provenance_dict())
    assert record.validate_record() is record
    assert record.downloaded_file_manifest[0].size_bytes == 128


def test_provenance_round_trip_is_lossless():
    record = AcquisitionProvenanceRecord.from_dict(valid_provenance_dict())
    restored = AcquisitionProvenanceRecord.from_json(record.to_json())
    assert restored.to_dict() == record.to_dict()


def test_zero_observation_blocked_query_is_legitimate_and_explicit():
    payload = valid_provenance_dict()
    payload.update(
        {
            "total_api_calls": 1,
            "total_scenes_returned": 0,
            "first_scene_date": None,
            "last_scene_date": None,
            "http_status_codes": [401],
            "downloaded_file_manifest": [],
            "response_payload_hash": digest(b"authentication denied"),
        }
    )
    record = AcquisitionProvenanceRecord.from_dict(payload)
    assert record.total_scenes_returned == 0
    assert record.http_status_codes == [401]


def test_missing_required_nested_checksum_is_rejected():
    payload = valid_provenance_dict()
    del payload["downloaded_file_manifest"][0]["checksum"]
    with pytest.raises(SchemaValidationError, match="missing required fields"):
        AcquisitionProvenanceRecord.from_dict(payload)


@pytest.mark.parametrize(
    "bad_hash",
    ["sha256:1234", "sha256:" + "g" * 64, "SHA256:" + "0" * 64],
)
def test_invalid_sha256_is_rejected(bad_hash):
    payload = valid_provenance_dict()
    payload["response_payload_hash"] = bad_hash
    with pytest.raises(SchemaValidationError, match="64 lowercase hex"):
        AcquisitionProvenanceRecord.from_dict(payload)


def test_non_utc_timestamp_and_invalid_http_code_are_rejected():
    payload = valid_provenance_dict()
    payload["first_scene_date"] = "2020-01-03T00:00:00+05:45"
    with pytest.raises(SchemaValidationError, match="explicit UTC offset"):
        AcquisitionProvenanceRecord.from_dict(payload)

    payload = valid_provenance_dict()
    payload["http_status_codes"] = [99]
    with pytest.raises(SchemaValidationError, match="100 to 599"):
        AcquisitionProvenanceRecord.from_dict(payload)


def test_query_parameters_require_spatial_temporal_and_sensor_fields():
    payload = valid_provenance_dict()
    payload["query_parameters"] = {"date_range": ["2020-01-01", "2020-12-31"]}
    with pytest.raises(SchemaValidationError, match="spatial_filter"):
        AcquisitionProvenanceRecord.from_dict(payload)


def test_scene_count_dates_and_api_calls_are_consistent():
    payload = valid_provenance_dict()
    payload["total_scenes_returned"] = 0
    with pytest.raises(SchemaValidationError, match="requires null scene dates"):
        AcquisitionProvenanceRecord.from_dict(payload)

    payload = valid_provenance_dict()
    payload["total_api_calls"] = 0
    with pytest.raises(SchemaValidationError, match="at least one API call"):
        AcquisitionProvenanceRecord.from_dict(payload)


def test_downloaded_filenames_must_be_unique():
    payload = valid_provenance_dict()
    payload["downloaded_file_manifest"].append(
        copy.deepcopy(payload["downloaded_file_manifest"][0])
    )
    with pytest.raises(SchemaValidationError, match="filenames must be unique"):
        AcquisitionProvenanceRecord.from_dict(payload)


def test_complete_dataset_manifest_validates_and_round_trips():
    manifest = DatasetManifest.from_dict(valid_manifest_dict())
    assert manifest.validate_record() is manifest
    assert manifest.evidence_class is EvidenceClass.AUTHENTICATED
    assert DatasetManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()


def test_manifest_rejects_unknown_evidence_class():
    payload = valid_manifest_dict()
    payload["evidence_class"] = "UNVERIFIED_BUT_PROBABLY_REAL"
    with pytest.raises(SchemaValidationError, match="invalid evidence_class"):
        DatasetManifest.from_dict(payload)


def test_manifest_requires_one_checksum_per_file():
    payload = valid_manifest_dict()
    payload["lake_entries"]["LK_01"]["checksums"] = {}
    with pytest.raises(SchemaValidationError, match="exactly match"):
        DatasetManifest.from_dict(payload)


def test_manifest_rejects_malformed_missingness_and_gap_statistics():
    payload = valid_manifest_dict()
    payload["lake_entries"]["LK_01"]["missingness_rate"] = 1.01
    with pytest.raises(SchemaValidationError, match="between 0 and 1"):
        DatasetManifest.from_dict(payload)

    payload = valid_manifest_dict()
    payload["gap_statistics"]["observed_missingness_percentage"]["sentinel1"] = 101.0
    with pytest.raises(SchemaValidationError, match="between 0 and 100"):
        DatasetManifest.from_dict(payload)


def test_validation_detects_post_construction_mutation():
    manifest = DatasetManifest.from_dict(valid_manifest_dict())
    mutated = copy.deepcopy(manifest)
    mutated.lake_entries["LK_01"].checksums.clear()
    with pytest.raises(SchemaValidationError, match="exactly match"):
        mutated.validate_record()
