"""Strict schema for authenticated external-data acquisition provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse


SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SchemaValidationError(ValueError):
    """Raised when a provenance or manifest record violates its schema."""


def _require_exact_fields(
    value: Mapping[str, Any], required: set[str], context: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise SchemaValidationError(
            f"{context} is missing required fields: {sorted(missing)}"
        )
    if unknown:
        raise SchemaValidationError(
            f"{context} contains unknown fields: {sorted(unknown)}"
        )


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")
    return value


def validate_sha256(value: Any, field_name: str = "checksum") -> str:
    """Validate the canonical ``sha256:<lowercase hex>`` representation."""
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SchemaValidationError(
            f"{field_name} must match 'sha256:' followed by 64 lowercase hex characters"
        )
    return value


def validate_utc_timestamp(value: Any, field_name: str) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit UTC offset."""
    _require_nonempty_string(value, field_name)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SchemaValidationError(f"{field_name} must carry an explicit UTC offset")
    return parsed


@dataclass
class DownloadedFileRecord:
    filename: str
    size_bytes: int
    checksum: str

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "DownloadedFileRecord":
        _require_nonempty_string(self.filename, "downloaded_file_manifest.filename")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise SchemaValidationError("downloaded_file_manifest.size_bytes must be an integer")
        if self.size_bytes < 0:
            raise SchemaValidationError("downloaded_file_manifest.size_bytes must be non-negative")
        validate_sha256(self.checksum, "downloaded_file_manifest.checksum")
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DownloadedFileRecord":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("downloaded_file_manifest entries must be objects")
        required = {"filename", "size_bytes", "checksum"}
        _require_exact_fields(value, required, "DownloadedFileRecord")
        return cls(**dict(value))


@dataclass
class ExecutionEnvironmentRecord:
    network_reachable: bool
    sandbox_bypass: bool
    python_version: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "ExecutionEnvironmentRecord":
        if type(self.network_reachable) is not bool:
            raise SchemaValidationError("execution_environment.network_reachable must be boolean")
        if type(self.sandbox_bypass) is not bool:
            raise SchemaValidationError("execution_environment.sandbox_bypass must be boolean")
        _require_nonempty_string(self.python_version, "execution_environment.python_version")
        validate_utc_timestamp(
            self.timestamp_utc, "execution_environment.timestamp_utc"
        )
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionEnvironmentRecord":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("execution_environment must be an object")
        required = {
            "network_reachable",
            "sandbox_bypass",
            "python_version",
            "timestamp_utc",
        }
        _require_exact_fields(value, required, "ExecutionEnvironmentRecord")
        return cls(**dict(value))


@dataclass
class AcquisitionProvenanceRecord:
    source: str
    api_endpoint: str
    authentication_method: str
    query_parameters: Dict[str, Any]
    total_api_calls: int
    total_scenes_returned: int
    first_scene_date: Optional[str]
    last_scene_date: Optional[str]
    http_status_codes: List[int]
    response_payload_hash: str
    downloaded_file_manifest: List[DownloadedFileRecord]
    execution_environment: ExecutionEnvironmentRecord

    REQUIRED_FIELDS = {
        "source",
        "api_endpoint",
        "authentication_method",
        "query_parameters",
        "total_api_calls",
        "total_scenes_returned",
        "first_scene_date",
        "last_scene_date",
        "http_status_codes",
        "response_payload_hash",
        "downloaded_file_manifest",
        "execution_environment",
    }

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "AcquisitionProvenanceRecord":
        _require_nonempty_string(self.source, "source")
        _require_nonempty_string(self.api_endpoint, "api_endpoint")
        endpoint = urlparse(self.api_endpoint)
        if endpoint.scheme not in {"https", "http"} or not endpoint.netloc:
            raise SchemaValidationError("api_endpoint must be an absolute HTTP(S) URL")
        _require_nonempty_string(self.authentication_method, "authentication_method")

        if not isinstance(self.query_parameters, dict):
            raise SchemaValidationError("query_parameters must be an object")
        required_query_fields = {"date_range", "spatial_filter"}
        missing_query = required_query_fields - set(self.query_parameters)
        if missing_query:
            raise SchemaValidationError(
                f"query_parameters is missing required fields: {sorted(missing_query)}"
            )
        if not ({"band_selection", "variables"} & set(self.query_parameters)):
            raise SchemaValidationError(
                "query_parameters must include band_selection or variables"
            )
        try:
            json.dumps(self.query_parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                "query_parameters must contain finite JSON-serializable values"
            ) from exc

        for field_name, value in (
            ("total_api_calls", self.total_api_calls),
            ("total_scenes_returned", self.total_scenes_returned),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"{field_name} must be a non-negative integer")

        if (self.first_scene_date is None) != (self.last_scene_date is None):
            raise SchemaValidationError(
                "first_scene_date and last_scene_date must either both be null or both be timestamps"
            )
        if self.total_scenes_returned > 0 and self.first_scene_date is None:
            raise SchemaValidationError(
                "nonzero total_scenes_returned requires first_scene_date and last_scene_date"
            )
        if self.total_scenes_returned == 0 and self.first_scene_date is not None:
            raise SchemaValidationError(
                "zero total_scenes_returned requires null scene dates"
            )
        if self.total_scenes_returned > 0 and self.total_api_calls == 0:
            raise SchemaValidationError(
                "nonzero total_scenes_returned requires at least one API call"
            )
        if self.first_scene_date is not None:
            first = validate_utc_timestamp(self.first_scene_date, "first_scene_date")
            last = validate_utc_timestamp(self.last_scene_date, "last_scene_date")
            if first > last:
                raise SchemaValidationError("first_scene_date must not be after last_scene_date")

        if not isinstance(self.http_status_codes, list):
            raise SchemaValidationError("http_status_codes must be a list")
        for status in self.http_status_codes:
            if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
                raise SchemaValidationError(
                    "http_status_codes entries must be integer HTTP status codes from 100 to 599"
                )

        validate_sha256(self.response_payload_hash, "response_payload_hash")
        if not isinstance(self.downloaded_file_manifest, list) or not all(
            isinstance(item, DownloadedFileRecord)
            for item in self.downloaded_file_manifest
        ):
            raise SchemaValidationError(
                "downloaded_file_manifest must contain DownloadedFileRecord objects"
            )
        for item in self.downloaded_file_manifest:
            item.validate_record()
        filenames = [item.filename for item in self.downloaded_file_manifest]
        if len(set(filenames)) != len(filenames):
            raise SchemaValidationError(
                "downloaded_file_manifest filenames must be unique"
            )
        if not isinstance(self.execution_environment, ExecutionEnvironmentRecord):
            raise SchemaValidationError(
                "execution_environment must be an ExecutionEnvironmentRecord"
            )
        self.execution_environment.validate_record()
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate_record()
        return asdict(self)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionProvenanceRecord":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("AcquisitionProvenanceRecord must be an object")
        _require_exact_fields(value, cls.REQUIRED_FIELDS, "AcquisitionProvenanceRecord")
        parsed = dict(value)
        files = parsed["downloaded_file_manifest"]
        if not isinstance(files, list):
            raise SchemaValidationError("downloaded_file_manifest must be a list")
        parsed["downloaded_file_manifest"] = [
            DownloadedFileRecord.from_dict(item) for item in files
        ]
        parsed["execution_environment"] = ExecutionEnvironmentRecord.from_dict(
            parsed["execution_environment"]
        )
        return cls(**parsed)

    @classmethod
    def from_json(cls, payload: str) -> "AcquisitionProvenanceRecord":
        if not isinstance(payload, str):
            raise SchemaValidationError("JSON payload must be a string")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError("Invalid provenance JSON") from exc
        return cls.from_dict(value)
