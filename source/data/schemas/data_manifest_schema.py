"""Strict schema for provenance-bearing scientific dataset manifests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, List, Mapping, Optional

from .provenance_schema import (
    SchemaValidationError,
    _require_exact_fields,
    _require_nonempty_string,
    validate_sha256,
    validate_utc_timestamp,
)


class EvidenceClass(str, Enum):
    AUTHENTICATED = "AUTHENTICATED"
    DERIVED_FROM_AUTHENTICATED = "DERIVED_FROM_AUTHENTICATED"
    SYNTHETIC_EXPLICIT = "SYNTHETIC_EXPLICIT"
    LEGACY_TAINTED = "LEGACY_TAINTED"
    UNKNOWN = "UNKNOWN"


def _validate_percentage(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field_name} must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric <= 100.0:
        raise SchemaValidationError(f"{field_name} must be between 0 and 100")
    return numeric


@dataclass
class LakeDataEntry:
    temporal_start: Optional[str]
    temporal_end: Optional[str]
    missingness_rate: float
    scene_count: int
    file_paths: List[str]
    checksums: Dict[str, str]

    REQUIRED_FIELDS = {
        "temporal_start",
        "temporal_end",
        "missingness_rate",
        "scene_count",
        "file_paths",
        "checksums",
    }

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "LakeDataEntry":
        if (self.temporal_start is None) != (self.temporal_end is None):
            raise SchemaValidationError(
                "temporal_start and temporal_end must either both be null or both be timestamps"
            )
        if self.temporal_start is not None:
            start = validate_utc_timestamp(self.temporal_start, "temporal_start")
            end = validate_utc_timestamp(self.temporal_end, "temporal_end")
            if start > end:
                raise SchemaValidationError("temporal_start must not be after temporal_end")
        if isinstance(self.missingness_rate, bool) or not isinstance(
            self.missingness_rate, (int, float)
        ):
            raise SchemaValidationError("missingness_rate must be numeric")
        if not 0.0 <= float(self.missingness_rate) <= 1.0:
            raise SchemaValidationError("missingness_rate must be between 0 and 1")
        if isinstance(self.scene_count, bool) or not isinstance(self.scene_count, int):
            raise SchemaValidationError("scene_count must be an integer")
        if self.scene_count < 0:
            raise SchemaValidationError("scene_count must be non-negative")
        if self.scene_count > 0 and self.temporal_start is None:
            raise SchemaValidationError(
                "nonzero scene_count requires temporal_start and temporal_end"
            )
        if not isinstance(self.file_paths, list) or not all(
            isinstance(path, str) and path.strip() for path in self.file_paths
        ):
            raise SchemaValidationError("file_paths must be a list of non-empty strings")
        if len(set(self.file_paths)) != len(self.file_paths):
            raise SchemaValidationError("file_paths must not contain duplicates")
        if not isinstance(self.checksums, dict):
            raise SchemaValidationError("checksums must be an object keyed by file path")
        if set(self.checksums) != set(self.file_paths):
            raise SchemaValidationError(
                "checksums keys must exactly match file_paths"
            )
        for path, checksum in self.checksums.items():
            validate_sha256(checksum, f"checksums[{path!r}]")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate_record()
        return {
            "temporal_start": self.temporal_start,
            "temporal_end": self.temporal_end,
            "missingness_rate": self.missingness_rate,
            "scene_count": self.scene_count,
            "file_paths": list(self.file_paths),
            "checksums": dict(self.checksums),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LakeDataEntry":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("LakeDataEntry must be an object")
        _require_exact_fields(value, cls.REQUIRED_FIELDS, "LakeDataEntry")
        return cls(**dict(value))


@dataclass
class ChannelProvenanceEntry:
    source_sensor: str
    raw_product: str
    transformation_pipeline: List[str]
    version: str

    REQUIRED_FIELDS = {
        "source_sensor",
        "raw_product",
        "transformation_pipeline",
        "version",
    }

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "ChannelProvenanceEntry":
        _require_nonempty_string(self.source_sensor, "source_sensor")
        _require_nonempty_string(self.raw_product, "raw_product")
        _require_nonempty_string(self.version, "version")
        if not isinstance(self.transformation_pipeline, list) or not all(
            isinstance(step, str) and step.strip()
            for step in self.transformation_pipeline
        ):
            raise SchemaValidationError(
                "transformation_pipeline must be a list of non-empty strings"
            )
        if not self.transformation_pipeline:
            raise SchemaValidationError("transformation_pipeline must not be empty")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate_record()
        return {
            "source_sensor": self.source_sensor,
            "raw_product": self.raw_product,
            "transformation_pipeline": list(self.transformation_pipeline),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelProvenanceEntry":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("ChannelProvenanceEntry must be an object")
        _require_exact_fields(value, cls.REQUIRED_FIELDS, "ChannelProvenanceEntry")
        return cls(**dict(value))


@dataclass
class GapStatistics:
    observed_missingness_percentage: Dict[str, float]
    declared_expected_gaps: Dict[str, float]

    REQUIRED_FIELDS = {
        "observed_missingness_percentage",
        "declared_expected_gaps",
    }

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "GapStatistics":
        if not isinstance(self.observed_missingness_percentage, dict):
            raise SchemaValidationError(
                "observed_missingness_percentage must be an object"
            )
        if not isinstance(self.declared_expected_gaps, dict):
            raise SchemaValidationError("declared_expected_gaps must be an object")
        if set(self.observed_missingness_percentage) != set(
            self.declared_expected_gaps
        ):
            raise SchemaValidationError(
                "observed and declared gap statistics must use identical sensor keys"
            )
        for sensor, value in self.observed_missingness_percentage.items():
            _require_nonempty_string(sensor, "gap-statistics sensor")
            _validate_percentage(value, f"observed_missingness_percentage[{sensor!r}]")
        for sensor, value in self.declared_expected_gaps.items():
            _validate_percentage(value, f"declared_expected_gaps[{sensor!r}]")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate_record()
        return {
            "observed_missingness_percentage": dict(
                self.observed_missingness_percentage
            ),
            "declared_expected_gaps": dict(self.declared_expected_gaps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GapStatistics":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("GapStatistics must be an object")
        _require_exact_fields(value, cls.REQUIRED_FIELDS, "GapStatistics")
        return cls(**dict(value))


@dataclass
class DatasetManifest:
    manifest_version: str
    created_at: str
    lake_entries: Dict[str, LakeDataEntry]
    channel_provenance: Dict[str, ChannelProvenanceEntry]
    gap_statistics: GapStatistics
    evidence_class: EvidenceClass

    REQUIRED_FIELDS = {
        "manifest_version",
        "created_at",
        "lake_entries",
        "channel_provenance",
        "gap_statistics",
        "evidence_class",
    }

    def __post_init__(self) -> None:
        self.validate_record()

    def validate_record(self) -> "DatasetManifest":
        if self.manifest_version != "2.0":
            raise SchemaValidationError("manifest_version must equal '2.0'")
        validate_utc_timestamp(self.created_at, "created_at")
        if not isinstance(self.lake_entries, dict):
            raise SchemaValidationError("lake_entries must be an object")
        for lake_id, entry in self.lake_entries.items():
            _require_nonempty_string(lake_id, "lake_entries key")
            if not isinstance(entry, LakeDataEntry):
                raise SchemaValidationError(
                    "lake_entries values must be LakeDataEntry objects"
                )
            entry.validate_record()
        if not isinstance(self.channel_provenance, dict):
            raise SchemaValidationError("channel_provenance must be an object")
        for channel, entry in self.channel_provenance.items():
            _require_nonempty_string(channel, "channel_provenance key")
            if not isinstance(entry, ChannelProvenanceEntry):
                raise SchemaValidationError(
                    "channel_provenance values must be ChannelProvenanceEntry objects"
                )
            entry.validate_record()
        if not isinstance(self.gap_statistics, GapStatistics):
            raise SchemaValidationError("gap_statistics must be a GapStatistics object")
        self.gap_statistics.validate_record()
        if not isinstance(self.evidence_class, EvidenceClass):
            raise SchemaValidationError("evidence_class must be an EvidenceClass value")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate_record()
        return {
            "manifest_version": self.manifest_version,
            "created_at": self.created_at,
            "lake_entries": {
                key: value.to_dict() for key, value in self.lake_entries.items()
            },
            "channel_provenance": {
                key: value.to_dict()
                for key, value in self.channel_provenance.items()
            },
            "gap_statistics": self.gap_statistics.to_dict(),
            "evidence_class": self.evidence_class.value,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("DatasetManifest must be an object")
        _require_exact_fields(value, cls.REQUIRED_FIELDS, "DatasetManifest")
        parsed = dict(value)
        lake_entries = parsed["lake_entries"]
        channel_entries = parsed["channel_provenance"]
        if not isinstance(lake_entries, Mapping):
            raise SchemaValidationError("lake_entries must be an object")
        if not isinstance(channel_entries, Mapping):
            raise SchemaValidationError("channel_provenance must be an object")
        parsed["lake_entries"] = {
            key: LakeDataEntry.from_dict(entry)
            for key, entry in lake_entries.items()
        }
        parsed["channel_provenance"] = {
            key: ChannelProvenanceEntry.from_dict(entry)
            for key, entry in channel_entries.items()
        }
        parsed["gap_statistics"] = GapStatistics.from_dict(parsed["gap_statistics"])
        try:
            parsed["evidence_class"] = EvidenceClass(parsed["evidence_class"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("invalid evidence_class") from exc
        return cls(**parsed)

    @classmethod
    def from_json(cls, payload: str) -> "DatasetManifest":
        if not isinstance(payload, str):
            raise SchemaValidationError("JSON payload must be a string")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError("Invalid dataset-manifest JSON") from exc
        return cls.from_dict(value)
