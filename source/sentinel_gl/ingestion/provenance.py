"""Provenance tracking and record models for remote sensing data ingestion."""
from __future__ import annotations
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class AcquisitionRecord:
    record_id: str
    lake_id: str
    modality: str  # optical, sar, weather, elevation
    provider_name: str
    product_identifier: str
    acquisition_timestamp: str  # ISO-8601 UTC
    query_timestamp: str        # ISO-8601 UTC
    roi_geometry: Dict[str, Any]
    status: str  # CATALOGUED, EXPORTED, HASH_VERIFIED, QA_ACCEPTED, NO_DATA, FAILED
    sha256: Optional[str] = None
    storage_bytes: int = 0
    missingness_mask: Optional[str] = None
    missingness_reason: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file with streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()
