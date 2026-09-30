"""
Sentinel-GL — Legacy Evidence Quarantine Verifier.
Mechanically prevents tainted legacy artifacts from satisfying publication claim checks.
Governed by INV-013, INV-015, INV-017, INV-026, INV-039.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import fnmatch
import json
import hashlib
import logging
import posixpath
import re

logger = logging.getLogger("sentinel_gl.quarantine")

DEFAULT_REGISTRY_PATH = Path("project/evolution/evidence_quarantine_registry.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QuarantinedArtifactError(Exception):
    """Raised when an artifact or run lineage is found in the evidence quarantine registry."""
    pass


@dataclass
class QuarantineEntry:
    entry_id: str
    target_path_or_pattern: str
    known_sha256: Optional[str]
    taint_reason: str
    quarantined_since: str
    replacement_contract: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.entry_id,
                self.target_path_or_pattern,
                self.taint_reason,
                self.quarantined_since,
                self.replacement_contract,
            )
        ):
            raise ValueError("quarantine entry text fields must be non-empty strings")
        if self.known_sha256 is not None and SHA256_RE.fullmatch(self.known_sha256) is None:
            raise ValueError("known_sha256 must be a 64-character lowercase SHA-256 hex string")


@dataclass
class QuarantineRegistry:
    version: str = "2.0"
    entries: List[QuarantineEntry] = field(default_factory=list)
    _path_patterns: Set[str] = field(default_factory=set, init=False)
    _known_hashes: Set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.version != "2.0":
            raise ValueError("quarantine registry version must equal '2.0'")
        if len({entry.entry_id for entry in self.entries}) != len(self.entries):
            raise ValueError("quarantine entry IDs must be unique")
        self._path_patterns = set(e.target_path_or_pattern for e in self.entries)
        self._known_hashes = set(e.known_sha256 for e in self.entries if e.known_sha256)

    @classmethod
    def load(cls, registry_path: Path = DEFAULT_REGISTRY_PATH) -> "QuarantineRegistry":
        if not registry_path.exists():
            raise FileNotFoundError(f"Quarantine registry not found at {registry_path}")
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("quarantined_artifacts"), list):
            raise ValueError("quarantine registry must contain a quarantined_artifacts list")
        entries = [QuarantineEntry(**e) for e in data["quarantined_artifacts"]]
        return cls(version=data.get("version", "2.0"), entries=entries)

    def is_quarantined(self, target_path: str, target_hash: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check whether a given path or hash matches any quarantine rule."""
        norm_path = self._normalise_path(target_path)
        for entry in self.entries:
            pattern = self._normalise_path(entry.target_path_or_pattern)
            prefix = pattern.rstrip("/")
            if (
                fnmatch.fnmatchcase(norm_path, pattern)
                or norm_path == prefix
                or norm_path.startswith(prefix + "/")
            ):
                return True, f"Matched path rule '{entry.target_path_or_pattern}': {entry.taint_reason}"
            if target_hash and entry.known_sha256 and target_hash == entry.known_sha256:
                return True, f"Matched SHA-256 rule '{entry.known_sha256}': {entry.taint_reason}"
        return False, None

    @staticmethod
    def _normalise_path(target_path: str) -> str:
        """Normalize separators, dot segments, absolute repo prefixes, and case."""
        raw = str(target_path).replace("\\", "/")
        while raw.startswith("./"):
            raw = raw[2:]
        normalized = posixpath.normpath(raw).lstrip("/")
        parts = normalized.split("/")
        markers = {"data", "models", "source", "results", "project"}
        for index, part in enumerate(parts):
            if part.casefold() in markers:
                normalized = "/".join(parts[index:])
                break
        return normalized.casefold()

    def assert_clean(self, target_path: str, target_hash: Optional[str] = None) -> None:
        """Assert that an artifact is not quarantined. Raises QuarantinedArtifactError if tainted."""
        quarantined, reason = self.is_quarantined(target_path, target_hash)
        if quarantined:
            raise QuarantinedArtifactError(
                f"REJECTED: Artifact '{target_path}' is quarantined and inadmissible for publication evidence. "
                f"Reason: {reason} (INV-039)."
            )


def assert_not_quarantined(path_or_hash: str, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """Convenience helper to assert that an artifact is admissible."""
    reg = QuarantineRegistry.load(registry_path)
    if SHA256_RE.fullmatch(path_or_hash):
        reg.assert_clean("<hash>", target_hash=path_or_hash)
    else:
        reg.assert_clean(path_or_hash)
