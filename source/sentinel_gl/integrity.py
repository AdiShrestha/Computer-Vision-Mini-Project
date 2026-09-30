"""Strict serialization and byte verification; checksums do not prove authenticity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def loads_strict(text: str):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_file(root: str | Path, relative: str, expected_sha256: str) -> Path:
    """Reject missing bytes, absolute/traversal paths, symlinks and hash changes."""
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("artifact path must be a nonempty relative path without traversal")
    path = root / rel
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("symlinks are not admissible evidence paths")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("artifact missing or outside evidence root")
    if file_hash(path) != expected_sha256:
        raise ValueError("artifact checksum mismatch")
    return path
