"""Strict serialization and byte verification; checksums do not prove authenticity."""
from __future__ import annotations
import hashlib
import json
import math
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
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number: " + value)
        return result
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant,
                      parse_float=finite_float)


def digest(value) -> str:
    """SHA-256 of Python sorted compact JSON (python-json-v1), not RFC 8785.

    Integers and floats keep their JSON spellings; tuples become JSON arrays.
    Cross-language verification must reproduce these bytes or migrate to an
    explicitly versioned serialization contract without rewriting old epochs.
    """
    def validate(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("canonical JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("canonical JSON values must have explicit builtin types")
    validate(value)
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
