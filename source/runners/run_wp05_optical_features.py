#!/usr/bin/env python3
"""Extract quarantined optical summaries from the accepted WP04 draft offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from sentinel_gl.integrity import file_hash  # noqa: E402
from sentinel_gl.wp05_optical_features import (  # noqa: E402
    build_feature_bundle,
    verify_feature_bundle,
    write_json,
)


def _runtime() -> dict:
    module = ROOT / "source/sentinel_gl/wp05_optical_features.py"
    lock = ROOT / "source/requirements.lock"
    schemas_dir = ROOT / "source/sentinel_gl/schemas"
    schema_hashes = {p.name: file_hash(p) for p in sorted(schemas_dir.glob("wp05_*.json"))}
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "executable_sha256": file_hash(Path(sys.executable)),
        "package_identity": "sentinel_gl.wp05_optical_features:development-only-v1",
        "package_sha256": file_hash(module),
        "requirements_lock_sha256": file_hash(lock),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "rasterio", "affine", "pyproj")},
        "schema_sha256": schema_hashes,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "randomness": "NONE_DETERMINISTIC",
        "network": "NOT_USED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="accepted WP04 run directory or bundle.v2.json path")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new output directory inside the repository")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        bundle = args.bundle if args.bundle.name == "bundle.v2.json" else args.bundle / "bundle.v2.json"
        if not bundle.is_absolute():
            bundle = root / bundle
        bundle = bundle.resolve(strict=True)
        output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
        output = output.resolve(strict=False)
        if not output.is_relative_to(root) or output == root:
            raise ValueError("output directory must remain inside the repository")
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        feature_doc = build_feature_bundle(root, bundle)
        verification = verify_feature_bundle(root, bundle, feature_doc)
        output.mkdir(parents=True, exist_ok=False)
        feature_path = output / "features.v1.json"
        verification_path = output / "verification.json"
        write_json(feature_path, feature_doc)
        write_json(verification_path, verification)
        runtime = _runtime()
        receipt = {
            "schema_version": 1,
            "status": "PASS_LOCAL_DEVELOPMENT_FEATURE_INTEGRITY_ONLY",
            "command": "run_wp05_optical_features",
            "bundle_path": bundle.relative_to(root).as_posix(),
            "bundle_sha256": file_hash(bundle),
            "output_files": [
                {"path": feature_path.relative_to(root).as_posix(),
                 "bytes": feature_path.stat().st_size, "sha256": file_hash(feature_path)},
                {"path": verification_path.relative_to(root).as_posix(),
                 "bytes": verification_path.stat().st_size, "sha256": file_hash(verification_path)},
            ],
            "record_count": len(feature_doc["records"]),
            "research_admissible": False,
            "research_status": "RESEARCH_NOT_READY",
            "plan_executable": False,
            "release_status": "BLOCKED",
            "runtime": runtime,
        }
        receipt_path = output / "execution_receipt.json"
        write_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"],
                          "feature_path": feature_path.relative_to(root).as_posix(),
                          "verification_path": verification_path.relative_to(root).as_posix(),
                          "receipt_path": receipt_path.relative_to(root).as_posix(),
                          "record_count": receipt["record_count"],
                          "research_admissible": False, "network": "NOT_USED"},
                         sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
                          "literal_error": str(exc), "research_admissible": False,
                          "network": "NOT_USED"}, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
