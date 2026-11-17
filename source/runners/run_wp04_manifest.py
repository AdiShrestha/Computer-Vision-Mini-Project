#!/usr/bin/env python3
"""Build or independently verify the retained-evidence WP04 draft offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from sentinel_gl.wp04_manifests import build_bundle, verify_bundle  # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _runtime() -> dict:
    lock = ROOT / "source/requirements.lock"
    module = ROOT / "source/sentinel_gl/wp04_manifests.py"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "package_identity": "sentinel_gl.wp04_manifests:offline-draft-v2",
        "package_sha256": _sha(module),
        "requirements_lock_sha256": _sha(lock),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": "NOT_USED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="materialize the bounded retained-source draft")
    build.add_argument("--root", type=Path, default=ROOT)
    build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="independently recompute bundle integrity and contract invariants")
    verify.add_argument("--root", type=Path, default=ROOT)
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        if args.command == "build":
            if not args.output.is_absolute() and ".." in args.output.parts:
                raise ValueError("output path traversal is not permitted")
            output = build_bundle(root, args.output)
            payload = {"command": "build", "status": "PASS_LOCAL_DEVELOPMENT_DRAFT",
                       "bundle_path": output.relative_to(root).as_posix(),
                       "research_admissible": False, "runtime": _runtime()}
        else:
            result = verify_bundle(root, args.bundle)
            payload = {"command": "verify", "result": result, "runtime": _runtime()}
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"command": args.command, "status": "FAIL",
                          "error_type": type(exc).__name__, "literal_error": str(exc),
                          "runtime": _runtime()}, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
