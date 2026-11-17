#!/usr/bin/env python3
"""Build or replay retained asynchronous WP05/WP06 diagnostic panels offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from sentinel_gl.admission import AdmissionError  # noqa: E402
from sentinel_gl.multimodal import build_multimodal_panel, verify_multimodal_panel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="assemble a new source-bound diagnostic panel")
    build.add_argument("--root", type=Path, default=ROOT)
    build.add_argument("--feature-bundle", type=Path, required=True)
    build.add_argument("--feature-receipt", type=Path, required=True)
    build.add_argument("--sar-execution", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--mode", choices=("research", "development-diagnostic"), default="research")
    verify = sub.add_parser("verify", help="independently replay a retained panel from source arrays")
    verify.add_argument("--root", type=Path, default=ROOT)
    verify.add_argument("--panel", type=Path, required=True)
    verify.add_argument("--mode", choices=("research", "development-diagnostic"), default="research")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if args.mode != "development-diagnostic":
            raise AdmissionError("research panel execution is blocked; explicit development-diagnostic mode is required")
        if args.command == "build":
            document = build_multimodal_panel(root, args.feature_bundle, args.feature_receipt,
                args.sar_execution, args.output_dir, mode=args.mode)
            output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
            result = {"status": "DEVELOPMENT_DIAGNOSTIC_PANEL_BUILT_AND_REPLAYED",
                "panel_path": (output / "panel.v1.json").relative_to(root).as_posix(),
                "receipt_path": (output / "execution_receipt.json").relative_to(root).as_posix(),
                "group_count": len(document["groups"]), "panel_count": len(document["panels"]),
                "research_admissible": False, "training_admitted": False, "network_requests": 0}
        else:
            result = verify_multimodal_panel(root, args.panel)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
            "literal_error": str(exc), "research_admissible": False,
            "training_admitted": False, "network_requests": 0},
            sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
