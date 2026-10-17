#!/usr/bin/env python3
"""Run the source-bound panel.v2, five-seed fit, and causal diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

from sentinel_gl.panel_v2 import load_panel_v2, write_panel_v2
from sentinel_gl.training_v2 import run_training_v2
from sentinel_gl.inference_v2 import run_inference_v2


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated nonnegative integers") from exc
    if len(seeds) < 5 or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds requires at least five distinct nonnegative integers")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path,
        help="root-relative observations.v2.json source manifest")
    parser.add_argument("--output", required=True, type=Path,
        help="new root-relative output directory; existing directories are refused")
    parser.add_argument("--expected-ready-sha256", required=True,
        help="Expected stable SHA-256 of observations.v2.json")
    parser.add_argument("--seeds", type=_seed_list, default=(0, 1, 2, 3, 4),
        help="at least five comma-separated deterministic training seeds")
    args = parser.parse_args()
    if (len(args.expected_ready_sha256) != 64 or
            any(ch not in "0123456789abcdef" for ch in args.expected_ready_sha256)):
        parser.error("--expected-ready-sha256 must be 64 lowercase hexadecimal characters")
    root = Path.cwd().resolve(strict=True)
    output = args.output if args.output.is_absolute() else root / args.output
    if not output.resolve(strict=False).is_relative_to(root):
        parser.error("--output must remain inside the workspace")
    if output.exists():
        parser.error("--output already exists; use a new attempt directory")

    panel = load_panel_v2(root, args.observations)
    if panel.manifest_sha256 != args.expected_ready_sha256:
        parser.error("observations manifest bytes differ from the expected input hash")
    if panel.document["domain_protocol"].get("publication_authorized") is not False:
        parser.error("domain protocol unexpectedly authorizes publication")
    output.mkdir(parents=True, exist_ok=False)
    panel_record = write_panel_v2(panel, output / "panel.v2.json")
    if hashlib.sha256(panel.manifest_path.read_bytes()).hexdigest() != args.expected_ready_sha256:
        parser.error("observations manifest changed after panel validation")
    training = run_training_v2(panel, output / "training", seeds=args.seeds)
    inference = run_inference_v2(panel, output / "training", output / "inference")
    if hashlib.sha256(panel.manifest_path.read_bytes()).hexdigest() != args.expected_ready_sha256:
        parser.error("observations manifest changed during diagnostics")
    summary = {
        "schema_version": 2,
        "status": "RETROSPECTIVE_DEVELOPMENT_ANALYSIS",
        "panel_v2": panel_record,
        "training_report": {"path": (output / "training/training_report.v2.json").relative_to(root).as_posix(),
            "seed_count": len(training["seeds"]),
            "selected_checkpoint_count": sum(x["selected_checkpoint_sha256"] is not None for x in training["seeds"])},
        "inference_report": {"path": (output / "inference/analysis_summary.v2.json").relative_to(root).as_posix(),
            "score_rows": inference["score_row_count"],
            "analysis_status": inference["analysis_status"],
            "publication_authorized": inference["publication_authorized"]},
        "source_manifest_sha256": panel.manifest_sha256,
        "expected_ready_sha256": args.expected_ready_sha256,
        "domain_protocol_sha256": panel.domain_protocol_sha256,
        "seeds": list(args.seeds),
        "runtime": {"python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(), "executable": sys.executable},
        "command_argv": [sys.executable, *sys.argv],
    }
    (output / "execution_summary.v2.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"],
        "observation_count": panel.document["observation_count"],
        "seed_count": len(args.seeds), "score_rows": inference["score_row_count"],
        "analysis_status": inference["analysis_status"],
        "publication_authorized": inference["publication_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
