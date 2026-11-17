#!/usr/bin/env python3
"""Run the source-bound panel.v2, five-seed fit, and causal diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

from sentinel_gl.panel_v2 import load_panel_v2, write_panel_v2
from sentinel_gl.training_v2 import run_training_v2, TrainingConfigV2
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
        help="Architect-declared stable SHA-256 of observations.v2.json")
    parser.add_argument("--seeds", type=_seed_list, default=(0, 1, 2, 3, 4),
        help="at least five comma-separated deterministic training seeds")
    parser.add_argument("--optical-representation", choices=("five_channel", "reflectance_only"),
                        default="five_channel", help="Fixed representation; sensitivity fits use a separate output epoch")
    args = parser.parse_args()
    if (len(args.expected_ready_sha256) != 64 or
            any(ch not in "0123456789abcdef" for ch in args.expected_ready_sha256)):
        parser.error("--expected-ready-sha256 must be 64 lowercase hexadecimal characters")
    root = Path.cwd().resolve(strict=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    source_paths = sorted({Path(m.__file__).resolve() for name, m in sys.modules.items()
        if name.startswith("sentinel_gl") and getattr(m, "__file__", "").endswith(".py")})
    source_identity = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_paths}
    output = args.output if args.output.is_absolute() else root / args.output
    if not output.resolve(strict=False).is_relative_to(root):
        parser.error("--output must remain inside the workspace")
    if output.exists():
        parser.error("--output already exists; use a new attempt directory")

    panel = load_panel_v2(root, args.observations, mode="observational")
    if panel.manifest_sha256 != args.expected_ready_sha256:
        parser.error("observations manifest bytes differ from the Architect-declared READY hash")
    if panel.document["domain_protocol"].get("publication_authorized") is not False:
        parser.error("domain protocol unexpectedly authorizes publication")
    output.mkdir(parents=True, exist_ok=False)
    panel_record = write_panel_v2(panel, output / "panel.v2.json")
    if hashlib.sha256(panel.manifest_path.read_bytes()).hexdigest() != args.expected_ready_sha256:
        parser.error("observations manifest changed after panel validation")
    training = run_training_v2(panel, output / "training", seeds=args.seeds,
        config=TrainingConfigV2(optical_representation=args.optical_representation))
    inference = run_inference_v2(panel, output / "training", output / "inference")
    if hashlib.sha256(panel.manifest_path.read_bytes()).hexdigest() != args.expected_ready_sha256:
        parser.error("observations manifest changed during diagnostics")
    if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != digest
           for p, digest in source_identity.items()):
        raise RuntimeError("executed scientific source changed during diagnostics; retain failed attempt")
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
        "optical_representation": args.optical_representation,
        "runtime": {"python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(), "executable": sys.executable},
        "command_argv": [sys.executable, *sys.argv],
        "source_identity": source_identity,
        "requirements_sha256": hashlib.sha256((root / "source/requirements.lock").read_bytes()).hexdigest(),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started_clock,
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
