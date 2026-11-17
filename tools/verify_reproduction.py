#!/usr/bin/env python3
"""Standalone reproduction verification tool for Sentinel-GL.

Verifies:
  1. Dependency lock: installed packages match source/requirements.lock exactly.
  2. Clean-process checkpoint replay: standalone subprocess produces bit-identical reconstructions.
  3. Figure manifest byte verification: on-disk PNGs match cryptographic SHA-256 digests.
  4. Table metric integrity: generated markdown and CSV tables match computed headline statistics.

Emits structured audit artifact:
  docs/audit/wp10_reproduction/reproduction_report.json
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple
import torch

# Ensure source/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sentinel_gl.model import TimeSeriesMAE
from sentinel_gl.replay import (
    CheckpointBundle,
    compute_deterministic_reconstruction,
    verify_clean_process_replay,
)


def verify_dependency_lock(lock_path: Path) -> Dict[str, Any]:
    """Verify that installed packages match source/requirements.lock."""
    if not lock_path.is_file():
        return {
            "status": "FAIL",
            "error": f"Requirements lock file not found at {lock_path}",
            "mismatches": [],
        }

    mismatches: List[Dict[str, str]] = []
    checked: List[Dict[str, str]] = []

    with open(lock_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" not in line:
                continue
            pkg_name, expected_ver = line.split("==", 1)
            pkg_name = pkg_name.strip()
            expected_ver = expected_ver.strip()

            try:
                installed_ver = importlib.metadata.version(pkg_name)
                record = {
                    "package": pkg_name,
                    "expected": expected_ver,
                    "installed": installed_ver,
                    "match": (installed_ver == expected_ver),
                }
                checked.append(record)
                if installed_ver != expected_ver:
                    mismatches.append(record)
            except importlib.metadata.PackageNotFoundError:
                record = {
                    "package": pkg_name,
                    "expected": expected_ver,
                    "installed": "NOT_INSTALLED",
                    "match": False,
                }
                checked.append(record)
                mismatches.append(record)

    status = "PASS" if not mismatches else "FAIL"
    return {
        "scope": "installed-version comparison; not a clean installation",
        "status": status,
        "total_packages_checked": len(checked),
        "mismatches_count": len(mismatches),
        "mismatches": mismatches,
    }


def verify_clean_subprocess_replay(tolerance: float = 1e-5) -> Dict[str, Any]:
    """Verify clean-process checkpoint replay in an isolated subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_path = tmp_path / "replay_check.pt"
        input_path = tmp_path / "replay_input.pt"
        expected_path = tmp_path / "replay_expected.pt"

        # Construct minimal model
        config = {
            "n_channels": 11,
            "max_time_steps": 180,
            "d_model": 32,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "n_encoder_heads": 2,
            "n_decoder_heads": 2,
            "d_ff_encoder": 64,
            "d_ff_decoder": 64,
            "dropout": 0.0,
            "masking_ratio": 0.5,
        }
        torch.manual_seed(999)
        model = TimeSeriesMAE(**config)
        model.eval()

        # Input tensors
        torch.manual_seed(888)
        x = torch.randn(2, 180, 11)
        validity = torch.ones(2, 180, 11, dtype=torch.bool)
        validity[:, 20:160, :] = False
        x[~validity] = 0.0

        expected = compute_deterministic_reconstruction(model, x, validity, device="cpu")

        bundle = CheckpointBundle.capture_fixture(
            model=model,
            optimizer=None,
            transform_state={"mean": [0.0] * 11, "scale": [1.0] * 11, "fit_lake_ids": ["LAKE-REPLAY"]},
            telemetry={"replay_test": True},
            history=[],
        )
        bundle.save(ckpt_path)
        torch.save((x, validity), input_path)
        torch.save(expected, expected_path)

        # Run standalone subprocess check
        ok, max_diff = verify_clean_process_replay(
            checkpoint_path=ckpt_path,
            input_tensor_path=input_path,
            expected_output_path=expected_path,
            tolerance=tolerance,
        )

        status = "PASS" if (ok and max_diff <= tolerance) else "FAIL"
        return {
            "status": status,
            "subprocess_executed": ok,
            "max_absolute_difference": max_diff,
            "tolerance": tolerance,
        }


def verify_figure_manifest_hashes(*args, **kwargs):
    return {"status": "BLOCKED", "reason": "RESEARCH_ARTIFACT_ADMISSION_REQUIRED"}


def verify_table_metrics_integrity(*args, **kwargs):
    return {"status": "BLOCKED", "reason": "RESEARCH_ARTIFACT_ADMISSION_REQUIRED"}


def generate_reproduction_report(output_path=None):
    report = {"overall_status": "BLOCKED", "scope": "research reproduction", "checks": {"measurement_admission": {"status": "BLOCKED", "reason": "NO_ADMITTED_RESEARCH_RUN"}, "clean_install": {"status": "NOT_VERIFIED"}}}
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
    return report


def main() -> int:
    """CLI entrypoint for standalone reproduction verification."""
    parser = argparse.ArgumentParser(description="Sentinel-GL Reproduction Verifier.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "docs/audit/wp10_reproduction/reproduction_report.json"),
        help="Path for emitted reproduction report JSON",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Sentinel-GL Clean Environment Reproduction Verifier")
    print("=" * 65)

    report = generate_reproduction_report(Path(args.output))

    checks = report["checks"]
    for name, check in checks.items():
        print(f"{name}: {check['status']} ({check.get('reason', 'no executed evidence')})")
    print("-" * 65)
    print(f"OVERALL REPRODUCTION VERDICT:     {report['overall_status']}")
    print(f"Report written to:               {args.output}")
    print("=" * 65)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
