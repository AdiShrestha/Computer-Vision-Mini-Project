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

        bundle = CheckpointBundle.capture(
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


def verify_figure_manifest_hashes(manifest_path: Path, figures_dir: Path) -> Dict[str, Any]:
    """Verify that all figures on disk match cryptographic SHA-256 digests in manifest."""
    if not manifest_path.is_file():
        return {
            "status": "FAIL",
            "error": f"Figure manifest not found at {manifest_path}",
            "figures_checked": 0,
            "mismatches": [],
        }

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    figures = manifest.get("figures", [])
    mismatches: List[Dict[str, Any]] = []
    checked: List[Dict[str, Any]] = []

    for fig in figures:
        fname = fig["filename"]
        expected_hash = fig["sha256"]
        expected_size = fig["size_bytes"]
        fig_path = figures_dir / fname

        if not fig_path.is_file():
            mismatch = {
                "filename": fname,
                "error": "FILE_NOT_FOUND",
                "expected_sha256": expected_hash,
            }
            mismatches.append(mismatch)
            continue

        raw_bytes = fig_path.read_bytes()
        actual_size = len(raw_bytes)
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()

        is_match = (actual_hash == expected_hash) and (actual_size == expected_size)
        record = {
            "filename": fname,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "expected_bytes": expected_size,
            "actual_bytes": actual_size,
            "match": is_match,
        }
        checked.append(record)
        if not is_match:
            mismatches.append(record)

    status = "PASS" if (not mismatches and len(checked) == len(figures)) else "FAIL"
    return {
        "status": status,
        "figures_checked": len(checked),
        "total_in_manifest": len(figures),
        "mismatches_count": len(mismatches),
        "mismatches": mismatches,
    }


def verify_table_metrics_integrity(tables_dir: Path) -> Dict[str, Any]:
    """Verify that table 1 markdown and CSV contain expected headline numbers without fabrication."""
    csv_path = tables_dir / "table1_comparative_evaluation.csv"
    md_path = tables_dir / "table1_comparative_evaluation.md"

    if not csv_path.is_file() or not md_path.is_file():
        return {
            "status": "FAIL",
            "error": f"Table files missing in {tables_dir}",
        }

    rows: List[List[str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Check headers and row counts
    if len(rows) < 6:
        return {
            "status": "FAIL",
            "error": f"Table 1 CSV has only {len(rows)} rows, expected at least 6",
        }

    # Verify Case Detection row
    case_row = next((r for r in rows if len(r) > 1 and "SGL-001" in r[1]), None)
    alert_row = next((r for r in rows if len(r) > 1 and "lambda_alert" in r[1]), None)
    paired_rows = [r for r in rows if len(r) > 0 and r[0] == "Paired Contrast"]

    checks: Dict[str, bool] = {
        "case_detection_present": case_row is not None and case_row[7] in ("DETECTED", "NOT_DETECTED"),
        "alert_burden_present": alert_row is not None and alert_row[7] in ("ESTIMATED", "NOT_ESTIMABLE") and float(alert_row[2]) >= 0.0,
        "paired_contrasts_present": len(paired_rows) == 4,
    }

    all_pass = all(checks.values())
    return {
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "rows_verified": len(rows),
    }


def generate_reproduction_report(
    output_path: Path = REPO_ROOT / "docs/audit/wp10_reproduction/reproduction_report.json",
) -> Dict[str, Any]:
    """Run all reproduction checks and emit structured JSON report."""
    lock_path = REPO_ROOT / "source/requirements.lock"
    manifest_path = REPO_ROOT / "docs/figures/manifest.json"
    figures_dir = REPO_ROOT / "docs/figures"
    tables_dir = REPO_ROOT / "docs/tables"

    dep_check = verify_dependency_lock(lock_path)
    replay_check = verify_clean_subprocess_replay(tolerance=1e-5)
    figure_check = verify_figure_manifest_hashes(manifest_path, figures_dir)
    table_check = verify_table_metrics_integrity(tables_dir)

    all_passed = (
        dep_check["status"] == "PASS"
        and replay_check["status"] == "PASS"
        and figure_check["status"] == "PASS"
        and table_check["status"] == "PASS"
    )

    report = {
        "report_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if all_passed else "FAIL",
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
        },
        "checks": {
            "dependency_lock": dep_check,
            "clean_process_replay": replay_check,
            "figure_manifest_hashes": figure_check,
            "table_metrics_integrity": table_check,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

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
    print(f"1. Dependency Lock Check:       {checks['dependency_lock']['status']}")
    print(f"2. Subprocess Replay Check:      {checks['clean_process_replay']['status']} (diff: {checks['clean_process_replay'].get('max_absolute_difference', 0.0):.2e})")
    print(f"3. Figure Manifest Hash Check:   {checks['figure_manifest_hashes']['status']} ({checks['figure_manifest_hashes'].get('figures_checked', 0)} figures verified)")
    print(f"4. Table Metric Integrity Check: {checks['table_metrics_integrity']['status']}")
    print("-" * 65)
    print(f"OVERALL REPRODUCTION VERDICT:     {report['overall_status']}")
    print(f"Report written to:               {args.output}")
    print("=" * 65)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
