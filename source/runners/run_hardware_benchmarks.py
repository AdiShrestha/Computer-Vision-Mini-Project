#!/usr/bin/env python3
"""Hardware Benchmarking & M3 Resource Trials Runner for Sentinel-GL.

Executes synchronized hardware profiling and performance trials across CPU
and Apple Silicon Metal Performance Shaders (MPS), verifying numerical equivalence
within float32 tolerance and recording high-resolution resource observations.

Emits structured artifact: docs/benchmarks/m3_trials.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

# Ensure source/ is discoverable under isolated Python runtimes (-s -B)
RUNNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sentinel_gl.hardware import get_device_profile, run_hardware_trials


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel-GL M3 Hardware Benchmarking Runner.")
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of timed benchmark trials per workload (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations for frequency stabilization (default: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["mps", "cpu", "auto", "all"],
        help="Target execution device (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "docs" / "benchmarks" / "m3_trials.json"),
        help="Target output path for JSON hardware report (default: docs/benchmarks/m3_trials.json).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowable absolute difference between CPU and MPS (default: 1e-4).",
    )
    args = parser.parse_args()

    out_path = Path(args.output).resolve()

    print("=" * 68)
    print("Sentinel-GL Hardware Benchmarking & M3 Resource Trials")
    print("=" * 68)
    print(f"Target Output Path:   {out_path}")
    print(f"Trial Configuration:  {args.trials} timed trials after {args.warmup} warmups")
    print(f"Batch Size:           {args.batch_size} (180 days, 11 channels)")
    print(f"Device Selection:     {args.device}")
    print(f"Numerical Tolerance:  {args.tolerance:.1e}")
    print("-" * 68)

    report = run_hardware_trials(
        n_trials=args.trials,
        warmup_trials=args.warmup,
        device=args.device,
        output_path=out_path,
        batch_size=args.batch_size,
        tolerance=args.tolerance,
    )

    prof = report["device_profile"]
    ram_str = f"{prof['total_ram_bytes'] / (1024**3):.1f} GB" if prof.get("total_ram_bytes") else "Unknown"
    print(f"Host Platform:        {prof['os_platform']} ({prof['arch']})")
    print(f"Chip & Architecture:  {prof['chip_family']} ({prof['cpu_count']} logical cores, {ram_str} RAM)")
    print(f"PyTorch Runtime:      v{prof['pytorch_version']} (MPS Built: {prof['mps_built']}, Available: {prof['mps_available']})")
    print("-" * 68)

    print("Summary Metrics by Workload & Device:")
    for group_key, s in report.get("summary", {}).items():
        dev = s.get("device", "unknown")
        workload = s.get("workload", "unknown")
        w_mean = s["wall_time_seconds"]["mean"] * 1000.0
        w_std = s["wall_time_seconds"]["std"] * 1000.0
        c_mean = s["cpu_time_seconds"]["mean"] * 1000.0
        rss_mb = s["peak_rss_bytes"]["mean"] / (1024 * 1024)
        mps_info = ""
        if s.get("mps_allocated_bytes") and s["mps_allocated_bytes"].get("mean"):
            mps_mb = s["mps_allocated_bytes"]["mean"] / (1024 * 1024)
            mps_info = f" | mps_alloc={mps_mb:5.1f} MB"
        print(f"  [{dev.upper():3s}] {workload:18s}: wall={w_mean:6.2f} ± {w_std:4.2f} ms | cpu={c_mean:6.2f} ms | rss={rss_mb:6.1f} MB{mps_info}")

    neq = report.get("numerical_equivalence", {})
    eq_status = neq.get("status", "UNKNOWN")
    max_diff = neq.get("max_absolute_difference", 0.0)
    tol = neq.get("tolerance", args.tolerance)
    print("-" * 68)
    print(f"Numerical Equivalence (CPU vs MPS): {eq_status} (max abs diff: {max_diff:.2e}, tol: {tol:.1e})")
    print(f"Passively Cooled Thermal Disclosure: Recorded in report JSON.")
    print("=" * 68)
    print(f"Report written to: {out_path}")

    return 0 if eq_status in ("PASS", "CPU_ONLY") else 1


if __name__ == "__main__":
    sys.exit(main())
