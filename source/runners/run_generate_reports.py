#!/usr/bin/env python3
"""Scientific Report, Figures & Manifest Generation Runner for Sentinel-GL.

Coordinates the end-to-end execution of:
1. Fig 1: Observation Cadence & Multi-Modal Coverage (C_obs)
2. Fig 2: Multi-Modal Feature Panel Traces & Pre-Event Trajectory
3. Fig 3: Anomaly Score Trajectories & Bounded Lead Time (Delta t_lead)
4. Fig 4: Negative-Control Alert Episodes & Follow-Up Exposure (lambda_alert)
5. Fig 5: 2^N Sensor Ablation Lattice Comparison
6. Manifest: Cryptographic SHA-256 byte digests & input lineage (docs/figures/manifest.json)
7. Evidence Tables: Table 1, Table 2, Table 3 (docs/tables/)

Emits reproducible artifacts from authentic pilot data without synthesized arrays.
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

from sentinel_gl.reports import run_full_report_generation


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel-GL Report, Figures & Manifest Runner.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(REPO_ROOT),
        help="Root repository directory (default: repo root).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "data"),
        help="Directory containing source data artifacts (default: data/).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "figures"),
        help="Target output directory for PNG figures (default: docs/figures/).",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "tables"),
        help="Target output directory for markdown/CSV tables (default: docs/tables/).",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Optional path to write JSON execution summary.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()

    print("=" * 68)
    print("Sentinel-GL Scientific Report, Figures & Manifest Generator")
    print("=" * 68)
    print(f"Data Source Directory:    {data_dir}")
    print(f"Figures Target Directory: {figures_dir}")
    print(f"Tables Target Directory:  {tables_dir}")
    print("-" * 68)

    summary = run_full_report_generation(
        root_dir=root_dir,
        data_dir=data_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )

    manifest = summary.get("figures_manifest", {})
    figures_count = manifest.get("figures_count", 0)
    print(f"Generated {figures_count} scientific figures successfully.")
    for fig in manifest.get("figures", []):
        fname = fig.get("filename")
        sha = fig.get("sha256", "")[:12]
        size = fig.get("size_bytes", 0)
        print(f"  - {fname:32s} (SHA256: {sha}..., {size:,} bytes)")

    print("-" * 68)
    print("Generated 3 evidence tables (Markdown & CSV):")
    print("  - table1_comparative_evaluation (.md & .csv)")
    print("  - table2_ablation_lattice       (.md & .csv)")
    print("  - table3_failure_taxonomy       (.md & .csv)")
    print("=" * 68)
    print(f"OVERALL GENERATION STATUS: {summary.get('status', 'FAIL')}")

    if args.output_summary:
        out_summary_p = Path(args.output_summary).resolve()
        out_summary_p.parent.mkdir(parents=True, exist_ok=True)
        out_summary_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Execution summary written to: {out_summary_p}")

    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
