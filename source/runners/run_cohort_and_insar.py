#!/usr/bin/env python3
"""Regional Cohort Expansion & InSAR Feasibility Runner for Sentinel-GL.

Coordinates the High Mountain Asia expanded candidate glacial lake and event
registries, executes spatial clustering and split leakage audits, and generates
the Sentinel-1 SLC InSAR feasibility dossier and technical report.

Artifacts:
  - data/lake_registry_expanded.csv
  - data/event_registry_expanded.csv
  - docs/insar/insar_feasibility_dossier.json
  - docs/insar/insar_feasibility_report.md
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

from sentinel_gl.cohort import (
    audit_cohort_integrity,
    export_expanded_registries,
    get_expanded_event_registry,
    get_expanded_lake_registry,
)
from sentinel_gl.insar import export_insar_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sentinel-GL Regional Cohort Expansion & InSAR Feasibility Runner."
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export expanded lake and event CSV registries to data directory.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit cohort integrity, spatial clustering, and split boundary isolation.",
    )
    parser.add_argument(
        "--insar",
        action="store_true",
        help="Generate InSAR feasibility dossier JSON and technical Markdown report.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all tasks: export registries, audit cohort integrity, and export InSAR artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "data"),
        help="Target directory for expanded CSV registries (default: data).",
    )
    parser.add_argument(
        "--output-insar",
        type=str,
        default=str(REPO_ROOT / "docs" / "insar"),
        help="Target directory for InSAR feasibility artifacts (default: docs/insar).",
    )
    args = parser.parse_args()

    # Determine actions: default to all if no specific action flag is provided
    run_all = args.all or (not args.export and not args.audit and not args.insar)
    do_export = args.export or run_all
    do_audit = args.audit or run_all
    do_insar = args.insar or run_all

    data_dir = Path(args.data_dir).resolve()
    insar_dir = Path(args.output_insar).resolve()

    print("=" * 70)
    print("Sentinel-GL Regional Cohort Expansion & InSAR Feasibility Runner")
    print("=" * 70)
    print(f"Data Registry Directory:   {data_dir}")
    print(f"InSAR Output Directory:    {insar_dir}")
    print(f"Execution Actions:         Export={do_export}, Audit={do_audit}, InSAR={do_insar}")
    print("-" * 70)

    # 1. Export expanded CSV registries
    if do_export:
        lake_csv = data_dir / "lake_registry_expanded.csv"
        event_csv = data_dir / "event_registry_expanded.csv"
        export_expanded_registries(lake_csv, event_csv)
        print(f"[1/3] Exported expanded lake registry:  {lake_csv}")
        print(f"      Exported expanded event registry: {event_csv}")

    # 2. Audit cohort integrity and spatial clustering
    if do_audit:
        lakes = get_expanded_lake_registry()
        events = get_expanded_event_registry()

        sample_splits = {
            "training": ["SGL-003", "SGL-006"],             # Hunza cluster
            "calibration": ["SGL-004", "SGL-007"],          # Gyirong cluster
            "evaluation": ["SGL-001", "SGL-002", "SGL-005", "SGL-008"], # Sikkim, Baige, Pumqu clusters
        }

        audit_res = audit_cohort_integrity(lakes, events, splits=sample_splits)
        print(f"[2/3] Cohort Integrity Audit Status:     {audit_res['status']}")
        print(f"      Curated Lakes:                    {audit_res['selection_bias']['n_lakes']} (4 Events, 4 Controls)")
        print(f"      Curated Historical Events:        {audit_res['selection_bias']['n_events']}")
        print(f"      Countries Represented:            {', '.join(audit_res['selection_bias']['countries'])}")
        print(f"      Basins Represented:               {', '.join(audit_res['selection_bias']['basins'])}")
        print(f"      Inter-Cluster Violations (<50km): {len(audit_res['inter_cluster_violations'])}")
        print(f"      Split Leakage Detected:           {audit_res['split_leakage_detected']}")

        if audit_res["status"] != "PASS":
            print(f"ERROR: Cohort integrity audit failed: {audit_res['errors']}", file=sys.stderr)
            return 1

    # 3. Export InSAR feasibility artifacts
    if do_insar:
        dossier_path, report_path = export_insar_artifacts(insar_dir)
        print(f"[3/3] Exported InSAR dossier JSON:      {dossier_path}")
        print(f"      Exported InSAR report Markdown:   {report_path}")

    print("-" * 70)
    print("Execution completed successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
