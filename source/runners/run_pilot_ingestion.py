#!/usr/bin/env python3
"""Observational remote-sensing and reanalysis data ingestion CLI runner.

Coordinates authentic satellite and atmospheric data catalogue queries
for South Lhonak (SGL-001) and Khangchung Tsho (SGL-002) pilot cohort.
Enforces strict provenance tracking, streaming SHA-256 digests, and
truthful telemetry.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

# Ensure source root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel_gl.ingestion.pipeline import PilotIngestionPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observational data pilot ingestion coordinator."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to directory containing lake_registry.csv (default: data).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-09-01",
        help="Start date for pilot observation window YYYY-MM-DD (default: 2023-09-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-10-03",
        help="End date for pilot observation window YYYY-MM-DD (default: 2023-10-03).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for pilot dossier JSON (default: <data-dir>/pilot_dossier.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    lake_registry_path = data_dir / "lake_registry.csv"

    if not lake_registry_path.exists():
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": f"Missing lake registry at {lake_registry_path}",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    pipeline = PilotIngestionPipeline(data_dir=data_dir)
    try:
        dossier = pipeline.run_pilot(
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except Exception as ex:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": f"Ingestion pipeline failure: {ex}",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else (data_dir / "pilot_dossier.json")
    )
    if args.output is not None and output_path != (data_dir / "pilot_dossier.json"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

    summary = {
        "status": "SUCCESS",
        "output": str(output_path),
        "pilot_window": dossier.get("pilot_window"),
        "records_count": dossier.get("records_count"),
        "storage_footprint_bytes": dossier.get("storage_footprint_bytes"),
        "cohort_summary": {
            lake_id: {
                "name": info.get("name"),
                "role": info.get("role"),
                "observations": info.get("observations"),
            }
            for lake_id, info in dossier.get("cohort_summary", {}).items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
