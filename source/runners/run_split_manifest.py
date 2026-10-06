#!/usr/bin/env python3
"""Split Manifest Runner for Sentinel-GL.

Coordinates authentic pilot observational records from data/pilot_dossier.json,
lake registry (data/lake_registry.csv), and event registry (data/event_registry.csv)
into a verifiable spatiotemporal split manifest.

Enforces four anti-leakage invariants:
1. Retrospective Decision Availability (t_acq < t_decision, zero future leakage).
2. Spatial Cluster Isolation (lakes < 50 km belong to the same cluster and are never split).
3. Temporal Context Purging (transition windows across split horizons are purged).
4. Pre-Event Quarantining (all observations on event lakes >= event cutoff are excluded).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Ensure source/ is discoverable under isolated Python runtimes (-s -B)
RUNNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sentinel_gl.splits import (
    SplitEngine,
    SplitManifest,
    parse_utc_timestamp,
    verify_retrospective_invariants,
)


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_lake_registry(filepath: Path) -> List[Dict[str, Any]]:
    """Load and validate lake registry CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Lake registry not found at {filepath}")

    lakes: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lake = {
                "lake_id": row["lake_id"].strip(),
                "rgi_id": row.get("rgi_id", "").strip(),
                "glims_id": row.get("glims_id", "").strip(),
                "name": row.get("name", "").strip(),
                "centroid_lat": float(row["centroid_lat"]),
                "centroid_lon": float(row["centroid_lon"]),
                "elevation_m": float(row.get("elevation_m", 0.0)),
                "dam_type": row.get("dam_type", "moraine_dammed").strip(),
                "area_km2_2020": float(row.get("area_km2_2020", 0.0)),
                "basin": row.get("basin", "").strip(),
                "country": row.get("country", "").strip(),
                "cluster_id": row.get("cluster_id", "").strip(),
                "cohort_role": row.get("cohort_role", "").strip(),
            }
            lakes.append(lake)
    return lakes


def load_event_cutoffs(filepath: Path) -> Dict[str, str]:
    """Derive conservative pre-event cutoffs from event registry CSV."""
    if not filepath.exists():
        return {}

    cutoffs: Dict[str, str] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lake_id = row["lake_id"].strip()
            onset_earliest = row.get("onset_earliest", "").strip()
            if onset_earliest:
                dt = parse_utc_timestamp(onset_earliest)
                # Conservative pre-event quarantine: beginning of event day (00:00:00 UTC)
                day_start_iso = dt.strftime("%Y-%m-%dT00:00:00Z")
                cutoffs[lake_id] = day_start_iso
    return cutoffs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatiotemporal split manifest generator and anti-leakage availability scheduler."
    )
    parser.add_argument(
        "--dossier",
        type=Path,
        default=Path("data/pilot_dossier.json"),
        help="Path to observational pilot dossier JSON (default: data/pilot_dossier.json).",
    )
    parser.add_argument(
        "--lake-registry",
        type=Path,
        default=Path("data/lake_registry.csv"),
        help="Path to lake registry CSV (default: data/lake_registry.csv).",
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=Path("data/event_registry.csv"),
        help="Path to event registry CSV (default: data/event_registry.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/split_manifest.json"),
        help="Destination path for split manifest JSON (default: data/split_manifest.json).",
    )
    parser.add_argument(
        "--split-id",
        type=str,
        default="SPLIT-PILOT-01",
        help="Identifier for the split manifest (default: SPLIT-PILOT-01).",
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        default=50.0,
        help="Spatial isolation buffer distance in km (default: 50.0).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=180,
        help="Retrospective context window duration in days (default: 180).",
    )
    parser.add_argument(
        "--stride-days",
        type=int,
        default=5,
        help="Cadence stride between consecutive decision timestamps in days (default: 5).",
    )
    parser.add_argument(
        "--min-obs-per-modality",
        type=int,
        default=2,
        help="Minimum required observations per modality for eligibility (default: 2).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD for decision schedule (default: from dossier).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD for decision schedule (default: from dossier).",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="Optional ISO timestamp for temporal train/test split boundary.",
    )
    parser.add_argument(
        "--test-cluster-ids",
        type=str,
        default=None,
        help="Comma-separated cluster IDs assigned to test evaluation split.",
    )
    parser.add_argument(
        "--val-cluster-ids",
        type=str,
        default=None,
        help="Comma-separated cluster IDs assigned to validation split.",
    )
    return parser.parse_args(argv)


def run_split_manifest(
    dossier_path: Path,
    lake_registry_path: Path,
    event_registry_path: Path,
    output_path: Path,
    split_id: str = "SPLIT-PILOT-01",
    buffer_km: float = 50.0,
    window_days: int = 180,
    stride_days: int = 5,
    min_obs_per_modality: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    split_date: Optional[str] = None,
    test_cluster_ids: Optional[Sequence[str]] = None,
    val_cluster_ids: Optional[Sequence[str]] = None,
) -> SplitManifest:
    """Execute split manifest generation with full cryptographic provenance."""
    dossier_path = dossier_path.resolve()
    lake_registry_path = lake_registry_path.resolve()
    event_registry_path = event_registry_path.resolve()
    output_path = output_path.resolve()

    if not dossier_path.exists():
        raise FileNotFoundError(f"Dossier not found at {dossier_path}")

    # 1. Compute input byte digests
    dossier_sha256 = compute_file_sha256(dossier_path)
    lake_reg_sha256 = compute_file_sha256(lake_registry_path)
    event_reg_sha256 = (
        compute_file_sha256(event_registry_path)
        if event_registry_path.exists()
        else None
    )

    # 2. Load lake and event registries
    lakes = load_lake_registry(lake_registry_path)
    event_cutoffs = load_event_cutoffs(event_registry_path)

    # 3. Load observational records from dossier
    with open(dossier_path, "r", encoding="utf-8") as f:
        dossier = json.load(f)

    records: List[Dict[str, Any]] = dossier.get("record_provenance", [])
    if not records:
        raise ValueError(f"Dossier at {dossier_path} contains zero records in record_provenance")

    # Group observations by lake
    observations_by_lake: Dict[str, List[Dict[str, Any]]] = {}
    quarantined_record_ids: List[str] = []
    quarantined_records_count = 0

    for r in records:
        lid = str(r["lake_id"])
        observations_by_lake.setdefault(lid, []).append(r)
        # Check if record falls into quarantine window
        cutoff = event_cutoffs.get(lid)
        if cutoff is not None:
            t_acq = parse_utc_timestamp(r["acquisition_timestamp"])
            t_cut = parse_utc_timestamp(cutoff)
            if t_acq >= t_cut:
                quarantined_record_ids.append(r["record_id"])
                quarantined_records_count += 1

    # 4. Determine schedule start and end dates
    pilot_window = dossier.get("pilot_window", {})
    if start_date is None:
        start_date = pilot_window.get("start_date", "2023-09-01")
    if end_date is None:
        end_date = pilot_window.get("end_date", "2023-10-03")

    if not start_date.endswith("Z") and "T" not in start_date:
        start_date = f"{start_date}T00:00:00Z"
    if not end_date.endswith("Z") and "T" not in end_date:
        end_date = f"{end_date}T00:00:00Z"

    # 5. Resolve cluster assignments
    # If test_cluster_ids is not explicitly specified, check cohort roles
    if test_cluster_ids is None:
        eval_roles = {"event_case_eval", "negative_control_eval", "event", "control"}
        all_eval = all(l.get("cohort_role", "") in eval_roles for l in lakes)
        if all_eval:
            # All lakes in registry are evaluation cohort members: assign their clusters to test
            test_clusters_set = {str(l["cluster_id"]) for l in lakes if l.get("cluster_id")}
            test_cluster_ids = sorted(test_clusters_set)
        else:
            test_cluster_ids = ()

    # 6. Initialize SplitEngine and construct split manifest
    engine = SplitEngine(
        buffer_km=buffer_km,
        window_days=window_days,
        stride_days=stride_days,
        min_obs_per_modality=min_obs_per_modality,
    )

    manifest = engine.build_spatiotemporal_split(
        split_id=split_id,
        lakes=lakes,
        observations_by_lake=observations_by_lake,
        start_date=start_date,
        end_date=end_date,
        split_date=split_date,
        test_cluster_ids=test_cluster_ids,
        val_cluster_ids=val_cluster_ids,
        event_cutoffs=event_cutoffs,
    )

    # 7. Enrich manifest metadata with cryptographic provenance and audit telemetry
    try:
        rel_dossier = str(dossier_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_dossier = dossier_path.name

    try:
        rel_lake_reg = str(lake_registry_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_lake_reg = lake_registry_path.name

    try:
        rel_event_reg = str(event_registry_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_event_reg = event_registry_path.name

    manifest.metadata.update(
        {
            "dossier_sha256": dossier_sha256,
            "lake_registry_sha256": lake_reg_sha256,
            "event_registry_sha256": event_reg_sha256,
            "source_dossier": rel_dossier,
            "source_lake_registry": rel_lake_reg,
            "source_event_registry": rel_event_reg,
            "total_input_records": len(records),
            "quarantined_records_count": quarantined_records_count,
            "quarantined_record_ids": quarantined_record_ids,
            "event_cutoffs": event_cutoffs,
            "schedule_start_date": start_date,
            "schedule_end_date": end_date,
        }
    )

    # 8. Write manifest to destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json(indent=2))

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    test_clusters = (
        [c.strip() for c in args.test_cluster_ids.split(",") if c.strip()]
        if args.test_cluster_ids
        else None
    )
    val_clusters = (
        [c.strip() for c in args.val_cluster_ids.split(",") if c.strip()]
        if args.val_cluster_ids
        else None
    )

    try:
        manifest = run_split_manifest(
            dossier_path=args.dossier,
            lake_registry_path=args.lake_registry,
            event_registry_path=args.event_registry,
            output_path=args.output,
            split_id=args.split_id,
            buffer_km=args.buffer_km,
            window_days=args.window_days,
            stride_days=args.stride_days,
            min_obs_per_modality=args.min_obs_per_modality,
            start_date=args.start_date,
            end_date=args.end_date,
            split_date=args.split_date,
            test_cluster_ids=test_clusters,
            val_cluster_ids=val_clusters,
        )
    except Exception as ex:
        err_out = {
            "status": "ERROR",
            "error": str(ex),
            "error_type": type(ex).__name__,
        }
        print(json.dumps(err_out, indent=2), file=sys.stderr)
        return 1

    summary = {
        "status": "SUCCESS",
        "output": str(args.output.resolve()),
        "split_id": manifest.split_id,
        "protocol": manifest.protocol,
        "clusters": manifest.clusters,
        "train_lakes": list(manifest.train_lakes),
        "val_lakes": list(manifest.val_lakes),
        "test_lakes": list(manifest.test_lakes),
        "summary": manifest.to_dict()["summary"],
        "quarantined_records_count": manifest.metadata.get("quarantined_records_count", 0),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
