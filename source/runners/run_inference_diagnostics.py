#!/usr/bin/env python3
"""Independent Inference, Diagnostics & Negative-Result Evaluation Runner for Sentinel-GL.

Executes independent headline metric recomputation, Holm-Bonferroni multiplicity
adjustment across all four fair baselines, exact Garwood Poisson confidence intervals,
operational failure taxonomy categorization, and spatial cluster-conditional aggregation.

Emits:
1. data/claims_table.json (conservative claims table, marking non-significant findings FAIL_TO_REJECT)
2. data/diagnostic_report.json (5-category operational failure taxonomy report)
3. data/cluster_conditional_metrics.json (basin/cluster conditional aggregation)
"""
from __future__ import annotations
import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Ensure source/ is discoverable under isolated Python runtimes (-s -B)
RUNNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sentinel_gl.features import MultiModalPanel, StaticTopography
from sentinel_gl.predictions import PredictionRecord
from sentinel_gl.inference import (
    HolmBonferroniResult,
    SignFlipTestResult,
    cluster_conditional_metrics,
    exact_sign_flip_test,
    holm_bonferroni_correction,
    poisson_rate_confidence_interval,
)
from sentinel_gl.diagnostics import (
    ALL_ERROR_CATEGORIES,
    DiagnosticReport,
    diagnose_prediction_ledger,
)
from sentinel_gl.evaluation import (
    BaselineComparisonClaim,
    ClaimsTable,
    generate_claims_table,
)

DEFAULT_TOPOS: Dict[str, StaticTopography] = {
    "SGL-001": StaticTopography(elevation_m=5200.0, moraine_slope_deg=28.5, catchment_area_km2=15.2),
    "SGL-002": StaticTopography(elevation_m=5250.0, moraine_slope_deg=26.0, catchment_area_km2=18.5),
}


def load_lake_registries(registry_path: Path) -> Tuple[Dict[str, StaticTopography], Dict[str, str]]:
    """Load lake static topography and cluster mapping from registry CSV."""
    topos: Dict[str, StaticTopography] = dict(DEFAULT_TOPOS)
    cluster_map: Dict[str, str] = {}

    if not registry_path.exists():
        return topos, cluster_map

    with open(registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row["lake_id"].strip()
            elev = float(row.get("elevation_m", 5200.0))
            area = float(row.get("area_km2_2020", 1.35))
            cid = row.get("cluster_id", f"CLUSTER_{lid}").strip()
            cluster_map[lid] = cid
            if lid not in topos:
                topos[lid] = StaticTopography(
                    elevation_m=elev,
                    moraine_slope_deg=25.0,
                    catchment_area_km2=max(1.0, area * 8.0),
                )
    return topos, cluster_map


def load_event_onsets(event_registry_path: Path) -> Dict[str, str]:
    """Load lake_id -> onset_date mapping from event registry CSV."""
    events: Dict[str, str] = {}
    if not event_registry_path.exists():
        return events

    with open(event_registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row["lake_id"].strip()
            onset = row.get("onset_earliest", "2023-10-04T00:00:00Z").strip()
            events[lid] = onset.split("T")[0]
    return events


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentinel-GL Independent Inference, Diagnostics, and Claims Table Runner."
    )
    parser.add_argument(
        "--predictions",
        dest="predictions",
        type=Path,
        default=Path("data/prediction_ledger.csv"),
        help="Path to prediction ledger CSV (default: data/prediction_ledger.csv).",
    )
    parser.add_argument(
        "--calibration",
        dest="calibration",
        type=Path,
        default=Path("data/calibration_thresholds.json"),
        help="Path to calibration thresholds JSON (default: data/calibration_thresholds.json).",
    )
    parser.add_argument(
        "--manifest",
        "--split-manifest",
        dest="manifest",
        type=Path,
        default=Path("data/split_manifest.json"),
        help="Path to split manifest JSON (default: data/split_manifest.json).",
    )
    parser.add_argument(
        "--panels",
        dest="panels",
        type=Path,
        default=Path("data/feature_panels.npz"),
        help="Path to feature panels archive NPZ (default: data/feature_panels.npz).",
    )
    parser.add_argument(
        "--lake-registry",
        "--registry",
        dest="lake_registry",
        type=Path,
        default=Path("data/lake_registry.csv"),
        help="Path to lake registry CSV (default: data/lake_registry.csv).",
    )
    parser.add_argument(
        "--event-registry",
        dest="event_registry",
        type=Path,
        default=Path("data/event_registry.csv"),
        help="Path to event registry CSV (default: data/event_registry.csv).",
    )
    parser.add_argument(
        "--output-claims",
        dest="output_claims",
        type=Path,
        default=Path("data/claims_table.json"),
        help="Path to output claims table JSON (default: data/claims_table.json).",
    )
    parser.add_argument(
        "--output-diagnostics",
        dest="output_diagnostics",
        type=Path,
        default=Path("data/diagnostic_report.json"),
        help="Path to output diagnostic report JSON (default: data/diagnostic_report.json).",
    )
    parser.add_argument(
        "--output-cluster-metrics",
        dest="output_cluster_metrics",
        type=Path,
        default=Path("data/cluster_conditional_metrics.json"),
        help="Path to output cluster conditional metrics JSON (default: data/cluster_conditional_metrics.json).",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=0.05,
        help="Significance level alpha for multiplicity and confidence intervals (default: 0.05).",
    )
    parser.add_argument(
        "--stride-days",
        dest="stride_days",
        type=int,
        default=5,
        help="Stride between retrospective decision windows in days (default: 5).",
    )
    return parser.parse_args(argv)


def run_inference_diagnostics(
    predictions_path: Path,
    calibration_path: Path,
    manifest_path: Path,
    panels_path: Path,
    lake_registry_path: Path,
    event_registry_path: Path,
    output_claims_path: Path,
    output_diagnostics_path: Path,
    output_cluster_metrics_path: Path,
    alpha: float = 0.05,
    stride_days: int = 5,
) -> Dict[str, Any]:
    """Execute independent inference recomputation, failure diagnosis, and claims generation."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction ledger not found at {predictions_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration artifact not found at {calibration_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {manifest_path}")
    if not panels_path.exists():
        raise FileNotFoundError(f"Feature panels archive not found at {panels_path}")

    # 1. Load calibration threshold
    with open(calibration_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    threshold = float(cal_data.get("calibrated_threshold", 2.643366))

    # 2. Load registries
    topos, cluster_map = load_lake_registries(lake_registry_path)
    events_map = load_event_onsets(event_registry_path)

    # 3. Load prediction ledger records
    all_records: List[PredictionRecord] = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_records.append(
                PredictionRecord(
                    sample_id=row["sample_id"],
                    lake_id=row["lake_id"],
                    decision_date=row["decision_date"],
                    window_start=row["window_start"],
                    window_end=row["window_end"],
                    model_id=row["model_id"],
                    ablation_id=row["ablation_id"],
                    score=float(row["score"]),
                    eligible=row["eligible"].lower() == "true",
                )
            )

    # Extract primary model (T-MAE full) and baseline records (full configuration)
    tmae_full_records = [r for r in all_records if r.model_id == "tmae" and r.ablation_id == "full"]

    baseline_names = ["climatology", "area_trend", "weather_only", "rpca"]
    baseline_records_map: Dict[str, List[PredictionRecord]] = {}
    for b_id in baseline_names:
        baseline_records_map[b_id] = [
            r for r in all_records if r.model_id == b_id and r.ablation_id == "full"
        ]

    # 4. Reconstruct panels for diagnostics
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    npz_data = np.load(panels_path)

    all_raw_windows = (
        manifest_data.get("train_windows", [])
        + manifest_data.get("val_windows", [])
        + manifest_data.get("test_windows", [])
    )

    panels_map: Dict[str, MultiModalPanel] = {}
    for w in all_raw_windows:
        wid = w["window_id"]
        lid = w["lake_id"]
        d_date = str(w["decision_timestamp"]).split("T")[0]
        topo = topos.get(lid, DEFAULT_TOPOS.get(lid, StaticTopography(5200.0, 25.0, 10.0)))

        dates_key = f"{wid}_dates"
        vals_key = f"{wid}_values"
        mask_key = f"{wid}_mask" if f"{wid}_mask" in npz_data else f"{wid}_masks"

        if vals_key not in npz_data:
            continue

        dates = tuple(str(d) for d in npz_data[dates_key])
        vals = npz_data[vals_key]
        mask = npz_data[mask_key]

        panel = MultiModalPanel(
            lake_id=lid,
            window_id=wid,
            start_date=dates[0],
            end_date=dates[-1],
            dates=dates,
            values=vals,
            mask=mask,
            static_metadata=topo,
        )
        panels_map[f"{lid}_{d_date}"] = panel
        panels_map[wid] = panel
        panels_map[lid] = panel

    # 5. Generate Conservative Claims Table
    claims_table = generate_claims_table(
        model_records=tmae_full_records,
        baseline_records=baseline_records_map,
        event_registry=events_map,
        cluster_map=cluster_map,
        threshold=threshold,
        alpha=alpha,
        stride_days=stride_days,
    )

    output_claims_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_claims_path, "w", encoding="utf-8") as f:
        json.dump(claims_table.to_dict(), f, indent=2)

    # 6. Run Operational Failure Taxonomy Diagnostics
    diagnostic_report = diagnose_prediction_ledger(
        records=tmae_full_records,
        panels=panels_map,
        events=events_map,
        threshold=threshold,
    )

    output_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_report.to_dict(), f, indent=2)

    # 7. Spatial Cluster Conditional Aggregation
    lake_metric_map: Dict[str, Dict[str, Any]] = {}
    for lid in sorted(set(r.lake_id for r in tmae_full_records)):
        lake_recs = [r for r in tmae_full_records if r.lake_id == lid]
        eligible_scores = [r.score for r in lake_recs if r.eligible]
        lake_metric_map[lid] = {
            "mean_score": float(np.mean(eligible_scores)) if eligible_scores else None,
            "peak_score": float(np.max(eligible_scores)) if eligible_scores else None,
            "n_eligible_windows": len(eligible_scores),
            "n_total_windows": len(lake_recs),
        }

    cluster_metrics_report = cluster_conditional_metrics(lake_metric_map, cluster_map)

    output_cluster_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cluster_metrics_path, "w", encoding="utf-8") as f:
        json.dump(cluster_metrics_report, f, indent=2)

    return {
        "status": "COMPLETED",
        "claims_table_hash": claims_table.report_hash,
        "diagnostics_hash": diagnostic_report.report_hash,
        "n_evaluations": diagnostic_report.total_evaluations,
        "n_failures": diagnostic_report.total_failures,
        "claims_status": claims_table.status,
        "output_claims": str(output_claims_path),
        "output_diagnostics": str(output_diagnostics_path),
        "output_cluster_metrics": str(output_cluster_metrics_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    res = run_inference_diagnostics(
        predictions_path=args.predictions,
        calibration_path=args.calibration,
        manifest_path=args.manifest,
        panels_path=args.panels,
        lake_registry_path=args.lake_registry,
        event_registry_path=args.event_registry,
        output_claims_path=args.output_claims,
        output_diagnostics_path=args.output_diagnostics,
        output_cluster_metrics_path=args.output_cluster_metrics,
        alpha=args.alpha,
        stride_days=args.stride_days,
    )
    print("=" * 70)
    print("Sentinel-GL Independent Inference, Diagnostics & Claims Table Complete")
    print(f"Claims Table Report Hash: {res['claims_table_hash']}")
    print(f"Diagnostic Report Hash:   {res['diagnostics_hash']}")
    print(f"Total Evaluations:        {res['n_evaluations']}")
    print(f"Total Diagnosed Failures: {res['n_failures']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
