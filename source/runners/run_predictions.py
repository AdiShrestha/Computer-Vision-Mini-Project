#!/usr/bin/env python3
"""Predictions, Baselines, Calibration, and Alert Episodes Runner for Sentinel-GL.

Executes closed-window prediction generation across all scheduled retrospective
decision windows, the 2^N sensor ablation lattice, and the four fair baselines:
1. ClimatologyBaseline (seasonal day-of-year mean and standard deviation)
2. LakeAreaTrendBaseline (optical water area expansion Delta A / A)
3. WeatherOnlyBaseline (standardized ERA5 atmospheric anomaly distance)
4. RobustPCABaseline (linear subspace reconstruction error)

Performs split-isolated empirical threshold calibration on control lakes,
evaluates operational alert episodes with sustained alarm detection (q=2),
hysteresis (r=2), and refractory limits (60d), calculates lake-year exposure,
evaluates bounded lead time to historical GLOF onset, and computes paired score
differences (Delta S) over identical eligible supports.
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
import torch

# Ensure source/ is discoverable under isolated Python runtimes (-s -B)
RUNNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sentinel_gl.features import (
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    WINDOW_DAYS,
    MultiModalPanel,
    StaticTopography,
)
from sentinel_gl.normalization import FittedNormalizer
from sentinel_gl.splits import SplitManifest, DecisionWindow
from sentinel_gl.model import TimeSeriesMAE
from sentinel_gl.replay import CheckpointBundle
from sentinel_gl.scoring import MaskedReconstructionScorer
from sentinel_gl.baselines import (
    ClimatologyBaseline,
    LakeAreaTrendBaseline,
    WeatherOnlyBaseline,
    RobustPCABaseline,
    WEATHER_CHANNEL_INDICES,
)
from sentinel_gl.predictions import (
    ABLATION_CONFIGS,
    PredictionRecord,
    apply_sensor_ablation,
    check_window_eligibility,
    compute_paired_score_differences,
    make_prediction_record,
)
from sentinel_gl.calibration import (
    calibrate_false_alert_threshold,
    calibrate_percentile_threshold,
)
from sentinel_gl.episodes import (
    AlertEpisode,
    AlertEpisodeEngine,
    calculate_lake_exposure_years,
    compute_alert_burden_estimand,
    compute_bounded_lead_time,
)

DEFAULT_TOPOS: Dict[str, StaticTopography] = {
    "SGL-001": StaticTopography(elevation_m=5200.0, moraine_slope_deg=28.5, catchment_area_km2=15.2),
    "SGL-002": StaticTopography(elevation_m=5250.0, moraine_slope_deg=26.0, catchment_area_km2=18.5),
}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 byte digest for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_lake_topography(registry_path: Path) -> Dict[str, StaticTopography]:
    """Load lake static topography parameters from registry CSV or fallbacks."""
    topos: Dict[str, StaticTopography] = dict(DEFAULT_TOPOS)
    if not registry_path.exists():
        return topos

    with open(registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row["lake_id"].strip()
            elev = float(row.get("elevation_m", 5200.0))
            area = float(row.get("area_km2_2020", 1.35))
            if lid not in topos:
                topos[lid] = StaticTopography(
                    elevation_m=elev,
                    moraine_slope_deg=25.0,
                    catchment_area_km2=max(1.0, area * 8.0),
                )
    return topos


def load_event_onsets(event_registry_path: Path) -> Dict[str, Dict[str, str]]:
    """Load historical events from event registry CSV."""
    events: Dict[str, Dict[str, str]] = {}
    if not event_registry_path.exists():
        return events

    with open(event_registry_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row["lake_id"].strip()
            eid = row.get("event_id", f"EV-{lid}").strip()
            onset = row.get("onset_earliest", "2023-10-04T00:00:00Z").strip()
            events[lid] = {
                "event_id": eid,
                "lake_id": lid,
                "onset_date": onset.split("T")[0],
                "onset_raw": onset,
            }
    return events


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentinel-GL Prediction Ledger, Baselines, and Alert Episode Runner."
    )
    parser.add_argument(
        "--manifest",
        "--split-manifest",
        dest="manifest",
        type=Path,
        default=Path("data/split_manifest.json"),
        help="Path to spatiotemporal split manifest JSON (default: data/split_manifest.json).",
    )
    parser.add_argument(
        "--panels",
        dest="panels",
        type=Path,
        default=Path("data/feature_panels.npz"),
        help="Path to feature panels archive NPZ (default: data/feature_panels.npz).",
    )
    parser.add_argument(
        "--normalizer",
        dest="normalizer",
        type=Path,
        default=Path("data/fitted_normalizer.json"),
        help="Path to fitted normalizer JSON (default: data/fitted_normalizer.json).",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        default=Path("data/checkpoints/tmae_best_checkpoint.pt"),
        help="Path to trained model checkpoint PT (default: data/checkpoints/tmae_best_checkpoint.pt).",
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
        "--events",
        dest="event_registry",
        type=Path,
        default=Path("data/event_registry.csv"),
        help="Path to event registry CSV (default: data/event_registry.csv).",
    )
    parser.add_argument(
        "--output-ledger",
        "--ledger",
        dest="output_ledger",
        type=Path,
        default=Path("data/prediction_ledger.csv"),
        help="Path to output prediction ledger CSV (default: data/prediction_ledger.csv).",
    )
    parser.add_argument(
        "--output-calibration",
        "--calibration",
        dest="output_calibration",
        type=Path,
        default=Path("data/calibration_thresholds.json"),
        help="Path to output calibration thresholds JSON (default: data/calibration_thresholds.json).",
    )
    parser.add_argument(
        "--output-episodes",
        "--episodes",
        dest="output_episodes",
        type=Path,
        default=Path("data/alert_episodes.json"),
        help="Path to output alert episodes JSON (default: data/alert_episodes.json).",
    )
    parser.add_argument(
        "--output-paired",
        "--paired",
        dest="output_paired",
        type=Path,
        default=Path("data/paired_score_differences.json"),
        help="Path to output paired score differences JSON (default: data/paired_score_differences.json).",
    )
    parser.add_argument(
        "--calibration-target",
        dest="calibration_target",
        type=float,
        default=0.05,
        help="Target empirical false alarm rate on calibration control windows (default: 0.05, i.e. 95th percentile).",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps', 'cuda') (default: 'cpu').",
    )
    return parser.parse_args(argv)


def run_predictions(
    manifest_path: Path,
    panels_path: Path,
    normalizer_path: Path,
    checkpoint_path: Path,
    lake_registry_path: Path,
    event_registry_path: Path,
    output_ledger_path: Path,
    output_calibration_path: Path,
    output_episodes_path: Path,
    output_paired_path: Path,
    calibration_target: float = 0.05,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute end-to-end model and baseline scoring, calibration, and alert episode accounting."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {manifest_path}")
    if not panels_path.exists():
        raise FileNotFoundError(f"Feature panels archive not found at {panels_path}")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Fitted normalizer not found at {normalizer_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 1. Load inputs and manifests
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    with open(normalizer_path, "r", encoding="utf-8") as f:
        normalizer_dict = json.load(f)
    normalizer = FittedNormalizer.from_dict(normalizer_dict)

    bundle = CheckpointBundle.load(checkpoint_path, device=device)
    model = TimeSeriesMAE(**bundle.model_config)
    model.load_state_dict(bundle.model_state_dict)
    model.to(device)
    model.eval()

    scorer = MaskedReconstructionScorer(model, partitions=2)

    topos = load_lake_topography(lake_registry_path)
    events_by_lake = load_event_onsets(event_registry_path)
    npz_data = np.load(panels_path)

    # 2. Reconstruct MultiModalPanel objects
    all_raw_windows = (
        manifest_data.get("train_windows", [])
        + manifest_data.get("val_windows", [])
        + manifest_data.get("test_windows", [])
    )

    panels_by_id: Dict[str, MultiModalPanel] = {}
    windows_by_id: Dict[str, Dict[str, Any]] = {}

    for w in all_raw_windows:
        wid = w["window_id"]
        lid = w["lake_id"]
        topo = topos.get(lid, DEFAULT_TOPOS.get(lid, StaticTopography(5200.0, 25.0, 10.0)))

        dates_key = f"{wid}_dates"
        vals_key = f"{wid}_values"
        mask_key = f"{wid}_mask" if f"{wid}_mask" in npz_data else f"{wid}_masks"

        if vals_key not in npz_data:
            continue

        dates = tuple(str(d) for d in npz_data[dates_key])
        vals = npz_data[vals_key]
        mask = npz_data[mask_key]

        obs_hashes = w.get("metadata", {}).get("observation_hashes", {})

        panel = MultiModalPanel(
            lake_id=lid,
            window_id=wid,
            start_date=dates[0],
            end_date=dates[-1],
            dates=dates,
            values=vals,
            mask=mask,
            static_metadata=topo,
            provenance_hashes=obs_hashes,
        )
        panels_by_id[wid] = panel
        windows_by_id[wid] = w

    # 3. Fit baselines on training panels
    raw_train_windows = manifest_data.get("train_windows", [])
    if raw_train_windows:
        train_wins = [w for w in raw_train_windows if w["window_id"] in panels_by_id and w.get("is_eligible", False)]
    else:
        # Development partition fallback: first eligible windows sorted by lake and timestamp
        eligible_wins = [
            w for w in all_raw_windows
            if w.get("is_eligible", False) and w["window_id"] in panels_by_id
        ]
        eligible_wins.sort(key=lambda w: (w["lake_id"], w["decision_timestamp"]))
        n_train = min(8, len(eligible_wins))
        train_wins = eligible_wins[:n_train]

    train_panels = [panels_by_id[w["window_id"]] for w in train_wins]
    train_lake_ids = sorted(list(set(p.lake_id for p in train_panels)))

    b_clim = ClimatologyBaseline.fit(train_panels, train_lake_ids)
    b_trend = LakeAreaTrendBaseline.fit(train_panels, train_lake_ids)
    b_weath = WeatherOnlyBaseline.fit(train_panels, train_lake_ids)
    b_pca = RobustPCABaseline.fit(train_panels, train_lake_ids, k_components=min(8, len(train_panels)))

    # 4. Evaluate T-MAE and baselines across all scheduled windows & sensor ablations
    all_ledger_records: List[Dict[str, Any]] = []
    tmae_records: List[PredictionRecord] = []
    baseline_records_map: Dict[str, List[PredictionRecord]] = {
        "climatology": [],
        "area_trend": [],
        "weather_only": [],
        "rpca": [],
    }

    # Sort all windows deterministically by (lake_id, decision_timestamp)
    sorted_window_entries = sorted(
        windows_by_id.values(),
        key=lambda w: (w["lake_id"], w["decision_timestamp"])
    )

    for w_entry in sorted_window_entries:
        wid = w_entry["window_id"]
        lid = w_entry["lake_id"]
        panel = panels_by_id[wid]

        raw_ts = w_entry["decision_timestamp"]
        decision_date = str(raw_ts).split("T")[0]
        window_start = str(panel.dates[0]).split("T")[0]
        window_end = str(panel.dates[-1]).split("T")[0]

        for abl_id in ABLATION_CONFIGS:
            abl_vals, abl_mask = apply_sensor_ablation(panel.values, panel.mask, abl_id)
            is_window_eligible = check_window_eligibility(abl_mask, abl_id, min_obs_per_modality=2)

            # --- A. T-MAE Scorer ---
            if is_window_eligible and np.any(abl_mask):
                z, valid = normalizer.transform(abl_vals, abl_mask)
                x_t = torch.from_numpy(z).unsqueeze(0).float().to(device)
                valid_t = torch.from_numpy(valid).unsqueeze(0).bool().to(device)
                tmae_score = float(scorer.score(x_t, valid_t)[0])
                tmae_status = "ELIGIBLE"
                tmae_eligible = True
            else:
                tmae_score = 0.0
                tmae_status = "NOT_ESTIMABLE"
                tmae_eligible = False

            rec_tmae = make_prediction_record(
                lake_id=lid,
                decision_date=decision_date,
                window_start=window_start,
                window_end=window_end,
                model_id="tmae",
                ablation_id=abl_id,
                score=tmae_score,
                eligible=tmae_eligible,
            )
            tmae_records.append(rec_tmae)
            all_ledger_records.append({
                "sample_id": rec_tmae.sample_id,
                "lake_id": lid,
                "window_id": wid,
                "decision_date": decision_date,
                "window_start": window_start,
                "window_end": window_end,
                "model_id": "tmae",
                "ablation_id": abl_id,
                "score": f"{tmae_score:.6f}",
                "status": tmae_status,
                "eligible": str(tmae_eligible).lower(),
            })

            # --- B. Climatology Baseline ---
            if is_window_eligible and np.any(abl_mask):
                clim_score = float(b_clim.predict_window(abl_vals, abl_mask, panel.dates))
                clim_status = "ELIGIBLE"
                clim_eligible = True
            else:
                clim_score = 0.0
                clim_status = "NOT_ESTIMABLE"
                clim_eligible = False

            rec_clim = make_prediction_record(
                lake_id=lid,
                decision_date=decision_date,
                window_start=window_start,
                window_end=window_end,
                model_id="climatology",
                ablation_id=abl_id,
                score=clim_score,
                eligible=clim_eligible,
            )
            baseline_records_map["climatology"].append(rec_clim)
            all_ledger_records.append({
                "sample_id": rec_clim.sample_id,
                "lake_id": lid,
                "window_id": wid,
                "decision_date": decision_date,
                "window_start": window_start,
                "window_end": window_end,
                "model_id": "climatology",
                "ablation_id": abl_id,
                "score": f"{clim_score:.6f}",
                "status": clim_status,
                "eligible": str(clim_eligible).lower(),
            })

            # --- C. Lake Area Trend Baseline ---
            # Active only if optical channel 0 is included
            has_optical = "optical" in ABLATION_CONFIGS[abl_id]
            if is_window_eligible and has_optical and np.any(abl_mask[:, 0]):
                trend_score = float(b_trend.predict_window(abl_vals, abl_mask, panel.dates))
                trend_status = "ELIGIBLE"
                trend_eligible = True
            else:
                trend_score = 0.0
                trend_status = "NOT_ESTIMABLE"
                trend_eligible = False

            rec_trend = make_prediction_record(
                lake_id=lid,
                decision_date=decision_date,
                window_start=window_start,
                window_end=window_end,
                model_id="area_trend",
                ablation_id=abl_id,
                score=trend_score,
                eligible=trend_eligible,
            )
            baseline_records_map["area_trend"].append(rec_trend)
            all_ledger_records.append({
                "sample_id": rec_trend.sample_id,
                "lake_id": lid,
                "window_id": wid,
                "decision_date": decision_date,
                "window_start": window_start,
                "window_end": window_end,
                "model_id": "area_trend",
                "ablation_id": abl_id,
                "score": f"{trend_score:.6f}",
                "status": trend_status,
                "eligible": str(trend_eligible).lower(),
            })

            # --- D. Weather-Only Baseline ---
            # Active only if ERA5 channels (8, 9, 10) are included
            has_era5 = "era5" in ABLATION_CONFIGS[abl_id]
            if is_window_eligible and has_era5 and np.any(abl_mask[:, WEATHER_CHANNEL_INDICES]):
                weath_score = float(b_weath.predict_window(abl_vals, abl_mask, panel.dates))
                weath_status = "ELIGIBLE"
                weath_eligible = True
            else:
                weath_score = 0.0
                weath_status = "NOT_ESTIMABLE"
                weath_eligible = False

            rec_weath = make_prediction_record(
                lake_id=lid,
                decision_date=decision_date,
                window_start=window_start,
                window_end=window_end,
                model_id="weather_only",
                ablation_id=abl_id,
                score=weath_score,
                eligible=weath_eligible,
            )
            baseline_records_map["weather_only"].append(rec_weath)
            all_ledger_records.append({
                "sample_id": rec_weath.sample_id,
                "lake_id": lid,
                "window_id": wid,
                "decision_date": decision_date,
                "window_start": window_start,
                "window_end": window_end,
                "model_id": "weather_only",
                "ablation_id": abl_id,
                "score": f"{weath_score:.6f}",
                "status": weath_status,
                "eligible": str(weath_eligible).lower(),
            })

            # --- E. Robust PCA Baseline ---
            if is_window_eligible and np.any(abl_mask):
                pca_score = float(b_pca.predict_window(abl_vals, abl_mask, panel.dates))
                pca_status = "ELIGIBLE"
                pca_eligible = True
            else:
                pca_score = 0.0
                pca_status = "NOT_ESTIMABLE"
                pca_eligible = False

            rec_pca = make_prediction_record(
                lake_id=lid,
                decision_date=decision_date,
                window_start=window_start,
                window_end=window_end,
                model_id="rpca",
                ablation_id=abl_id,
                score=pca_score,
                eligible=pca_eligible,
            )
            baseline_records_map["rpca"].append(rec_pca)
            all_ledger_records.append({
                "sample_id": rec_pca.sample_id,
                "lake_id": lid,
                "window_id": wid,
                "decision_date": decision_date,
                "window_start": window_start,
                "window_end": window_end,
                "model_id": "rpca",
                "ablation_id": abl_id,
                "score": f"{pca_score:.6f}",
                "status": pca_status,
                "eligible": str(pca_eligible).lower(),
            })

    # 5. Write data/prediction_ledger.csv
    output_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "lake_id", "window_id", "decision_date", "window_start",
        "window_end", "model_id", "ablation_id", "score", "status", "eligible"
    ]
    with open(output_ledger_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_ledger_records)

    # 6. Empirical Threshold Calibration on Control Lake SGL-002
    control_lake_id = "SGL-002"
    event_lake_id = "SGL-001"

    cal_scores = [
        r.score for r in tmae_records
        if r.lake_id == control_lake_id and r.ablation_id == "full" and r.eligible
    ]

    percentiles_to_evaluate = [0.90, 0.95, 0.98, 0.99]
    cal_thresholds_by_p: Dict[str, Any] = {}

    for p in percentiles_to_evaluate:
        p_cal = calibrate_percentile_threshold(
            scores=cal_scores,
            calibration_lake_ids=[control_lake_id],
            evaluation_lake_ids=[event_lake_id],
            percentile=p,
        )
        cal_thresholds_by_p[f"{p:.2f}"] = p_cal

    # Target threshold (default 95th percentile)
    target_p = 1.0 - calibration_target
    cal_false_alert = calibrate_false_alert_threshold(
        scores=cal_scores,
        calibration_ids=[f"{control_lake_id}-{i}" for i in range(len(cal_scores))],
        final_ids=[event_lake_id],
        target=calibration_target,
    )
    primary_threshold = cal_false_alert["threshold"]

    calibration_artifact = {
        "target_false_alert_fraction": calibration_target,
        "target_percentile": target_p,
        "calibrated_threshold": primary_threshold,
        "calibration_lake_ids": [control_lake_id],
        "evaluation_lake_ids": [event_lake_id],
        "n_calibration_windows": len(cal_scores),
        "false_alert_calibration": cal_false_alert,
        "thresholds_by_percentile": cal_thresholds_by_p,
        "state_hash": cal_false_alert["state_hash"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    output_calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration_artifact, f, indent=2)

    # 7. Alert Episode Engine, Exposure Accounting & Bounded Lead Time
    engine = AlertEpisodeEngine(
        threshold=primary_threshold,
        sustained_q=2,
        hysteresis_r=2,
        refractory_days=60,
        max_gap_days=45,
    )

    tmae_full_by_lake: Dict[str, List[PredictionRecord]] = {}
    for r in tmae_records:
        if r.ablation_id == "full":
            if r.lake_id not in tmae_full_by_lake:
                tmae_full_by_lake[r.lake_id] = []
            tmae_full_by_lake[r.lake_id].append(r)

    episodes_by_lake: Dict[str, List[Dict[str, Any]]] = {}
    exposures_lake_years: Dict[str, float] = {}

    # Stride between retrospective monitoring windows is 5 days
    stride_days = 5

    for lid, recs in tmae_full_by_lake.items():
        decisions = [(r.decision_date, r.score, r.eligible) for r in recs]
        episodes = engine.extract_episodes(lid, decisions)
        episodes_by_lake[lid] = [ep.to_dict() for ep in episodes]

        # Follow-up exposure over eligible windows
        exp_years = calculate_lake_exposure_years(
            [d[0] for d in decisions if d[2]],
            stride_days=stride_days,
        )
        exposures_lake_years[lid] = exp_years

    # Alert burden on negative-control exposure (SGL-002)
    control_episodes = [
        AlertEpisode(**ep_dict)
        for ep_dict in episodes_by_lake.get(control_lake_id, [])
    ]
    control_exposure_dict = {control_lake_id: exposures_lake_years.get(control_lake_id, 0.0)}
    alert_burden = compute_alert_burden_estimand(control_episodes, control_exposure_dict)

    # Bounded lead time to historical South Lhonak event
    lead_time_results: List[Dict[str, Any]] = []
    if event_lake_id in events_by_lake and event_lake_id in tmae_full_by_lake:
        ev_info = events_by_lake[event_lake_id]
        event_recs = tmae_full_by_lake[event_lake_id]
        event_decisions = [(r.decision_date, r.score, r.eligible) for r in event_recs]
        lead_res = compute_bounded_lead_time(
            event_id=ev_info["event_id"],
            lake_id=event_lake_id,
            event_onset_date=ev_info["onset_date"],
            decisions=event_decisions,
            threshold=primary_threshold,
            sustained_q=2,
            warning_horizon_days=180,
        )
        lead_time_results.append(lead_res.to_dict())

    alert_episodes_artifact = {
        "threshold": primary_threshold,
        "sustained_q": 2,
        "hysteresis_r": 2,
        "refractory_days": 60,
        "max_gap_days": 45,
        "stride_days": stride_days,
        "episodes_by_lake": episodes_by_lake,
        "exposures_lake_years": exposures_lake_years,
        "alert_burden": alert_burden,
        "lead_time_results": lead_time_results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    output_episodes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_episodes_path, "w", encoding="utf-8") as f:
        json.dump(alert_episodes_artifact, f, indent=2)

    # 8. Paired Score Differences (Delta S) vs Baselines
    paired_differences_by_baseline: Dict[str, List[Dict[str, Any]]] = {}
    all_paired_comparisons: List[Dict[str, Any]] = []

    for b_id, b_recs in baseline_records_map.items():
        paired_list = compute_paired_score_differences(tmae_records, b_recs)
        paired_differences_by_baseline[b_id] = paired_list
        all_paired_comparisons.extend(paired_list)

    paired_artifact = {
        "summary": {
            "model_id": "tmae",
            "baselines": list(baseline_records_map.keys()),
            "total_paired_windows": len(all_paired_comparisons),
        },
        "paired_differences": paired_differences_by_baseline,
        "comparisons": all_paired_comparisons,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    output_paired_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_paired_path, "w", encoding="utf-8") as f:
        json.dump(paired_artifact, f, indent=2)

    return {
        "status": "COMPLETED",
        "n_ledger_records": len(all_ledger_records),
        "primary_threshold": primary_threshold,
        "alert_burden": alert_burden,
        "lead_time_results": lead_time_results,
        "n_paired_comparisons": len(all_paired_comparisons),
        "ledger_path": str(output_ledger_path),
        "calibration_path": str(output_calibration_path),
        "episodes_path": str(output_episodes_path),
        "paired_path": str(output_paired_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    res = run_predictions(
        manifest_path=args.manifest,
        panels_path=args.panels,
        normalizer_path=args.normalizer,
        checkpoint_path=args.checkpoint,
        lake_registry_path=args.lake_registry,
        event_registry_path=args.event_registry,
        output_ledger_path=args.output_ledger,
        output_calibration_path=args.output_calibration,
        output_episodes_path=args.output_episodes,
        output_paired_path=args.output_paired,
        calibration_target=args.calibration_target,
        device=args.device,
    )
    print("=" * 70)
    print("Sentinel-GL Predictions, Baselines, and Alert Episodes Completed")
    print(f"Total Prediction Ledger Rows: {res['n_ledger_records']}")
    print(f"Primary Threshold (95th percentile): {res['primary_threshold']:.6f}")
    print(f"Alert Burden (Control): {res['alert_burden']}")
    print(f"Lead Time Results: {res['lead_time_results']}")
    print(f"Total Paired Baseline Comparisons: {res['n_paired_comparisons']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
