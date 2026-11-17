#!/usr/bin/env python3
"""Run the frozen SR-1 2x2x2 FIT/VAL component ablation; never score EVAL."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source" / "runners"))
import analyze_sr1_true_pipeline as analysis  # noqa: E402
from sentinel_gl.sr1_features import apply_robust_scaler, fit_robust_scaler  # noqa: E402
from sentinel_gl.sr1_model import SR1TransformerMAE, train_factorized_sr1_model  # noqa: E402

PROTOCOL = ROOT / "project/sr1_factorial_ablation_protocol_v1.json"
PARENT_PROTOCOL = ROOT / "project/sr1_protocol_true_v4.json"
PARENT_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/PROTOCOL_FREEZE_004.json"
SWEEP_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/FACTORIAL_PROTOCOL_FREEZE_001.json"
ANALYSIS_REPORT = ROOT / "data/derived/sr1_multiyear/attempt_006/analysis_001/analysis_report.json"
MANIFEST = ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json"
PRIMARY_HISTORIES = ROOT / "data/derived/sr1_multiyear/attempt_006/analysis_001/training_histories.json"
OUTPUT = ROOT / "data/derived/sr1_multiyear/attempt_006/sensitivity_002"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"immutable output already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text())
    freeze = json.loads(SWEEP_FREEZE.read_text())
    parent_freeze = json.loads(PARENT_FREEZE.read_text())
    if freeze.get("sha256") != sha(PROTOCOL):
        raise ValueError("factorial protocol differs from its freeze record")
    if parent_freeze.get("sha256") != sha(PARENT_PROTOCOL):
        raise ValueError("parent protocol differs from its freeze record")
    primary = json.loads(ANALYSIS_REPORT.read_text())
    if primary.get("protocol_sha256") != sha(PARENT_PROTOCOL):
        raise ValueError("primary analysis protocol binding mismatch")
    manifest_rows, _ = analysis.load_manifest(MANIFEST)
    if sha(MANIFEST) != primary.get("manifest_sha256"):
        raise ValueError("observation manifest differs from the primary analysis")
    fit_ids = list(primary["split"]["FIT_ids"])
    val_ids = list(primary["split"]["VAL_ids"])
    eval_ids = set(primary["split"]["EVAL_ids"])
    expected = set(fit_ids) | set(val_ids)
    rows = [row for row in manifest_rows if row.get("observation_id") in expected
            and row.get("role") in {"FIT", "VAL"}]
    if {row["observation_id"] for row in rows} != expected or expected & eval_ids:
        raise ValueError("exact FIT/VAL roles are absent or overlap EVAL")
    if not fit_ids or not val_ids:
        raise ValueError("FIT/VAL split must be non-empty")

    zone_paths = {site: info["path"] for site, info in primary["zones"].items()}
    zones_by_site = {}
    for site, relative in zone_paths.items():
        with np.load(ROOT / relative, allow_pickle=False) as zone_data:
            zones_by_site[site] = {"zones": zone_data["zones"]}
    analysis.tokenize_all(rows, zones_by_site)
    usable = [row for row in rows if "features" in row]
    usable_fit_ids = [oid for oid in fit_ids if any(r.get("observation_id") == oid for r in usable)]
    usable_val_ids = [oid for oid in val_ids if any(r.get("observation_id") == oid for r in usable)]
    if not usable_fit_ids or not usable_val_ids:
        raise ValueError("no supported observations in one or both declared roles")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "checkpoints").mkdir()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_rows = []
    baseline_cell = (True, True, 8)
    primary_histories = json.loads(PRIMARY_HISTORIES.read_text())
    seeds = (0, 1, 2, 3, 4)

    for channel_full, zones_enabled, context in itertools.product((False, True), (False, True), (1, 8)):
        cell_id = f"channels_{'5' if channel_full else '3'}_zones_{'fit' if zones_enabled else 'constant2'}_context_{context}"
        if (channel_full, zones_enabled, context) == baseline_cell:
            for item in primary_histories:
                hist = item["history"]
                run_rows.append({
                    "cell_id": cell_id, "seed": hist["seed"],
                    "channels": 5, "zones_enabled": True, "context_scenes_max": 8,
                    "parameter_count": hist["parameter_count"],
                    "best_val_masked_mse": hist["best_val_loss"],
                    "best_epoch": hist["best_epoch"],
                    "stopping_status": hist["stopping_status"],
                    "checkpoint": hist["checkpoint_path"],
                    "checkpoint_sha256": hist["checkpoint_sha256"],
                    "origin": "RETAINED_PRIMARY_ANALYSIS_001_BASELINE_CELL",
                })
            continue

        channel_count = 5 if channel_full else 3
        cell_records = []
        for source in usable:
            row = dict(source)
            row["features"] = np.asarray(source["features"][:, :channel_count], dtype=np.float64)
            row["observed"] = np.asarray(source["observed"][:, :channel_count], dtype=bool)
            if not zones_enabled:
                row["zones"] = np.full_like(source["zones"], 2, dtype=np.int64)
            cell_records.append(row)
        scaler = fit_robust_scaler(cell_records, usable_fit_ids)
        scaler_path = OUTPUT / f"scaler_{cell_id}.json"
        write(scaler_path, scaler)
        for row in cell_records:
            row["std_features"] = apply_robust_scaler(
                row["features"], row["observed"], scaler).astype(np.float32)
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SR1TransformerMAE(in_channels=channel_count, d_model=64, nhead=4,
                                      num_encoder_layers=3, num_decoder_layers=1,
                                      num_patches=100)
            history = train_factorized_sr1_model(
                model, cell_records,
                fit_observation_ids=usable_fit_ids,
                validation_observation_ids=usable_val_ids,
                max_epochs=150, device=device, seed=seed,
                max_context_scenes=context, plateau_patience=20,
            )
            checkpoint = OUTPUT / "checkpoints" / f"{cell_id}_seed_{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "cell_id": cell_id, "in_channels": channel_count,
                        "zones_enabled": zones_enabled, "max_context_scenes": context,
                        "parameter_count": model.count_parameters(),
                        "protocol_sha256": sha(PROTOCOL),
                        "parent_protocol_sha256": sha(PARENT_PROTOCOL),
                        "fit_observation_ids": usable_fit_ids,
                        "validation_observation_ids": usable_val_ids,
                        "scaler_sha256": sha(scaler_path), "history": history}, checkpoint)
            run_rows.append({
                "cell_id": cell_id, "seed": seed, "channels": channel_count,
                "zones_enabled": zones_enabled, "context_scenes_max": context,
                "parameter_count": model.count_parameters(),
                "best_val_masked_mse": history["best_val_loss"],
                "best_epoch": history["best_epoch"],
                "epochs_completed": history["epochs_completed"],
                "stopping_status": history["stopping_status"],
                "optimizer_steps": history["optimizer_steps"],
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha(checkpoint),
                "scaler": scaler_path.relative_to(ROOT).as_posix(),
                "scaler_sha256": sha(scaler_path),
                "origin": "NEW_FACTORIAL_RUN",
            })
            write(OUTPUT / "run_history.json", run_rows)
            print(json.dumps({"cell_id": cell_id, "seed": seed,
                              "best_val_masked_mse": history["best_val_loss"],
                              "best_epoch": history["best_epoch"],
                              "stopping_status": history["stopping_status"]}), flush=True)

    expected_cells = 8
    cell_counts = {cell: sum(1 for row in run_rows if row["cell_id"] == cell)
                   for cell in sorted({row["cell_id"] for row in run_rows})}
    result = {
        "schema_version": 1,
        "status": "LOCAL_FIT_VAL_FACTORIAL_DIAGNOSTIC; EXPLORATORY; NOT_INDEPENDENTLY_REVIEWED",
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha(PROTOCOL),
        "protocol_freeze_record": SWEEP_FREEZE.relative_to(ROOT).as_posix(),
        "parent_protocol_sha256": sha(PARENT_PROTOCOL),
        "manifest_sha256": sha(MANIFEST),
        "primary_analysis_report_sha256": sha(ANALYSIS_REPORT),
        "fit_ids": usable_fit_ids, "validation_ids": usable_val_ids,
        "evaluation_access": "NO_EVAL_SCORING_OR_SUMMARY_PERFORMED",
        "device": device, "expected_factorial_cells": expected_cells,
        "cell_counts": cell_counts, "results": run_rows,
        "interpretation_limits": [
            "VAL selects checkpoints and is not an independent test set.",
            "Primary SR-1 outcomes were already inspected; this is post hoc exploratory work.",
            "These validation losses do not estimate event detection, generalization, causal component effects, or feature importance.",
            "The five-channel zone-conditioned context-eight cell is reused from the primary execution rather than retrained."
        ],
        "protocol_metadata": protocol,
    }
    write(OUTPUT / "factorial_report.json", result)
    write(OUTPUT / "run_history.json", run_rows)
    print(json.dumps({"status": result["status"], "runs": len(run_rows),
                      "cells": cell_counts,
                      "output": OUTPUT.relative_to(ROOT).as_posix()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
