#!/usr/bin/env python3
"""Re-execute primary, capacity, and factorial FIT/VAL fits with corrected attention.

Only exact frozen FIT and VAL records are tokenized or passed to model code. The
original outputs are never read or modified. Every fit result or failure is
retained under one caller-supplied, previously nonexistent output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source" / "runners"))
import analyze_sr1_true_pipeline as analysis  # noqa: E402
from sentinel_gl.sr1_features import apply_robust_scaler, fit_robust_scaler  # noqa: E402
from sentinel_gl.sr1_model import SR1TransformerMAE, train_factorized_sr1_model  # noqa: E402

MANIFEST = ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json"
ANALYSIS_REPORT = ROOT / "data/derived/sr1_multiyear/attempt_006/analysis_001/analysis_report.json"
PROTOCOL = ROOT / "project/sr1_protocol_true_v4.json"
AMENDMENT = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/REEXECUTION_PROTOCOL_001.json"
FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/REEXECUTION_PROTOCOL_FREEZE_001.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def jobs():
    # Primary five-seed reference cell.
    yield {"family": "primary", "cell": "channels_5_zones_fit_context_8",
           "channels": 5, "zones": True, "context": 8, "width": 64, "seed": 0}
    for seed in range(1, 5):
        yield {"family": "primary", "cell": "channels_5_zones_fit_context_8",
               "channels": 5, "zones": True, "context": 8, "width": 64, "seed": seed}
    # Capacity endpoints; width 64 is the primary reference above.
    for width in (32, 128):
        for seed in range(5):
            yield {"family": "capacity", "cell": f"width_{width}",
                   "channels": 5, "zones": True, "context": 8, "width": width, "seed": seed}
    # Frozen 2x2x2 factorial, with the primary all-on cell reused.
    for channels in (3, 5):
        for zones in (False, True):
            for context in (1, 8):
                if (channels, zones, context) == (5, True, 8):
                    continue
                label = f"channels_{channels}_zones_{'fit' if zones else 'constant2'}_context_{context}"
                for seed in range(5):
                    yield {"family": "factorial", "cell": label,
                           "channels": channels, "zones": zones, "context": context,
                           "width": 64, "seed": seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="new repository-relative directory; never reused")
    args = ap.parse_args()
    output = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("output directory must be inside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")

    report = json.loads(ANALYSIS_REPORT.read_text())
    amendment_bytes = AMENDMENT.read_bytes()
    freeze = json.loads(FREEZE.read_text())
    if freeze.get("amendment_sha256") != hashlib.sha256(amendment_bytes).hexdigest():
        raise ValueError("re-execution amendment differs from its pre-run freeze")
    if freeze.get("trainer_sha256") != sha(ROOT / "source/sentinel_gl/sr1_model.py"):
        raise ValueError("trainer differs from its pre-run freeze")
    if freeze.get("runner_sha256") != sha(Path(__file__)):
        raise ValueError("runner differs from its pre-run freeze")
    manifest_rows, _ = analysis.load_manifest(MANIFEST)
    fit_ids = list(report["split"]["FIT_ids"])
    val_ids = list(report["split"]["VAL_ids"])
    if set(fit_ids) & set(val_ids):
        raise ValueError("frozen FIT and VAL IDs overlap")
    allowed = set(fit_ids) | set(val_ids)
    role_rows = [r for r in manifest_rows if r.get("observation_id") in allowed
                 and r.get("role") in {"FIT", "VAL"}]
    if {r["observation_id"] for r in role_rows} != allowed:
        raise ValueError("the exact frozen FIT/VAL records are not available")
    zones_by_site = {}
    for site, info in report["zones"].items():
        with np.load(ROOT / info["path"], allow_pickle=False) as stored:
            zones_by_site[site] = {"zones": stored["zones"]}
    analysis.tokenize_all(role_rows, zones_by_site)
    usable = [r for r in role_rows if "features" in r]
    fit_usable = [oid for oid in fit_ids if any(r["observation_id"] == oid for r in usable)]
    val_usable = [oid for oid in val_ids if any(r["observation_id"] == oid for r in usable)]
    if not fit_usable or not val_usable:
        raise ValueError("no supported FIT or VAL rows")

    output.mkdir(parents=True)
    entries = []
    write(output / "run_manifest.json", {
        "status": "RUNNING", "script_sha256": sha(Path(__file__)),
        "manifest_sha256": sha(MANIFEST), "analysis_report_sha256": sha(ANALYSIS_REPORT),
        "parent_protocol_sha256": sha(PROTOCOL), "trainer_sha256": sha(ROOT / "source/sentinel_gl/sr1_model.py"),
        "fit_ids": sorted(fit_usable), "validation_ids": sorted(val_usable),
        "eval_ids_accessed": False, "jobs_expected": 50, "jobs_completed": 0,
        "jobs_failed": 0, "jobs": entries,
    })
    for index, job in enumerate(jobs(), start=1):
        label = f"{job['family']}__{job['cell']}__seed_{job['seed']}"
        cell_rows = []
        channels = job["channels"]
        for source in usable:
            row = dict(source)
            row["features"] = np.asarray(source["features"][:, :channels], dtype=np.float64)
            row["observed"] = np.asarray(source["observed"][:, :channels], dtype=bool)
            if not job["zones"]:
                row["zones"] = np.full_like(source["zones"], 2, dtype=np.int64)
            cell_rows.append(row)
        scaler = fit_robust_scaler(cell_rows, fit_usable)
        scaler_path = output / "scalers" / f"{job['family']}__{job['cell']}.json"
        write(scaler_path, scaler)
        for row in cell_rows:
            row["std_features"] = apply_robust_scaler(
                row["features"], row["observed"], scaler).astype(np.float32)
        torch.manual_seed(job["seed"])
        np.random.seed(job["seed"])
        model = SR1TransformerMAE(in_channels=channels, d_model=job["width"], nhead=4,
                                  num_encoder_layers=3, num_decoder_layers=1, num_patches=100)
        started = time.time()
        try:
            history = train_factorized_sr1_model(
                model, cell_rows, fit_observation_ids=fit_usable,
                validation_observation_ids=val_usable, max_epochs=150, device="cpu",
                seed=job["seed"], max_context_scenes=job["context"], plateau_patience=20,
            )
            checkpoint = output / "checkpoints" / f"{label}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "job": job,
                        "fit_observation_ids": sorted(fit_usable),
                        "validation_observation_ids": sorted(val_usable),
                        "parent_protocol_sha256": sha(PROTOCOL),
                        "trainer_sha256": sha(ROOT / "source/sentinel_gl/sr1_model.py"),
                        "scaler_sha256": sha(scaler_path), "history": history}, checkpoint)
            entry = {**job, "status": "PASS_COMPLETE_VAL", "best_val_loss": history["best_val_loss"],
                     "best_epoch": history["best_epoch"], "epochs_completed": history["epochs_completed"],
                     "optimizer_steps": history["optimizer_steps"], "stopping_status": history["stopping_status"],
                     "validation_blocks": history["validation_block_count"],
                     "validation_blocks_scored": history["validation_blocks_scored"],
                     "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                     "checkpoint_sha256": sha(checkpoint), "scaler": scaler_path.relative_to(ROOT).as_posix(),
                     "scaler_sha256": sha(scaler_path), "elapsed_seconds": round(time.time() - started, 3)}
        except Exception as exc:
            entry = {**job, "status": "FAILED_RETAINED", "error_type": type(exc).__name__,
                     "error": str(exc), "elapsed_seconds": round(time.time() - started, 3)}
        entries.append(entry)
        write(output / "run_manifest.json", {
            "status": "RUNNING", "script_sha256": sha(Path(__file__)),
            "manifest_sha256": sha(MANIFEST), "analysis_report_sha256": sha(ANALYSIS_REPORT),
            "parent_protocol_sha256": sha(PROTOCOL), "trainer_sha256": sha(ROOT / "source/sentinel_gl/sr1_model.py"),
            "fit_ids": sorted(fit_usable), "validation_ids": sorted(val_usable),
            "eval_ids_accessed": False, "jobs_expected": 50,
            "jobs_completed": len(entries), "jobs_failed": sum(x["status"] != "PASS_COMPLETE_VAL" for x in entries),
            "jobs": entries,
        })
        print(json.dumps({"job": index, "label": label, "status": entry["status"],
                          "best_val_loss": entry.get("best_val_loss"), "elapsed_seconds": entry["elapsed_seconds"]}),
              flush=True)
    final = json.loads((output / "run_manifest.json").read_text())
    final["status"] = "PASS_COMPLETE_VAL" if all(x["status"] == "PASS_COMPLETE_VAL" for x in entries) else "COMPLETED_WITH_RETAINED_FAILURES"
    write(output / "run_manifest.json", final)
    return 0 if final["status"] == "PASS_COMPLETE_VAL" else 2

if __name__ == "__main__":
    raise SystemExit(main())
