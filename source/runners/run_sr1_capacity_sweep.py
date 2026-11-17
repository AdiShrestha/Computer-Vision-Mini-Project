#!/usr/bin/env python3
"""Run a frozen SR-1 width sweep on FIT/VAL only; never score EVAL."""
from __future__ import annotations

import hashlib
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

PROTOCOL = ROOT / "project/sr1_capacity_sweep_protocol_v1.json"
PARENT_PROTOCOL = ROOT / "project/sr1_protocol_true_v4.json"
PARENT_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/PROTOCOL_FREEZE_004.json"
SWEEP_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/CAPACITY_PROTOCOL_FREEZE_001.json"
ANALYSIS_REPORT = ROOT / "data/derived/sr1_multiyear/attempt_006/analysis_001/analysis_report.json"
MANIFEST = ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json"
OUTPUT = ROOT / "data/derived/sr1_multiyear/attempt_006/sensitivity_001"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"immutable output already exists: {OUTPUT}")
    frozen = json.loads(PROTOCOL.read_text())
    sweep_freeze = json.loads(SWEEP_FREEZE.read_text())
    if sweep_freeze.get("sha256") != sha(PROTOCOL):
        raise ValueError("capacity protocol differs from its pre-execution freeze record")
    parent_freeze = json.loads(PARENT_FREEZE.read_text())
    if parent_freeze.get("sha256") != sha(PARENT_PROTOCOL):
        raise ValueError("parent SR-1 protocol no longer matches its freeze record")
    report = json.loads(ANALYSIS_REPORT.read_text())
    if report.get("protocol_sha256") != sha(PARENT_PROTOCOL):
        raise ValueError("primary execution was not produced under the frozen parent protocol")
    manifest_rows, manifest = analysis.load_manifest(MANIFEST)
    if sha(MANIFEST) != report.get("manifest_sha256"):
        raise ValueError("source manifest differs from the primary execution")
    fit_ids = list(report["split"]["FIT_ids"])
    val_ids = list(report["split"]["VAL_ids"])
    if set(fit_ids) & set(val_ids):
        raise ValueError("FIT and VAL IDs overlap")
    if not fit_ids or not val_ids:
        raise ValueError("the frozen FIT/VAL populations are empty")

    # Restrict all subsequent computation to the preregistered FIT and VAL IDs.
    allowed = set(fit_ids) | set(val_ids)
    records = [row for row in manifest_rows
               if row.get("observation_id") in allowed and row.get("role") in {"FIT", "VAL"}]
    if {row["observation_id"] for row in records} != allowed:
        raise ValueError("the retained inputs do not contain the exact frozen FIT/VAL IDs")

    old_zone_paths = {site: info["path"] for site, info in report["zones"].items()}
    zones_by_site = {}
    for site, relative in old_zone_paths.items():
        path = ROOT / relative
        with np.load(path, allow_pickle=False) as stored:
            zones_by_site[site] = {"zones": stored["zones"]}
    analysis.tokenize_all(records, zones_by_site)
    usable = [row for row in records if "features" in row]
    usable_fit_ids = [oid for oid in fit_ids if any(row.get("observation_id") == oid for row in usable)]
    usable_val_ids = [oid for oid in val_ids if any(row.get("observation_id") == oid for row in usable)]
    if not usable_fit_ids or not usable_val_ids:
        raise ValueError("no supported FIT/VAL observations remain")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "checkpoints").mkdir()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for width in (32, 128):
        for seed in (0, 1, 2, 3, 4):
            scaler = fit_robust_scaler(usable, usable_fit_ids)
            scaler_path = OUTPUT / f"scaler_d{width}_seed{seed}.json"
            write(scaler_path, scaler)
            run_records = []
            for source_row in usable:
                row = dict(source_row)
                row["std_features"] = apply_robust_scaler(
                    row["features"], row["observed"], scaler).astype(np.float32)
                run_records.append(row)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SR1TransformerMAE(d_model=width, nhead=4,
                                      num_encoder_layers=3, num_decoder_layers=1,
                                      num_patches=100)
            history = train_factorized_sr1_model(
                model, run_records,
                fit_observation_ids=usable_fit_ids,
                validation_observation_ids=usable_val_ids,
                max_epochs=150, device=device, seed=seed,
                max_context_scenes=8, plateau_patience=20,
            )
            checkpoint = OUTPUT / "checkpoints" / f"sr1_tmae_d{width}_seed_{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "d_model": width, "parameter_count": model.count_parameters(),
                        "protocol_sha256": sha(PROTOCOL),
                        "parent_protocol_sha256": sha(PARENT_PROTOCOL),
                        "fit_observation_ids": usable_fit_ids,
                        "validation_observation_ids": usable_val_ids,
                        "scaler_sha256": sha(scaler_path), "history": history}, checkpoint)
            rows.append({
                "d_model": width, "seed": seed,
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
            })
            write(OUTPUT / "run_history.json", rows)
            print(json.dumps({"d_model": width, "seed": seed,
                              "best_val_masked_mse": history["best_val_loss"],
                              "best_epoch": history["best_epoch"],
                              "stopping_status": history["stopping_status"]}), flush=True)

    reference = json.loads((ROOT / "data/derived/sr1_multiyear/attempt_006/analysis_001/training_histories.json").read_text())
    width64 = [{"seed": item["history"]["seed"],
                "parameter_count": item["history"]["parameter_count"],
                "best_val_masked_mse": item["history"]["best_val_loss"],
                "best_epoch": item["history"]["best_epoch"],
                "checkpoint_sha256": item["history"]["checkpoint_sha256"],
                "origin": "RETAINED_PRIMARY_ANALYSIS_001_REFERENCE_CELL"}
               for item in reference]
    output = {
        "schema_version": 1,
        "status": "LOCAL_FIT_VAL_CAPACITY_DIAGNOSTIC; EXPLORATORY; NOT_INDEPENDENTLY_REVIEWED",
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha(PROTOCOL),
        "protocol_freeze_record": SWEEP_FREEZE.relative_to(ROOT).as_posix(),
        "parent_protocol_sha256": sha(PARENT_PROTOCOL),
        "manifest_sha256": sha(MANIFEST),
        "primary_analysis_report_sha256": sha(ANALYSIS_REPORT),
        "fit_ids": usable_fit_ids, "validation_ids": usable_val_ids,
        "evaluation_access": "NO_EVAL_SCORING_OR_SUMMARY_PERFORMED",
        "device": device,
        "results": rows,
        "width_64_reference": width64,
        "interpretation_limits": [
            "VAL is a checkpoint-selection set and not an independent test set.",
            "Primary SR-1 results were already inspected; this is post hoc exploratory sensitivity.",
            "No event-detection, generalization, capacity-adequacy, or confirmatory claim is supported.",
            "The factorial component-ablation grid remains incomplete."
        ],
        "protocol_metadata": frozen,
    }
    write(OUTPUT / "capacity_report.json", output)
    write(OUTPUT / "run_history.json", rows)
    print(json.dumps({"status": output["status"], "runs": len(rows),
                      "output": OUTPUT.relative_to(ROOT).as_posix()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
