#!/usr/bin/env python3
"""Recompute held-in VAL masked-MSE from retained SR-1 sensitivity checkpoints.

Uses FIT/VAL preprocessing and the declared model architecture, with a separate
inference-only evaluation loop. It does not retrain or access EVAL rows, and is
not independent verification of the architecture or feature implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source" / "runners"))
import analyze_sr1_true_pipeline as analysis  # noqa: E402
from sentinel_gl.sr1_features import apply_robust_scaler  # noqa: E402
from sentinel_gl.sr1_model import SR1TransformerMAE  # noqa: E402

BASE = Path("data/derived/sr1_multiyear/attempt_006")
MANIFEST = BASE / "observations.sr1.json"
ANALYSIS_REPORT = BASE / "analysis_001/analysis_report.json"
CAPACITY_REPORT = BASE / "sensitivity_001/capacity_report.json"
FACTORIAL_REPORT = BASE / "sensitivity_002/factorial_report.json"
OUTPUT = Path("project/remediation/2026-10-26/sr1_true_implementation_001/CHECKPOINT_VAL_REPLAY_003.json")
ABS_TOL = 1e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def roles_and_token_rows(primary: dict[str, Any], manifest_rows: list[dict[str, Any]]):
    split = primary["split"]
    fit_ids, val_ids, eval_ids = set(split["FIT_ids"]), set(split["VAL_ids"]), set(split["EVAL_ids"])
    allowed = fit_ids | val_ids
    if fit_ids & val_ids or allowed & eval_ids:
        raise ValueError("primary split membership overlaps")
    records = [row for row in manifest_rows
               if row.get("observation_id") in allowed and row.get("role") in {"FIT", "VAL"}]
    if {row["observation_id"] for row in records} != allowed:
        raise ValueError("manifest does not contain exact FIT/VAL split IDs")
    zones_by_site = {}
    for site, record in primary["zones"].items():
        with np.load(ROOT / record["path"], allow_pickle=False) as archive:
            zones_by_site[site] = {"zones": archive["zones"]}
    analysis.tokenize_all(records, zones_by_site)
    usable = {row["observation_id"]: row for row in records if "features" in row}
    if not (fit_ids & usable.keys()) or not (val_ids & usable.keys()):
        raise ValueError("no supported FIT or VAL rows")
    return fit_ids & usable.keys(), val_ids & usable.keys(), usable


def validation_loss(model: SR1TransformerMAE, rows: list[dict[str, Any]], context: int, device: str = "cpu") -> tuple[float | None, int, int, list[dict[str, Any]]]:
    by_site: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_site.setdefault(str(row.get("site_id", "")), []).append(row)
    blocks = []
    for site in sorted(by_site):
        ordered = sorted(by_site[site], key=lambda item: item["acquired_at"])
        for start in range(0, len(ordered), context):
            blocks.append(ordered[start:start + context])
    losses = []
    total_targets = 0
    omitted = []
    unsupported = []
    model.to(device).eval()
    with torch.no_grad():
        for block in blocks:
            x_np = np.stack([np.asarray(row["std_features"], dtype=np.float32) for row in block])
            valid_channels = np.stack([np.asarray(row["observed"], dtype=bool) for row in block])
            observed = valid_channels.any(axis=-1)
            zones = np.stack([np.asarray(row["zones"], dtype=np.int64) for row in block])
            times = np.stack([np.asarray(row["time_feats"], dtype=np.float32) for row in block])
            x = torch.as_tensor(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).unsqueeze(0)
            obs = torch.as_tensor(observed, dtype=torch.bool).unsqueeze(0)
            channel_valid = torch.as_tensor(valid_channels, dtype=torch.bool).unsqueeze(0)
            zone_tensor = torch.as_tensor(zones, dtype=torch.long).unsqueeze(0)
            time_tensor = torch.as_tensor(times, dtype=torch.float32).unsqueeze(0)
            masked = torch.zeros_like(obs)
            for t in range(obs.shape[1]):
                valid_patch = torch.nonzero(obs[0, t], as_tuple=False).flatten()
                if valid_patch.numel():
                    masked[0, t, valid_patch[::2]] = True
            target_mask = masked.unsqueeze(-1) & channel_valid
            count = int(target_mask.sum().item())
            if count == 0:
                unsupported.append({
                    "observation_ids": [r["observation_id"] for r in block],
                    "disposition": "NO_MASKED_VALID_CHANNEL_TARGETS",
                })
                continue
            prediction, _ = model(x, obs, masked, zone_tensor, time_tensor, modality_id=0)
            loss = (prediction[target_mask] - x[target_mask]).square().mean()
            value = float(loss.cpu())
            if not math.isfinite(value):
                omitted.append({
                    "observation_ids": [r["observation_id"] for r in block],
                    "finite_weights": all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()),
                    "input_finite_after_nan_to_num": bool(torch.isfinite(x).all()),
                    "time_features_finite": bool(torch.isfinite(time_tensor).all()),
                    "prediction_finite": bool(torch.isfinite(prediction).all()),
                    "masked_valid_channel_targets": count,
                    "disposition": "NONFINITE_BLOCK; ORIGINAL_TRAINER_OMITS_THIS_BLOCK",
                })
                continue
            losses.append(value)
            total_targets += count
    return (sum(losses) / len(losses) if losses else None), len(losses), total_targets, omitted, unsupported


def prepare_rows(usable: dict[str, dict[str, Any]], ids: set[str], scaler_path: Path, channel_count: int, zones_enabled: bool) -> list[dict[str, Any]]:
    scaler = load(scaler_path)
    result = []
    for observation_id in sorted(ids):
        source = usable[observation_id]
        row = dict(source)
        row["features"] = np.asarray(source["features"][:, :channel_count], dtype=np.float64)
        row["observed"] = np.asarray(source["observed"][:, :channel_count], dtype=bool)
        if not zones_enabled:
            row["zones"] = np.full_like(source["zones"], 2, dtype=np.int64)
        row["std_features"] = apply_robust_scaler(row["features"], row["observed"], scaler).astype(np.float32)
        result.append(row)
    return result


def load_checkpoint(path: Path, expected_hash: str, d_model: int, channel_count: int) -> SR1TransformerMAE:
    file_path = ROOT / path
    if sha(file_path) != expected_hash:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    artifact = torch.load(file_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("state_dict"), dict):
        raise ValueError(f"checkpoint payload has unexpected structure: {path}")
    model = SR1TransformerMAE(in_channels=channel_count, d_model=d_model, nhead=4,
                              num_encoder_layers=3, num_decoder_layers=1,
                              num_patches=100)
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model


def main() -> None:
    primary, capacity, factorial = load(ANALYSIS_REPORT), load(CAPACITY_REPORT), load(FACTORIAL_REPORT)
    manifest_rows, _ = analysis.load_manifest(ROOT / MANIFEST)
    if sha(ROOT / MANIFEST) != primary["manifest_sha256"]:
        raise ValueError("manifest hash does not match the primary report")
    if sha(ROOT / ANALYSIS_REPORT) != capacity["primary_analysis_report_sha256"]:
        raise ValueError("capacity report is bound to a different primary analysis report")
    if sha(ROOT / ANALYSIS_REPORT) != factorial["primary_analysis_report_sha256"]:
        raise ValueError("factorial report is bound to a different primary analysis report")
    fit_ids, val_ids, usable = roles_and_token_rows(primary, manifest_rows)
    primary_histories = load(BASE / "analysis_001/training_histories.json")
    primary_by_seed = {int(item["history"]["seed"]): item for item in primary_histories}

    jobs: dict[str, dict[str, Any]] = {}
    for row in capacity["results"]:
        jobs[f"capacity_d{row['d_model']}_seed{row['seed']}"] = {
            "row": row, "channels": 5, "zones": True, "context": 8,
            "scaler": Path(row["scaler"]), "checkpoint": Path(row["checkpoint"]),
            "checkpoint_hash": row["checkpoint_sha256"], "d_model": int(row["d_model"]),
        }
    for seed, item in primary_by_seed.items():
        hist = item["history"]
        jobs[f"primary_d64_seed{seed}"] = {
            "row": {"seed": seed, "best_val_masked_mse": hist["best_val_loss"], "checkpoint_sha256": hist["checkpoint_sha256"]},
            "channels": 5, "zones": True, "context": 8,
            "scaler": Path(primary["scaler_path"]), "checkpoint": Path(hist["checkpoint_path"]),
            "checkpoint_hash": hist["checkpoint_sha256"], "d_model": 64,
        }
    for row in factorial["results"]:
        cell = row["cell_id"]
        if row.get("origin") == "RETAINED_PRIMARY_ANALYSIS_001_BASELINE_CELL":
            continue
        jobs[f"factorial_{cell}_seed{row['seed']}"] = {
            "row": row, "channels": int(row["channels"]), "zones": bool(row["zones_enabled"]),
            "context": int(row["context_scenes_max"]), "scaler": Path(row["scaler"]),
            "checkpoint": Path(row["checkpoint"]), "checkpoint_hash": row["checkpoint_sha256"],
            "d_model": 64,
        }
    results = []
    for index, (job_id, job) in enumerate(sorted(jobs.items()), 1):
        print(f"starting {index}/{len(jobs)} {job_id}", flush=True)
        prepared = prepare_rows(usable, val_ids, job["scaler"], job["channels"], job["zones"])
        model = load_checkpoint(job["checkpoint"], job["checkpoint_hash"], job["d_model"], job["channels"])
        recomputed, finite_blocks, targets, omitted_blocks, unsupported_blocks = validation_loss(model, prepared, job["context"])
        recorded = float(job["row"]["best_val_masked_mse"])
        delta = recomputed - recorded if recomputed is not None else None
        if delta is None:
            status = "NO_FINITE_VAL_BLOCKS"
        elif abs(delta) > ABS_TOL:
            status = "MISMATCH_OVER_1E-6"
        elif omitted_blocks:
            status = "MATCH_ON_FINITE_BLOCKS_WITH_NONFINITE_OMISSIONS"
        else:
            status = "MATCH_ALL_VAL_BLOCKS_WITHIN_1E-6"
        results.append({
            "job_id": job_id, "seed": int(job["row"]["seed"]), "checkpoint": job["checkpoint"].as_posix(),
            "checkpoint_sha256": job["checkpoint_hash"], "channels": job["channels"],
            "zones_enabled": job["zones"], "context_scenes_max": job["context"],
            "recorded_best_val_masked_mse": recorded, "recomputed_finite_block_mean_val_masked_mse": recomputed,
            "difference": delta, "absolute_tolerance": ABS_TOL, "status": status,
            "finite_validation_blocks": finite_blocks, "nonfinite_validation_blocks": omitted_blocks,
            "no_valid_target_blocks": unsupported_blocks,
            "masked_valid_channel_targets_in_finite_blocks": targets,
        })
        print(f"{index}/{len(jobs)} {job_id} {status} Δ={delta:.3g}", flush=True)

    report = {
        "schema_version": 1,
        "status": "PASS_ALL_BLOCKS_MATCH" if all(row["status"] == "MATCH_ALL_VAL_BLOCKS_WITHIN_1E-6" for row in results) else "NONFINITE_OR_MISMATCH_RETAINED; REVIEW_REQUIRED",
        "replay_script": "tools/replay_sr1_sensitivity_checkpoints.py",
        "replay_script_sha256": sha(Path(__file__).resolve()),
        "scope": {
            "evaluation_rows_read": 0, "training_reproduced": False,
            "architecture_and_feature_implementation_independently_reimplemented": False,
            "claim": "Inference-only recomputation of checkpoint-selected VAL masked MSE using a separate evaluation loop over retained FIT-fitted preprocessing and the declared model implementation.",
            "not_claimed": ["independent training reproduction", "independent architecture review", "generalization", "scientific component effect"],
        },
        "inputs": {str(path): sha(ROOT / path) for path in (MANIFEST, ANALYSIS_REPORT, CAPACITY_REPORT, FACTORIAL_REPORT)},
        "fit_observations_used_for_scaler_replay": len(fit_ids),
        "validation_observations_scored": len(val_ids),
        "unique_checkpoints_replayed": len(results),
        "matched_all_blocks": sum(row["status"] == "MATCH_ALL_VAL_BLOCKS_WITHIN_1E-6" for row in results),
        "matched_after_original_trainer_omissions": sum(row["status"] == "MATCH_ON_FINITE_BLOCKS_WITH_NONFINITE_OMISSIONS" for row in results),
        "mismatched_or_unestimable": sum(row["status"] in {"MISMATCH_OVER_1E-6", "NO_FINITE_VAL_BLOCKS"} for row in results),
        "nonfinite_validation_blocks_total": sum(len(row["nonfinite_validation_blocks"]) for row in results),
        "no_valid_target_blocks_total": sum(len(row["no_valid_target_blocks"]) for row in results),
        "results": results,
    }
    out = ROOT / OUTPUT
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha(out), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
