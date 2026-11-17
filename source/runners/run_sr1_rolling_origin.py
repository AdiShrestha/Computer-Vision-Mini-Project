#!/usr/bin/env python3
"""Run frozen chronological SR-1 origins; FIT/VAL are refit per target year."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source" / "runners"))
import analyze_sr1_true_pipeline as analysis  # noqa: E402
from sentinel_gl.sr1_features import apply_robust_scaler, fit_robust_scaler  # noqa: E402
from sentinel_gl.sr1_model import SR1TransformerMAE, train_factorized_sr1_model  # noqa: E402

PROTOCOL = ROOT / "project/sr1_rolling_origin_protocol_v3.json"
PARENT_PROTOCOL = ROOT / "project/sr1_protocol_true_v4.json"
PARENT_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/PROTOCOL_FREEZE_004.json"
ROLL_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/ROLLING_ORIGIN_PROTOCOL_FREEZE_003.json"
MANIFEST = ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json"
OUTPUT = ROOT / "data/derived/sr1_multiyear/attempt_006/rolling_origin_003"
UNCERTAIN_START = datetime.fromisoformat("2023-10-02T18:30:00+00:00")
UNCERTAIN_END = datetime.fromisoformat("2023-10-04T18:30:00+00:00")


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


def assign_origin(records: list[dict], target_year: int) -> tuple[list[dict], list[str], list[str], list[str]]:
    included = [dict(row) for row in records if row["dt"].year <= target_year]
    if target_year == 2019:
        # The only historical training year is split chronologically per site.
        fit_ids, val_ids = [], []
        for site in sorted({r["site_id"] for r in included if r["dt"].year == 2018}):
            prior = sorted([r for r in included if r["site_id"] == site and r["dt"].year == 2018],
                           key=lambda row: (row["dt"], row["observation_id"]))
            cut = math.floor(0.75 * len(prior))
            if cut < 1 or cut >= len(prior):
                raise ValueError(f"2018 chronological FIT/VAL split unsupported for {site}")
            fit_ids.extend(r["observation_id"] for r in prior[:cut])
            val_ids.extend(r["observation_id"] for r in prior[cut:])
        for row in included:
            if row["observation_id"] in fit_ids:
                row["role"] = "FIT"
            elif row["observation_id"] in val_ids:
                row["role"] = "VAL"
            elif row["dt"].year == target_year:
                row["role"] = "EVAL"
            else:
                raise ValueError("unexpected pre-target role in 2019 origin")
    else:
        fit_years = set(range(2018, target_year - 1))
        val_year = target_year - 1
        fit_ids, val_ids = [], []
        for row in included:
            if row["dt"].year in fit_years:
                row["role"] = "FIT"
                fit_ids.append(row["observation_id"])
            elif row["dt"].year == val_year:
                row["role"] = "VAL"
                val_ids.append(row["observation_id"])
            elif row["dt"].year == target_year:
                row["role"] = "EVAL"
            else:
                raise ValueError("origin contains an undeclared role year")
    if target_year == 2023:
        for row in included:
            if row["dt"].year == 2023 and UNCERTAIN_START <= row["dt"] < UNCERTAIN_END:
                row["role"] = "EVENT_UNCERTAIN"
    eval_ids = [r["observation_id"] for r in included if r["role"] == "EVAL"]
    if not fit_ids or not val_ids or not eval_ids:
        raise ValueError(f"origin {target_year} has an empty FIT/VAL/EVAL role")
    if set(fit_ids) & set(val_ids) or (set(fit_ids) | set(val_ids)) & set(eval_ids):
        raise ValueError(f"origin {target_year} role IDs overlap")
    if max(r["dt"] for r in included if r["role"] == "FIT") >= min(r["dt"] for r in included if r["role"] == "VAL"):
        raise ValueError(f"origin {target_year} FIT is not strictly earlier than VAL")
    if max(r["dt"] for r in included if r["role"] == "VAL") >= min(r["dt"] for r in included if r["role"] in {"EVAL", "EVENT_UNCERTAIN"}):
        raise ValueError(f"origin {target_year} VAL is not strictly earlier than target-year observations")
    return included, sorted(fit_ids), sorted(val_ids), sorted(eval_ids)


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"immutable output already exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text())
    freeze = json.loads(ROLL_FREEZE.read_text())
    parent_freeze = json.loads(PARENT_FREEZE.read_text())
    if freeze.get("sha256") != sha(PROTOCOL):
        raise ValueError("rolling-origin protocol differs from its pre-execution freeze")
    if parent_freeze.get("sha256") != sha(PARENT_PROTOCOL):
        raise ValueError("parent protocol differs from its freeze record")
    records, manifest = analysis.load_manifest(MANIFEST)
    if manifest.get("origin") != "observational":
        raise ValueError("rolling-origin analysis refuses non-observational manifest")
    records = [row for row in records if "observation_id" in row]
    manifest_sha = sha(MANIFEST)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "origins").mkdir()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    result_origins, all_eval_scores = [], []

    for target_year in protocol["target_years"]:
        origin_records, fit_ids, val_ids, eval_ids = assign_origin(records, target_year)
        origin_root = OUTPUT / "origins" / str(target_year)
        origin_root.mkdir()
        zone_info = analysis.fit_zones_from_fit(origin_records, origin_root)
        zones_by_site = {site: {"zones": item["zones"]} for site, item in zone_info.items()}
        analysis.tokenize_all(origin_records, zones_by_site)
        usable = [r for r in origin_records if "features" in r]
        usable_fit = sorted(r["observation_id"] for r in usable if r["role"] == "FIT")
        usable_val = sorted(r["observation_id"] for r in usable if r["role"] == "VAL")
        usable_eval = {r["observation_id"] for r in usable if r["role"] == "EVAL"}
        if not usable_fit or not usable_val or not usable_eval:
            raise ValueError(f"origin {target_year}: supported FIT/VAL/EVAL rows are incomplete")
        scaler = fit_robust_scaler(usable, usable_fit)
        scaler_path = origin_root / "fit_scaler.json"
        write(scaler_path, scaler)
        for row in usable:
            row["std_features"] = apply_robust_scaler(
                row["features"], row["observed"], scaler).astype(np.float32)
        by_site = {site: sorted([r for r in usable if r["site_id"] == site],
                                key=lambda row: row["acquired_at"])
                   for site in sorted({r["site_id"] for r in usable})}
        seed_rows, checkpoints = [], []
        (origin_root / "checkpoints").mkdir()
        (origin_root / "scores").mkdir()
        for seed in (0, 1, 2, 3, 4):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SR1TransformerMAE(d_model=64, nhead=4,
                                      num_encoder_layers=3, num_decoder_layers=1,
                                      num_patches=100)
            history = train_factorized_sr1_model(
                model, usable, fit_observation_ids=usable_fit,
                validation_observation_ids=usable_val,
                max_epochs=150, device=device, seed=seed,
                max_context_scenes=8, plateau_patience=20,
            )
            checkpoint = origin_root / "checkpoints" / f"sr1_tmae_seed_{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "target_year": target_year,
                        "parameter_count": model.count_parameters(),
                        "protocol_sha256": sha(PROTOCOL),
                        "parent_protocol_sha256": sha(PARENT_PROTOCOL),
                        "fit_observation_ids": usable_fit,
                        "validation_observation_ids": usable_val,
                        "scaler_sha256": sha(scaler_path),
                        "history": history}, checkpoint)
            predictions, score_info, _ = analysis.score_one_seed(
                model, usable, by_site, set(usable_fit), seed, device)
            target_scores = [row for row in predictions
                             if row["observation_id"] in usable_eval]
            score_path = origin_root / "scores" / f"eval_scores_seed_{seed}.json"
            write(score_path, target_scores)
            record = {
                "target_year": target_year, "seed": seed,
                "fit_ids": usable_fit, "validation_ids": usable_val,
                "eval_ids": sorted(usable_eval),
                "parameter_count": model.count_parameters(),
                "best_val_masked_mse": history["best_val_loss"],
                "best_epoch": history["best_epoch"],
                "epochs_completed": history["epochs_completed"],
                "stopping_status": history["stopping_status"],
                "optimizer_steps": history["optimizer_steps"],
                "score_c_status": score_info["score_c_status"],
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha(checkpoint),
                "scaler": scaler_path.relative_to(ROOT).as_posix(),
                "scaler_sha256": sha(scaler_path),
                "score_path": score_path.relative_to(ROOT).as_posix(),
                "score_sha256": sha(score_path),
                "score_rows": len(target_scores),
            }
            seed_rows.append(record)
            checkpoints.append(record)
            all_eval_scores.extend(target_scores)
            write(origin_root / "training_history.json", seed_rows)
            print(json.dumps({"target_year": target_year, "seed": seed,
                              "best_val_masked_mse": history["best_val_loss"],
                              "score_c_status": score_info["score_c_status"],
                              "eval_score_rows": len(target_scores)}), flush=True)
        result_origins.append({
            "target_year": target_year,
            "fit_ids": usable_fit, "validation_ids": usable_val,
            "eval_ids": sorted(usable_eval),
            "event_uncertain_ids": sorted(r["observation_id"] for r in usable
                                            if r["role"] == "EVENT_UNCERTAIN"),
            "zones": {site: {k: v for k, v in info.items() if k not in {"zones", "path"}}
                      | {"path": info["path"].relative_to(ROOT).as_posix()}
                      for site, info in zone_info.items()},
            "scaler_path": scaler_path.relative_to(ROOT).as_posix(),
            "scaler_sha256": sha(scaler_path),
            "seeds": seed_rows,
        })
        write(OUTPUT / "origins" / f"{target_year}_summary.json", result_origins[-1])

    # E3-like seasonal summaries are retained as diagnostic-only distributions.
    autumn = []
    for row in all_eval_scores:
        dt = analysis.parse_time(row["acquired_at"])
        if dt.month in (9, 10, 11) and row["role"] == "EVAL":
            autumn.append({"target_year": dt.year, "site_id": row["site_id"],
                           "observation_id": row["observation_id"],
                           "acquired_at": row["acquired_at"], "seed": row["seed"],
                           "score_a_zone0_1": row["score_a_zone0_1"],
                           "score_b_zone0_1": row["score_b_zone0_1"],
                           "score_c": row["score_c"],
                           "score_a_status": row["score_a_status"],
                           "score_b_status": row["score_b_status"]})
    write(OUTPUT / "autumn_eval_scores.json", autumn)
    result = {
        "schema_version": 1,
        "status": "LOCAL_ROLLING_ORIGIN_EXECUTION; EXPLORATORY; E3_NOT_ADMITTED",
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha(PROTOCOL),
        "execution_script": Path(__file__).relative_to(ROOT).as_posix(),
        "execution_script_sha256": sha(Path(__file__)),
        "protocol_freeze_record": ROLL_FREEZE.relative_to(ROOT).as_posix(),
        "parent_protocol_sha256": sha(PARENT_PROTOCOL),
        "manifest_sha256": manifest_sha,
        "device": device,
        "origin_count": len(result_origins),
        "model_runs": sum(len(origin["seeds"]) for origin in result_origins),
        "origins": result_origins,
        "autumn_score_path": (OUTPUT / "autumn_eval_scores.json").relative_to(ROOT).as_posix(),
        "autumn_score_rows": len(autumn),
        "E3": {"value": None, "status": "NOT_ADMITTED",
               "reason": "Control identity/event status and physical candidate-zone validation are unresolved; 2018 has no prior out-of-sample origin; 2023 was development-exposed; cross-origin Score-C comparability and independent reproduction remain unreviewed."},
        "interpretation_limits": [
            "All outcomes were previously inspected; all origins are exploratory.",
            "Controls are not independently verified as event-free.",
            "Candidate zones lack independent geomorphic validation.",
            "The 2018 reference year has no preceding source year and remains unavailable out of sample.",
            "No threshold calibration, detection delay, false-alarm rate, causal effect, or confirmatory inference is computed.",
            "Score-C is transformed using FIT-only moments within each origin; independent cross-origin comparability is not established."
        ]
    }
    write(OUTPUT / "rolling_origin_report.json", result)
    print(json.dumps({"status": result["status"], "models": result["model_runs"],
                      "autumn_score_rows": len(autumn), "E3": result["E3"],
                      "output": OUTPUT.relative_to(ROOT).as_posix()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
