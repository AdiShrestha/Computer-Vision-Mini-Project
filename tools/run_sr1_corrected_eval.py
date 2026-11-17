#!/usr/bin/env python3
"""Score the frozen corrected SR-1 checkpoints on the frozen 141-scene corpus.

This is a new, immutable descriptive evaluation package. It does not retrain,
modify the parent protocol, or interpret the two-lake record as confirmation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
spec = importlib.util.spec_from_file_location(
    "sr1_analysis", ROOT / "source/runners/analyze_sr1_true_pipeline.py")
analysis = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(analysis)

ATTEMPT = ROOT / "data/derived/sr1_multiyear/attempt_006"
REEXEC = ATTEMPT / "reexecution_001"
OUT = ATTEMPT / "corrected_eval_001"
FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_EVAL_PROTOCOL_FREEZE_001.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    if OUT.exists():
        # The first pre-score launch stopped while inspecting zone metadata and
        # created only an empty replay directory. Preserve its recorded failure
        # while allowing this same frozen package to resume before any scores.
        allowed = {"replay", "SETUP_ATTEMPT_001.json"}
        if set(p.name for p in OUT.iterdir()) - allowed:
            raise SystemExit(f"immutable output already contains package outputs: {OUT}")
    if not FREEZE.exists():
        raise SystemExit("corrected EVAL protocol freeze is required before scoring")
    frozen = json.loads(FREEZE.read_text())
    if frozen.get("status") != "FROZEN_BEFORE_CORRECTED_EVAL_SCORING":
        raise SystemExit("protocol freeze does not authorize corrected EVAL scoring")
    manifest_path = ATTEMPT / "observations.sr1.json"
    run_path = REEXEC / "run_manifest.json"
    audit_path = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_REEXECUTION_AUDIT_001.json"
    for path, key in [(manifest_path, "observations_manifest_sha256"),
                      (run_path, "reexecution_manifest_sha256"),
                      (audit_path, "reexecution_audit_sha256")]:
        if sha(path) != frozen["inputs"][key]:
            raise ValueError(f"frozen input hash mismatch: {path}")
    records, _ = analysis.load_manifest(manifest_path)
    usable = [r for r in records if "reflectance" in r]
    if len(usable) != 141:
        raise ValueError(f"expected 141 observed scenes, found {len(usable)}")
    role_counts = {role: sum(r["role"] == role for r in usable)
                   for role in ("FIT", "VAL", "EVENT_UNCERTAIN", "EVAL")}
    if role_counts != {"FIT": 117, "VAL": 17, "EVENT_UNCERTAIN": 0, "EVAL": 7}:
        raise ValueError(f"frozen role membership differs: {role_counts}")
    fit_ids = sorted(r["observation_id"] for r in usable if r["role"] == "FIT")
    val_ids = sorted(r["observation_id"] for r in usable if r["role"] == "VAL")
    eval_ids = sorted(r["observation_id"] for r in usable if r["role"] == "EVAL")
    zone_data = {}
    for site in sorted({r["site_id"] for r in usable}):
        zp = ROOT / "data/raw/sr1_multiyear/attempt_006/zones" / f"{site}_fit_only_zones.npz"
        with np.load(zp, allow_pickle=False) as z:
            zone_fit_ids = sorted(r["observation_id"] for r in usable
                                  if r["site_id"] == site and r["role"] == "FIT")
            zone_data[site] = {"zones": z["zones"].copy(), "path": zp,
                               "sha256": sha(zp), "fit_observation_ids": zone_fit_ids}
    analysis.tokenize_all(usable, zone_data)
    scaler_path = REEXEC / "scalers/primary__channels_5_zones_fit_context_8.json"
    scaler = json.loads(scaler_path.read_text())
    if sorted(scaler["fit_observation_ids"]) != fit_ids:
        raise ValueError("corrected primary scaler was not fitted on exactly frozen FIT IDs")
    for row in usable:
        row["std_features"] = analysis.apply_robust_scaler(
            row["features"], row["observed"], scaler).astype(np.float32)
    by_site = {site: sorted([r for r in usable if r["site_id"] == site], key=lambda r: r["acquired_at"])
               for site in sorted({r["site_id"] for r in usable})}
    run = json.loads(run_path.read_text())
    jobs = [j for j in run["jobs"] if j.get("family") == "primary"]
    if len(jobs) != 5 or {j["seed"] for j in jobs} != set(range(5)):
        raise ValueError("corrective run does not contain five primary seed checkpoints")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay").mkdir()
    predictions, replay_files, seed_summaries = [], [], []
    for job in sorted(jobs, key=lambda j: j["seed"]):
        cp = ROOT / job["checkpoint"]
        sp = ROOT / job["scaler"]
        if sha(cp) != job["checkpoint_sha256"] or sha(sp) != job["scaler_sha256"]:
            raise ValueError("checkpoint/scaler hash differs from corrected 50-fit manifest")
        if sha(sp) != sha(scaler_path):
            raise ValueError("primary seed checkpoints do not share the frozen primary FIT scaler")
        payload = torch.load(cp, map_location=device, weights_only=False)
        torch.manual_seed(int(job["seed"]))
        model = analysis.SR1TransformerMAE(d_model=64, nhead=4, num_encoder_layers=3,
                                           num_decoder_layers=1, num_patches=100).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        scores, info, replay = analysis.score_one_seed(
            model, usable, by_site, set(fit_ids), int(job["seed"]), device)
        for row in scores:
            row["checkpoint_sha256"] = sha(cp)
        predictions.extend(scores)
        rows = replay["rows"]
        ordered = [r["observation_id"] for r in rows]
        rp = OUT / "replay" / f"seed_{job['seed']}_score_replay.npz"
        np.savez_compressed(rp, observation_ids=np.asarray(ordered, dtype="U128"),
            site_ids=np.asarray([r["site_id"] for r in rows], dtype="U64"),
            acquired_at=np.asarray([r["acquired_at"] for r in rows], dtype="U40"),
            roles=np.asarray([r["role"] for r in rows], dtype="U32"),
            target=np.stack([r["target"] for r in rows]), observed=np.stack([r["observed"] for r in rows]),
            zones=np.stack([r["zones"] for r in rows]), masked=np.stack([r["masked"] for r in rows]),
            prediction=np.stack([r["prediction"] for r in rows]),
            projected=np.stack([replay["projected"][oid] for oid in ordered]).astype(np.float64),
            fit_observation_ids=np.asarray(fit_ids, dtype="U128"))
        replay_files.append({"path": rp.relative_to(ROOT).as_posix(), "sha256": sha(rp), "rows": len(rows)})
        seed_summaries.append({"seed": int(job["seed"]), "checkpoint_sha256": sha(cp),
                               "selected_val_masked_mse": float(job["best_val_loss"]),
                               "score_info": info})
    fitted = analysis.fit_descriptive_baselines(usable, fit_ids, random_state=0)
    baseline_rows = analysis.score_descriptive_baselines(usable, fit_ids, fitted)
    bmap = {r["observation_id"]: r for r in baseline_rows}
    for row in predictions:
        row["baselines"] = bmap[row["observation_id"]]
    analysis.add_page_cusum(predictions, kappa=0.5, threshold=4.0)
    save_json(OUT / "predictions.json", predictions)
    import joblib
    model_path = OUT / "baseline_models.joblib"
    joblib.dump(fitted, model_path, compress=3)
    save_json(OUT / "baseline_suite.json", {
        "status": fitted["status"], "fit_observation_ids": fitted["fit_observation_ids"],
        "fit_token_count": fitted["fit_token_count"], "detector_fit_token_count": fitted["detector_fit_token_count"],
        "isolation_forest_status": fitted["isolation_forest_status"],
        "one_class_svm_status": fitted["one_class_svm_status"],
        "rpca_status": fitted["rpca_status"], "model_path": model_path.relative_to(ROOT).as_posix(),
        "model_sha256": sha(model_path), "thresholds": {
            "note": "The parent protocol defines fixed descriptive CUSUM and does not define VAL-calibrated operating thresholds. No threshold was tuned from EVAL.",
            "val_q90_descriptive_only": {}}})
    # Report descriptive VAL q90 cut-points without rebranding them as
    # operational alarm thresholds or changing the frozen parent detector.
    methods = ["score_a_zone0_1", "score_b_zone0_1", "score_c"]
    for method in methods:
        values = [r.get(method) for r in predictions if r["role"] == "VAL" and r.get(method) is not None]
        finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        # seed-level rows are correlated; this is a descriptive reference only.
        if finite.size:
            suite_path = OUT / "baseline_suite.json"
            suite = json.loads(suite_path.read_text())
            suite["thresholds"]["val_q90_descriptive_only"][method] = {
                "value": float(np.quantile(finite, 0.9)), "n_seed_scene_scores": int(finite.size),
                "status": "DESCRIPTIVE_NOT_AN_OPERATIONAL_THRESHOLD"}
            save_json(suite_path, suite)
    estimands = {
        "E1": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No independent event-bracket review, delivery-availability evidence, or operationally calibrated alarm declaration."},
        "E2": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No independently ascertained event-free control lake-years or measured eligible exposure; VAL exceedance counts are not exposure rates."},
        "E3": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No supported independent rolling-origin models for the requested reference years; these checkpoints were fit/selected using earlier years."},
        "E4": {"value": "See per-scene Zone 0+1 and Zone 2 masked reconstruction summaries in predictions.json", "status": "CONDITIONAL_DESCRIPTIVE", "reason": "Candidate FIT-derived spectral zones lack independent physical geometry validation; scenes/patches are temporally and spatially dependent."},
        "E5": {"value": None, "status": "NOT_ESTIMABLE", "reason": "Descriptive score summaries and common finite VAL quantiles do not establish a fair, independently calibrated warning operating point."}}
    report = {"schema_version": 1, "status": "CORRECTED_EVAL_SCORED_EXPLORATORY_RESEARCH_GATES_OPEN",
        "protocol_freeze_sha256": sha(FREEZE), "parent_protocol_sha256": frozen["inputs"]["parent_protocol_sha256"],
        "observations_manifest_sha256": sha(manifest_path), "reexecution_manifest_sha256": sha(run_path),
        "reexecution_audit_sha256": sha(audit_path), "primary_scaler_sha256": sha(scaler_path),
        "split": {"counts": role_counts, "fit_ids": fit_ids, "val_ids": val_ids, "eval_ids": eval_ids,
                  "event_uncertain_ids": [], "disjoint": True},
        "zones": {s: {"path": v["path"].relative_to(ROOT).as_posix(), "sha256": v["sha256"],
                      "fit_observation_ids": v["fit_observation_ids"]} for s,v in zone_data.items()},
        "seeds": seed_summaries, "device": device, "prediction_rows": len(predictions),
        "baseline_suite_path": (OUT / "baseline_suite.json").relative_to(ROOT).as_posix(),
        "predictions_path": (OUT / "predictions.json").relative_to(ROOT).as_posix(),
        "arithmetic_replays": replay_files, "estimands": estimands,
        "limitations": ["Scores are retrospective and exploratory; 2023 outcomes were previously inspected.",
          "VAL thresholds are descriptive quantiles over correlated seed-scene rows, not operational thresholds.",
          "EVAL has seven acquisitions and no independently reviewed event/control labels.",
          "The parent protocol's baseline definitions are retained (OC-SVM nu=0.05, RPCA tol=1e-7, CUSUM without resets).",
          "Harmonic and ridge baselines requested in the PI directive are not part of the frozen parent analysis and are not fabricated or silently added."]}
    save_json(OUT / "analysis_report.json", report)
    print(json.dumps({"status": report["status"], "counts": role_counts,
        "prediction_rows": len(predictions), "device": device, "output": OUT.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
