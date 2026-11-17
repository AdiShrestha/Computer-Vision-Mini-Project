#!/usr/bin/env python3
"""Corrected SR-1 analysis under protocol amendment PLOS-001.

Stages (run in order; each writes into a new immutable analysis directory):

  zones   independent Hi-MAG inventory zones and spectral-zone agreement
  train   10 primary seeds, capacity and factorial cells, extended-budget check
  score   cross-mask scoring, references, ablations, comparators, CUSUM,
          constructed positive control
  replay  independent re-computation of predictions and Score A from checkpoints

The training and scoring stages refuse to run unless the amendment and the
code files named in its freeze record have the recorded SHA-256 digests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.sr1_corpus import SITES, digest, load_manifest  # noqa: E402
from sentinel_gl.sr1_features import (  # noqa: E402
    apply_robust_scaler, candidate_water_area_baseline, compute_guarded_indices, fit_robust_scaler,
    temporal_features, tokenize_reflectance_patches,
)
from sentinel_gl.sr1_geometry import (  # noqa: E402
    HI_MAG_IDS, HI_MAG_RECORD, HI_MAG_ZIP_SHA256, inventory_zone_map, load_hi_mag_outline, mask_agreement,
    spectral_fit_zone_map,
)

ATTEMPT = "attempt_007"
ANALYSIS = "plos_001"
MANIFEST = ROOT / f"data/derived/sr1_multiyear/{ATTEMPT}/observations.sr1.json"
OUT = ROOT / f"data/derived/sr1_multiyear/{ATTEMPT}/{ANALYSIS}"
AMENDMENT = ROOT / "docs/protocol/SR1_AMENDMENT_PLOS_001.json"
FREEZE = ROOT / "docs/protocol/SR1_AMENDMENT_PLOS_001.freeze.json"
HI_MAG_2017 = ROOT / "data/raw/external/hi_mag_zenodo_4275164/extracted/Hi-MAG database/Hi_MAG_database_2017"
HI_MAG_ZIP = ROOT / "data/raw/external/hi_mag_zenodo_4275164/Hi-MAG_database.zip"
PRIMARY_SEEDS = tuple(range(10))
TORCH_THREADS = 2
WORKERS = 4


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_CORRECTED_REEXECUTION":
        raise SystemExit("amendment freeze record does not authorize execution")
    if digest(AMENDMENT) != freeze["amendment_sha256"]:
        raise SystemExit("amendment changed after its freeze record")
    for path, expected in freeze["code_sha256"].items():
        if digest(ROOT / path) != expected:
            raise SystemExit(f"code changed after freeze: {path}")
    return freeze


def load_corpus():
    records, manifest = load_manifest(MANIFEST, ROOT)
    usable = [r for r in records if "reflectance" in r]
    if len(usable) != len(records):
        raise SystemExit("manifest contains failed exports; the frozen corpus has none")
    return usable, manifest


def load_zones() -> dict:
    zones = {}
    for site in SITES:
        with np.load(OUT / "zones" / f"{site}_inventory_zones.npz", allow_pickle=False) as z:
            zones[site] = z["zones"].copy()
    return zones


def tokenize(records: list[dict], zones: dict) -> None:
    for site in SITES:
        rows = sorted([r for r in records if r["site_id"] == site], key=lambda r: r["acquired_at"])
        previous = None
        for row in rows:
            tok = tokenize_reflectance_patches(row["reflectance"], row["valid"], row["scl"], zones[site])
            row["features"], row["observed"] = tok["features"], tok["observed"]
            row["zones"] = tok["zones"].astype(np.int64)
            row["time_feats"] = temporal_features(row["dt"], previous)
            row["water_area"] = candidate_water_area_baseline(row["reflectance"], row["valid"], row["scl"], zones[site])
            previous = row["dt"]


def split_ids(records):
    fit = sorted(r["observation_id"] for r in records if r["role"] == "FIT")
    val = sorted(r["observation_id"] for r in records if r["role"] == "VAL")
    ev = sorted(r["observation_id"] for r in records if r["role"] == "EVAL")
    if {r["role"] for r in records} - {"FIT", "VAL", "EVAL"}:
        raise SystemExit("unexpected role in frozen corpus")
    return fit, val, ev


# ---------------------------------------------------------------- zones stage
def stage_zones() -> None:
    if (OUT / "zones").exists():
        raise SystemExit("zones already exist; analysis directories are immutable")
    if digest(HI_MAG_ZIP) != HI_MAG_ZIP_SHA256:
        raise SystemExit("Hi-MAG archive hash mismatch")
    records, manifest = load_corpus()
    report = {"source": HI_MAG_RECORD, "hi_mag_zip_sha256": HI_MAG_ZIP_SHA256, "layer": "Hi_MAG_database_2017",
              "manifest": rel(MANIFEST), "manifest_sha256": digest(MANIFEST), "sites": {}}
    (OUT / "zones").mkdir(parents=True)
    for site in SITES:
        outline, attrs = load_hi_mag_outline(HI_MAG_2017, HI_MAG_IDS[site])
        rows = [r for r in records if r["site_id"] == site]
        grid = rows[0]["grid"]
        if any(r["grid"]["transform"] != grid["transform"] for r in rows):
            raise SystemExit(f"{site} rows do not share one grid")
        zinfo = inventory_zone_map(outline, grid)
        fit_rows = [r for r in rows if r["role"] == "FIT"]
        spectral = spectral_fit_zone_map(fit_rows, compute_guarded_indices)
        path = OUT / "zones" / f"{site}_inventory_zones.npz"
        np.savez_compressed(path, zones=zinfo["zones"], lake=zinfo["lake"], buffer=zinfo["buffer"],
                            spectral_fit_core=spectral, origin=np.asarray("independent_inventory_hi_mag_2017"))
        report["sites"][site] = {
            "hi_mag_id": HI_MAG_IDS[site], "hi_mag_attributes": attrs, "grid": grid,
            "outline_area_km2": zinfo["outline_area_km2"],
            "outline_fraction_inside_window": zinfo["outline_fraction_inside_window"],
            "rasterized_lake_pixels": zinfo["rasterized_lake_pixels"],
            "rasterized_lake_area_km2": zinfo["rasterized_lake_area_km2"],
            "buffer_pixels": int(zinfo["buffer"].sum()),
            "spectral_fit_core_vs_inventory_lake": mask_agreement(spectral, zinfo["lake"]),
            "zones_path": rel(path), "zones_sha256": digest(path)}
    write_json(OUT / "zones" / "zones_report.json", report)
    print(json.dumps({s: {k: v for k, v in d.items() if k in ("outline_fraction_inside_window", "rasterized_lake_area_km2")}
                      for s, d in report["sites"].items()}, indent=2))


# ---------------------------------------------------------------- train stage
def _jobs():
    for seed in PRIMARY_SEEDS:
        yield {"family": "primary", "cell": "channels_5_zones_inventory_context_8", "channels": 5,
               "zones": True, "context": 8, "width": 64, "seed": seed, "max_epochs": 150}
    for width in (32, 128):
        for seed in range(5):
            yield {"family": "capacity", "cell": f"width_{width}", "channels": 5, "zones": True,
                   "context": 8, "width": width, "seed": seed, "max_epochs": 150}
    for channels in (3, 5):
        for zones in (False, True):
            for context in (1, 8):
                if (channels, zones, context) == (5, True, 8):
                    continue
                cell = f"channels_{channels}_zones_{'inventory' if zones else 'constant'}_context_{context}"
                for seed in range(5):
                    yield {"family": "factorial", "cell": cell, "channels": channels, "zones": zones,
                           "context": context, "width": 64, "seed": seed, "max_epochs": 150}


def _cell_rows(records, job):
    rows = []
    for r in records:
        if r["role"] not in ("FIT", "VAL"):
            continue
        row = {k: r[k] for k in ("observation_id", "role", "site_id", "acquired_at", "dt", "time_feats")}
        row["features"] = np.asarray(r["features"][:, :job["channels"]], dtype=np.float64)
        row["observed"] = np.asarray(r["observed"][:, :job["channels"]], dtype=bool)
        row["zones"] = r["zones"] if job["zones"] else np.full_like(r["zones"], 2)
        rows.append(row)
    return rows


def _train_worker(payload):
    import torch
    torch.set_num_threads(TORCH_THREADS)
    from sentinel_gl.sr1_model import SR1TransformerMAE, train_factorized_sr1_model
    job, rows, fit_ids, val_ids, checkpoint = payload
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    model = SR1TransformerMAE(in_channels=job["channels"], d_model=job["width"], nhead=4,
                              num_encoder_layers=3, num_decoder_layers=1, num_patches=100)
    started = time.time()
    try:
        history = train_factorized_sr1_model(model, rows, fit_observation_ids=fit_ids,
                                             validation_observation_ids=val_ids, max_epochs=job["max_epochs"],
                                             device="cpu", seed=job["seed"], max_context_scenes=job["context"],
                                             plateau_patience=20, mask_ratio=0.5)
        torch.save({"state_dict": model.state_dict(), "job": job, "history": history}, checkpoint)
        return {**job, "status": "COMPLETED", "elapsed_seconds": round(time.time() - started, 3),
                "checkpoint": rel(checkpoint), "checkpoint_sha256": digest(checkpoint),
                **{k: history[k] for k in ("best_val_loss", "best_epoch", "epochs_completed", "stopping_status",
                                           "optimizer_steps", "validation_block_count", "validation_blocks_scored",
                                           "validation_target_entries", "validation_target_entries_by_block",
                                           "train_block_count", "parameter_count", "val_loss_history",
                                           "train_loss_history")}}
    except Exception as exc:  # retained, never hidden
        return {**job, "status": "FAILED_RETAINED", "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": round(time.time() - started, 3)}


def stage_train() -> None:
    freeze = verify_freeze()
    if (OUT / "training").exists():
        raise SystemExit("training output exists; analysis directories are immutable")
    from sentinel_gl.sr1_scoring import validation_reference_mse
    records, _ = load_corpus()
    tokenize(records, load_zones())
    fit_ids, val_ids, eval_ids = split_ids(records)
    (OUT / "training" / "checkpoints").mkdir(parents=True)
    payloads, scalers, refs = [], {}, {}
    jobs = list(_jobs())
    for job in jobs:
        key = (job["channels"], job["zones"])
        rows = _cell_rows(records, job)
        if key not in scalers:
            scaler = fit_robust_scaler(rows, fit_ids)
            sp = OUT / "training" / "scalers" / f"channels_{job['channels']}.json"
            if not sp.exists():
                write_json(sp, scaler)
            scalers[key] = (scaler, sp)
        scaler, sp = scalers[key]
        for row in rows:
            row["std_features"] = apply_robust_scaler(row["features"], row["observed"], scaler).astype(np.float32)
        if job["channels"] not in refs:
            refs[job["channels"]] = validation_reference_mse(rows, fit_ids, val_ids)
        job["scaler"], job["scaler_sha256"] = rel(sp), digest(sp)
        slim = [{k: r[k] for k in ("observation_id", "role", "site_id", "acquired_at", "std_features",
                                     "observed", "zones", "time_feats")} for r in rows]
        label = f"{job['family']}__{job['cell']}__seed_{job['seed']}"
        payloads.append((job, slim, fit_ids, val_ids, OUT / "training" / "checkpoints" / f"{label}.pt"))
    started = datetime.now(timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        entries = []
        for entry in pool.map(_train_worker, payloads):
            entries.append(entry)
            print(json.dumps({k: entry.get(k) for k in ("family", "cell", "seed", "status", "best_val_loss",
                                                          "best_epoch", "stopping_status", "elapsed_seconds")}), flush=True)
    # Extended-budget check: primary fits that stopped on the 150-epoch cap are refit with 450 epochs.
    extended_payloads = []
    for entry, payload in zip(entries, payloads):
        if entry["family"] == "primary" and entry.get("stopping_status") == "MAX_EPOCH_BUDGET_REACHED":
            job = {**payload[0], "family": "extended_budget", "max_epochs": 450}
            label = f"extended_budget__{job['cell']}__seed_{job['seed']}"
            extended_payloads.append((job, payload[1], fit_ids, val_ids,
                                      OUT / "training" / "checkpoints" / f"{label}.pt"))
    if extended_payloads:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for entry in pool.map(_train_worker, extended_payloads):
                entries.append(entry)
                print(json.dumps({k: entry.get(k) for k in ("family", "seed", "status", "best_val_loss",
                                                              "best_epoch", "stopping_status")}), flush=True)
    write_json(OUT / "training" / "run_manifest.json", {
        "status": "COMPLETED" if all(e["status"] == "COMPLETED" for e in entries) else "COMPLETED_WITH_RETAINED_FAILURES",
        "amendment_sha256": freeze["amendment_sha256"], "freeze_sha256": digest(FREEZE),
        "manifest_sha256": digest(MANIFEST), "fit_ids": fit_ids, "val_ids": val_ids,
        "eval_ids_accessed": False, "device": "cpu", "torch_threads_per_worker": TORCH_THREADS,
        "workers": WORKERS, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "validation_reference_mse_by_channels": {str(k): v for k, v in refs.items()},
        "jobs_expected": len(jobs), "extended_budget_jobs": len(extended_payloads), "jobs": entries})


# ---------------------------------------------------------------- score stage
def _primary_models(run, device):
    import torch
    from sentinel_gl.sr1_model import SR1TransformerMAE
    models = {}
    for entry in run["jobs"]:
        if entry["family"] != "primary":
            continue
        if entry["status"] != "COMPLETED":
            raise SystemExit(f"primary seed {entry['seed']} failed; scoring would be incomplete")
        cp = ROOT / entry["checkpoint"]
        if digest(cp) != entry["checkpoint_sha256"]:
            raise SystemExit("checkpoint hash differs from run manifest")
        payload = torch.load(cp, map_location=device, weights_only=False)
        model = SR1TransformerMAE(d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=1, num_patches=100)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models[entry["seed"]] = (model, entry)
    if sorted(models) != list(PRIMARY_SEEDS):
        raise SystemExit("primary seeds incomplete")
    return models


def _projected_record(row, projected, observed):
    return {"observation_id": row["observation_id"], "doy": row["dt"].timetuple().tm_yday,
            "projected": projected, "patch_observed": observed}


def stage_score() -> None:
    freeze = verify_freeze()
    import torch
    torch.set_num_threads(8)
    from sentinel_gl.sr1_baselines_true import add_page_cusum, fit_descriptive_baselines, score_descriptive_baselines
    from sentinel_gl.sr1_positive_control import FRACTIONS, inject_drainage
    from sentinel_gl.sr1_scoring import (
        ABLATIONS, REFERENCE_PREDICTORS, add_score_c_per_site, entry_errors, fit_latent_pca, model_passes,
        patch_folds, reference_predictions, score_a, score_b_for_scene,
    )
    if (OUT / "scores").exists():
        raise SystemExit("score output exists; analysis directories are immutable")
    run = json.loads((OUT / "training" / "run_manifest.json").read_text())
    records, _ = load_corpus()
    zones = load_zones()
    tokenize(records, zones)
    fit_ids, val_ids, eval_ids = split_ids(records)
    scaler_path = OUT / "training" / "scalers" / "channels_5.json"
    scaler = json.loads(scaler_path.read_text())
    if scaler["fit_observation_ids"] != fit_ids:
        raise SystemExit("primary scaler was not fitted on the frozen FIT IDs")
    for r in records:
        r["std_features"] = apply_robust_scaler(r["features"], r["observed"], scaler).astype(np.float32)
    by_site = {s: sorted([r for r in records if r["site_id"] == s], key=lambda r: r["acquired_at"]) for s in SITES}
    fit_set = set(fit_ids)
    rows_by_id = {r["observation_id"]: r for r in records}
    folds = patch_folds()
    device = "cpu"
    models = _primary_models(run, device)
    (OUT / "scores" / "replay").mkdir(parents=True)

    # Seed-independent reference predictors on the identical hidden entries.
    recon = {}
    for site, chrono in by_site.items():
        for idx, row in enumerate(chrono):
            preds, info = reference_predictions(chrono, idx, fit_set, folds)
            target = np.asarray(row["std_features"], np.float64)
            tv = np.asarray(row["observed"], bool) & np.isfinite(target)
            recon[row["observation_id"]] = {
                "observation_id": row["observation_id"], "site_id": site, "role": row["role"],
                "acquired_at": row["acquired_at"], "reference_info": info,
                "references": {name: entry_errors(preds[name], target, tv, row["zones"]) for name in REFERENCE_PREDICTORS},
                "model_by_seed": {}}

    # Comparators fitted once on FIT.
    fitted = fit_descriptive_baselines(records, fit_ids, random_state=0)
    baseline_rows = {r["observation_id"]: r for r in score_descriptive_baselines(records, fit_ids, fitted)}

    predictions, seed_info = [], []
    pc_targets = [r for r in records if r["role"] in ("VAL", "EVAL")]
    pc_rows = []
    perturbed_cache = {}
    for row in pc_targets:
        for fraction in FRACTIONS:
            arrays, info = inject_drainage(row["reflectance"], row["valid"], row["scl"], zones[row["site_id"]], fraction)
            if arrays is None:
                perturbed_cache[(row["observation_id"], fraction)] = (None, info)
                continue
            tok = tokenize_reflectance_patches(arrays["reflectance"], arrays["valid"], arrays["scl"], zones[row["site_id"]])
            prow = dict(row)
            prow.update({"features": tok["features"], "observed": tok["observed"], "zones": tok["zones"].astype(np.int64),
                         "origin": arrays["origin"],
                         "water_area": candidate_water_area_baseline(arrays["reflectance"], arrays["valid"], arrays["scl"], zones[row["site_id"]])})
            prow["std_features"] = apply_robust_scaler(prow["features"], prow["observed"], scaler).astype(np.float32)
            perturbed_cache[(row["observation_id"], fraction)] = (prow, info)

    def pc_baselines(prow):
        site_rows = [prow if r["observation_id"] == prow["observation_id"] else r for r in by_site[prow["site_id"]]]
        out = score_descriptive_baselines(site_rows, fit_ids, fitted, score_ids={prow["observation_id"]})
        return out[0]

    pc_baseline_cache = {key: (pc_baselines(v[0]) if v[0] is not None else None) for key, v in perturbed_cache.items()}

    for seed, (model, entry) in sorted(models.items()):
        passes, latents, observed_patch = {}, {}, {}
        for site, chrono in by_site.items():
            for idx, row in enumerate(chrono):
                out = model_passes(model, chrono, idx, folds, device)
                passes[row["observation_id"]] = out
                latents[row["observation_id"]] = out["latent"]
                observed_patch[row["observation_id"]] = out["target_observed_patch"]
        pca, pca_info = fit_latent_pca(latents, observed_patch, fit_set)
        projected = {oid: pca.transform(v) for oid, v in latents.items()}
        fit_refs = {site: [_projected_record(r, projected[r["observation_id"]], observed_patch[r["observation_id"]])
                           for r in chrono if r["observation_id"] in fit_set] for site, chrono in by_site.items()}
        seed_rows = []
        for site, chrono in by_site.items():
            for idx, row in enumerate(chrono):
                oid = row["observation_id"]
                out = passes[oid]
                target = np.asarray(row["std_features"], np.float64)
                tv = out["target_channel_valid"]
                a = score_a(out["model"], target, tv, row["zones"])
                b, b_status, b_n = score_b_for_scene(_projected_record(row, projected[oid], observed_patch[oid]),
                                                     fit_refs[site], row["zones"])
                recon[oid]["model_by_seed"][str(seed)] = {name: entry_errors(out[name], target, tv, row["zones"]) for name in ABLATIONS}
                seed_rows.append({"observation_id": oid, "site_id": site, "acquired_at": row["acquired_at"],
                                  "role": row["role"], "seed": seed, **a,
                                  "score_b_zone0_1": b, "score_b_status": b_status, "score_b_eligible_patches": b_n,
                                  "observed_patches": int(out["target_observed_patch"].sum()),
                                  "observed_zone01_patches": int((out["target_observed_patch"] & np.isin(row["zones"], [0, 1])).sum()),
                                  "water_candidate_area_km2": row["water_area"]["candidate_area_km2"],
                                  "water_area_valid_pixels": row["water_area"]["valid_pixels"],
                                  "baselines": baseline_rows[oid], "checkpoint_sha256": entry["checkpoint_sha256"]})
        moments = add_score_c_per_site(seed_rows)
        predictions.extend(seed_rows)
        seed_info.append({"seed": seed, "checkpoint_sha256": entry["checkpoint_sha256"], "pca": pca_info,
                          "score_c_moments": moments})
        order = [r["observation_id"] for site in SITES for r in by_site[site]]
        np.savez_compressed(OUT / "scores" / "replay" / f"seed_{seed}.npz",
                            observation_ids=np.asarray(order, dtype="U128"),
                            target=np.stack([rows_by_id[o]["std_features"] for o in order]),
                            channel_valid=np.stack([passes[o]["target_channel_valid"] for o in order]),
                            zones=np.stack([rows_by_id[o]["zones"] for o in order]),
                            folds=folds,
                            **{f"pred_{name}": np.stack([passes[o][name] for o in order]) for name in ABLATIONS},
                            latent=np.stack([latents[o] for o in order]), projected=np.stack([projected[o] for o in order]))
        # Constructed positive control on held-out scenes with this seed's model, PCA and moments.
        moments_by_site = {s: m["moments"] for s, m in moments.items()}
        for row in pc_targets:
            oid, site = row["observation_id"], row["site_id"]
            chrono = by_site[site]
            idx = next(i for i, r in enumerate(chrono) if r["observation_id"] == oid)
            clean = next(r for r in seed_rows if r["observation_id"] == oid)
            for fraction in FRACTIONS:
                prow, info = perturbed_cache[(oid, fraction)]
                base = {"seed": seed, "observation_id": oid, "site_id": site, "role": row["role"],
                        "acquired_at": row["acquired_at"], "fraction": fraction, "injection": info}
                if prow is None:
                    pc_rows.append(base | {"status": info["status"]})
                    continue
                pchrono = list(chrono)
                pchrono[idx] = prow
                out = model_passes(model, pchrono, idx, folds, device)
                target = np.asarray(prow["std_features"], np.float64)
                a = score_a(out["model"], target, out["target_channel_valid"], prow["zones"])
                pproj = pca.transform(out["latent"])
                b, b_status, _ = score_b_for_scene(_projected_record(prow, pproj, out["target_observed_patch"]),
                                                   fit_refs[site], prow["zones"])
                m = moments_by_site.get(site)
                c = None
                if m and a["score_a_zone0_1"] and a["score_a_zone0_1"] > 0 and b is not None:
                    c = 0.5 * (math.log(a["score_a_zone0_1"]) - m["ln_a_median"]) / m["ln_a_scale"] + \
                        0.5 * (b - m["b_median"]) / m["b_scale"]
                pb = pc_baseline_cache[(oid, fraction)]
                pc_rows.append(base | {"status": "SCORED", "origin": prow["origin"],
                    "clean": {"score_a_zone0_1": clean["score_a_zone0_1"], "score_b_zone0_1": clean["score_b_zone0_1"],
                              "score_c": clean["score_c"], "water_candidate_area_km2": clean["water_candidate_area_km2"],
                              **{k: clean["baselines"][k] for k in ("seasonal_climatology_residual", "spatial_persistence_residual",
                                                                      "isolation_forest_score", "one_class_svm_score", "missingness_only_score")}},
                    "perturbed": {"score_a_zone0_1": a["score_a_zone0_1"], "score_b_zone0_1": b, "score_c": c,
                                  "water_candidate_area_km2": prow["water_area"]["candidate_area_km2"],
                                  **{k: pb[k] for k in ("seasonal_climatology_residual", "spatial_persistence_residual",
                                                         "isolation_forest_score", "one_class_svm_score", "missingness_only_score")}}})
        print(json.dumps({"seed": seed, "rows": len(seed_rows)}), flush=True)
    add_page_cusum(predictions, kappa=0.5, threshold=4.0)
    write_json(OUT / "scores" / "predictions.json", predictions)
    write_json(OUT / "scores" / "reconstruction.json", list(recon.values()))
    write_json(OUT / "scores" / "positive_control.json", pc_rows)
    write_json(OUT / "scores" / "score_report.json", {
        "amendment_sha256": freeze["amendment_sha256"], "freeze_sha256": digest(FREEZE),
        "manifest_sha256": digest(MANIFEST), "run_manifest_sha256": digest(OUT / "training" / "run_manifest.json"),
        "primary_scaler_sha256": digest(scaler_path), "fit_ids": fit_ids, "val_ids": val_ids, "eval_ids": eval_ids,
        "seeds": seed_info, "device": device, "comparators": {k: v for k, v in fitted.items()
                                                              if k not in ("isolation_forest", "one_class_svm")},
        "files": {name: digest(OUT / "scores" / name) for name in ("predictions.json", "reconstruction.json", "positive_control.json")},
        "replay_files": {p.name: digest(p) for p in sorted((OUT / "scores" / "replay").glob("*.npz"))}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["zones", "train", "score"])
    args = ap.parse_args()
    {"zones": stage_zones, "train": stage_train, "score": stage_score}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
