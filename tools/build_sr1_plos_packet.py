#!/usr/bin/env python3
"""Build the result packet from which every manuscript number, table and figure is drawn.

Inputs are the hash-bound outputs of run_sr1_plos_analysis.py and the corpus
manifest; the output is docs/results/sr1_plos_packet.json. Nothing in this
file is typed in by hand: every value is computed here from retained rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.sr1_corpus import SITES, load_manifest, parse_time  # noqa: E402
from sentinel_gl.sr1_geometry import HI_MAG_IDS, inventory_zone_map, load_hi_mag_outline  # noqa: E402
from sentinel_gl.sr1_stats import paired_ratio_difference, sample_sd  # noqa: E402

ATT = ROOT / "data/derived/sr1_multiyear/attempt_007"
OUT = ATT / "plos_001"
PARENT_MANIFEST = ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json"
HI_MAG_2017 = ROOT / "data/raw/external/hi_mag_zenodo_4275164/extracted/Hi-MAG database/Hi_MAG_database_2017"
PACKET = ROOT / "docs/results/sr1_plos_packet.json"
ROLES = ("FIT", "VAL", "EVAL")
SCORE_FIELDS = [
    ("score_a_zone0_1", "Score A, zone 0+1 (MSE)"), ("score_b_zone0_1", "Score B (latent distance)"),
    ("score_c", "Score C (robust z)"), ("score_a_zone_contrast", "Score A contrast, zone 0+1 minus zone 2"),
    ("water_candidate_area_km2", "Candidate water area (km²)"),
    ("baselines.seasonal_climatology_residual", "Seasonal residual"),
    ("baselines.spatial_persistence_residual", "Temporal persistence residual"),
    ("baselines.isolation_forest_score", "Isolation Forest score"),
    ("baselines.one_class_svm_score", "One-class SVM score"),
    ("baselines.missingness_only_score", "Missingness-only score"),
    ("page_cusum.statistic", "CUSUM statistic on Score C"),
]
PC_METHODS = [("score_a_zone0_1", 1), ("score_b_zone0_1", 1), ("score_c", 1), ("water_candidate_area_km2", -1),
              ("seasonal_climatology_residual", 1), ("spatial_persistence_residual", 1),
              ("isolation_forest_score", 1), ("one_class_svm_score", 1), ("missingness_only_score", 1)]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(row, path):
    v = row
    for k in path.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    return v if v is None or (isinstance(v, (int, float)) and math.isfinite(v)) else None


def summary(values):
    x = np.asarray([v for v in values if v is not None], float)
    if not len(x):
        return {"n": 0, "median": None, "q1": None, "q3": None}
    return {"n": int(len(x)), "median": float(np.median(x)), "q1": float(np.quantile(x, 0.25)),
            "q3": float(np.quantile(x, 0.75))}


def autumn_fit(row):
    return row["role"] == "FIT" and row["acquired_at"][:4] in {"2018", "2019", "2020", "2021"} and row["acquired_at"][5:7] in {"10", "11"}


def corpus_section(records, manifest):
    out = {"scenes_total": len(records), "by_site_role": {}, "by_site_year": {}, "processing_baselines": {},
           "providers": {}, "relative_orbits": {}, "zero_observed_patch_scenes": {}, "clear_fraction_median": {},
           "observed_patches_quartiles": None, "months_2022": {}}
    for site in SITES:
        rows = [r for r in records if r["site_id"] == site]
        out["by_site_role"][site] = {role: sum(r["role"] == role for r in rows) for role in ROLES}
        out["by_site_year"][site] = dict(sorted(Counter(r["dt"].year for r in rows).items()))
        out["months_2022"][site] = sorted({r["dt"].strftime("%Y-%m-%d") for r in rows if r["dt"].year == 2022})
        out["relative_orbits"][site] = dict(Counter(re.search(r"_R(\d{3})_", r["product_uri"]).group(1) for r in rows))
        for role in ROLES:
            rr = [r for r in rows if r["role"] == role]
            out["clear_fraction_median"].setdefault(site, {})[role] = float(np.median(
                [(np.isin(r["scl"], [4, 5, 6]) & r["valid"].all(axis=0)).mean() for r in rr])) if rr else None
    out["processing_baselines"] = dict(sorted(Counter(r["processing_baseline"] for r in records).items()))
    out["providers"] = dict(Counter(r["provider"] for r in records))
    dates = defaultdict(set)
    for r in records:
        dates[r["site_id"]].add(r["dt"].date())
    out["shared_acquisition_dates"] = len(dates[SITES[0]] & dates[SITES[1]])
    out["catalogue_inventory"] = manifest.get("catalogue_inventory")
    out["selection_rules"] = manifest.get("selection")
    out["date_window"] = manifest.get("date_window")
    return out


def main() -> int:
    records, manifest = load_manifest(ATT / "observations.sr1.json", ROOT)
    zones_report = json.loads((OUT / "zones/zones_report.json").read_text())
    run = json.loads((OUT / "training/run_manifest.json").read_text())
    preds = json.loads((OUT / "scores/predictions.json").read_text())
    recon = json.loads((OUT / "scores/reconstruction.json").read_text())
    pc = json.loads((OUT / "scores/positive_control.json").read_text())
    score_report = json.loads((OUT / "scores/score_report.json").read_text())
    replay = json.loads((OUT / "scores/replay_report.json").read_text())
    freeze = json.loads((ROOT / "docs/protocol/SR1_AMENDMENT_PLOS_001.freeze.json").read_text())
    by_id = {r["observation_id"]: r for r in records}
    seeds = sorted({r["seed"] for r in preds})
    packet = {"schema_version": 1, "status": "RESULT_PACKET_EXPLORATORY_RETROSPECTIVE",
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "inputs": {}, "corpus": corpus_section(records, manifest)}
    for p in [ATT / "observations.sr1.json", OUT / "zones/zones_report.json", OUT / "training/run_manifest.json",
              OUT / "scores/predictions.json", OUT / "scores/reconstruction.json", OUT / "scores/positive_control.json",
              OUT / "scores/score_report.json", OUT / "scores/replay_report.json",
              ROOT / "docs/protocol/SR1_AMENDMENT_PLOS_001.json", ROOT / "docs/protocol/SR1_AMENDMENT_PLOS_001.freeze.json"]:
        packet["inputs"][p.relative_to(ROOT).as_posix()] = sha(p)

    # Observed-patch support per scene (from the seed-0 rows; support is seed independent).
    s0 = [r for r in preds if r["seed"] == seeds[0]]
    obs = np.asarray([r["observed_patches"] for r in s0])
    packet["corpus"]["observed_patches_quartiles"] = {"min": int(obs.min()), "q1": float(np.quantile(obs, .25)),
                                                       "median": float(np.median(obs)), "q3": float(np.quantile(obs, .75)),
                                                       "max": int(obs.max())}
    zero = defaultdict(Counter)
    for r in s0:
        if r["observed_patches"] == 0:
            zero[r["site_id"]][r["role"]] += 1
    packet["corpus"]["zero_observed_patch_scenes"] = {s: dict(zero[s]) for s in SITES}
    packet["corpus"]["zero_observed_patch_scenes_total"] = int(sum(sum(c.values()) for c in zero.values()))
    packet["corpus"]["scenes_without_observed_zone01_patch"] = int(sum(r["observed_zone01_patches"] == 0 for r in s0))

    # Geometry.
    geo = {"source": zones_report["source"], "layer": zones_report["layer"], "sites": {}}
    parent = json.loads(PARENT_MANIFEST.read_text())
    for site in SITES:
        z = zones_report["sites"][site]
        outline, _ = load_hi_mag_outline(HI_MAG_2017, HI_MAG_IDS[site])
        old_grid = next(r["grid"] for r in parent["observations"] if r.get("site_id") == site and "grid" in r)
        old = inventory_zone_map(outline, old_grid)
        with np.load(ROOT / z["zones_path"]) as arr:
            zones_px = arr["zones"]
        geo["sites"][site] = {
            "hi_mag_id": z["hi_mag_id"], "hi_mag_area_km2": z["outline_area_km2"],
            "hi_mag_elevation_m": z["hi_mag_attributes"].get("GL_Elev"), "hi_mag_type": z["hi_mag_attributes"].get("GL_Type"),
            "outline_fraction_inside_new_window": z["outline_fraction_inside_window"],
            "outline_fraction_inside_previous_window": old["outline_fraction_inside_window"],
            "window_shift_east_m": z["grid"]["transform"][2] - old_grid["transform"][2],
            "window_shift_north_m": z["grid"]["transform"][5] - old_grid["transform"][5],
            "rasterized_lake_area_km2": z["rasterized_lake_area_km2"], "buffer_pixels": z["buffer_pixels"],
            "zone_pixel_counts": {str(k): int((zones_px == k).sum()) for k in (0, 1, 2)},
            "spectral_vs_inventory": z["spectral_fit_core_vs_inventory_lake"]}
    packet["geometry"] = geo

    # Patch-zone counts from the scoring rows (zone labels per patch are static per site).
    packet["geometry"]["patch_zone_counts"] = {}
    for site in SITES:
        rp = np.load(OUT / "scores/replay" / f"seed_{seeds[0]}.npz")
        ids = rp["observation_ids"].tolist()
        i = next(k for k, oid in enumerate(ids) if oid.startswith(site))
        pz = rp["zones"][i]
        packet["geometry"]["patch_zone_counts"][site] = {str(k): int((pz == k).sum()) for k in (0, 1, 2)}

    # Training.
    jobs = run["jobs"]
    cells = defaultdict(list)
    for j in jobs:
        cells[(j["family"], j["cell"])].append(j)
    cell_rows = []
    for (family, cell), js in sorted(cells.items()):
        done = [j for j in js if j["status"] == "COMPLETED"]
        losses = [j["best_val_loss"] for j in done]
        cell_rows.append({"family": family, "cell": cell, "channels": js[0]["channels"], "zones": js[0]["zones"],
                          "context": js[0]["context"], "width": js[0]["width"], "max_epochs": js[0]["max_epochs"],
                          "seeds": len(js), "completed": len(done), "failed": len(js) - len(done),
                          "mean_val_mse": float(np.mean(losses)) if losses else None,
                          "sample_sd_val_mse": sample_sd(losses), "min_val_mse": min(losses) if losses else None,
                          "max_val_mse": max(losses) if losses else None,
                          "stopping": dict(Counter(j.get("stopping_status") for j in done)),
                          "best_epochs": [j["best_epoch"] for j in done],
                          "validation_blocks_per_fit": sorted({j["validation_block_count"] for j in done}),
                          "validation_blocks_scored_per_fit": sorted({j["validation_blocks_scored"] for j in done}),
                          "validation_target_entries_per_fit": sorted({j["validation_target_entries"] for j in done}),
                          "parameter_count": done[0]["parameter_count"] if done else None,
                          "validation_reference_mse": run["validation_reference_mse_by_channels"][str(js[0]["channels"])]})
    primary = sorted((j for j in jobs if j["family"] == "primary"), key=lambda j: j["seed"])
    extended = {j["seed"]: j for j in jobs if j["family"] == "extended_budget"}
    packet["training"] = {
        "cells": cell_rows, "fits_total": len(jobs), "fits_completed": sum(j["status"] == "COMPLETED" for j in jobs),
        "primary_by_seed": [{"seed": j["seed"], "best_val_mse": j["best_val_loss"], "best_epoch": j["best_epoch"],
                             "epochs_completed": j["epochs_completed"], "stopping_status": j["stopping_status"],
                             "extended_budget": ({"best_val_mse": extended[j["seed"]]["best_val_loss"],
                                                  "best_epoch": extended[j["seed"]]["best_epoch"],
                                                  "stopping_status": extended[j["seed"]]["stopping_status"]}
                                                 if j["seed"] in extended else None)} for j in primary],
        "cpu_seconds_total": float(sum(j["elapsed_seconds"] for j in jobs)),
        "fit_ids": len(run["fit_ids"]), "val_ids": len(run["val_ids"]),
        "validation_reference_mse_by_channels": run["validation_reference_mse_by_channels"],
        "workers": run["workers"], "torch_threads_per_worker": run["torch_threads_per_worker"],
        "fits_at_epoch_cap": sum(j.get("stopping_status") == "MAX_EPOCH_BUDGET_REACHED" for j in jobs if j["family"] != "extended_budget")}

    # Reconstruction comparison (pooled MSE, seed-averaged SSE per scene).
    names_ref = list(recon[0]["references"])
    names_abl = list(next(iter(recon[0]["model_by_seed"].values())))
    scene_rows = []
    for rec in recon:
        n = rec["references"][names_ref[0]]["n_all"]
        if n == 0:
            continue
        row = {"observation_id": rec["observation_id"], "site_id": rec["site_id"], "role": rec["role"],
               "dt": parse_time(rec["acquired_at"]), "n": n}
        for name in names_ref:
            row[f"sse_{name}"] = rec["references"][name]["sse_all"]
        for name in names_abl:
            per_seed = [rec["model_by_seed"][str(s)][name]["sse_all"] for s in seeds]
            row[f"sse_{name}"] = float(np.mean(per_seed))
            for s, v in zip(seeds, per_seed):
                row[f"sse_{name}__seed{s}"] = v
        scene_rows.append(row)
    recon_out = {"predictors": names_ref, "ablations": names_abl, "by_role": {}, "differences": {}}
    for role in ROLES:
        rr = [r for r in scene_rows if r["role"] == role]
        n = sum(r["n"] for r in rr)
        entry = {"scenes": len(rr), "entries": n, "pooled_mse": {}}
        for name in names_ref + names_abl:
            entry["pooled_mse"][name] = sum(r[f"sse_{name}"] for r in rr) / n if n else None
        for name in names_abl:
            per_seed = [sum(r[f"sse_{name}__seed{s}"] for r in rr) / n for s in seeds] if n else []
            entry["pooled_mse"][name + "__seed_min"] = min(per_seed) if per_seed else None
            entry["pooled_mse"][name + "__seed_max"] = max(per_seed) if per_seed else None
        for site in SITES:
            rs = [r for r in rr if r["site_id"] == site]
            ns = sum(r["n"] for r in rs)
            entry.setdefault("by_site", {})[site] = {"scenes": len(rs), "entries": ns, "pooled_mse": {
                name: (sum(r[f"sse_{name}"] for r in rs) / ns if ns else None) for name in names_ref + names_abl}}
        recon_out["by_role"][role] = entry
        diffs = {}
        for other in names_ref + names_abl[1:]:
            strata = {site: [r for r in rr if r["site_id"] == site] for site in SITES}
            diffs[other] = paired_ratio_difference(strata, a="model", b=other, block_days=30.0, n_boot=4000, seed=0)
        recon_out["differences"][role] = diffs
    val_d = recon_out["differences"]["VAL"]
    recon_out["rule_R1_model_uses_context"] = all(
        val_d[a]["status"] == "ESTIMATED" and val_d[a]["upper"] < 0 for a in names_abl[1:])
    recon_out["rule_R2_model_beats_seasonal_plus_offset"] = {
        role: (recon_out["differences"][role]["patch_seasonal_mean_plus_visible_offset"]["status"] == "ESTIMATED"
               and recon_out["differences"][role]["patch_seasonal_mean_plus_visible_offset"]["upper"] < 0)
        for role in ("VAL", "EVAL")}
    packet["reconstruction"] = recon_out

    # Scene-level (seed-median) score summaries.
    scene_vals = defaultdict(dict)
    for field, _ in SCORE_FIELDS:
        grouped = defaultdict(list)
        for r in preds:
            v = get(r, field)
            if v is not None:
                grouped[r["observation_id"]].append(float(v))
        for oid, vals in grouped.items():
            scene_vals[oid][field] = float(np.median(vals))
    groups = {"FIT_VAL": lambda r: r["role"] in ("FIT", "VAL"), "AUTUMN_FIT_2018_2021": autumn_fit,
              "EVAL": lambda r: r["role"] == "EVAL"}
    table = []
    for field, label in SCORE_FIELDS:
        row = {"field": field, "label": label}
        for site in SITES:
            for gname, sel in groups.items():
                ids = [r["observation_id"] for r in s0 if r["site_id"] == site and sel(r)]
                row[f"{site}|{gname}"] = summary([scene_vals[i].get(field) for i in ids])
        table.append(row)
    packet["scores"] = {"table": table, "groups": list(groups), "status_counts": {
        "score_a": dict(Counter(r["score_a_status"] for r in preds)),
        "score_b": dict(Counter(r["score_b_status"] for r in preds)),
        "score_c": dict(Counter(r["score_c_status"] for r in preds)),
        "seasonal_residual": dict(Counter(r["baselines"]["seasonal_status"] for r in preds if r["seed"] == seeds[0])),
        "persistence": dict(Counter(r["baselines"]["persistence_status"] for r in preds if r["seed"] == seeds[0]))},
        "seed_scene_rows": len(preds), "seeds": seeds,
        "eval_scenes": [{"observation_id": r["observation_id"], "site_id": r["site_id"], "date": r["acquired_at"][:10],
                         **{f: scene_vals[r["observation_id"]].get(f) for f, _ in SCORE_FIELDS}}
                        for r in sorted(s0, key=lambda x: (x["site_id"], x["acquired_at"])) if r["role"] == "EVAL"]}
    comps = []
    for site in SITES:
        for field in ("score_a_zone0_1", "score_b_zone0_1", "score_c", "water_candidate_area_km2"):
            fv = [scene_vals[r["observation_id"]].get(field) for r in s0 if r["site_id"] == site and autumn_fit(r)]
            ev = [scene_vals[r["observation_id"]].get(field) for r in s0 if r["site_id"] == site and r["role"] == "EVAL"]
            fv, ev = [v for v in fv if v is not None], [v for v in ev if v is not None]
            comps.append({"site": site, "field": field, "fit_autumn_median": float(np.median(fv)) if fv else None,
                          "eval_median": float(np.median(ev)) if ev else None, "n_fit": len(fv), "n_eval": len(ev)})
    packet["scores"]["autumn_comparisons"] = comps
    packet["scores"]["score_c_moments"] = {str(s["seed"]): s["score_c_moments"] for s in score_report["seeds"]}
    packet["scores"]["pca_explained_variance_ratio_sum"] = {
        str(s["seed"]): float(sum(s["pca"]["explained_variance_ratio"])) for s in score_report["seeds"]}

    # CUSUM alarm history.
    alarms = {}
    majority = len(seeds) // 2 + 1
    for site in SITES:
        for role in ROLES:
            by_scene = defaultdict(int)
            scenes = sorted({(r["acquired_at"], r["observation_id"]) for r in preds if r["site_id"] == site and r["role"] == role})
            for r in preds:
                if r["site_id"] == site and r["role"] == role and r["page_cusum"]["alarm_declared"]:
                    by_scene[r["observation_id"]] += 1
            maj = sorted(a for a, oid in scenes if by_scene.get(oid, 0) >= majority)
            any_ = sorted(a for a, oid in scenes if by_scene.get(oid, 0) >= 1)
            alarms[f"{site}|{role}"] = {"scenes": len(scenes), "scenes_any_seed": len(any_), "scenes_majority": len(maj),
                                        "first_any_seed": any_[0][:10] if any_ else None,
                                        "first_majority": maj[0][:10] if maj else None,
                                        "seed_scene_alarms": int(sum(by_scene.values()))}
    packet["cusum"] = {"kappa": 0.5, "h": 4.0, "majority_seeds": majority, "alarms": alarms}

    # Positive control.
    fit_thresholds = {}
    for site in SITES:
        for field, direction in PC_METHODS:
            key = field if field in ("score_a_zone0_1", "score_b_zone0_1", "score_c", "water_candidate_area_km2") else f"baselines.{field}"
            vals = [scene_vals[r["observation_id"]].get(key) for r in s0 if r["site_id"] == site and r["role"] == "FIT"]
            vals = [v for v in vals if v is not None]
            q = 0.95 if direction > 0 else 0.05
            fit_thresholds[f"{site}|{field}"] = float(np.quantile(vals, q)) if vals else None
    pc_scene = defaultdict(lambda: defaultdict(lambda: {"clean": [], "perturbed": []}))
    status_counts = Counter()
    for r in pc:
        status_counts[(r["fraction"], r["status"])] += 1
        if r["status"] != "SCORED":
            continue
        for field, _ in PC_METHODS:
            pc_scene[(r["observation_id"], r["site_id"], r["fraction"])][field]["clean"].append(r["clean"][field])
            pc_scene[(r["observation_id"], r["site_id"], r["fraction"])][field]["perturbed"].append(r["perturbed"][field])
    pc_out = {"fractions": sorted({r["fraction"] for r in pc}), "fit_thresholds": fit_thresholds,
              "status_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(status_counts.items())}, "methods": {}}
    for field, direction in PC_METHODS:
        per_fraction = {}
        for fraction in pc_out["fractions"]:
            clean_hits = pert_hits = n = 0
            changes = []
            for (oid, site, frac), vals in pc_scene.items():
                if frac != fraction:
                    continue
                c = [v for v in vals[field]["clean"] if v is not None]
                p = [v for v in vals[field]["perturbed"] if v is not None]
                thr = fit_thresholds[f"{site}|{field}"]
                if not c or not p or thr is None:
                    continue
                cm, pm = float(np.median(c)), float(np.median(p))
                n += 1
                clean_hits += (cm > thr) if direction > 0 else (cm < thr)
                pert_hits += (pm > thr) if direction > 0 else (pm < thr)
                changes.append(pm - cm)
            per_fraction[str(fraction)] = {"scenes": n, "clean_detection_rate": clean_hits / n if n else None,
                                           "perturbed_detection_rate": pert_hits / n if n else None,
                                           "median_paired_change": float(np.median(changes)) if changes else None,
                                           "fraction_changed_in_expected_direction": (
                                               float(np.mean([(d > 0) if direction > 0 else (d < 0) for d in changes]))
                                               if changes else None),
                                           "registers": bool(n and pert_hits > clean_hits)}
        pc_out["methods"][field] = {"direction": direction, "by_fraction": per_fraction}
    packet["positive_control"] = pc_out

    packet["estimands"] = {
        "E1": {"name": "Event-onset detection timing", "status": "NOT_ESTIMABLE",
               "reason": "No independent review of the event bracket, no delivery-availability record and no calibrated alarm operating point."},
        "E2": {"name": "False-alert burden per event-free lake-year", "status": "NOT_ESTIMABLE",
               "reason": "No independently verified event-free control lake-years or measured exposure."},
        "E3": {"name": "Out-of-sample seasonal percentile of post-event scores", "status": "NOT_ESTIMABLE",
               "reason": "No rolling-origin models: every checkpoint was fitted on 2018-2021 and early-2023 scenes."},
        "E4": {"name": "Within-scene zone 0+1 versus zone 2 contrast", "status": "DESCRIPTIVE_CONDITIONAL",
               "reason": "Zones follow the 2017 Hi-MAG outline, not a 2018-2023 shoreline; patches are dependent."},
        "E5": {"name": "Incremental warning value at a common operating point", "status": "NOT_ESTIMABLE",
               "reason": "No common calibrated operating point and no verified controls."}}
    versions = {}
    for pkg in ("torch", "numpy", "scipy", "scikit-learn", "rasterio", "pyproj", "shapely", "pyshp", "python-docx", "matplotlib"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    packet["software"] = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                          "packages": versions}
    packet["replay"] = replay
    packet["protocol"] = {"amendment": "docs/protocol/SR1_AMENDMENT_PLOS_001.json",
                          "amendment_sha256": freeze["amendment_sha256"], "frozen_utc": freeze["frozen_utc"]}
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"packet": PACKET.relative_to(ROOT).as_posix(), "sha256": sha(PACKET),
                      "R1": recon_out["rule_R1_model_uses_context"], "R2": recon_out["rule_R2_model_beats_seasonal_plus_offset"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
