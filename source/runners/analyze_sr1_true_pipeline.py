#!/usr/bin/env python3
"""Strict split, feature, training and descriptive scoring for an SR-1 attempt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import rasterio
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.sr1_features import (  # noqa: E402
    apply_robust_scaler, candidate_water_area_baseline, compute_guarded_indices,
    fit_robust_scaler, temporal_features, tokenize_reflectance_patches,
)
from sentinel_gl.sr1_model import SR1TransformerMAE, train_factorized_sr1_model  # noqa: E402
from sentinel_gl.sr1_baselines_true import (  # noqa: E402
    add_page_cusum, fit_descriptive_baselines, score_descriptive_baselines,
)

PROTOCOL = ROOT / "project/sr1_protocol_true_v4.json"
PROTOCOL_FREEZE = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/PROTOCOL_FREEZE_004.json"
T0 = datetime.fromisoformat("2023-10-02T18:30:00+00:00")
T1 = datetime.fromisoformat("2023-10-04T18:30:00+00:00")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def role_for_time(dt: datetime) -> str:
    if dt.year in (2018, 2019, 2020, 2021):
        return "FIT"
    if dt.year == 2022:
        return "VAL"
    if dt.year == 2023 and dt.month <= 6:
        return "FIT"
    if dt.year == 2023 and 7 <= dt.month <= 9:
        return "VAL"
    if T0 <= dt < T1:
        return "EVENT_UNCERTAIN"
    if dt >= T1:
        return "EVAL"
    return "OUT_OF_PROTOCOL_WINDOW"


def verify_protocol() -> str:
    freeze = json.loads(PROTOCOL_FREEZE.read_text())
    actual = digest(PROTOCOL)
    if actual != freeze["sha256"]:
        raise ValueError("protocol changed after the pre-scoring freeze record")
    if freeze.get("status") != "FROZEN_BEFORE_NEW_EMPIRICAL_SCORING":
        raise ValueError("protocol freeze record does not authorize this analysis")
    return actual


def verify_range_receipts(row: dict) -> None:
    if row.get("provider") != "Earth Search C1 L2A":
        return
    for role in ("B03", "B08", "B11", "SCL"):
        receipts = row.get("range_receipts", {}).get(role, {}).get("range_receipts", [])
        if not receipts:
            raise ValueError(f"no byte-range receipt for {row['item_id']} {role}")
        for receipt in receipts:
            if receipt.get("http_status") != 206:
                raise ValueError(f"non-206 COG request in {row['item_id']} {role}")
            content_range = receipt.get("content_range", "")
            if not re.fullmatch(r"bytes \d+-\d+/\d+", content_range):
                raise ValueError(f"malformed Content-Range in {row['item_id']} {role}")
            if receipt.get("body_status") != "COMPLETE_RESPONSE" or not receipt.get("sha256"):
                raise ValueError(f"unverified range payload in {row['item_id']} {role}")


def verify_native_metadata_receipts(row: dict, item: dict, attempt_root: Path) -> None:
    for role in ("product_metadata", "granule_metadata"):
        receipt = row.get("metadata_receipts", {}).get(role)
        if not receipt or receipt.get("http_status") != 200 or receipt.get("body_status") != "COMPLETE_RESPONSE":
            raise ValueError(f"missing successful native metadata receipt for {row['item_id']} {role}")
        if receipt.get("url") != item.get("assets", {}).get(role, {}).get("href"):
            raise ValueError(f"native metadata receipt URL mismatch for {row['item_id']} {role}")
        path = ROOT / receipt["path"]
        if digest(path) != receipt.get("sha256"):
            raise ValueError(f"native metadata file hash mismatch for {row['item_id']} {role}")
        if receipt.get("received_bytes") != path.stat().st_size:
            raise ValueError(f"native metadata response length mismatch for {row['item_id']} {role}")


def load_manifest(manifest_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("origin") != "observational":
        raise ValueError("empirical analysis refuses a non-observational manifest")
    attempt_root = ROOT / "data/raw/sr1_multiyear" / manifest_path.parent.name
    by_parent_id = {}
    old = json.loads((ROOT / "data/derived/expansion_2023_recentered/observations.v2.json").read_text())
    for row in old["observations"]:
        if row.get("sensor") == "SENTINEL_2_OPTICAL":
            by_parent_id[row["observation_id"]] = row
    records = []
    for row in manifest.get("observations", []):
        if row.get("status") == "EXPORT_FAILED":
            records.append({**row, "role": role_for_time(parse_time(row["stac_datetime"]))})
            continue
        site, oid = row.get("site_id"), row.get("item_id")
        if site not in {"PILOT-SOUTH-LHONAK", "PILOT-KHANGCHUNG-TSHO"}:
            raise ValueError("manifest contains an undeclared site")
        if row.get("origin") != "observational":
            raise ValueError(f"constructed or unknown origin in empirical row {oid}")
        values_path, receipt_path = ROOT / row["values_file"], ROOT / row["receipt_file"]
        if digest(values_path) != row["values_sha256"] or digest(receipt_path) != row["receipt_sha256"]:
            raise ValueError(f"values or receipt hash mismatch for {oid}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("site_id") != site or receipt.get("values_sha256") != row["values_sha256"]:
            raise ValueError(f"receipt-to-row identity mismatch for {oid}")
        if parse_time(receipt["acquired_at"]) != parse_time(row["acquired_at"]):
            raise ValueError(f"manifest UTC time differs from receipt for {oid}")
        if row.get("provider") == "Earth Search C1 L2A":
            item_path = attempt_root / "provider" / site / oid / "item.json"
            if digest(item_path) != row["item_sha256"]:
                raise ValueError(f"STAC item hash mismatch for {oid}")
            item = json.loads(item_path.read_text())
            if parse_time(item["properties"]["datetime"]) != parse_time(row["acquired_at"]):
                raise ValueError(f"STAC datetime differs from bound manifest time for {oid}")
            verify_range_receipts(row)
            verify_native_metadata_receipts(row, item, attempt_root)
        else:
            parent_id = oid
            parent = by_parent_id.get(parent_id)
            if not parent:
                raise ValueError(f"2023 row does not bind to source manifest observation {oid}")
            parent_receipt = ROOT / parent["receipt_file"]
            parent_values = ROOT / parent["values_file"]
            if digest(parent_receipt) != row["parent_receipt_sha256"] or digest(parent_values) != row["parent_values_sha256"]:
                raise ValueError(f"retained 2023 parent binding mismatch for {oid}")
            if parse_time(parent["acquired_at"]) != parse_time(row["acquired_at"]):
                raise ValueError(f"2023 source-manifest timestamp mismatch for {oid}")
            parent_receipt_data=json.loads(parent_receipt.read_text())
            native=parent_receipt_data.get("native",{})
            if (row.get("processing_baseline")!=native.get("baseline")
                    or row.get("parent_native_radiometry")!=native.get("radiometry")
                    or parse_time(row.get("native_sensing_time"))!=parse_time(native.get("native_sensing_time"))
                    or row.get("metadata_receipts")!=native.get("receipts")):
                raise ValueError(f"2023 native source metadata differs from exact parent receipt for {oid}")
            for metadata in row.get("metadata_receipts",{}).values():
                mpath=ROOT/metadata["path"]
                if digest(mpath)!=metadata.get("sha256") or metadata.get("http_status")!=200:
                    raise ValueError(f"2023 native metadata receipt invalid for {oid}")
        with np.load(values_path, allow_pickle=False) as data:
            reflectance = data["reflectance"].astype(np.float64)
            valid = data["valid"].astype(bool)
            scl = data["scl"].astype(np.uint8)
        if reflectance.shape != (3, 160, 160) or valid.shape != reflectance.shape or scl.shape != (160, 160):
            raise ValueError(f"input array dimensions mismatch for {oid}")
        dt = parse_time(row["acquired_at"])
        records.append({**row, "observation_id": f"{site}__{oid}", "dt": dt,
                        "role": role_for_time(dt), "reflectance": reflectance,
                        "valid": valid, "scl": scl, "parent_receipt": row.get("receipt_sha256")})
    ids = [r["observation_id"] for r in records if "observation_id" in r]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate observation IDs in active manifest")
    return records, manifest


def fit_zones_from_fit(records: list[dict], attempt_root: Path) -> dict[str, dict]:
    from scipy.ndimage import distance_transform_edt
    fit = [r for r in records if r.get("role") == "FIT"]
    if not fit:
        raise ValueError("no FIT observations available for zone construction")
    zones_by_site = {}
    for site in sorted({r["site_id"] for r in fit}):
        scene_rows = [r for r in fit if r["site_id"] == site]
        votes = np.zeros((160, 160), dtype=np.uint32)
        support = np.zeros((160, 160), dtype=np.uint32)
        for row in scene_rows:
            clear = np.isin(row["scl"], [4, 5, 6]) & row["valid"].all(axis=0)
            ndwi, mndwi = compute_guarded_indices(*row["reflectance"], delta=0.002)
            defined = clear & np.isfinite(ndwi) & np.isfinite(mndwi)
            support[defined] += 1
            water = defined & (row["scl"] == 6) & (ndwi > 0) & (mndwi > 0)
            votes[water] += 1
        frequency = np.divide(votes, support, out=np.zeros_like(votes, dtype=np.float64), where=support > 0)
        core = (frequency >= 0.5) & (support >= min(3, len(scene_rows)))
        buffer = (distance_transform_edt(~core) * 20.0 <= 500.0) & ~core if core.any() else np.zeros_like(core)
        zones = np.full((160, 160), 2, dtype=np.uint8)
        zones[buffer] = 1
        zones[core] = 0
        path = attempt_root / "zones" / f"{site}_fit_only_zones.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, zones=zones, core_water_candidate=core,
                            moraine_buffer_candidate=buffer, support_count=support,
                            water_vote_count=votes, origin=np.asarray("observational_fit_only"))
        zones_by_site[site] = {"zones": zones, "path": path,
                               "sha256": digest(path), "fit_observation_ids": sorted(r["observation_id"] for r in scene_rows),
                               "supported_pixels": int((support > 0).sum()),
                               "candidate_core_pixels": int(core.sum()),
                               "status": "FIT_ONLY_SPECTRAL_ZONES_NOT_INDEPENDENTLY_GEOMETRY_VALIDATED"}
    return zones_by_site


def tokenize_all(records: list[dict], zones_by_site: dict[str, dict]) -> None:
    for site in zones_by_site:
        zone_map = zones_by_site[site]["zones"]
        rows = sorted([r for r in records if r.get("site_id") == site and "reflectance" in r],
                      key=lambda r: r["acquired_at"])
        previous = None
        for row in rows:
            token = tokenize_reflectance_patches(row["reflectance"], row["valid"], row["scl"], zone_map,
                                                 patch_size=16, min_support=0.25, delta=0.002)
            row["features"] = token["features"]
            row["observed"] = token["observed"]
            row["feature_status"] = token["statuses"]
            row["zones"] = token["zones"].astype(np.int64)
            row["valid_counts"] = token["valid_counts"]
            row["time_feats"] = temporal_features(row["dt"], previous)
            row["water_area"] = candidate_water_area_baseline(
                row["reflectance"], row["valid"], row["scl"], zone_map, delta=0.002)
            previous = row["dt"]


def make_context(model, chronological: list[dict], target_index: int, device: str, *, mask_target=False, seed=0):
    from torch import tensor
    start = max(0, target_index - 7)
    context = chronological[start:target_index + 1]
    x_np = np.stack([r["std_features"] for r in context]).astype(np.float32)
    channel_valid = np.stack([r["observed"] for r in context]).astype(bool)
    obs = channel_valid.any(axis=-1)
    zones = np.stack([r["zones"] for r in context]).astype(np.int64)
    times = np.stack([r["time_feats"] for r in context]).astype(np.float32)
    x = tensor(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device).unsqueeze(0)
    o = tensor(obs, dtype=torch.bool, device=device).unsqueeze(0)
    z = tensor(zones, dtype=torch.long, device=device).unsqueeze(0)
    t = tensor(times, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.zeros_like(o)
    target_obs = np.flatnonzero(obs[-1])
    if mask_target and len(target_obs):
        rng = np.random.default_rng(seed)
        selected = target_obs[rng.permutation(len(target_obs))[:max(1, len(target_obs) // 2)]]
        m[0, -1, torch.as_tensor(selected, dtype=torch.long, device=device)] = True
    with torch.no_grad():
        pred, latent = model(x, o, m, z, t, modality_id=0)
    return context, pred[0, -1].cpu().numpy(), latent[0, -1].cpu().numpy(), m[0, -1].cpu().numpy()


def fit_score_b_pca(model, records, by_site, fit_ids, device):
    latents = {}
    for site, chrono in by_site.items():
        for idx, row in enumerate(chrono):
            _, _, latent, _ = make_context(model, chrono, idx, device)
            latents[row["observation_id"]] = latent.astype(np.float64)
    fit_latents = np.concatenate([latents[oid] for oid in fit_ids], axis=0)
    if fit_latents.shape[0] < 9 or not np.isfinite(fit_latents).all():
        raise ValueError("FIT latent bank cannot support the frozen PCA rank 8")
    pca = PCA(n_components=8, svd_solver="full")
    pca.fit(fit_latents)
    projected = {oid: pca.transform(value) for oid, value in latents.items()}
    pca_info = {"rank": 8, "fit_observation_ids": sorted(fit_ids),
                "fit_latent_rows": int(fit_latents.shape[0]),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "components_sha256": hashlib.sha256(pca.components_.tobytes()).hexdigest()}
    return projected, pca_info


def score_one_seed(model, records, by_site, fit_ids, seed, device):
    projected, pca_info = fit_score_b_pca(model, records, by_site, fit_ids, device)
    rows = []
    replay = {"rows": [], "projected": projected}
    for site, chrono in by_site.items():
        fit_rows = [r for r in chrono if r["observation_id"] in fit_ids]
        fit_doys = {r["observation_id"]: r["dt"].timetuple().tm_yday for r in fit_rows}
        for idx, row in enumerate(chrono):
            _, pred, _, masked = make_context(model, chrono, idx, device,
                                               mask_target=True, seed=seed + idx)
            target = row["std_features"]
            channel_valid = row["observed"] & np.isfinite(target)
            patch_zone = row["zones"]
            patch_errors = np.full(len(patch_zone), np.nan, dtype=np.float64)
            for p in np.flatnonzero(masked):
                channels = channel_valid[p]
                if channels.any():
                    patch_errors[p] = float(np.mean((pred[p, channels] - target[p, channels]) ** 2))
            target_patch = np.isin(patch_zone, [0, 1]) & np.isfinite(patch_errors)
            background_patch = (patch_zone == 2) & np.isfinite(patch_errors)
            a_target = float(np.mean(patch_errors[target_patch])) if target_patch.any() else None
            a_background = float(np.mean(patch_errors[background_patch])) if background_patch.any() else None
            patch_scores = []
            doy = row["dt"].timetuple().tm_yday
            for p in np.flatnonzero(np.isin(patch_zone, [0, 1])):
                refs = []
                for ref in fit_rows:
                    if ref["observation_id"] == row["observation_id"]:
                        continue
                    rdoy = fit_doys[ref["observation_id"]]
                    dd = abs(doy - rdoy)
                    if min(dd, 366 - dd) <= 45:
                        ref_latent = projected[ref["observation_id"]][p]
                        query_latent = projected[row["observation_id"]][p]
                        refs.append(float(np.linalg.norm(query_latent - ref_latent)))
                if len(refs) >= 5:
                    patch_scores.append(float(np.mean(np.sort(refs)[:5])))
            b_target = float(np.mean(patch_scores)) if patch_scores else None
            rows.append({"observation_id": row["observation_id"], "site_id": site,
                         "acquired_at": row["acquired_at"], "role": row["role"], "seed": seed,
                         "score_a_zone0_1": a_target,
                         "score_a_zone0_1_ln": (math.log(a_target) if a_target is not None and a_target > 0 else None),
                         "score_a_zone2": a_background,
                         "score_a_zone2_ln": (math.log(a_background) if a_background is not None and a_background > 0 else None),
                         "score_a_status": "ESTIMATED" if a_target is not None else "NO_SUPPORTED_TARGET_PATCHES",
                         "score_a_target_patch_count": int(target_patch.sum()),
                         "score_a_background_patch_count": int(background_patch.sum()),
                         "score_b_zone0_1": b_target,
                         "score_b_status": "ESTIMATED" if b_target is not None else "INSUFFICIENT_FIT_NEIGHBORS",
                         "score_a_zone_contrast": (a_target - a_background if a_target is not None and a_background is not None else None),
                         "water_candidate_area_km2": row["water_area"]["candidate_area_km2"],
                         "water_area_status": row["water_area"]["status"]})
            replay["rows"].append({
                "observation_id": row["observation_id"], "site_id": site,
                "acquired_at": row["acquired_at"], "role": row["role"],
                "target": target.astype(np.float32),
                "observed": row["observed"].astype(bool),
                "zones": patch_zone.astype(np.int8),
                "masked": masked.astype(bool),
                "prediction": pred.astype(np.float32),
            })
    # Score-C moments are estimated from FIT only and applied unchanged elsewhere.
    fit_scores = [r for r in rows if r["role"] == "FIT" and r["score_a_zone0_1"] is not None and r["score_a_zone0_1"] > 0
                  and r["score_b_zone0_1"] is not None]
    if len(fit_scores) < 2:
        c_status = "NOT_ESTIMABLE_INSUFFICIENT_FIT_SCORE_PAIRS"
        c_moments = None
        for row in rows:
            row["score_c"] = None
    else:
        ln_a = np.log(np.asarray([r["score_a_zone0_1"] for r in fit_scores], dtype=np.float64))
        b = np.asarray([r["score_b_zone0_1"] for r in fit_scores], dtype=np.float64)
        med_a, mad_a = float(np.median(ln_a)), float(np.median(np.abs(ln_a - np.median(ln_a))))
        med_b, mad_b = float(np.median(b)), float(np.median(np.abs(b - np.median(b))))
        if mad_a <= np.finfo(float).eps or mad_b <= np.finfo(float).eps:
            c_status = "NOT_ESTIMABLE_DEGENERATE_FIT_MAD"
            c_moments = {"ln_a_median": med_a, "ln_a_mad": mad_a,
                         "b_median": med_b, "b_mad": mad_b,
                         "fit_observation_ids": sorted(r["observation_id"] for r in fit_scores)}
            for row in rows:
                row["score_c"] = None
        else:
            c_status = "ESTIMATED_FIT_ONLY_TRANSFORM"
            c_moments = {"ln_a_median": med_a, "ln_a_scale": 1.4826 * mad_a,
                         "b_median": med_b, "b_scale": 1.4826 * mad_b,
                         "fit_observation_ids": sorted(r["observation_id"] for r in fit_scores)}
            for row in rows:
                if row["score_a_zone0_1"] is not None and row["score_a_zone0_1"] > 0 and row["score_b_zone0_1"] is not None:
                    za = (math.log(row["score_a_zone0_1"]) - med_a) / (1.4826 * mad_a)
                    zb = (row["score_b_zone0_1"] - med_b) / (1.4826 * mad_b)
                    row["score_c"] = 0.5 * za + 0.5 * zb
                else:
                    row["score_c"] = None
    return rows, {"pca": pca_info, "score_c_status": c_status, "score_c_fit_moments": c_moments}, replay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="attempt_004")
    ap.add_argument("--analysis", default="analysis_001")
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    if not re.fullmatch(r"attempt_[0-9]{3}", args.attempt) or not re.fullmatch(r"analysis_[0-9]{3}", args.analysis):
        raise SystemExit("attempt and analysis IDs must use immutable attempt_NNN / analysis_NNN names")
    protocol_hash = verify_protocol()
    manifest_path = ROOT / "data/derived/sr1_multiyear" / args.attempt / "observations.sr1.json"
    records, manifest = load_manifest(manifest_path)
    output = ROOT / "data/derived/sr1_multiyear" / args.attempt / args.analysis
    if output.exists():
        raise SystemExit(f"analysis output already exists: {output}")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    attempt_root = ROOT / "data/raw/sr1_multiyear" / args.attempt
    zones_by_site = fit_zones_from_fit(records, attempt_root)
    tokenize_all(records, zones_by_site)
    usable = [r for r in records if "features" in r]
    fit_ids = sorted(r["observation_id"] for r in usable if r["role"] == "FIT")
    val_ids = sorted(r["observation_id"] for r in usable if r["role"] == "VAL")
    if not fit_ids or not val_ids:
        raise ValueError("frozen FIT/VAL split does not contain both populations")
    scaler = fit_robust_scaler(usable, fit_ids)
    scaler_payload = {k: v for k, v in scaler.items()}
    scaler_path = output / "fit_scaler.json"
    write(scaler_path, scaler_payload)
    for row in usable:
        row["std_features"] = apply_robust_scaler(row["features"], row["observed"], scaler).astype(np.float32)
    by_site = {site: sorted([r for r in usable if r["site_id"] == site], key=lambda r: r["acquired_at"])
               for site in sorted({r["site_id"] for r in usable})}
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("five distinct execution seeds are required")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    summaries, all_predictions, replay_files = [], [], []
    for seed in seeds:
        # Seed before parameter construction as well as inside the trainer, so
        # each requested seed has an independently initialized parameter set.
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SR1TransformerMAE(d_model=64, nhead=4, num_encoder_layers=3,
                                  num_decoder_layers=1, num_patches=100)
        history = train_factorized_sr1_model(
            model, usable, fit_observation_ids=fit_ids, validation_observation_ids=val_ids,
            max_epochs=args.max_epochs, device=device, seed=seed, max_context_scenes=8,
        )
        checkpoint_path = output / "checkpoints" / f"sr1_tmae_seed_{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "seed": seed,
                    "parameter_count": model.count_parameters(), "protocol_sha256": protocol_hash,
                    "fit_observation_ids": fit_ids, "validation_observation_ids": val_ids,
                    "scaler_sha256": digest(scaler_path), "history": history}, checkpoint_path)
        history["checkpoint_path"] = checkpoint_path.relative_to(ROOT).as_posix()
        history["checkpoint_sha256"] = digest(checkpoint_path)
        predictions, score_info, replay = score_one_seed(model, usable, by_site, set(fit_ids), seed, device)
        for row in predictions:
            row["checkpoint_sha256"] = history["checkpoint_sha256"]
        all_predictions.extend(predictions)
        summaries.append({"history": history, "score_info": score_info})
        replay_dir = output / "replay"
        replay_dir.mkdir(exist_ok=True)
        replay_path = replay_dir / f"seed_{seed}_arithmetic_replay.npz"
        replay_rows = replay["rows"]
        ordered_ids = [r["observation_id"] for r in replay_rows]
        projected = replay["projected"]
        np.savez_compressed(
            replay_path,
            observation_ids=np.asarray(ordered_ids, dtype="U128"),
            site_ids=np.asarray([r["site_id"] for r in replay_rows], dtype="U64"),
            acquired_at=np.asarray([r["acquired_at"] for r in replay_rows], dtype="U40"),
            roles=np.asarray([r["role"] for r in replay_rows], dtype="U32"),
            target=np.stack([r["target"] for r in replay_rows]),
            observed=np.stack([r["observed"] for r in replay_rows]),
            zones=np.stack([r["zones"] for r in replay_rows]),
            masked=np.stack([r["masked"] for r in replay_rows]),
            prediction=np.stack([r["prediction"] for r in replay_rows]),
            projected=np.stack([projected[oid] for oid in ordered_ids]).astype(np.float64),
            fit_observation_ids=np.asarray(fit_ids, dtype="U128"),
            protocol_sha256=np.asarray(protocol_hash),
            pca_components_sha256=np.asarray(score_info["pca"]["components_sha256"]),
        )
        replay_files.append({"seed": seed, "path": replay_path.relative_to(ROOT).as_posix(),
                             "sha256": digest(replay_path), "rows": len(replay_rows)})
        for row in predictions:
            row["arithmetic_replay_path"] = replay_path.relative_to(ROOT).as_posix()
        write(output / "training_histories.json", summaries)
    # Descriptive baselines are fitted only from complete FIT target-patch tokens.
    fitted_baselines = fit_descriptive_baselines(usable, fit_ids, random_state=0)
    baseline_rows = score_descriptive_baselines(usable, fit_ids, fitted_baselines)
    baseline_by_id = {r["observation_id"]: r for r in baseline_rows}
    for prediction in all_predictions:
        prediction["baselines"] = baseline_by_id[prediction["observation_id"]]
    add_page_cusum(all_predictions, kappa=0.5, threshold=4.0)
    write(output / "predictions.json", all_predictions)
    import joblib
    baseline_model_path=output/"checkpoints"/"sr1_descriptive_baselines.joblib"
    joblib.dump(fitted_baselines,baseline_model_path,compress=3)
    baseline_info={k:v for k,v in fitted_baselines.items() if k not in {"isolation_forest","one_class_svm","rpca_basis"}}
    baseline_info.update({"checkpoint_path":baseline_model_path.relative_to(ROOT).as_posix(),
                          "checkpoint_sha256":digest(baseline_model_path),
                          "predictions_path":(output/"predictions.json").relative_to(ROOT).as_posix(),
                          "seasonal_status":"FIT_ONLY_SAME_LOCATION_CIRCULAR_DOY_WINDOW_45D_MIN_5",
                          "persistence_status":"CAUSAL_PRIOR_OBSERVATION_SAME_PATCH",
                          "page_cusum":{"kappa":0.5,"threshold":4.0,
                              "status":"EXPLORATORY_FIXED_RULE_NOT_OPERATIONALLY_CALIBRATED"}})
    write(output/"baseline_suite.json",baseline_info)
    # Estimands fail closed when their populations/validation conditions are absent.
    estimands = {
        "E1": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No frozen independent alarm operating point and delivery-availability evidence; event bracket is not independently reviewed."},
        "E2": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No independently ascertained event-free control lake-years or measured eligible exposure."},
        "E3": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No separate rolling-origin model per 2018-2022 reference year; a single fitted model would expose training years."},
        "E4": {"value": "Per-scene Score-A zone contrasts are retained in predictions.json", "status": "CONDITIONAL_DESCRIPTIVE", "reason": "Zones use FIT-only candidate spectral composites and lack independent lake/moraine geometry validation."},
        "E5": {"value": None, "status": "NOT_ESTIMABLE", "reason": "No independently calibrated common operating point and complete fair-baseline predictions with matched availability."},
    }
    report = {
        "schema_version": 1, "status": "LOCAL_EXPLORATORY_EXECUTION_WITH_OPEN_RESEARCH_BLOCKERS",
        "protocol_sha256": protocol_hash, "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(manifest_path), "split": {"FIT_ids": fit_ids, "VAL_ids": val_ids,
              "EVAL_ids": sorted(r["observation_id"] for r in usable if r["role"] == "EVAL"),
              "EVENT_UNCERTAIN_ids": sorted(r["observation_id"] for r in usable if r["role"] == "EVENT_UNCERTAIN"),
              "disjoint": not (set(fit_ids) & set(val_ids))},
        "scaler_path": scaler_path.relative_to(ROOT).as_posix(), "scaler_sha256": digest(scaler_path),
        "zones": {site: {k: v for k, v in info.items() if k not in {"zones", "path"}}
                  | {"path": info["path"].relative_to(ROOT).as_posix()} for site, info in zones_by_site.items()},
        "seeds": seeds, "device": device, "model_parameter_count": SR1TransformerMAE().count_parameters(),
        "predictions_path": (output / "predictions.json").relative_to(ROOT).as_posix(),
        "arithmetic_replay_files": replay_files,
        "training_history_path": (output / "training_histories.json").relative_to(ROOT).as_posix(),
        "estimands": estimands,
        "limitations": ["Two declared ROIs only; no independent event/control support.",
                        "No rolling-origin years for E3; autumn ranks remain unestimated.",
                        "Descriptive baseline scores and fixed-rule CUSUM are exploratory; no alarm operating point or factorial ablation is independently validated.",
                        "No independent terrain/lake geometry validation or source-specific redistribution review."],
        "baseline_suite": baseline_info,
        "missing_selected_scenes": [r for r in records if r.get("status") == "EXPORT_FAILED"],
    }
    write(output / "analysis_report.json", report)
    print(json.dumps({"status": report["status"], "fit": len(fit_ids), "val": len(val_ids),
                      "eval": len(report["split"]["EVAL_ids"]), "seeds": seeds,
                      "device": device, "output": output.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
