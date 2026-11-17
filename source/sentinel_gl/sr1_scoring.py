"""Scoring for the corrected SR-1 analysis (protocol amendment PLOS-001).

* Cross-mask reconstruction: the 100 patches of a 10 x 10 window are split
  into two fixed checkerboard folds. For each target scene, each fold's
  observed patches are hidden in turn while the other fold and up to seven
  earlier scenes stay visible, so every observed target patch is hidden
  exactly once and the folds are the same for every seed.
* Score A: per hidden patch, mean squared error over its observed channels;
  averaged over zone 0+1 patches (and separately zone 2).
* Score B: same-patch distance between rank-8 PCA projections of latent
  vectors (unmasked pass) and the five nearest FIT acquisitions of the same
  site within +/-45 days of day of year. Query and reference patches must be
  observed; PCA is fitted on FIT latents of observed patches only.
* Score C: 0.5 z(ln A) + 0.5 z(B) with median / 1.4826 MAD moments fitted on
  the FIT scenes of the same site.
* Reference predictors are evaluated on exactly the same hidden entries:
  FIT median (zero in standardized units), per-patch FIT mean, per-patch
  seasonal FIT mean (+/-45 d), seasonal mean plus the target scene's
  visible-fold offset, and a calendar-only site mean (+/-45 d, no patch
  identity).
* Context ablation: the same targets reconstructed with all visible inputs set
  to zero, and with every observed patch of every context scene hidden.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

GRID_SIDE = 10
MAX_CONTEXT = 8
SEASON_WINDOW_DAYS = 45
SCORE_B_K = 5
PCA_RANK = 8
REFERENCE_PREDICTORS = ("fit_median", "patch_mean", "patch_seasonal_mean",
                        "patch_seasonal_mean_plus_visible_offset", "calendar_site_mean")
ABLATIONS = ("model", "model_visible_inputs_zeroed", "model_all_context_hidden")


def patch_folds(n_patches: int = 100, side: int = GRID_SIDE) -> np.ndarray:
    if n_patches != side * side:
        raise ValueError("checkerboard folds require a square patch grid")
    return np.array([((p // side) + (p % side)) % 2 for p in range(n_patches)], dtype=np.int8)


def doy_distance(a: int, b: int) -> int:
    d = abs(int(a) - int(b))
    return min(d, 366 - d)


def _context(chrono: list[dict], idx: int, device: str, max_context: int = MAX_CONTEXT):
    start = max(0, idx - max_context + 1)
    ctx = chrono[start: idx + 1]
    x_np = np.stack([r["std_features"] for r in ctx]).astype(np.float32)
    cv = np.stack([r["observed"] for r in ctx]).astype(bool) & np.isfinite(x_np)
    obs = cv.any(axis=-1)
    zones = np.stack([r["zones"] for r in ctx]).astype(np.int64)
    times = np.stack([r["time_feats"] for r in ctx]).astype(np.float32)
    x = torch.as_tensor(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0), device=device).unsqueeze(0)
    o = torch.as_tensor(obs, device=device).unsqueeze(0)
    z = torch.as_tensor(zones, device=device).unsqueeze(0)
    t = torch.as_tensor(times, device=device).unsqueeze(0)
    return x, o, z, t, cv[-1]


@torch.no_grad()
def model_passes(model, chrono: list[dict], idx: int, folds: np.ndarray, device: str = "cpu") -> dict:
    """Cross-mask reconstruction, both ablations and the unmasked latent for one scene."""
    x, o, z, t, target_cv = _context(chrono, idx, device)
    target_obs = o[0, -1].cpu().numpy()
    n, c = target_cv.shape
    out = {name: np.full((n, c), np.nan, dtype=np.float64) for name in ABLATIONS}
    for fold in (0, 1):
        sel = target_obs & (folds == fold)
        if not sel.any():
            continue
        m = torch.zeros_like(o)
        m[0, -1, torch.as_tensor(np.flatnonzero(sel), device=device)] = True
        pred, _ = model(x, o, m, z, t, modality_id=0)
        out["model"][sel] = pred[0, -1].cpu().numpy()[sel]
        xz = x.clone()
        xz[~m] = 0.0
        pred0, _ = model(xz, o, m, z, t, modality_id=0)
        out["model_visible_inputs_zeroed"][sel] = pred0[0, -1].cpu().numpy()[sel]
    if target_obs.any():
        pall, _ = model(x, o, o.clone(), z, t, modality_id=0)
        out["model_all_context_hidden"][target_obs] = pall[0, -1].cpu().numpy()[target_obs]
    _, latent = model(x, o, torch.zeros_like(o), z, t, modality_id=0)
    out["latent"] = latent[0, -1].cpu().numpy().astype(np.float64)
    out["target_observed_patch"] = target_obs
    out["target_channel_valid"] = target_cv
    return out


def reference_predictions(chrono: list[dict], idx: int, fit_ids: set[str], folds: np.ndarray,
                          window: int = SEASON_WINDOW_DAYS) -> tuple[dict, dict]:
    """Parameter-free predictors for the target scene from FIT scenes of the same site."""
    query = chrono[idx]
    refs = [r for r in chrono if r["observation_id"] in fit_ids and r["observation_id"] != query["observation_id"]]
    target = np.asarray(query["std_features"], dtype=np.float64)
    tv = np.asarray(query["observed"], bool) & np.isfinite(target)
    n, c = target.shape
    if not refs:
        raise ValueError("no FIT reference scene for this site")
    vals = np.stack([np.asarray(r["std_features"], dtype=np.float64) for r in refs])
    valid = np.stack([np.asarray(r["observed"], bool) for r in refs]) & np.isfinite(vals)
    filled = np.where(valid, vals, 0.0)
    qdoy = query["dt"].timetuple().tm_yday
    near = np.array([doy_distance(qdoy, r["dt"].timetuple().tm_yday) <= window for r in refs])

    def mean_over(sel):
        cnt = valid[sel].sum(axis=0)
        tot = filled[sel].sum(axis=0)
        return np.divide(tot, cnt, out=np.full((n, c), np.nan), where=cnt > 0)

    patch_mean = mean_over(np.ones(len(refs), bool))
    seasonal = mean_over(near) if near.any() else np.full((n, c), np.nan)
    cal_sel = near if near.any() else np.ones(len(refs), bool)
    cal_cnt = valid[cal_sel].sum(axis=(0, 1))
    calendar = np.divide(filled[cal_sel].sum(axis=(0, 1)), cal_cnt, out=np.zeros(c), where=cal_cnt > 0)
    calendar = np.broadcast_to(calendar, (n, c)).copy()
    fallback = {"patch_mean_to_calendar": int((~np.isfinite(patch_mean) & tv).sum()),
                "seasonal_to_patch_mean": int((~np.isfinite(seasonal) & np.isfinite(patch_mean) & tv).sum()),
                "seasonal_to_calendar": int((~np.isfinite(seasonal) & ~np.isfinite(patch_mean) & tv).sum())}
    patch_mean = np.where(np.isfinite(patch_mean), patch_mean, calendar)
    seasonal = np.where(np.isfinite(seasonal), seasonal, patch_mean)
    offset_pred = seasonal.copy()
    offset_status = {}
    for fold in (0, 1):
        visible = tv & (folds != fold)[:, None]
        offset = np.zeros(c)
        for ch in range(c):
            if visible[:, ch].any():
                offset[ch] = float(np.mean(target[visible[:, ch], ch] - seasonal[visible[:, ch], ch]))
        offset_status[f"fold_{fold}_visible_entries"] = int(visible.sum())
        offset_pred[folds == fold] = seasonal[folds == fold] + offset
    preds = {"fit_median": np.zeros((n, c)), "patch_mean": patch_mean,
             "patch_seasonal_mean": seasonal,
             "patch_seasonal_mean_plus_visible_offset": offset_pred,
             "calendar_site_mean": calendar}
    info = {"fit_reference_scenes": len(refs), "seasonal_reference_scenes": int(near.sum()),
            "fallback_entries": fallback, **offset_status}
    return preds, info


def entry_errors(pred: np.ndarray, target: np.ndarray, tv: np.ndarray, zones: np.ndarray) -> dict:
    """Sum of squared errors and entry counts on hidden target entries, by zone group."""
    se = np.where(tv, (np.nan_to_num(pred) - np.nan_to_num(target)) ** 2, 0.0)
    if (tv & ~np.isfinite(pred)).any():
        raise ValueError("predictor left a target entry without a finite prediction")
    z01 = np.isin(zones, [0, 1])[:, None] & tv
    z2 = (zones == 2)[:, None] & tv
    return {"sse_all": float(se.sum()), "n_all": int(tv.sum()),
            "sse_zone01": float(se[z01].sum()), "n_zone01": int(z01.sum()),
            "sse_zone2": float(se[z2].sum()), "n_zone2": int(z2.sum())}


def score_a(pred: np.ndarray, target: np.ndarray, tv: np.ndarray, zones: np.ndarray) -> dict:
    patch_err = np.full(len(zones), np.nan)
    for p in range(len(zones)):
        ch = tv[p]
        if ch.any() and np.isfinite(pred[p, ch]).all():
            patch_err[p] = float(np.mean((pred[p, ch] - target[p, ch]) ** 2))
    z01 = np.isin(zones, [0, 1]) & np.isfinite(patch_err)
    z2 = (zones == 2) & np.isfinite(patch_err)
    a01 = float(np.mean(patch_err[z01])) if z01.any() else None
    a2 = float(np.mean(patch_err[z2])) if z2.any() else None
    return {"score_a_zone0_1": a01, "score_a_zone2": a2,
            "score_a_zone_contrast": (a01 - a2) if a01 is not None and a2 is not None else None,
            "score_a_status": "ESTIMATED" if a01 is not None else "NOT_ESTIMABLE_NO_OBSERVED_ZONE_0_1_PATCH",
            "score_a_zone01_patches": int(z01.sum()), "score_a_zone2_patches": int(z2.sum())}


def fit_latent_pca(latents: dict[str, np.ndarray], patch_observed: dict[str, np.ndarray],
                   fit_ids: set[str], rank: int = PCA_RANK):
    rows = [latents[oid][patch_observed[oid]] for oid in sorted(fit_ids) if patch_observed[oid].any()]
    bank = np.concatenate(rows, axis=0) if rows else np.empty((0, 1))
    if bank.shape[0] <= rank or not np.isfinite(bank).all():
        raise ValueError("FIT latent bank cannot support the PCA rank")
    pca = PCA(n_components=rank, svd_solver="full").fit(bank)
    return pca, {"rank": rank, "fit_latent_rows_observed_patches_only": int(bank.shape[0]),
                 "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}


def score_b_for_scene(query: dict, refs: list[dict], zones: np.ndarray, *, k: int = SCORE_B_K,
                      window_days: int = SEASON_WINDOW_DAYS) -> tuple[Any, str, int]:
    """Mean over eligible zone 0/1 patches of the mean distance to the k nearest FIT neighbours.

    ``query`` and each ref carry ``observation_id``, ``doy``, ``projected`` (N, r)
    and ``patch_observed`` (N,). A patch is eligible only if it is observed in
    the query and has at least ``k`` observed same-patch FIT references within
    the circular day-of-year window (excluding the query itself).
    """
    zone01 = np.isin(np.asarray(zones), [0, 1])
    candidates = np.flatnonzero(zone01 & np.asarray(query["patch_observed"], bool))
    if not len(candidates):
        return None, "NOT_ESTIMABLE_NO_OBSERVED_ZONE_0_1_PATCH", 0
    seasonal = [r for r in refs if r["observation_id"] != query["observation_id"]
                and doy_distance(query["doy"], r["doy"]) <= window_days]
    scores = []
    for p in candidates:
        dists = [float(np.linalg.norm(query["projected"][p] - r["projected"][p]))
                 for r in seasonal if r["patch_observed"][p]]
        if len(dists) >= k:
            scores.append(float(np.mean(np.sort(dists)[:k])))
    if not scores:
        return None, "NOT_ESTIMABLE_LT_K_OBSERVED_FIT_NEIGHBOURS", 0
    return float(np.mean(scores)), "ESTIMATED", len(scores)


def add_score_c_per_site(rows: list[dict]) -> dict:
    """Score C with FIT-only moments estimated separately for each site."""
    moments = {}
    for site in sorted({r["site_id"] for r in rows}):
        fit = [r for r in rows if r["site_id"] == site and r["role"] == "FIT"
               and r["score_a_zone0_1"] is not None and r["score_a_zone0_1"] > 0 and r["score_b_zone0_1"] is not None]
        status, m = "NOT_ESTIMABLE_LT_2_FIT_SCORE_PAIRS", None
        if len(fit) >= 2:
            la = np.log([r["score_a_zone0_1"] for r in fit])
            b = np.asarray([r["score_b_zone0_1"] for r in fit], float)
            ma, mb = float(np.median(la)), float(np.median(b))
            sa = 1.4826 * float(np.median(np.abs(la - ma)))
            sb = 1.4826 * float(np.median(np.abs(b - mb)))
            if sa > np.finfo(float).eps and sb > np.finfo(float).eps:
                status = "ESTIMATED_FIT_ONLY_PER_SITE"
                m = {"ln_a_median": ma, "ln_a_scale": sa, "b_median": mb, "b_scale": sb, "n_fit_scenes": len(fit)}
            else:
                status = "NOT_ESTIMABLE_DEGENERATE_FIT_MAD"
        moments[site] = {"status": status, "moments": m}
        for r in rows:
            if r["site_id"] != site:
                continue
            ok = m is not None and r["score_a_zone0_1"] is not None and r["score_a_zone0_1"] > 0 and r["score_b_zone0_1"] is not None
            r["score_c"] = (0.5 * (math.log(r["score_a_zone0_1"]) - m["ln_a_median"]) / m["ln_a_scale"]
                            + 0.5 * (r["score_b_zone0_1"] - m["b_median"]) / m["b_scale"]) if ok else None
            r["score_c_status"] = "ESTIMATED" if ok else ("NOT_ESTIMABLE_MISSING_A_OR_B" if m is not None else status)
    return moments


def validation_reference_mse(rows: list[dict], fit_ids: list[str], val_ids: list[str]) -> dict:
    """Per-patch seasonal FIT mean evaluated on the trainer's VAL targets (every other observed patch)."""
    by_site: dict[str, list[dict]] = {}
    for r in rows:
        by_site.setdefault(r["site_id"], []).append(r)
    fit, val = set(fit_ids), set(val_ids)
    sse = {name: 0.0 for name in ("fit_median", "patch_mean", "patch_seasonal_mean")}
    n = 0
    for site, site_rows in by_site.items():
        chrono = sorted(site_rows, key=lambda r: r["acquired_at"])
        dummy_folds = np.zeros(len(chrono[0]["zones"]), dtype=np.int8)
        for idx, row in enumerate(chrono):
            if row["observation_id"] not in val:
                continue
            target = np.asarray(row["std_features"], dtype=np.float64)
            tv = np.asarray(row["observed"], bool) & np.isfinite(target)
            obs_patch = np.flatnonzero(tv.any(axis=1))
            mask = np.zeros(len(tv), bool)
            mask[obs_patch[::2]] = True
            entries = tv & mask[:, None]
            if not entries.any():
                continue
            preds, _ = reference_predictions(chrono, idx, fit, dummy_folds)
            for name in sse:
                sse[name] += float(((preds[name] - target)[entries] ** 2).sum())
            n += int(entries.sum())
    return {name: (value / n if n else None) for name, value in sse.items()} | {"n_entries": n}
