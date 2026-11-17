"""Milestone SR-1 Feature Extraction, Lake-Focused ROIs, Guarded Indices & Robust Scaling.

Governed by plan.md Milestone SR-1 and project/contracts/SR02_registry_and_rois.md.
Eliminates:
- 98.53% index variance explosion via denominator guard delta and non-negative reflectance floor.
- Spatial dilution via 160x160 lake-focused window and 16x16 patch tokenization.
- Monsoon cloud aliasing via >=25% valid support patch gating.
- Out-of-distribution drift via cyclical DOY features (no monotonic time).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CHANNELS_OPTICAL = ["B03", "B08", "B11", "NDWI", "MNDWI"]
CHANNELS_SAR = ["SIGMA0_VV_DB", "SIGMA0_VH_DB"]

# Cutoff: 2 October 2023, 18:30:00 UTC (Sikkim 3 October 00:00 IST)
PRE_BREACH_CUTOFF_UTC = "2023-10-02T18:30:00Z"
EVENT_WINDOW_END_UTC = "2023-10-04T18:30:00Z"

DELTA_DENOMINATOR = 0.002
REFLECTANCE_FLOOR = 0.001


def parse_utc(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def compute_guarded_indices(
    b03: np.ndarray,
    b08: np.ndarray,
    b11: np.ndarray,
    delta: float = DELTA_DENOMINATOR,
    floor: float = REFLECTANCE_FLOOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute guarded NDWI/MNDWI from caller-supplied reflectance means.

    Valid negative reflectances are preserved. The legacy ``floor`` argument is
    accepted for source compatibility but intentionally has no effect; clipping
    is available only as an explicitly named sensitivity analysis. A denominator
    with absolute value below ``delta`` is undefined and returns NaN.
    """
    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    a, b08, b11 = (np.asarray(value, dtype=np.float64) for value in (b03, b08, b11))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        den_ndwi = a + b08
        den_mndwi = a + b11
        ok_ndwi = np.isfinite(a) & np.isfinite(b08) & np.isfinite(den_ndwi) & (np.abs(den_ndwi) >= delta)
        ok_mndwi = np.isfinite(a) & np.isfinite(b11) & np.isfinite(den_mndwi) & (np.abs(den_mndwi) >= delta)
        ndwi = np.full(np.broadcast_shapes(a.shape, b08.shape), np.nan, dtype=np.float64)
        mndwi = np.full(np.broadcast_shapes(a.shape, b11.shape), np.nan, dtype=np.float64)
        np.divide(a - b08, den_ndwi, out=ndwi, where=ok_ndwi)
        np.divide(a - b11, den_mndwi, out=mndwi, where=ok_mndwi)
    return ndwi, mndwi


def tokenize_reflectance_patches(
    reflectance: np.ndarray,
    valid: np.ndarray,
    scl: np.ndarray,
    zone_map: np.ndarray,
    *,
    patch_size: int = 16,
    min_support: float = 0.25,
    delta: float = DELTA_DENOMINATOR,
) -> Dict[str, Any]:
    """Aggregate source reflectance first, then derive patch-mean indices.

    Input bands are B03/B08/B11 in float64 reflectance. ``observed`` is
    channel-specific so undefined index denominators remain missing without
    discarding otherwise observed reflectance channels.
    """
    values = np.asarray(reflectance, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    classes = np.asarray(scl)
    zones = np.asarray(zone_map)
    if values.ndim != 3 or values.shape[0] != 3 or mask.shape != values.shape:
        raise ValueError("reflectance and validity must have matching (3,H,W) shapes")
    if classes.shape != values.shape[1:] or zones.shape != classes.shape:
        raise ValueError("SCL and zone maps must match the reflectance grid")
    if patch_size <= 0 or values.shape[1] % patch_size or values.shape[2] % patch_size:
        raise ValueError("patch size must divide both spatial dimensions")
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be in [0,1]")
    nrows, ncols = values.shape[1] // patch_size, values.shape[2] // patch_size
    count = nrows * ncols
    features = np.full((count, 5), np.nan, dtype=np.float64)
    observed = np.zeros((count, 5), dtype=bool)
    statuses = [["INSUFFICIENT_VALID_SUPPORT"] * 5 for _ in range(count)]
    valid_counts = np.zeros(count, dtype=np.int32)
    patch_zones = np.full(count, 2, dtype=np.int8)
    coords = []
    clear = np.isin(classes, [4, 5, 6]) & mask.all(axis=0) & np.isfinite(values).all(axis=0)
    index = 0
    for pr in range(nrows):
        for pc in range(ncols):
            rs = slice(pr * patch_size, (pr + 1) * patch_size)
            cs = slice(pc * patch_size, (pc + 1) * patch_size)
            patch_clear = clear[rs, cs]
            n_valid = int(patch_clear.sum())
            valid_counts[index] = n_valid
            coords.append((pr, pc))
            zone_patch = zones[rs, cs]
            water_fraction = float(np.mean(zone_patch == 0))
            moraine_fraction = float(np.mean(zone_patch == 1))
            patch_zones[index] = 0 if water_fraction >= 0.15 else (1 if moraine_fraction >= 0.30 else 2)
            if n_valid < math.ceil(min_support * patch_size * patch_size):
                index += 1
                continue
            means = np.array([np.mean(values[ch, rs, cs][patch_clear], dtype=np.float64) for ch in range(3)], dtype=np.float64)
            features[index, :3] = means
            observed[index, :3] = np.isfinite(means)
            for ch in range(3):
                statuses[index][ch] = "OBSERVED" if observed[index, ch] else "NONFINITE_PATCH_MEAN"
            ndwi, mndwi = compute_guarded_indices(means[0], means[1], means[2], delta=delta)
            for col, (value, denominator) in enumerate(((ndwi, means[0] + means[1]),
                                                         (mndwi, means[0] + means[2])), start=3):
                if np.isfinite(value):
                    features[index, col] = float(value)
                    observed[index, col] = True
                    statuses[index][col] = "OBSERVED"
                else:
                    statuses[index][col] = "UNDEFINED_INDEX_DENOMINATOR"
            index += 1
    return {"features": features, "observed": observed, "statuses": statuses,
            "valid_counts": valid_counts, "zones": patch_zones, "coords": coords,
            "supported_patch_count": int(observed[:, :3].all(axis=1).sum()),
            "index_undefined_counts": {"NDWI": int((~observed[:, 3]).sum()),
                                       "MNDWI": int((~observed[:, 4]).sum())}}


def candidate_water_area_baseline(
    reflectance: np.ndarray, valid: np.ndarray, scl: np.ndarray, zone_map: np.ndarray,
    *, delta: float = DELTA_DENOMINATOR, pixel_area_m2: float = 400.0,
) -> Dict[str, Any]:
    """Report a spectral candidate-area baseline, never a validated shoreline."""
    values = np.asarray(reflectance, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).all(axis=0)
    classes = np.asarray(scl)
    zones = np.asarray(zone_map)
    if values.ndim != 3 or values.shape[0] != 3 or classes.shape != values.shape[1:] or zones.shape != classes.shape:
        raise ValueError("baseline inputs must share B03/B08/B11 and spatial dimensions")
    supported = mask & np.isin(classes, [4, 5, 6]) & np.isfinite(values).all(axis=0)
    ndwi, mndwi = compute_guarded_indices(values[0], values[1], values[2], delta=delta)
    defined = supported & np.isfinite(ndwi) & np.isfinite(mndwi)
    candidate = defined & np.isin(zones, [0, 1]) & (ndwi > 0) & (mndwi > 0)
    n_pixels = int(candidate.sum())
    return {"status": "CANDIDATE_SPECTRAL_AREA_NOT_PHYSICAL_WATER_VALIDATION",
            "candidate_pixels": n_pixels,
            "candidate_area_km2": n_pixels * float(pixel_area_m2) / 1_000_000.0,
            "valid_pixels": int(supported.sum()), "defined_index_pixels": int(defined.sum()),
            "zone_pixels": int(np.isin(zones, [0, 1]).sum()),
            "rule": "NDWI > 0 AND MNDWI > 0 within Zone 0/1; clear SCL 4/5/6; absolute denominator >= delta",
            "delta": float(delta)}


def fit_robust_scaler(records: List[Dict[str, Any]], fit_observation_ids: List[str]) -> Dict[str, Any]:
    """Fit channel moments only from an explicit, verified set of FIT records."""
    by_id = {record.get("observation_id"): record for record in records}
    if len(by_id) != len(records) or None in by_id:
        raise ValueError("every scene must have a unique observation_id")
    if not fit_observation_ids or len(set(fit_observation_ids)) != len(fit_observation_ids):
        raise ValueError("FIT observation IDs must be a non-empty unique list")
    missing = set(fit_observation_ids) - set(by_id)
    if missing:
        raise ValueError(f"FIT IDs absent from records: {sorted(missing)}")
    fit_set = set(fit_observation_ids)
    if any(by_id[oid].get("role") != "FIT" for oid in fit_set):
        raise ValueError("scaler input includes an observation not independently declared FIT")
    arrays, masks = [], []
    for oid in fit_observation_ids:
        features = np.asarray(by_id[oid]["features"], dtype=np.float64)
        observed = np.asarray(by_id[oid]["observed"], dtype=bool)
        if features.ndim != 2 or observed.shape != features.shape:
            raise ValueError(f"malformed feature/missingness arrays for {oid}")
        arrays.append(features)
        masks.append(observed & np.isfinite(features))
    values = np.concatenate(arrays, axis=0)
    valid = np.concatenate(masks, axis=0)
    channels = values.shape[1]
    medians, scales, counts = [], [], []
    for channel in range(channels):
        x = values[valid[:, channel], channel]
        if not len(x):
            raise ValueError(f"FIT support contains no valid values for channel {channel}")
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            raise ValueError(f"FIT MAD is degenerate for channel {channel}; no hidden scale fallback")
        medians.append(median)
        scales.append(scale)
        counts.append(int(len(x)))
    return {"medians": medians, "scales": scales, "sample_count_by_channel": counts,
            "fit_observation_ids": sorted(fit_set), "fit_observation_count": len(fit_set),
            "fit_role_check": "ALL_INPUT_IDS_DECLARED_FIT"}


def apply_robust_scaler(features: np.ndarray, observed: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64).copy()
    mask = np.asarray(observed, dtype=bool)
    medians = np.asarray(scaler["medians"], dtype=np.float64)
    scales = np.asarray(scaler["scales"], dtype=np.float64)
    if values.shape != mask.shape or values.shape[-1] != len(medians) or len(medians) != len(scales):
        raise ValueError("feature/scaler dimensions disagree")
    transformed = np.full(values.shape, np.nan, dtype=np.float64)
    np.subtract(values, medians, out=transformed, where=mask & np.isfinite(values))
    np.divide(transformed, scales, out=transformed, where=mask & np.isfinite(values))
    return transformed


def compute_water_area_km2(
    values_5ch: np.ndarray,
    scl: np.ndarray,
    zone_map: Optional[np.ndarray] = None,
    mndwi_thresh: float = 0.05,
    b08_thresh: float = 0.16,
    pixel_area_m2: float = 400.0,
) -> Dict[str, float]:
    """Extract physical liquid water surface area within the core lake basin (Zone 0)."""
    valid_qa = np.isin(scl, [4, 5, 6]) & np.isfinite(values_5ch[0])
    clear_pixels = int(valid_qa.sum())
    clear_fraction = float(clear_pixels / (values_5ch.shape[1] * values_5ch.shape[2]))

    core_mask = (zone_map == 0) if zone_map is not None else np.ones_like(scl, dtype=bool)
    core_liquid_qa = float((valid_qa & core_mask).sum() / max(1, core_mask.sum()))

    # Require scene clear_fraction >= 0.45 and non-frozen lake support (core_liquid_qa >= 0.25)
    if clear_fraction < 0.45 or core_liquid_qa < 0.25:
        return {
            "water_pixel_count": 0,
            "water_area_km2": float("nan"),
            "clear_pixel_count": clear_pixels,
            "clear_fraction": clear_fraction,
        }

    # Count liquid water pixels strictly inside Zone 0 (no artificial division by drained area)
    spectral_water = (
        core_mask
        & valid_qa
        & (
            (scl == 6)
            | (
                (values_5ch[4] > mndwi_thresh)
                & (values_5ch[3] > 0.0)
                & (values_5ch[1] < b08_thresh)
                & (values_5ch[0] > 0.02)
            )
        )
    )
    pixel_count = int(spectral_water.sum())
    area_km2 = float(pixel_count * pixel_area_m2 / 1e6)

    return {
        "water_pixel_count": pixel_count,
        "water_area_km2": area_km2,
        "clear_pixel_count": clear_pixels,
        "clear_fraction": clear_fraction,
    }


def tokenize_scene_patches(
    values_ch: np.ndarray,  # shape (C, 160, 160)
    valid_mask: np.ndarray,  # shape (C, 160, 160)
    scl: np.ndarray,  # shape (160, 160)
    zone_map: np.ndarray,  # shape (160, 160) - 0: water, 1: moraine, 2: background
    patch_size: int = 16,
    min_support: float = 0.25,
) -> Dict[str, Any]:
    """Partition 160x160 scene into 10x10 patches (100 tokens)."""
    H, W = values_ch.shape[1], values_ch.shape[2]
    n_pr = H // patch_size  # 10
    n_pc = W // patch_size  # 10
    num_patches = n_pr * n_pc  # 100
    C = values_ch.shape[0]
    
    patch_features = np.zeros((num_patches, C), dtype=np.float32)
    patch_observed = np.zeros(num_patches, dtype=bool)
    patch_valid_counts = np.zeros(num_patches, dtype=np.int32)
    patch_snow_fractions = np.zeros(num_patches, dtype=np.float32)
    patch_zones = np.zeros(num_patches, dtype=np.int32)
    patch_coords = []

    # Scene-wide clear fraction and measured bare-moraine (Zone 1, SCL 5) reference for dewatered lake bed
    all_clear = np.isin(scl, [4, 5, 6]) & valid_mask.all(axis=0)
    scene_clear_frac = float(all_clear.sum() / (H * W))
    moraine_bare = (zone_map == 1) & (scl == 5) & valid_mask.all(axis=0)
    if not moraine_bare.any():
        moraine_bare = (zone_map >= 1) & all_clear
    sediment_ref = np.array(
        [values_ch[ch][moraine_bare].mean() if moraine_bare.any() else 0.0 for ch in range(C)],
        dtype=np.float32,
    )
    
    idx = 0
    for pr in range(n_pr):
        for pc in range(n_pc):
            r_start, r_end = pr * patch_size, (pr + 1) * patch_size
            c_start, c_end = pc * patch_size, (pc + 1) * patch_size
            
            p_scl = scl[r_start:r_end, c_start:c_end]
            p_zone = zone_map[r_start:r_end, c_start:c_end]
            p_val = values_ch[:, r_start:r_end, c_start:c_end]
            p_mask = valid_mask[:, r_start:r_end, c_start:c_end]
            
            clear_px = np.isin(p_scl, [4, 5, 6]) & p_mask.all(axis=0)
            n_clear = int(clear_px.sum())
            total_px = patch_size * patch_size
            clear_frac = n_clear / total_px
            
            snow_px = (p_scl == 11)
            snow_frac = float(snow_px.sum() / total_px)
            
            water_frac = (p_zone == 0).mean()
            moraine_frac = (p_zone == 1).mean()
            if water_frac >= 0.15:
                dom_zone = 0
            elif moraine_frac >= 0.30:
                dom_zone = 1
            else:
                dom_zone = 2
                
            patch_zones[idx] = dom_zone
            patch_valid_counts[idx] = n_clear
            patch_snow_fractions[idx] = snow_frac
            patch_coords.append((pr, pc))
            
            if dom_zone == 0 and scene_clear_frac >= 0.45:
                core_px = (p_zone == 0)
                cld_free = ~np.isin(p_scl, [0, 1, 2, 3, 7, 8, 9, 10]) & p_mask.all(axis=0)
                n_core_obs = int((core_px & cld_free).sum())
                n_core_water = int((core_px & (p_scl == 6) & p_mask.all(axis=0)).sum())
                n_core_ice = int((core_px & (p_scl == 11) & p_mask.all(axis=0)).sum())
                
                # Check if patch is seasonally snow/ice covered (winter freeze)
                if n_core_obs > 0 and (n_core_ice / n_core_obs) >= 0.25:
                    valid_surf = np.isin(p_scl, [4, 5, 6, 11]) & p_mask.all(axis=0)
                    if valid_surf.any():
                        patch_observed[idx] = True
                        for ch in range(C):
                            patch_features[idx, ch] = float(p_val[ch][valid_surf].mean())
                    else:
                        patch_observed[idx] = False
                        patch_features[idx, :] = np.nan
                elif n_core_obs > 0:
                    # Non-frozen lake: water retention is fraction of cloud-free core pixels that remain water
                    w_retention = float(n_core_water / n_core_obs)
                    patch_observed[idx] = True
                    for ch in range(C):
                        w_mean = float(p_val[ch][clear_px].mean()) if n_clear > 0 else float(sediment_ref[ch])
                        patch_features[idx, ch] = w_retention * w_mean + (1.0 - w_retention) * float(sediment_ref[ch])
                elif clear_frac >= min_support and n_clear > 0:
                    patch_observed[idx] = True
                    for ch in range(C):
                        patch_features[idx, ch] = p_val[ch][clear_px].mean()
                else:
                    patch_observed[idx] = False
                    patch_features[idx, :] = np.nan
            elif clear_frac >= min_support and n_clear > 0:
                patch_observed[idx] = True
                for ch in range(C):
                    patch_features[idx, ch] = p_val[ch][clear_px].mean()
            else:
                patch_observed[idx] = False
                patch_features[idx, :] = np.nan
                
            idx += 1
            
    return {
        "features": patch_features,
        "observed": patch_observed,
        "valid_counts": patch_valid_counts,
        "snow_fractions": patch_snow_fractions,
        "zones": patch_zones,
        "coords": patch_coords,
        "supported_patch_count": int(patch_observed.sum()),
    }


def fit_robust_scaler_legacy_pre_cutoff(
    all_pre_patch_features: np.ndarray,  # shape (N_tokens, C)
    all_pre_observed: np.ndarray,        # shape (N_tokens,)
) -> Dict[str, Any]:
    """Fit robust median and MAD strictly on pre-cutoff observed tokens.
    
    standardized = (x - median) / (1.4826 * MAD)
    Zero singularity risk; outliers cannot inflate the scale.
    """
    valid_tokens = all_pre_patch_features[all_pre_observed]
    C = valid_tokens.shape[1]
    
    medians = np.zeros(C, dtype=np.float32)
    mads = np.zeros(C, dtype=np.float32)
    iqrs = np.zeros(C, dtype=np.float32)
    
    for ch in range(C):
        col = valid_tokens[:, ch]
        col_finite = col[np.isfinite(col)]
        if len(col_finite) == 0:
            medians[ch] = 0.0
            mads[ch] = 1.0
            iqrs[ch] = 1.0
            continue
        med = float(np.median(col_finite))
        mad = float(np.median(np.abs(col_finite - med)))
        q25 = float(np.percentile(col_finite, 25))
        q75 = float(np.percentile(col_finite, 75))
        
        medians[ch] = med
        # Guard against zero MAD
        scale = max(1.4826 * mad, 1e-4)
        mads[ch] = scale
        iqrs[ch] = max(q75 - q25, 1e-4)
        
    return {
        "medians": medians.tolist(),
        "mads": mads.tolist(),
        "iqrs": iqrs.tolist(),
        "sample_tokens_fitted": int(len(valid_tokens)),
    }


def temporal_features(
    dt: datetime,
    prev_dt: Optional[datetime] = None,
) -> np.ndarray:
    """Cyclical day-of-year and delta-t (NO monotonic elapsed time)."""
    doy = dt.timetuple().tm_yday
    rad = 2.0 * math.pi * (doy - 1) / 365.25
    sin_doy = math.sin(rad)
    cos_doy = math.cos(rad)
    
    delta_days = 0.0
    if prev_dt is not None:
        delta_days = max(0.0, (dt - prev_dt).total_seconds() / 86400.0)
    norm_delta = min(delta_days / 30.0, 5.0)  # normalized gap
    
    return np.array([sin_doy, cos_doy, norm_delta], dtype=np.float32)
