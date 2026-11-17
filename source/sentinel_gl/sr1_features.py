"""SR-1 patch features: guarded water indices, patch tokens, FIT-only scaling.

* Reflectance is native signed bottom-of-atmosphere reflectance; valid
  negative values are kept (no reflectance floor).
* B03/B08/B11 are averaged over the clear pixels (SCL 4, 5, 6) of each
  16 x 16 pixel patch; a patch is observed only with >= 25% clear support.
* NDWI and MNDWI are computed from the patch-mean reflectances. A denominator
  whose absolute value is below delta = 0.002 is undefined and the channel is
  missing with reason ``UNDEFINED_INDEX_DENOMINATOR``.
* Patch zone labels: 0 if >= 15% of the patch's pixels are in zone 0 (lake),
  else 1 if >= 30% are in zone 1 (500 m buffer), else 2 (background).
* Robust scaling uses the median and 1.4826 x MAD of FIT observations only.
* Time features are cyclic day of year and the capped revisit gap.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

CHANNELS_OPTICAL = ["B03", "B08", "B11", "NDWI", "MNDWI"]
DELTA_DENOMINATOR = 0.002
PATCH_SIZE = 16
MIN_PATCH_SUPPORT = 0.25
ZONE0_PATCH_FRACTION = 0.15
ZONE1_PATCH_FRACTION = 0.30
CLEAR_SCL_CLASSES = (4, 5, 6)


def compute_guarded_indices(
    b03: np.ndarray,
    b08: np.ndarray,
    b11: np.ndarray,
    delta: float = DELTA_DENOMINATOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute guarded NDWI/MNDWI from caller-supplied reflectance means.

    Valid negative reflectances are preserved. A denominator with absolute
    value below ``delta`` is undefined and returns NaN.
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
    patch_size: int = PATCH_SIZE,
    min_support: float = MIN_PATCH_SUPPORT,
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
    clear = np.isin(classes, CLEAR_SCL_CLASSES) & mask.all(axis=0) & np.isfinite(values).all(axis=0)
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
            patch_zones[index] = 0 if water_fraction >= ZONE0_PATCH_FRACTION else (1 if moraine_fraction >= ZONE1_PATCH_FRACTION else 2)
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
    supported = mask & np.isin(classes, CLEAR_SCL_CLASSES) & np.isfinite(values).all(axis=0)
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


def temporal_features(
    dt: datetime,
    prev_dt: Optional[datetime] = None,
) -> np.ndarray:
    """Cyclic day of year and revisit gap min(dt/30 days, 5); no monotonic time."""
    doy = dt.timetuple().tm_yday
    rad = 2.0 * math.pi * (doy - 1) / 365.25
    sin_doy = math.sin(rad)
    cos_doy = math.cos(rad)
    
    delta_days = 0.0
    if prev_dt is not None:
        delta_days = max(0.0, (dt - prev_dt).total_seconds() / 86400.0)
    norm_delta = min(delta_days / 30.0, 5.0)  # normalized gap
    
    return np.array([sin_doy, cos_doy, norm_delta], dtype=np.float32)
