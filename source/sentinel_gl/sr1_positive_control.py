"""Constructed positive control: simulated partial lake drainage.

The perturbation replaces the clear pixels of a contiguous part of the
inventory lake zone (zone 0 pixels, taken from east to west, i.e. from the
dam end) with the scene's own median clear background (zone 2) reflectance in
each band, and labels them SCL 5 (bare soil). Cloudy or invalid pixels are not
touched. Fractions are fixed in the protocol amendment. Every perturbed array
carries ``origin = constructed_positive_control`` and is never written into an
observational manifest.
"""
from __future__ import annotations

import numpy as np

ORIGIN = "constructed_positive_control"
FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def inject_drainage(reflectance: np.ndarray, valid: np.ndarray, scl: np.ndarray,
                    zone_pixels: np.ndarray, fraction: float) -> tuple[dict | None, dict]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    values = np.asarray(reflectance, dtype=np.float64)
    good = np.asarray(valid, bool).all(axis=0) & np.isin(scl, [4, 5, 6]) & np.isfinite(values).all(axis=0)
    lake = np.asarray(zone_pixels) == 0
    background = (np.asarray(zone_pixels) == 2) & good
    if not background.any():
        return None, {"status": "NOT_APPLICABLE_NO_CLEAR_BACKGROUND"}
    rows, cols = np.nonzero(lake)
    order = np.lexsort((rows, -cols))  # east (largest column) first, then north to south
    n_select = int(round(fraction * len(order)))
    chosen = np.zeros_like(lake)
    chosen[rows[order[:n_select]], cols[order[:n_select]]] = True
    replace = chosen & good
    if not replace.any():
        return None, {"status": "NOT_APPLICABLE_NO_CLEAR_LAKE_PIXEL_IN_SELECTION"}
    fill = np.array([np.median(values[b][background]) for b in range(values.shape[0])])
    out = values.copy()
    out[:, replace] = fill[:, None]
    new_scl = np.asarray(scl).copy()
    new_scl[replace] = 5
    info = {"status": "APPLIED", "origin": ORIGIN, "fraction": float(fraction),
            "selected_lake_pixels": int(chosen.sum()), "replaced_clear_pixels": int(replace.sum()),
            "lake_pixels": int(lake.sum()), "clear_lake_pixels": int((lake & good).sum()),
            "fill_reflectance": fill.tolist()}
    return {"reflectance": out, "valid": np.asarray(valid, bool).copy(), "scl": new_scl, "origin": ORIGIN}, info
