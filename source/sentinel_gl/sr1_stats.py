"""Dependence-aware uncertainty for scene-level reconstruction comparisons.

Scenes of one lake are serially dependent, so intervals use a moving-block
bootstrap over the chronologically ordered scenes of each site; sites are
resampled separately (stratified) and their sums combined. The block length
in scenes is chosen so a block spans about ``block_days`` days at the site's
median revisit gap within the analysed subset. Intervals are percentile
intervals. With very few scenes they are wide and unstable, and fewer than
``min_scenes`` scenes yields no interval (status NOT_ESTIMABLE).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np


def block_length(dates: list[datetime], block_days: float) -> int:
    if len(dates) < 2:
        return 1
    ordered = sorted(dates)
    gaps = [(b - a).total_seconds() / 86400.0 for a, b in zip(ordered, ordered[1:])]
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return 1
    return int(max(1, min(len(dates), math.ceil(block_days / median_gap))))


def _resample_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
    idx = np.concatenate([np.arange(s, s + length) for s in starts])
    return idx[:n]


def paired_ratio_difference(strata: dict[str, list[dict]], *, a: str, b: str, block_days: float = 30.0,
                            n_boot: int = 4000, alpha: float = 0.05, seed: int = 0,
                            min_scenes: int = 3) -> dict:
    """Bootstrap the difference of pooled MSEs, sum(sse_a)/sum(n) - sum(sse_b)/sum(n).

    ``strata`` maps a site to its chronologically ordered scene records, each
    with ``dt``, ``n`` and ``sse_<name>`` for both predictors.
    """
    sites = {s: sorted(rows, key=lambda r: r["dt"]) for s, rows in strata.items() if rows}
    total = sum(len(v) for v in sites.values())
    n_all = sum(r["n"] for v in sites.values() for r in v)

    def stat(groups):
        sa = sum(r[f"sse_{a}"] for g in groups for r in g)
        sb = sum(r[f"sse_{b}"] for g in groups for r in g)
        nn = sum(r["n"] for g in groups for r in g)
        return (sa - sb) / nn if nn else math.nan

    point = stat(list(sites.values())) if n_all else None
    lengths = {s: block_length([r["dt"] for r in v], block_days) for s, v in sites.items()}
    result = {"estimate": point, "n_scenes": total, "n_entries": n_all, "block_lengths": lengths,
              "block_days": block_days, "n_boot": n_boot, "seed": seed, "alpha": alpha,
              "method": "stratified moving-block bootstrap over scenes; percentile interval"}
    if total < min_scenes or not n_all:
        return result | {"lower": None, "upper": None, "status": f"NOT_ESTIMABLE_LT_{min_scenes}_SCENES"}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        groups = []
        for s, rows in sites.items():
            idx = _resample_indices(len(rows), lengths[s], rng)
            groups.append([rows[i] for i in idx])
        value = stat(groups)
        if math.isfinite(value):
            draws.append(value)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return result | {"lower": float(lo), "upper": float(hi), "status": "ESTIMATED",
                     "fraction_draws_a_lower": float(np.mean(np.asarray(draws) < 0))}


def sample_sd(values) -> float | None:
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    return float(np.std(x, ddof=1)) if len(x) >= 2 else None
