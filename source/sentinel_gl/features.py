"""Explicit missingness, fit scope and trailing-window identity."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping
import numpy as np
from .integrity import digest
from .contracts import instance_ids


def observed_array(values, valid):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError("measurements must be real numeric values, not strings or booleans")
    values = values.astype(np.float64)
    valid = np.asarray(valid)
    if values.ndim != 2 or not values.size or valid.shape != values.shape or valid.dtype != np.bool_:
        raise ValueError("values and boolean valid mask must be matching nonempty T,C matrices")
    if np.any(valid & ~np.isfinite(values)) or np.any(~valid & ~np.isnan(values)):
        raise ValueError("valid entries must be finite; invalid entries must be NaN")
    return values, valid


@dataclass(frozen=True)
class FitScope:
    allowed_lake_ids: tuple[str, ...]
    forbidden_lake_ids: tuple[str, ...]

    def __post_init__(self):
        for ids in (self.allowed_lake_ids, self.forbidden_lake_ids):
            if not isinstance(ids, tuple):
                raise ValueError("fit and holdout IDs must be nonempty unique tuples")
            instance_ids(ids)
        if set(self.allowed_lake_ids) & set(self.forbidden_lake_ids):
            raise ValueError("fit lakes overlap final evaluation lakes")

    def check(self, lake_ids):
        ids = instance_ids(tuple(lake_ids.keys()) if isinstance(lake_ids, Mapping) else lake_ids)
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(self.allowed_lake_ids):
            raise ValueError("fit data include undeclared or duplicate lake IDs")


class FeatureNormalizer:
    """Fit observed-element z scores once; missing inputs become zero plus masks.

    Constant observed channels use scale one and are recorded. Entirely absent
    channels are an error, requiring an explicit schema amendment. This class
    checks declared IDs, not provider truth or an external split manifest.
    """
    def __init__(self, scope: FitScope):
        self._scope = scope
        self._state = None

    @property
    def scope(self):
        return self._scope

    @property
    def state(self):
        return self._state

    def fit(self, panels: Mapping[str, tuple[np.ndarray, np.ndarray]]):
        if self.state is not None:
            raise RuntimeError("normalizer already fitted; create a new transform for an amendment")
        self.scope.check(panels)
        arrays = [observed_array(*panels[key])[0] for key in sorted(panels)]
        if len({a.shape[1] for a in arrays}) != 1:
            raise ValueError("inconsistent channel count")
        stack = np.concatenate(arrays)
        if np.any(np.isfinite(stack).sum(axis=0) == 0):
            raise ValueError("entirely missing training channel")
        # Scale before calculating variance: squaring tiny physical values can
        # otherwise underflow and falsely label a nonconstant channel constant.
        magnitude = np.nanmax(np.abs(stack), axis=0)
        factor = np.where(magnitude == 0, 1., magnitude)
        scaled = stack / factor
        mean = np.nanmean(scaled, axis=0) * factor
        std = np.nanstd(scaled, axis=0) * factor
        constant = np.nanmin(stack, axis=0) == np.nanmax(stack, axis=0)
        if np.any((std == 0) & ~constant):
            raise ValueError("nonconstant channel variance cannot be represented")
        scale = np.where(constant, 1., std)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("nonfinite fitted transform")
        self._state = MappingProxyType({"mean": tuple(mean.tolist()), "scale": tuple(scale.tolist()),
            "constant_channels": tuple(np.flatnonzero(constant).tolist()),
            "fit_lake_ids": tuple(sorted(panels))})
        return self

    @property
    def state_hash(self):
        if self.state is None:
            raise RuntimeError("normalizer has not been fitted")
        return digest(dict(self.state))

    def transform(self, values, valid):
        if self.state is None:
            raise RuntimeError("normalizer has not been fitted")
        values, valid = observed_array(values, valid)
        if values.shape[1] != len(self.state["mean"]):
            raise ValueError("channel count differs from fitted transform")
        out = (values - self.state["mean"]) / self.state["scale"]
        out = np.where(valid, out, 0.).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("nonfinite normalized input")
        return out, valid.copy()


@dataclass(frozen=True)
class Window:
    start_index: int
    stop_index: int
    start_date: str
    latest_observation_date: str
    decision_date: str

    @property
    def earliest_available_date(self):
        """Earliest permissible date, not proof an alarm was actually executed."""
        return self.decision_date


def trailing_windows(dates, available_dates, length: int, stride: int):
    """One row per date; available_dates records latest source availability per row.

    Windows include indices [start, stop). The legacy-named decision_date is the latest of the
    final observation date and all source availability dates in the window.
    It is an earliest permissible date, not an actual execution timestamp.
    Daily gaps in the calendar must be inserted explicitly before this stage.
    """
    if type(length) is not int or length < 2 or type(stride) is not int or stride < 1:
        raise ValueError("length >= 2 and stride >= 1 must be integers")
    ds = [date.fromisoformat(str(x)) for x in dates]
    av = [date.fromisoformat(str(x)) for x in available_dates]
    if not ds or len(ds) != len(av):
        raise ValueError("observation and availability dates must be equal and nonempty")
    if len(ds) < length:
        raise ValueError("insufficient calendar support for a complete trailing window")
    if any((b-a).days != 1 for a, b in zip(ds, ds[1:])):
        raise ValueError("daily panel requires ordered consecutive calendar dates")
    if any(a < d for d, a in zip(ds, av)):
        raise ValueError("availability cannot precede observation")
    windows = []
    for start in range(0, len(ds)-length+1, stride):
        stop = start+length
        windows.append(Window(start, stop, ds[start].isoformat(), ds[stop-1].isoformat(),
                              max(ds[stop-1], *av[start:stop]).isoformat()))
    return windows
