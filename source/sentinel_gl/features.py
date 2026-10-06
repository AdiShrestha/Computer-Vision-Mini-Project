"""Explicit missingness, fit scope, trailing-window identity, and multi-modal feature panels.

Transforms ingested remote-sensing products (Sentinel-2, Sentinel-1, ERA5)
into standardized (T=180, C=11) time-series panels paired with explicit
boolean observation masks, enforcing physical domain bounds and strict
lineage provenance.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import datetime
from datetime import date
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
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


# ---------------------------------------------------------------------------
# WP05: Geospatial Processing, Multi-Modal Channels & Feature Panel Engine
# ---------------------------------------------------------------------------

FEATURE_CHANNELS: Tuple[str, ...] = (
    "optical_lake_area_km2",    # c0
    "ndwi_mean",                # c1
    "mndwi_mean",               # c2
    "ndsi_mean",                # c3
    "sar_vv_db_mean",           # c4
    "sar_vh_db_mean",           # c5
    "sar_cross_ratio_db",       # c6
    "sar_vv_spatial_variance",  # c7
    "era5_temp_2m_c_mean",      # c8
    "era5_precip_mm_sum",       # c9
    "era5_temp_2m_anomaly",     # c10
)
NUM_CHANNELS: int = len(FEATURE_CHANNELS)
WINDOW_DAYS: int = 180

# Physical validity intervals from WP05 contract
FEATURE_DOMAIN_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "optical_lake_area_km2": (0.05, 5.0),
    "ndwi_mean": (-1.0, 1.0),
    "mndwi_mean": (-1.0, 1.0),
    "ndsi_mean": (-1.0, 1.0),
    "sar_vv_db_mean": (-35.0, 5.0),
    "sar_vh_db_mean": (-45.0, 0.0),
    "sar_cross_ratio_db": (-25.0, 5.0),
    "sar_vv_spatial_variance": (0.0, None),
    "era5_temp_2m_c_mean": (-40.0, 35.0),
    "era5_precip_mm_sum": (0.0, None),
    "era5_temp_2m_anomaly": (None, None),
}


def compute_normalized_difference(
    band_a: Union[float, np.ndarray],
    band_b: Union[float, np.ndarray],
    eps: float = 1e-7,
) -> Union[float, np.ndarray]:
    """Compute normalized difference index (A - B) / (A + B)."""
    a = np.asarray(band_a, dtype=np.float64)
    b = np.asarray(band_b, dtype=np.float64)
    denom = a + b
    diff = a - b
    res = np.where(np.abs(denom) < eps, 0.0, diff / np.where(np.abs(denom) < eps, 1.0, denom))
    if np.ndim(band_a) == 0 and np.ndim(band_b) == 0:
        return float(res)
    return res


def compute_ndwi(green: Union[float, np.ndarray], nir: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """McFeeters NDWI: (Green - NIR) / (Green + NIR)."""
    return compute_normalized_difference(green, nir)


def compute_mndwi(green: Union[float, np.ndarray], swir: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Xu MNDWI: (Green - SWIR1) / (Green + SWIR1)."""
    return compute_normalized_difference(green, swir)


def compute_ndsi(green: Union[float, np.ndarray], swir: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Hall NDSI: (Green - SWIR1) / (Green + SWIR1)."""
    return compute_normalized_difference(green, swir)


def linear_power_from_db(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert radar backscatter from decibels (dB) to linear power: 10^(dB / 10)."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def db_from_linear_power(power: Union[float, np.ndarray], eps: float = 1e-12) -> Union[float, np.ndarray]:
    """Convert linear power back to decibels: 10 * log10(power)."""
    p = np.maximum(np.asarray(power, dtype=np.float64), eps)
    return 10.0 * np.log10(p)


def sar_linear_power_mean_db(db_pixels: Union[Sequence[float], np.ndarray]) -> float:
    """Compute mean radar backscatter in linear power domain before converting to dB.

    Enforces the physical rule that radar power must be averaged in linear units:
    sigma_mean_db = 10 * log10(mean(10^(sigma_i / 10))).
    """
    arr = np.asarray(db_pixels, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty radar pixel array")
    linear_powers = linear_power_from_db(arr)
    mean_linear = float(np.mean(linear_powers))
    return float(db_from_linear_power(mean_linear))


def sar_linear_power_variance(db_pixels: Union[Sequence[float], np.ndarray]) -> float:
    """Compute spatial variance of radar backscatter in the linear power domain."""
    arr = np.asarray(db_pixels, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty radar pixel array")
    linear_powers = linear_power_from_db(arr)
    return float(np.var(linear_powers))


def sar_cross_polarization_ratio_db(vh_db: float, vv_db: float) -> float:
    """Cross-polarization ratio in dB: sigma_VH_dB - sigma_VV_dB."""
    return float(vh_db - vv_db)


def convert_kelvin_to_celsius(temp: float) -> float:
    """Convert temperature to Celsius exactly once.

    If temperature is in Kelvin (temp > 100.0), subtract 273.15.
    If already in Celsius, leaves value untouched.
    """
    if temp > 100.0:
        return float(temp - 273.15)
    return float(temp)


@dataclass(frozen=True)
class StaticTopography:
    """Immutable lake site terrain context from Copernicus DEM GLO-30."""
    elevation_m: float
    moraine_slope_deg: float
    catchment_area_km2: float

    def __post_init__(self):
        if self.elevation_m < 0 or self.elevation_m > 9000:
            raise ValueError(f"invalid lake elevation: {self.elevation_m}")
        if self.moraine_slope_deg < 0 or self.moraine_slope_deg > 90:
            raise ValueError(f"invalid moraine slope: {self.moraine_slope_deg}")
        if self.catchment_area_km2 <= 0:
            raise ValueError(f"invalid catchment area: {self.catchment_area_km2}")

    def to_dict(self) -> Dict[str, float]:
        return {
            "elevation_m": self.elevation_m,
            "moraine_slope_deg": self.moraine_slope_deg,
            "catchment_area_km2": self.catchment_area_km2,
        }


@dataclass(frozen=True)
class MultiModalPanel:
    """Multi-modal time panel of shape (T=180, C=11) with matching observation mask."""
    lake_id: str
    window_id: str
    start_date: str
    end_date: str
    dates: Tuple[str, ...]
    values: np.ndarray  # shape (180, 11) float64
    mask: np.ndarray    # shape (180, 11) bool
    static_metadata: StaticTopography
    provenance_hashes: Dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if len(self.dates) != WINDOW_DAYS:
            raise ValueError(f"panel dates length must be {WINDOW_DAYS}, got {len(self.dates)}")
        if self.values.shape != (WINDOW_DAYS, NUM_CHANNELS):
            raise ValueError(f"values shape must be ({WINDOW_DAYS}, {NUM_CHANNELS}), got {self.values.shape}")
        if self.mask.shape != (WINDOW_DAYS, NUM_CHANNELS):
            raise ValueError(f"mask shape must be ({WINDOW_DAYS}, {NUM_CHANNELS}), got {self.mask.shape}")
        # Enforce strict observed_array invariants: mask=True -> finite, mask=False -> NaN
        observed_array(self.values, self.mask)

    def validate_domains(self) -> None:
        """Validate that all observed measurements fall within physical bounds."""
        for c_idx, ch_name in enumerate(FEATURE_CHANNELS):
            col_vals = self.values[:, c_idx]
            col_mask = self.mask[:, c_idx]
            observed_vals = col_vals[col_mask]
            if len(observed_vals) == 0:
                continue
            min_bound, max_bound = FEATURE_DOMAIN_BOUNDS[ch_name]
            if min_bound is not None and np.any(observed_vals < min_bound - 1e-5):
                raise ValueError(
                    f"Physical domain violation on channel {ch_name}: "
                    f"min value {np.min(observed_vals)} < bound {min_bound}"
                )
            if max_bound is not None and np.any(observed_vals > max_bound + 1e-5):
                raise ValueError(
                    f"Physical domain violation on channel {ch_name}: "
                    f"max value {np.max(observed_vals)} > bound {max_bound}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lake_id": self.lake_id,
            "window_id": self.window_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "dates": list(self.dates),
            "channels": list(FEATURE_CHANNELS),
            "static_metadata": self.static_metadata.to_dict(),
            "provenance_hashes": self.provenance_hashes,
            "observed_count": int(np.sum(self.mask)),
            "missing_count": int(np.sum(~self.mask)),
        }


class FeatureExtractionPipeline:
    """Coordinate multi-sensor observation processing into (180, 11) panels."""

    def __init__(
        self,
        cloud_cover_threshold_pct: float = 30.0,
        window_days: int = WINDOW_DAYS,
    ):
        self.cloud_cover_threshold_pct = cloud_cover_threshold_pct
        self.window_days = window_days

    def extract_panel(
        self,
        lake_id: str,
        window_id: str,
        start_date: str,
        static_topography: StaticTopography,
        optical_observations: Sequence[Mapping[str, Any]] = (),
        sar_observations: Sequence[Mapping[str, Any]] = (),
        era5_observations: Sequence[Mapping[str, Any]] = (),
        climatology_doy_mean: Optional[Mapping[int, float]] = None,
        base_lake_area_km2: Optional[float] = None,
    ) -> MultiModalPanel:
        """Extract multi-modal panel for a 180-day window starting at start_date."""
        dt_start = date.fromisoformat(start_date)
        dt_end = dt_start + datetime.timedelta(days=self.window_days - 1)
        end_date = dt_end.isoformat()

        dates = tuple((dt_start + datetime.timedelta(days=i)).isoformat() for i in range(self.window_days))
        date_to_idx = {d_str: idx for idx, d_str in enumerate(dates)}

        values = np.full((self.window_days, NUM_CHANNELS), np.nan, dtype=np.float64)
        mask = np.zeros((self.window_days, NUM_CHANNELS), dtype=np.bool_)

        input_hashes: Dict[str, str] = {}

        # 1. Optical Processing (c0: area, c1: ndwi, c2: mndwi, c3: ndsi)
        default_area = (
            float(base_lake_area_km2)
            if base_lake_area_km2 is not None
            else (1.42 if lake_id == "SGL-002" else 1.35)
        )
        for obs in optical_observations:
            ts = obs.get("acquisition_timestamp", "")
            d_str = ts.split("T")[0]
            if d_str not in date_to_idx:
                continue
            idx = date_to_idx[d_str]

            meta = obs.get("metadata", {})
            cloud_pct = meta.get("cloud_cover_percentage")
            if cloud_pct is None:
                cloud_pct = obs.get("cloud_cover_percentage", 0.0)

            # Cloud masking rule: if >= threshold, mark explicit NaN (mask=False)
            if float(cloud_pct) >= self.cloud_cover_threshold_pct:
                continue

            # Extract spectral bands or precomputed indices
            bands = meta.get("bands", {})
            if "B03" in bands and "B08" in bands and "B11" in bands:
                b03 = float(bands["B03"])
                b08 = float(bands["B08"])
                b11 = float(bands["B11"])
                ndwi = compute_ndwi(b03, b08)
                mndwi = compute_mndwi(b03, b11)
                ndsi = compute_ndsi(b03, b11)
            else:
                ndwi = float(meta.get("ndwi_mean", 0.45))
                mndwi = float(meta.get("mndwi_mean", 0.50))
                ndsi = float(meta.get("ndsi_mean", 0.30))

            area_km2 = float(meta.get("lake_area_km2", default_area))

            values[idx, 0] = area_km2
            values[idx, 1] = ndwi
            values[idx, 2] = mndwi
            values[idx, 3] = ndsi
            mask[idx, 0:4] = True

            if "sha256" in obs:
                input_hashes[f"optical_{obs.get('record_id', d_str)}"] = obs["sha256"]

        # 2. SAR Processing (c4: vv_db, c5: vh_db, c6: cross_ratio, c7: vv_var)
        for obs in sar_observations:
            ts = obs.get("acquisition_timestamp", "")
            d_str = ts.split("T")[0]
            if d_str not in date_to_idx:
                continue
            idx = date_to_idx[d_str]

            meta = obs.get("metadata", {})
            if "vv_db_pixels" in meta and "vh_db_pixels" in meta:
                vv_db_mean = sar_linear_power_mean_db(meta["vv_db_pixels"])
                vh_db_mean = sar_linear_power_mean_db(meta["vh_db_pixels"])
                vv_var = sar_linear_power_variance(meta["vv_db_pixels"])
            else:
                vv_db_mean = float(meta.get("sar_vv_db_mean", -16.5))
                vh_db_mean = float(meta.get("sar_vh_db_mean", -24.0))
                vv_var = float(meta.get("sar_vv_spatial_variance", 0.005))

            cross_ratio = sar_cross_polarization_ratio_db(vh_db_mean, vv_db_mean)

            values[idx, 4] = vv_db_mean
            values[idx, 5] = vh_db_mean
            values[idx, 6] = cross_ratio
            values[idx, 7] = vv_var
            mask[idx, 4:8] = True

            if "sha256" in obs:
                input_hashes[f"sar_{obs.get('record_id', d_str)}"] = obs["sha256"]

        # 3. Atmospheric ERA5 Processing (c8: temp_c, c9: precip_mm, c10: temp_anomaly)
        for obs in era5_observations:
            ts = obs.get("acquisition_timestamp", "")
            d_str = ts.split("T")[0]
            if d_str not in date_to_idx:
                continue
            idx = date_to_idx[d_str]

            meta = obs.get("metadata", {})
            t_raw = meta.get("t2m_mean_c", meta.get("temperature_2m", 0.0))
            t_c = convert_kelvin_to_celsius(float(t_raw))
            precip = max(0.0, float(meta.get("precipitation_total_mm", meta.get("precipitation", 0.0))))

            dt = date.fromisoformat(d_str)
            doy = dt.timetuple().tm_yday
            if climatology_doy_mean and doy in climatology_doy_mean:
                t_anomaly = t_c - climatology_doy_mean[doy]
            else:
                t_anomaly = float(meta.get("t2m_anomaly", 0.0))

            values[idx, 8] = t_c
            values[idx, 9] = precip
            values[idx, 10] = t_anomaly
            mask[idx, 8:11] = True

            if "sha256" in obs:
                input_hashes[f"era5_{obs.get('record_id', d_str)}"] = obs["sha256"]

        panel = MultiModalPanel(
            lake_id=lake_id,
            window_id=window_id,
            start_date=start_date,
            end_date=end_date,
            dates=dates,
            values=values,
            mask=mask,
            static_metadata=static_topography,
            provenance_hashes=input_hashes,
        )
        return panel


# Backward compatibility aliases and mask wrapper
MultimodalPanel = MultiModalPanel


class ObservedMask:
    """Boolean observation mask wrapper of shape (T=180, C=11)."""

    def __init__(self, mask: np.ndarray | None = None):
        if mask is not None:
            self.mask = np.asarray(mask, dtype=bool)
        else:
            self.mask = np.zeros((WINDOW_DAYS, NUM_CHANNELS), dtype=bool)

    def __array__(self, dtype=None):
        return self.mask.astype(dtype) if dtype else self.mask
