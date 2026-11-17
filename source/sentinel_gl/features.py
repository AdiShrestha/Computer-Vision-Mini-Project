"""Numerical missingness/fit/window primitives and admitted analytical fixture grids."""
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
# Analytical feature transforms and explicitly missing panels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelDefinition:
    name: str
    modality: str
    units: str
    required_inputs: Tuple[str, ...]
    derivation: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def to_dict(self):
        return {"name": self.name, "modality": self.modality, "units": self.units,
                "required_inputs": list(self.required_inputs), "derivation": self.derivation,
                "lower_bound": self.lower_bound, "upper_bound": self.upper_bound}


@dataclass(frozen=True)
class ChannelSchema:
    schema_id: str
    version: int
    channels: Tuple[ChannelDefinition, ...]

    def __post_init__(self):
        if not self.schema_id or type(self.version) is not int or self.version < 1 or not self.channels:
            raise ValueError("channel schema requires an ID, positive version and channels")
        names = [channel.name for channel in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("channel names must be unique; duplicate derived indices are forbidden")
        if any(channel.modality not in {"optical", "sar", "weather", "static"} for channel in self.channels):
            raise ValueError("unknown channel modality")

    @property
    def channel_names(self):
        return tuple(channel.name for channel in self.channels)

    @property
    def schema_hash(self):
        return digest(self.to_dict())

    def to_dict(self):
        return {"schema_id": self.schema_id, "version": self.version,
                "channels": [channel.to_dict() for channel in self.channels]}


_CHANNELS = {
    "optical_candidate_roi_area_km2": ChannelDefinition("optical_candidate_roi_area_km2", "optical", "km2", ("water", "pixel_area_m2"), "sum(candidate_mask * pixel_area_m2) / 1e6", 0.0),
    "ndwi_mean": ChannelDefinition("ndwi_mean", "optical", "1", ("B03", "B08"), "mean finite (B03-B08)/(B03+B08)"),
    "mndwi_mean": ChannelDefinition("mndwi_mean", "optical", "1", ("B03", "B11"), "mean finite (B03-B11)/(B03+B11); algebraically identical to NDSI"),
    "sar_vv_db_mean": ChannelDefinition("sar_vv_db_mean", "sar", "dB", ("VV",), "10*log10(mean positive linear sigma0)"),
    "sar_vh_db_mean": ChannelDefinition("sar_vh_db_mean", "sar", "dB", ("VH",), "10*log10(mean positive linear sigma0)"),
    "sar_cross_ratio_db": ChannelDefinition("sar_cross_ratio_db", "sar", "dB", ("VV", "VH"), "mean VH dB - mean VV dB on paired positive support"),
    "sar_vv_linear_variance": ChannelDefinition("sar_vv_linear_variance", "sar", "sigma0^2", ("VV",), "variance of signed linear sigma0"),
    "era5_temp_2m_c_mean": ChannelDefinition("era5_temp_2m_c_mean", "weather", "degC", ("temperature",), "mean over declared spatial support"),
    "era5_precip_mm_sum": ChannelDefinition("era5_precip_mm_sum", "weather", "mm/support", ("precipitation",), "mean provider accumulation over declared support", 0.0),
    "four_band_water_candidate_area_km2": ChannelDefinition("four_band_water_candidate_area_km2", "optical", "km2", ("B03", "B04", "B08", "B11", "water", "pixel_area_m2"), "unvalidated four-band development candidate", 0.0),
    "awei_sh_mean": ChannelDefinition("awei_sh_mean", "optical", "1", ("B02", "B03", "B08", "B11", "B12"), "unvalidated AWEI_sh development feature"),
}
CHANNEL_MODALITIES = {name: definition.modality for name, definition in _CHANNELS.items()}


def channel_definition(name: str) -> ChannelDefinition:
    try:
        return _CHANNELS[name]
    except KeyError as exc:
        raise ValueError("unknown versioned channel: " + str(name)) from exc

DEFAULT_CHANNEL_SCHEMA = ChannelSchema("sentinel_gl_panel_v2", 2, tuple(
    _CHANNELS[name] for name in (
        "optical_candidate_roi_area_km2", "ndwi_mean", "mndwi_mean",
        "sar_vv_db_mean", "sar_vh_db_mean", "sar_cross_ratio_db",
        "sar_vv_linear_variance", "era5_temp_2m_c_mean", "era5_precip_mm_sum")))
OPTIONAL_CANDIDATE_SCHEMA = ChannelSchema("sentinel_gl_optional_candidates_v3", 3,
    DEFAULT_CHANNEL_SCHEMA.channels + (_CHANNELS["four_band_water_candidate_area_km2"],
                                        _CHANNELS["awei_sh_mean"]))
FEATURE_CHANNELS: Tuple[str, ...] = DEFAULT_CHANNEL_SCHEMA.channel_names
NUM_CHANNELS: int = len(FEATURE_CHANNELS)  # size of the named default schema, not a panel requirement
FEATURE_DOMAIN_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    channel.name: (channel.lower_bound, channel.upper_bound)
    for channel in DEFAULT_CHANNEL_SCHEMA.channels
    if channel.lower_bound is not None or channel.upper_bound is not None
}


def compute_normalized_difference(
    band_a: Union[float, np.ndarray],
    band_b: Union[float, np.ndarray],
    eps: float = 0.0,
) -> Union[float, np.ndarray]:
    """Compute normalized difference, leaving a zero/unsupported denominator NaN.

    A positive epsilon is an explicit sensitivity choice; no universal reflectance
    bound or threshold is implied by this numerical helper.
    """
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps < 0:
        raise ValueError("ratio support tolerance must be finite and nonnegative")
    a = np.asarray(band_a, dtype=np.float64)
    b = np.asarray(band_b, dtype=np.float64)
    denom = a + b
    diff = a - b
    res = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype=np.float64)
    np.divide(diff, denom, out=res, where=np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > eps))
    if np.ndim(band_a) == 0 and np.ndim(band_b) == 0:
        return float(res)
    return res


def compute_ndwi(green: Union[float, np.ndarray], nir: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """McFeeters NDWI: (Green - NIR) / (Green + NIR)."""
    return compute_normalized_difference(green, nir)


def compute_mndwi(green: Union[float, np.ndarray], swir: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Xu MNDWI: (Green - SWIR1) / (Green + SWIR1)."""
    return compute_normalized_difference(green, swir)


def linear_power_from_db(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert radar backscatter from decibels (dB) to linear power: 10^(dB / 10)."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def db_from_linear_power(power: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert positive finite linear power to dB; other cells remain unavailable NaN."""
    p = np.asarray(power, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p) & (p > 0)
    out[valid] = 10.0 * np.log10(p[valid])
    return float(out) if out.ndim == 0 else out


def sar_linear_power_mean_db(db_pixels: Union[Sequence[float], np.ndarray]) -> float:
    """Compute mean radar backscatter in linear power domain before converting to dB.

    Enforces the physical rule that radar power must be averaged in linear units:
    sigma_mean_db = 10 * log10(mean(10^(sigma_i / 10))).
    """
    arr = np.asarray(db_pixels, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
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


def convert_kelvin_to_celsius(temp: float, *, unit: str) -> float:
    """Convert using declared units; magnitude never establishes units."""
    if type(temp) not in (int, float) or not math.isfinite(temp):
        raise ValueError("temperature must be finite numeric data")
    if unit == "K":
        return float(temp - 273.15)
    if unit == "degC":
        return float(temp)
    raise ValueError("unknown temperature unit")


@dataclass(frozen=True)
class StaticTopography:
    """Immutable lake site terrain context from Copernicus DEM GLO-30."""
    elevation_m: float
    moraine_slope_deg: float
    catchment_area_km2: float

    def __post_init__(self):
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (self.elevation_m,self.moraine_slope_deg,self.catchment_area_km2)):
            raise ValueError("topography must be finite numeric data")
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
    """Versioned, schedule-sized panel with cell status, units and source lineage."""
    lake_id: str
    window_id: str
    start_date: str
    end_date: str
    dates: Tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray
    static_metadata: Optional[StaticTopography]
    provenance_hashes: Dict[str, str] = dataclasses.field(default_factory=dict)
    origin: str = "unverified"
    absence_reasons: Dict[str, str] = dataclasses.field(default_factory=dict)
    channel_names: Tuple[str, ...] = FEATURE_CHANNELS
    channel_schema_id: str = DEFAULT_CHANNEL_SCHEMA.schema_id
    channel_schema_version: int = DEFAULT_CHANNEL_SCHEMA.version
    channel_schema_hash: str = DEFAULT_CHANNEL_SCHEMA.schema_hash
    units: Mapping[str, str] = dataclasses.field(default_factory=dict)
    statuses: Optional[np.ndarray] = None
    lineage: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schedule_mode: str = "DECLARED_SCHEDULE_NO_EXECUTION_EVIDENCE"

    def __post_init__(self):
        from datetime import date
        if not self.dates or any(not isinstance(value, str) for value in self.dates):
            raise ValueError("panel requires an explicit nonempty schedule")
        parsed_dates = tuple(date.fromisoformat(value) for value in self.dates)
        if tuple(sorted(set(parsed_dates))) != parsed_dates:
            raise ValueError("schedule dates must be unique and increasing")
        if self.start_date != self.dates[0] or self.end_date != self.dates[-1]:
            raise ValueError("panel date bounds must match the declared schedule")
        names = tuple(self.channel_names)
        if not names or len(names) != len(set(names)) or any(name not in _CHANNELS for name in names):
            raise ValueError("panel channel names must be unique members of the versioned channel catalog")
        if type(self.channel_schema_version) is not int or self.channel_schema_version < 1:
            raise ValueError("panel channel schema version must be a positive integer")
        expected_schema_hash = digest({"schema_id": self.channel_schema_id,
            "version": self.channel_schema_version,
            "channels": [_CHANNELS[name].to_dict() for name in names]})
        if expected_schema_hash != self.channel_schema_hash:
            raise ValueError("panel channel schema ID/version/hash do not bind its declared channels")
        values, mask = observed_array(self.values, self.mask)
        expected_shape = (len(self.dates), len(names))
        if values.shape != expected_shape:
            raise ValueError(f"panel values shape must match schedule/channels {expected_shape}, got {values.shape}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "channel_names", names)
        expected_units = {name: _CHANNELS[name].units for name in names}
        if self.units and dict(self.units) != expected_units:
            raise ValueError("panel units must exactly match its versioned channel definitions")
        object.__setattr__(self, "units", expected_units)
        if self.statuses is None:
            raise ValueError("panel requires explicit per-cell statuses; missingness cannot imply a reason")
        statuses = np.asarray(self.statuses, dtype=object)
        if statuses.shape != expected_shape or any(not isinstance(value, str) or not value for value in statuses.flat):
            raise ValueError("panel requires one explicit nonempty status per schedule/channel cell")
        if np.any(mask & (statuses != "OBSERVED")) or np.any(~mask & (statuses == "OBSERVED")):
            raise ValueError("panel cell status conflicts with its explicit observation mask")
        object.__setattr__(self, "statuses", statuses)
        if not isinstance(self.lineage, Mapping):
            raise ValueError("panel lineage must be a mapping")
        if not self.schedule_mode:
            raise ValueError("schedule mode must be explicit")

    def validate_domains(self) -> None:
        """Check only explicit physical domains; spectral ratios are unbounded here."""
        for c_idx, ch_name in enumerate(self.channel_names):
            col_vals = self.values[:, c_idx]
            col_mask = self.mask[:, c_idx]
            observed_vals = col_vals[col_mask]
            if len(observed_vals) == 0:
                continue
            definition = _CHANNELS[ch_name]
            min_bound, max_bound = definition.lower_bound, definition.upper_bound
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
            "schedule_mode": self.schedule_mode,
            "channels": list(self.channel_names),
            "channel_schema_id": self.channel_schema_id,
            "channel_schema_version": self.channel_schema_version,
            "channel_schema_hash": self.channel_schema_hash,
            "units": dict(self.units),
            "statuses": self.statuses.tolist(),
            "lineage": dict(self.lineage),
            "static_metadata": self.static_metadata.to_dict() if self.static_metadata else None,
            "origin": self.origin,
            "absence_reasons": dict(self.absence_reasons),
            "provenance_hashes": self.provenance_hashes,
            "observed_count": int(np.sum(self.mask)),
            "missing_count": int(np.sum(~self.mask)),
        }


class FeatureExtractionPipeline:
    """Build fixture panels from an explicit schedule and admitted measurement bytes."""
    def __init__(self, window_days=None, *, admission=None, channel_schema=DEFAULT_CHANNEL_SCHEMA):
        if window_days is not None and (type(window_days) is not int or window_days < 1):
            raise ValueError("explicit schedule length must be a positive integer")
        if not isinstance(channel_schema, ChannelSchema):
            raise ValueError("a versioned channel schema is required")
        self.window_days = window_days
        self.admission = admission
        self.channel_schema = channel_schema

    def extract_panel(self, lake_id, window_id, start_date=None, static_topography=None,
                      optical_observations=(), sar_observations=(), era5_observations=(),
                      climatology_doy_mean=None, base_lake_area_km2=None, *, schedule_dates=None):
        from .admission import Absence, AdmissionError, timestamp
        if climatology_doy_mean is not None or base_lake_area_km2 is not None:
            raise AdmissionError("unadmitted area/climatology context")
        if schedule_dates is not None:
            dates = tuple(schedule_dates)
            if not dates or any(not isinstance(value, str) for value in dates):
                raise ValueError("an explicit nonempty schedule_dates sequence is required")
            parsed = tuple(date.fromisoformat(value) for value in dates)
            if tuple(sorted(set(parsed))) != parsed:
                raise ValueError("schedule dates must be unique and increasing")
            if start_date is not None and date.fromisoformat(start_date) != parsed[0]:
                raise ValueError("start_date conflicts with explicit schedule")
        else:
            if self.window_days is None or start_date is None:
                raise ValueError("provide explicit schedule_dates or both start_date and schedule length")
            start = date.fromisoformat(start_date)
            dates = tuple((start + datetime.timedelta(days=i)).isoformat() for i in range(self.window_days))
        channel_names = self.channel_schema.channel_names
        channel_index = {name: index for index, name in enumerate(channel_names)}
        values = np.full((len(dates), len(channel_names)), np.nan, dtype=np.float64)
        mask = np.zeros(values.shape, dtype=bool)
        statuses = np.full(values.shape, "UNAVAILABLE_NO_ADMITTED_SUPPORT", dtype=object)
        reasons, hashes, lineage = {}, {}, {}
        first_schedule_day = date.fromisoformat(dates[0])
        consumed = set()
        for modality, records in (("optical", optical_observations), ("sar", sar_observations), ("weather", era5_observations)):
            for n, record in enumerate(records):
                key = modality + ":" + str(record.get("measurement_id", n))
                if self.admission is None:
                    reasons[key] = "NO_ADMITTED_MEASUREMENT_BYTES"
                    continue
                admitted = self.admission.measurement(record)
                if isinstance(admitted, Absence):
                    reasons[key] = admitted.reason
                    continue
                if record['lake_id'] != lake_id:
                    raise AdmissionError("measurement lake mismatch")
                if record['measurement_id'] in consumed:
                    raise AdmissionError("duplicate measurement")
                consumed.add(record['measurement_id'])
                day = timestamp(record['support_end']).date().isoformat()
                if day not in dates:
                    reasons[key] = "OUTSIDE_CLOSED_WINDOW"
                    continue
                close = timestamp(day + "T23:59:59.999999+00:00")
                if (timestamp(record['available_at']) > close or
                        timestamp(record['support_start']).date() < first_schedule_day):
                    reasons[key] = "UNAVAILABLE_AT_WINDOW_CLOSE_OR_SUPPORT_OUTSIDE_WINDOW"
                    continue
                with np.load(self.admission.artifact(record['product_id']), allow_pickle=False) as product:
                    fields = {k: product[k] for k in product.files}
                qa, roi = fields.pop('valid'), fields.pop('roi')
                if qa.dtype != np.bool_ or roi.dtype != np.bool_ or qa.ndim != 2 or qa.shape != roi.shape or not qa.size:
                    raise AdmissionError("explicit matching QA/ROI masks required")
                usable = qa & roi
                if not usable.any():
                    reasons[key] = "NO_VALID_ROI_PIXELS"
                    continue
                units = record['units']
                row = np.full(len(channel_names), np.nan, dtype=np.float64)
                row_status = np.full(len(channel_names), "UNAVAILABLE_NOT_PRODUCED_BY_MODALITY", dtype=object)

                def pixels(name, expected_unit=None, *, positive=False):
                    if name not in fields:
                        return None
                    if name not in units:
                        raise AdmissionError("measured input lacks explicit units: " + name)
                    if expected_unit is not None and units[name] != expected_unit:
                        raise AdmissionError("unexpected units for measured input: " + name)
                    array = np.asarray(fields.pop(name))
                    if array.shape != qa.shape or array.dtype.kind not in "iuf" or not np.isfinite(array[usable]).all():
                        raise AdmissionError("invalid measured pixels: " + name)
                    selected = array[usable].astype(np.float64)
                    if positive and np.any(selected <= 0):
                        raise AdmissionError("pixel area must be positive on usable support")
                    return selected

                def put(name, value, status="OBSERVED"):
                    if name in channel_index:
                        idx = channel_index[name]
                        if value is not None and np.isfinite(value):
                            row[idx], row_status[idx] = float(value), status
                        else:
                            row_status[idx] = (status if status != "OBSERVED"
                                               else "UNAVAILABLE_NONFINITE_DERIVATION")

                if modality == 'optical':
                    band_values = {name: pixels(name, "1") for name in ("B02", "B03", "B04", "B08", "B11", "B12")}
                    area_values = pixels("pixel_area_m2", "m2", positive=True)
                    water = fields.pop("water", None)
                    if water is not None and (water.dtype != np.bool_ or water.shape != qa.shape):
                        raise AdmissionError("water mask must be a matching boolean raster")
                    if water is not None and area_values is not None:
                        put("optical_candidate_roi_area_km2", float(area_values[water[usable]].sum() / 1e6))
                    else:
                        put("optical_candidate_roi_area_km2", None, "UNAVAILABLE_MASK_OR_AREA")
                    for out_name, (left, right) in {
                        "ndwi_mean": ("B03", "B08"), "mndwi_mean": ("B03", "B11")
                    }.items():
                        a, b = band_values[left], band_values[right]
                        if a is None or b is None:
                            missing = [name for name, array in ((left, a), (right, b)) if array is None]
                            put(out_name, None, "UNAVAILABLE_MISSING_" + "_".join(missing))
                            continue
                        index = compute_normalized_difference(a, b)
                        finite = np.isfinite(index)
                        put(out_name, float(np.mean(index[finite])) if finite.any() else None,
                            "OBSERVED" if finite.any() else "UNDEFINED_RATIO_SUPPORT")
                    for optional_name, required in (
                        ("four_band_water_candidate_area_km2", ("B03", "B04", "B08", "B11")),
                        ("awei_sh_mean", ("B02", "B03", "B08", "B11", "B12")),
                    ):
                        if optional_name not in channel_index:
                            continue
                        missing = [name for name in required if band_values.get(name) is None]
                        if missing:
                            put(optional_name, None, "UNAVAILABLE_MISSING_" + "_".join(missing))
                        else:
                            put(optional_name, None, "UNAVAILABLE_UNVALIDATED_DEVELOPMENT_RULE")
                    for name in tuple(fields):
                        if name not in {"valid", "roi"}:
                            raise AdmissionError("unknown optical product field: " + name)
                elif modality == 'sar':
                    unit_vv, unit_vh = units.get("VV"), units.get("VH")
                    if unit_vv not in {"dB", "sigma0"} or unit_vh != unit_vv:
                        raise AdmissionError("VV/VH must have matching declared dB or signed sigma0 units")
                    vv, vh = pixels("VV", unit_vv), pixels("VH", unit_vh)
                    if vv is None or vh is None:
                        for out_name in ("sar_vv_db_mean", "sar_vh_db_mean", "sar_cross_ratio_db", "sar_vv_linear_variance"):
                            put(out_name, None, "UNAVAILABLE_MISSING_POLARIZATION")
                    else:
                        vv_linear = linear_power_from_db(vv) if unit_vv == "dB" else vv
                        vh_linear = linear_power_from_db(vh) if unit_vh == "dB" else vh
                        vv_mean = float(np.mean(vv_linear))
                        vh_mean = float(np.mean(vh_linear))
                        vv_status = ("OBSERVED" if np.isfinite(vv_mean) and vv_mean > 0 else
                                     "UNAVAILABLE_NONPOSITIVE_POWER" if np.isfinite(vv_mean) else
                                     "UNAVAILABLE_NONFINITE_POWER")
                        vh_status = ("OBSERVED" if np.isfinite(vh_mean) and vh_mean > 0 else
                                     "UNAVAILABLE_NONPOSITIVE_POWER" if np.isfinite(vh_mean) else
                                     "UNAVAILABLE_NONFINITE_POWER")
                        put("sar_vv_db_mean", db_from_linear_power(vv_mean), vv_status)
                        put("sar_vh_db_mean", db_from_linear_power(vh_mean), vh_status)
                        put("sar_cross_ratio_db", db_from_linear_power(vh_mean) - db_from_linear_power(vv_mean)
                            if vh_mean > 0 and vv_mean > 0 else None,
                            "OBSERVED" if vh_mean > 0 and vv_mean > 0 else "UNAVAILABLE_NONPOSITIVE_POWER")
                        vv_variance = float(np.var(vv_linear))
                        put("sar_vv_linear_variance", vv_variance,
                            "OBSERVED" if np.isfinite(vv_variance) else "UNAVAILABLE_NONFINITE_POWER")
                else:
                    unit = units.get('temperature')
                    t = pixels('temperature', unit)
                    if t is None:
                        put("era5_temp_2m_c_mean", None, "UNAVAILABLE_MISSING_TEMPERATURE")
                    elif unit not in ('K','degC'):
                        raise AdmissionError("unknown temperature unit")
                    else:
                        put("era5_temp_2m_c_mean", convert_kelvin_to_celsius(float(t.mean()), unit=unit))
                    pu = units.get('precipitation')
                    precip = pixels("precipitation", pu) if pu else None
                    if precip is None:
                        put("era5_precip_mm_sum", None, "UNAVAILABLE_MISSING_PRECIPITATION")
                    elif pu not in ('m','mm'):
                        raise AdmissionError("unknown precipitation unit")
                    elif (precip < 0).any():
                        raise AdmissionError("negative precipitation")
                    else:
                        put("era5_precip_mm_sum", float(precip.mean() * (1000 if pu == 'm' else 1)))
                    for name in tuple(fields):
                        if name not in {"valid", "roi"}:
                            raise AdmissionError("unknown weather product field: " + name)
                if fields:
                    raise AdmissionError("unknown product bands")
                i = dates.index(day)
                produced = np.isfinite(row)
                if (mask[i] & produced).any():
                    raise AdmissionError("multiple contributions require a declared aggregation policy")
                values[i, produced], mask[i, produced] = row[produced], True
                for name, idx in channel_index.items():
                    if row_status[idx] == "UNAVAILABLE_NOT_PRODUCED_BY_MODALITY":
                        continue
                    if np.isfinite(row[idx]):
                        statuses[i, idx] = "OBSERVED"
                        lineage[f"{day}|{name}"] = {
                            "measurement_id": record["measurement_id"], "product_id": record["product_id"],
                            "support_start": record["support_start"], "support_end": record["support_end"],
                            "available_at": record["available_at"], "status": "OBSERVED"}
                    else:
                        statuses[i, idx] = row_status[idx]
                        reasons[f"{day}|{name}"] = row_status[idx]
                hashes[record['measurement_id']] = digest(record)
                hashes[record['product_id']] = self.admission.manifest['artifacts'][record['product_id']]['sha256']
        return MultiModalPanel(lake_id, window_id, dates[0], dates[-1], dates, values, mask,
            static_topography, hashes, self.admission.mode if self.admission else 'unverified', reasons,
            channel_names=channel_names, channel_schema_id=self.channel_schema.schema_id,
            channel_schema_version=self.channel_schema.version,
            channel_schema_hash=self.channel_schema.schema_hash,
            units={name: _CHANNELS[name].units for name in channel_names}, statuses=statuses,
            lineage=lineage, schedule_mode="RETROSPECTIVE_AS_OF_SIMULATION")


class ObservedMask:
    """Boolean observation mask wrapper for an explicitly declared shape."""

    def __init__(self, mask: np.ndarray):
        self.mask = np.asarray(mask)
        if self.mask.dtype != np.bool_ or self.mask.ndim != 2:
            raise ValueError("observation mask must be an explicit boolean T,C matrix")

    def __array__(self, dtype=None):
        return self.mask.astype(dtype) if dtype else self.mask
