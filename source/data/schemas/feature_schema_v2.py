"""
Sentinel-GL — Canonical Feature Schema v2.
Defines physical channel metadata, sensor sources, valid ranges, and split-isolated preprocessing rules.
Governed by INV-005, INV-020, INV-021, INV-022.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
import hashlib
import math
import re


CHANNEL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MISSING_VALUE_STRATEGIES = {
    "linear_interpolate",
    "seasonal_mean",
    "indicator",
    "explicit_missing",
}
FITTED_TRANSFORMS = {"robust_scaler", "min_max", "z_score"}


@dataclass(frozen=True)
class FeatureChannel:
    channel_id: str
    sensor: str
    product: str
    band_or_variable: str
    physical_quantity: str
    physical_unit: str
    valid_range: Tuple[float, float]
    missing_value_strategy: str  # validated against MISSING_VALUE_STRATEGIES
    is_temporal: bool  # True for time series, False for static terrain
    temporal_cadence: str
    requires_fitted_transform: bool  # True if requires scaler fitted on training lakes
    transform_type: Optional[str]  # "robust_scaler", "min_max", "z_score", None
    description: str

    def __post_init__(self) -> None:
        if CHANNEL_ID_RE.fullmatch(self.channel_id) is None:
            raise ValueError(f"Invalid channel_id: {self.channel_id!r}")
        for field_name in (
            "sensor",
            "product",
            "band_or_variable",
            "physical_quantity",
            "physical_unit",
            "temporal_cadence",
            "description",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.valid_range, tuple)
            or len(self.valid_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in self.valid_range)
            or any(not math.isfinite(float(value)) for value in self.valid_range)
            or self.valid_range[0] >= self.valid_range[1]
        ):
            raise ValueError("valid_range must contain two finite increasing numbers")
        if self.missing_value_strategy not in MISSING_VALUE_STRATEGIES:
            raise ValueError(
                f"Unsupported missing_value_strategy: {self.missing_value_strategy!r}"
            )
        if self.is_temporal and self.temporal_cadence == "static":
            raise ValueError("temporal channels cannot declare static cadence")
        if not self.is_temporal and self.temporal_cadence != "static":
            raise ValueError("static channels must declare static cadence")
        if self.requires_fitted_transform:
            if self.transform_type not in FITTED_TRANSFORMS:
                raise ValueError("fitted channels must declare a supported transform_type")
        elif self.transform_type is not None:
            raise ValueError("non-fitted channels must use transform_type=None")

    @property
    def transform_fit_scope(self) -> str:
        return "training_split_only" if self.requires_fitted_transform else "not_applicable"

    @property
    def missingness_indicator_id(self) -> str:
        return f"{self.channel_id}__valid"


@dataclass
class FeatureSchemaV2:
    schema_version: str = "2.0"
    channels: List[FeatureChannel] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate_schema()

    def validate_schema(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.channels, list) or not self.channels:
            raise ValueError("channels must be a non-empty list")
        if not all(isinstance(channel, FeatureChannel) for channel in self.channels):
            raise ValueError("channels must contain FeatureChannel objects")
        channel_ids = [channel.channel_id for channel in self.channels]
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("channel_id values must be unique")

    @classmethod
    def default_multisensor_schema(cls) -> "FeatureSchemaV2":
        """Construct the canonical HKH multi-sensor feature schema."""
        channels = [
            # Radar (Sentinel-1 GRD)
            FeatureChannel(
                channel_id="s1_vv_backscatter",
                sensor="Sentinel-1",
                product="COPERNICUS/S1_GRD",
                band_or_variable="VV",
                physical_quantity="Radar Backscatter",
                physical_unit="dB",
                valid_range=(-35.0, 5.0),
                missing_value_strategy="linear_interpolate",
                is_temporal=True,
                temporal_cadence="irregular_scene",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Copolarized backscatter coefficient, surface roughness / ice cover proxy",
            ),
            FeatureChannel(
                channel_id="s1_vh_backscatter",
                sensor="Sentinel-1",
                product="COPERNICUS/S1_GRD",
                band_or_variable="VH",
                physical_quantity="Radar Backscatter",
                physical_unit="dB",
                valid_range=(-40.0, 0.0),
                missing_value_strategy="linear_interpolate",
                is_temporal=True,
                temporal_cadence="irregular_scene",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Crosspolarized backscatter coefficient, volume scattering proxy",
            ),
            # Optical Water Indices (Sentinel-2 MSI)
            FeatureChannel(
                channel_id="s2_ndwi",
                sensor="Sentinel-2",
                product="COPERNICUS/S2_SR_HARMONIZED",
                band_or_variable="(B3-B8)/(B3+B8)",
                physical_quantity="Normalized Difference Water Index",
                physical_unit="dimensionless",
                valid_range=(-1.0, 1.0),
                missing_value_strategy="indicator",
                is_temporal=True,
                temporal_cadence="irregular_scene",
                requires_fitted_transform=False,
                transform_type=None,
                description="McFeeters NDWI for open water surface delineation",
            ),
            FeatureChannel(
                channel_id="s2_mndwi",
                sensor="Sentinel-2",
                product="COPERNICUS/S2_SR_HARMONIZED",
                band_or_variable="(B3-B11)/(B3+B11)",
                physical_quantity="Modified NDWI",
                physical_unit="dimensionless",
                valid_range=(-1.0, 1.0),
                missing_value_strategy="indicator",
                is_temporal=True,
                temporal_cadence="irregular_scene",
                requires_fitted_transform=False,
                transform_type=None,
                description="Xu MNDWI suppressing shadow / built / moraine noise",
            ),
            FeatureChannel(
                channel_id="s2_lake_area",
                sensor="Sentinel-2",
                product="Derived_MNDWI_Threshold",
                band_or_variable="area_km2",
                physical_quantity="Lake Surface Area",
                physical_unit="km2",
                valid_range=(0.01, 10.0),
                missing_value_strategy="linear_interpolate",
                is_temporal=True,
                temporal_cadence="irregular_scene",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Delineated lake polygon surface area",
            ),
            # Thermal (MODIS MOD11A1)
            FeatureChannel(
                channel_id="modis_lst_day",
                sensor="MODIS",
                product="MODIS/061/MOD11A1",
                band_or_variable="LST_Day_1km",
                physical_quantity="Land Surface Temperature (Day)",
                physical_unit="Kelvin",
                valid_range=(220.0, 320.0),
                missing_value_strategy="seasonal_mean",
                is_temporal=True,
                temporal_cadence="daily",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Daytime land surface temperature",
            ),
            FeatureChannel(
                channel_id="modis_lst_night",
                sensor="MODIS",
                product="MODIS/061/MOD11A1",
                band_or_variable="LST_Night_1km",
                physical_quantity="Land Surface Temperature (Night)",
                physical_unit="Kelvin",
                valid_range=(210.0, 310.0),
                missing_value_strategy="seasonal_mean",
                is_temporal=True,
                temporal_cadence="daily",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Nighttime land surface temperature",
            ),
            # Meteorology (ERA5-Land)
            FeatureChannel(
                channel_id="era5_2m_temp",
                sensor="ERA5-Land",
                product="reanalysis-era5-land",
                band_or_variable="2m_temperature",
                physical_quantity="2m Air Temperature",
                physical_unit="Kelvin",
                valid_range=(220.0, 315.0),
                missing_value_strategy="linear_interpolate",
                is_temporal=True,
                temporal_cadence="daily",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Daily mean near-surface air temperature",
            ),
            FeatureChannel(
                channel_id="era5_total_precip",
                sensor="ERA5-Land",
                product="reanalysis-era5-land",
                band_or_variable="total_precipitation",
                physical_quantity="Total Precipitation",
                physical_unit="m/day",
                valid_range=(0.0, 0.5),
                missing_value_strategy="indicator",
                is_temporal=True,
                temporal_cadence="daily",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Daily cumulative precipitation",
            ),
            FeatureChannel(
                channel_id="era5_freezing_level",
                sensor="ERA5",
                product="reanalysis-era5-single-levels",
                band_or_variable="zero_degree_level",
                physical_quantity="Freezing Level Altitude",
                physical_unit="m_asl",
                valid_range=(0.0, 7500.0),
                missing_value_strategy="linear_interpolate",
                is_temporal=True,
                temporal_cadence="provider_timestep",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Altitude of the 0°C isothermal surface",
            ),
            # Topography (NASADEM / SRTM) - Static channels
            FeatureChannel(
                channel_id="topo_elevation",
                sensor="NASADEM",
                product="NASA/NASADEM_HGT/001",
                band_or_variable="elevation",
                physical_quantity="Lake Surface Altitude",
                physical_unit="m_asl",
                valid_range=(2000.0, 7000.0),
                missing_value_strategy="explicit_missing",
                is_temporal=False,
                temporal_cadence="static",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Mean lake water surface elevation above sea level",
            ),
            FeatureChannel(
                channel_id="topo_slope_surrounding",
                sensor="NASADEM",
                product="NASA/NASADEM_HGT/001",
                band_or_variable="slope",
                physical_quantity="Surrounding Moraine Slope",
                physical_unit="degrees",
                valid_range=(0.0, 85.0),
                missing_value_strategy="explicit_missing",
                is_temporal=False,
                temporal_cadence="static",
                requires_fitted_transform=True,
                transform_type="robust_scaler",
                description="Mean terrain slope within 1km buffer of lake perimeter",
            ),
        ]
        return cls(channels=channels)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def temporal_channels(self) -> List[FeatureChannel]:
        return [c for c in self.channels if c.is_temporal]

    @property
    def static_channels(self) -> List[FeatureChannel]:
        return [c for c in self.channels if not c.is_temporal]

    def compute_schema_hash(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the schema definition."""
        self.validate_schema()
        serialized = json.dumps(
            {
                "schema_version": self.schema_version,
                "channels": [asdict(c) for c in self.channels],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_matrix(self, matrix_dict: Dict[str, Any]) -> List[str]:
        """Validate a feature matrix against the schema."""
        import numpy as np

        errors: List[str] = []
        if not isinstance(matrix_dict, dict):
            return ["Feature matrix must be a dictionary keyed by channel and validity-mask IDs"]

        expected_keys = {
            key
            for channel in self.channels
            for key in (channel.channel_id, channel.missingness_indicator_id)
        }
        unexpected = sorted(set(matrix_dict) - expected_keys)
        if unexpected:
            errors.append(f"Unexpected matrix keys: {unexpected}")

        reference_shape = None
        for channel in self.channels:
            if channel.channel_id not in matrix_dict:
                errors.append(f"Missing required channel: {channel.channel_id}")
                continue
            if channel.missingness_indicator_id not in matrix_dict:
                errors.append(
                    f"Missing required validity mask: {channel.missingness_indicator_id}"
                )
                continue
            try:
                arr = np.asarray(matrix_dict[channel.channel_id], dtype=float)
            except (TypeError, ValueError):
                errors.append(f"Channel {channel.channel_id} must contain numeric values")
                continue
            mask = np.asarray(matrix_dict[channel.missingness_indicator_id])
            if arr.ndim == 0 or arr.size == 0:
                errors.append(f"Channel {channel.channel_id} must be a non-empty array")
                continue
            if reference_shape is None:
                reference_shape = arr.shape
            elif arr.shape != reference_shape:
                errors.append(
                    f"Channel {channel.channel_id} shape {arr.shape} does not match {reference_shape}"
                )
            if mask.shape != arr.shape:
                errors.append(
                    f"Validity mask {channel.missingness_indicator_id} shape {mask.shape} does not match {arr.shape}"
                )
                continue
            if mask.dtype != np.bool_:
                errors.append(
                    f"Validity mask {channel.missingness_indicator_id} must have boolean dtype"
                )
                continue
            finite_mask = np.isfinite(arr)
            if np.any(mask & ~finite_mask):
                errors.append(
                    f"Channel {channel.channel_id} marks non-finite values as valid"
                )
            if np.any(~mask & ~np.isnan(arr)):
                errors.append(
                    f"Channel {channel.channel_id} must encode invalid entries as NaN"
                )
            low, high = channel.valid_range
            checked = arr[mask & finite_mask]
            if checked.size:
                min_val = float(np.min(checked))
                max_val = float(np.max(checked))
                if min_val < low or max_val > high:
                    errors.append(
                        f"Channel {channel.channel_id} out of physical range [{low}, {high}]: observed min={min_val}, max={max_val}"
                    )
        return errors
