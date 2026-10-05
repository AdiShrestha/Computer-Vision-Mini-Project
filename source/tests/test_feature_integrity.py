"""Feature integrity, algebraic transformations, and missingness propagation tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Synthetic spectra, radar backscatters, and weather values in this file verify
algebraic correctness, linear power averaging, physical bounds, and missingness
masks in offline unit tests only. They do not represent real observations and
must never be used to support scientific claims.
"""
from __future__ import annotations
import dataclasses
import math
import numpy as np
import pytest

from sentinel_gl.features import (
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    WINDOW_DAYS,
    observed_array,
    compute_normalized_difference,
    compute_ndwi,
    compute_mndwi,
    compute_ndsi,
    linear_power_from_db,
    db_from_linear_power,
    sar_linear_power_mean_db,
    sar_linear_power_variance,
    sar_cross_polarization_ratio_db,
    convert_kelvin_to_celsius,
    StaticTopography,
    MultiModalPanel,
    FeatureExtractionPipeline,
)


def test_optical_index_algebraic_correctness():
    # 1. Normalized difference basics
    assert compute_normalized_difference(1.0, 0.0) == 1.0
    assert compute_normalized_difference(0.0, 1.0) == -1.0
    assert compute_normalized_difference(0.5, 0.5) == 0.0
    assert compute_normalized_difference(0.0, 0.0) == 0.0

    # 2. NDWI = (Green - NIR) / (Green + NIR)
    # Clear water typically has high green, low NIR -> positive NDWI
    assert compute_ndwi(0.30, 0.10) == pytest.approx(0.20 / 0.40)
    # Dense vegetation has low green, high NIR -> negative NDWI
    assert compute_ndwi(0.05, 0.45) == pytest.approx(-0.40 / 0.50)

    # 3. MNDWI = (Green - SWIR) / (Green + SWIR)
    assert compute_mndwi(0.25, 0.05) == pytest.approx(0.20 / 0.30)

    # 4. NDSI = (Green - SWIR) / (Green + SWIR)
    assert compute_ndsi(0.40, 0.10) == pytest.approx(0.30 / 0.50)

    # 5. Array vectorization
    g = np.array([0.2, 0.4, 0.0])
    n = np.array([0.1, 0.4, 0.0])
    res = compute_ndwi(g, n)
    assert np.allclose(res, [(0.2 - 0.1) / 0.3, 0.0, 0.0])


def test_sar_linear_power_averaging_and_conversions():
    # 1. Roundtrip dB <-> linear power
    assert linear_power_from_db(0.0) == pytest.approx(1.0)
    assert linear_power_from_db(10.0) == pytest.approx(10.0)
    assert linear_power_from_db(-20.0) == pytest.approx(0.01)

    assert db_from_linear_power(1.0) == pytest.approx(0.0)
    assert db_from_linear_power(10.0) == pytest.approx(10.0)
    assert db_from_linear_power(0.01) == pytest.approx(-20.0)

    # 2. Linear power averaging vs naive arithmetic mean in dB
    # Consider two pixels: -10 dB (power 0.1) and -20 dB (power 0.01)
    # Naive arithmetic mean in dB would be: (-10 + -20) / 2 = -15.0 dB
    # Correct physical linear mean: (0.1 + 0.01) / 2 = 0.055 -> 10 * log10(0.055) = -12.59637 dB
    pixels = [-10.0, -20.0]
    expected_linear_mean_db = 10.0 * math.log10(0.055)
    computed_mean_db = sar_linear_power_mean_db(pixels)
    assert computed_mean_db == pytest.approx(expected_linear_mean_db, abs=1e-5)
    assert abs(computed_mean_db - (-15.0)) > 2.0  # Must NOT be naive arithmetic mean!

    # 3. Spatial variance of linear power
    var_p = sar_linear_power_variance(pixels)
    expected_var = float(np.var([0.1, 0.01]))
    assert var_p == pytest.approx(expected_var, abs=1e-7)
    assert var_p >= 0.0

    # 4. Cross polarization ratio: VH - VV in dB
    cross = sar_cross_polarization_ratio_db(vh_db=-24.0, vv_db=-16.0)
    assert cross == -8.0


def test_kelvin_to_celsius_conversion_exactness():
    # Temperature in Kelvin (> 100.0) converts to Celsius
    assert convert_kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert convert_kelvin_to_celsius(300.0) == pytest.approx(26.85)
    assert convert_kelvin_to_celsius(250.0) == pytest.approx(-23.15)

    # Already in Celsius: untouched (never double-converted)
    assert convert_kelvin_to_celsius(12.5) == 12.5
    assert convert_kelvin_to_celsius(-8.0) == -8.0


def test_static_topography_immutability():
    topo = StaticTopography(elevation_m=5200.0, moraine_slope_deg=28.5, catchment_area_km2=15.2)
    assert topo.elevation_m == 5200.0
    assert topo.moraine_slope_deg == 28.5
    assert topo.catchment_area_km2 == 15.2

    # Immutability: frozen dataclass cannot be modified
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        topo.elevation_m = 4800.0  # type: ignore

    # Bounds validation on invalid parameters
    with pytest.raises(ValueError, match="invalid lake elevation"):
        StaticTopography(elevation_m=-50.0, moraine_slope_deg=20.0, catchment_area_km2=10.0)
    with pytest.raises(ValueError, match="invalid moraine slope"):
        StaticTopography(elevation_m=5000.0, moraine_slope_deg=100.0, catchment_area_km2=10.0)
    with pytest.raises(ValueError, match="invalid catchment area"):
        StaticTopography(elevation_m=5000.0, moraine_slope_deg=20.0, catchment_area_km2=-2.0)


def test_explicit_nan_missingness_mask_propagation():
    pipeline = FeatureExtractionPipeline(cloud_cover_threshold_pct=30.0, window_days=180)
    topo = StaticTopography(elevation_m=5200.0, moraine_slope_deg=25.0, catchment_area_km2=12.0)

    # Window from 2023-01-01 to 2023-06-29 (180 days)
    start_date = "2023-01-01"

    optical_obs = [
        # Clear optical pass on 2023-01-10 (Day 9)
        {
            "record_id": "S2-CLEAR",
            "acquisition_timestamp": "2023-01-10T04:30:00Z",
            "cloud_cover_percentage": 5.0,
            "metadata": {
                "bands": {"B03": 0.25, "B08": 0.05, "B11": 0.02},
                "lake_area_km2": 1.35,
            },
            "sha256": "hash_clear",
        },
        # Cloudy optical pass on 2023-01-20 (Day 19) -> 65% cloud -> MUST BE CLOUD MASKED (NaN)
        {
            "record_id": "S2-CLOUDY",
            "acquisition_timestamp": "2023-01-20T04:30:00Z",
            "cloud_cover_percentage": 65.0,
            "metadata": {
                "bands": {"B03": 0.50, "B08": 0.50, "B11": 0.40},
                "lake_area_km2": 1.35,
            },
            "sha256": "hash_cloudy",
        },
    ]

    sar_obs = [
        # SAR pass on 2023-01-15 (Day 14)
        {
            "record_id": "S1-PASS",
            "acquisition_timestamp": "2023-01-15T12:00:00Z",
            "metadata": {
                "vv_db_pixels": [-16.0, -17.0, -15.5],
                "vh_db_pixels": [-24.0, -25.0, -23.5],
            },
            "sha256": "hash_sar",
        }
    ]

    era5_obs = [
        # Daily ERA5 on 2023-01-01 to 2023-01-03
        {
            "record_id": "ERA5-01",
            "acquisition_timestamp": "2023-01-01T12:00:00Z",
            "metadata": {"t2m_mean_c": -5.2, "precipitation_total_mm": 1.4, "t2m_anomaly": -0.8},
            "sha256": "hash_era5_01",
        },
        {
            "record_id": "ERA5-02",
            "acquisition_timestamp": "2023-01-02T12:00:00Z",
            "metadata": {"t2m_mean_c": -4.8, "precipitation_total_mm": 0.0, "t2m_anomaly": -0.4},
            "sha256": "hash_era5_02",
        },
    ]

    panel = pipeline.extract_panel(
        lake_id="SGL-001",
        window_id="WIN-TEST-01",
        start_date=start_date,
        static_topography=topo,
        optical_observations=optical_obs,
        sar_observations=sar_obs,
        era5_observations=era5_obs,
    )

    assert panel.values.shape == (180, 11)
    assert panel.mask.shape == (180, 11)

    # 1. Strict observed_array consistency: where mask=False, value must be NaN; where mask=True, value must be finite
    assert np.all(np.isnan(panel.values) == ~panel.mask)
    assert np.all(np.isfinite(panel.values) == panel.mask)

    # 2. Clear optical pass (Day 9 = 2023-01-10) is valid and populated
    day_clear = 9
    assert np.all(panel.mask[day_clear, 0:4] == True)
    assert panel.values[day_clear, 0] == 1.35  # area
    assert panel.values[day_clear, 1] > 0.0   # ndwi
    assert panel.values[day_clear, 2] > 0.0   # mndwi

    # 3. Cloudy optical pass (Day 19 = 2023-01-20) is NaN with mask=False (ZERO IMPUTATION FORBIDDEN)
    day_cloudy = 19
    assert np.all(panel.mask[day_cloudy, 0:4] == False)
    assert np.all(np.isnan(panel.values[day_cloudy, 0:4]))

    # 4. SAR pass (Day 14 = 2023-01-15) is valid and populated
    day_sar = 14
    assert np.all(panel.mask[day_sar, 4:8] == True)
    assert panel.values[day_sar, 4] < 0.0  # vv_db_mean
    assert panel.values[day_sar, 5] < 0.0  # vh_db_mean
    assert panel.values[day_sar, 7] >= 0.0 # vv_spatial_variance

    # 5. Days without SAR (e.g. Day 0) must be NaN with mask=False
    assert np.all(panel.mask[0, 4:8] == False)
    assert np.all(np.isnan(panel.values[0, 4:8]))

    # 6. Provenance hashes recorded
    assert "optical_S2-CLEAR" in panel.provenance_hashes
    assert "sar_S1-PASS" in panel.provenance_hashes
    assert "era5_ERA5-01" in panel.provenance_hashes

    # 7. Validates physical domain intervals
    panel.validate_domains()


def test_physical_domain_violations_detected():
    topo = StaticTopography(elevation_m=5200.0, moraine_slope_deg=25.0, catchment_area_km2=12.0)
    dates = tuple(f"2023-01-{i:02d}" for i in range(1, 181))
    values = np.full((180, 11), np.nan, dtype=np.float64)
    mask = np.zeros((180, 11), dtype=np.bool_)

    # Insert an unphysical area: 15.0 km2 (> 5.0)
    values[0, 0] = 15.0
    mask[0, 0] = True

    panel = MultiModalPanel(
        lake_id="TEST-01",
        window_id="W-01",
        start_date="2023-01-01",
        end_date="2023-06-29",
        dates=dates,
        values=values,
        mask=mask,
        static_metadata=topo,
    )

    with pytest.raises(ValueError, match="Physical domain violation on channel optical_lake_area_km2"):
        panel.validate_domains()


def test_panel_immutability_under_future_appends():
    pipeline = FeatureExtractionPipeline()
    topo = StaticTopography(elevation_m=5200.0, moraine_slope_deg=25.0, catchment_area_km2=12.0)

    start_date = "2023-01-01"

    # Historical observations inside [2023-01-01, 2023-06-29]
    hist_optical = [
        {
            "record_id": "S2-01",
            "acquisition_timestamp": "2023-02-01T04:30:00Z",
            "metadata": {"lake_area_km2": 1.30, "ndwi_mean": 0.45, "mndwi_mean": 0.50, "ndsi_mean": 0.35},
        }
    ]
    hist_sar = [
        {
            "record_id": "S1-01",
            "acquisition_timestamp": "2023-02-05T12:00:00Z",
            "metadata": {"sar_vv_db_mean": -16.0, "sar_vh_db_mean": -24.0, "sar_vv_spatial_variance": 0.002},
        }
    ]

    panel_initial = pipeline.extract_panel(
        lake_id="SGL-001",
        window_id="WIN-IMMUTABLE-01",
        start_date=start_date,
        static_topography=topo,
        optical_observations=hist_optical,
        sar_observations=hist_sar,
    )

    # Future observations arrived months later (e.g. July, August 2023)
    future_optical = list(hist_optical) + [
        {
            "record_id": "S2-FUTURE",
            "acquisition_timestamp": "2023-08-15T04:30:00Z",
            "metadata": {"lake_area_km2": 1.45, "ndwi_mean": 0.55, "mndwi_mean": 0.60, "ndsi_mean": 0.40},
        }
    ]
    future_sar = list(hist_sar) + [
        {
            "record_id": "S1-FUTURE",
            "acquisition_timestamp": "2023-09-01T12:00:00Z",
            "metadata": {"sar_vv_db_mean": -14.0, "sar_vh_db_mean": -21.0, "sar_vv_spatial_variance": 0.005},
        }
    ]

    panel_re_extracted = pipeline.extract_panel(
        lake_id="SGL-001",
        window_id="WIN-IMMUTABLE-01",
        start_date=start_date,
        static_topography=topo,
        optical_observations=future_optical,
        sar_observations=future_sar,
    )

    # Verify past feature panel remains 100% byte-for-byte identical
    assert np.array_equal(panel_initial.mask, panel_re_extracted.mask)
    # Check NaN and numerical equality
    assert np.array_equal(np.isnan(panel_initial.values), np.isnan(panel_re_extracted.values))
    valid_idx = panel_initial.mask
    assert np.allclose(panel_initial.values[valid_idx], panel_re_extracted.values[valid_idx])
