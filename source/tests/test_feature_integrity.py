"""Feature integrity, algebraic transformations, and missingness propagation tests.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Synthetic spectra, radar backscatters, and weather values in this file verify
algebraic correctness, linear power averaging, physical bounds, and missingness
masks in offline unit tests only. They do not represent real observations and
must never be used to support scientific claims.
"""
from __future__ import annotations
import dataclasses
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from sentinel_gl.features import (
    DEFAULT_CHANNEL_SCHEMA,
    OPTIONAL_CANDIDATE_SCHEMA,
    FEATURE_CHANNELS,
    NUM_CHANNELS,
    observed_array,
    compute_normalized_difference,
    compute_ndwi,
    compute_mndwi,
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
    assert math.isnan(compute_normalized_difference(0.0, 0.0))

    # 2. NDWI = (Green - NIR) / (Green + NIR)
    # Clear water typically has high green, low NIR -> positive NDWI
    assert compute_ndwi(0.30, 0.10) == pytest.approx(0.20 / 0.40)
    # Dense vegetation has low green, high NIR -> negative NDWI
    assert compute_ndwi(0.05, 0.45) == pytest.approx(-0.40 / 0.50)

    # 3. MNDWI = (Green - SWIR) / (Green + SWIR)
    assert compute_mndwi(0.25, 0.05) == pytest.approx(0.20 / 0.30)

    # MNDWI and NDSI use the same B03/B11 expression and are one channel.
    assert compute_mndwi(0.40, 0.10) == pytest.approx(0.30 / 0.50)
    assert FEATURE_CHANNELS.count("mndwi_mean") == 1
    assert all("ndsi" not in name for name in FEATURE_CHANNELS)
    assert compute_normalized_difference(-0.1, 0.2) == pytest.approx(-3.0)

    # 5. Array vectorization
    g = np.array([0.2, 0.4, 0.0])
    n = np.array([0.1, 0.4, 0.0])
    res = compute_ndwi(g, n)
    assert np.allclose(res[:2], [(0.2 - 0.1) / 0.3, 0.0])
    assert np.isnan(res[2])


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
    assert convert_kelvin_to_celsius(273.15,unit='K')==pytest.approx(0.)
    assert convert_kelvin_to_celsius(12.5,unit='degC')==12.5
    assert convert_kelvin_to_celsius(12.5,unit='K')==pytest.approx(-260.65)
    with pytest.raises(ValueError):convert_kelvin_to_celsius(273.15,unit='unknown')



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


def test_explicit_nan_missingness_mask_propagation(tmp_path):
    from admission_fixtures import measurement_fixture
    ctx, record = measurement_fixture(tmp_path)
    p = FeatureExtractionPipeline(2, admission=ctx).extract_panel('ANALYTICAL-SITE','fixture','2023-01-01',optical_observations=[record])
    assert np.array_equal(np.isnan(p.values), ~p.mask)
    assert p.values[1,1] == pytest.approx(.5)
    assert p.values[1,0] == pytest.approx(.0001)
    assert not p.mask[0].any()



def test_physical_domain_violations_detected():
    dates=("2023-01-01", "2023-01-02")
    values=np.full((len(dates),NUM_CHANNELS),np.nan); mask=np.zeros_like(values,dtype=bool)
    values[0,0]=-1.;mask[0,0]=True
    statuses=np.full(mask.shape,"UNAVAILABLE_TEST_FIXTURE",dtype=object);statuses[mask]="OBSERVED"
    panel=MultiModalPanel('fixture','window',dates[0],dates[-1],dates,values,mask,None,
        origin='fixture',statuses=statuses)
    with pytest.raises(ValueError,match='domain violation'):panel.validate_domains()
    panel.values[0,0]=15.
    panel.validate_domains()  # No unsupported upper bound on real lake area.



def test_panel_immutability_under_future_appends(tmp_path):
    from admission_fixtures import measurement_fixture
    from sentinel_gl.admission import AdmissionContext
    ctx,r=measurement_fixture(tmp_path,suffix='past')
    future,rf=measurement_fixture(tmp_path,suffix='future',day='2023-08-01')
    before=FeatureExtractionPipeline(2,admission=ctx).extract_panel('ANALYTICAL-SITE','w','2023-01-01',optical_observations=[r])
    ctx.manifest['artifacts'].update(future.manifest['artifacts']);ctx.manifest['measurements'].update(future.manifest['measurements'])
    after=FeatureExtractionPipeline(2,admission=ctx).extract_panel('ANALYTICAL-SITE','w','2023-01-01',optical_observations=[r,rf])
    assert np.array_equal(before.mask,after.mask)
    assert np.allclose(before.values,after.values,equal_nan=True)
    assert before.provenance_hashes==after.provenance_hashes



def test_optical_cloud_obscuration_masking():
    # Catalogue cloud percentages are not pixel QA or evidence of physical values.
    for percentage in (0.,65.,None):
        p=FeatureExtractionPipeline(1).extract_panel('fixture','w','2023-01-01',optical_observations=[{'metadata':{'cloud_cover_percentage':percentage,'lake_area_km2':1.35}}])
        assert not p.mask.any() and np.isnan(p.values).all()



def test_feature_input_sensitivity(tmp_path):
    from admission_fixtures import measurement_fixture
    from sentinel_gl.integrity import file_hash
    ctx,r=measurement_fixture(tmp_path)
    before=FeatureExtractionPipeline(2,admission=ctx).extract_panel('ANALYTICAL-SITE','w','2023-01-01',optical_observations=[r])
    path=ctx.artifact(r['product_id'])
    with np.load(path,allow_pickle=False) as npz:fields={k:npz[k] for k in npz.files}
    fields['B03'][0,0]=.5
    np.savez(path,**fields)
    with pytest.raises(ValueError,match='checksum'):ctx.measurement(r)
    ctx.manifest['artifacts'][r['product_id']]['sha256']=file_hash(path)
    after=FeatureExtractionPipeline(2,admission=ctx).extract_panel('ANALYTICAL-SITE','w','2023-01-01',optical_observations=[r])
    assert after.values[1,1]==pytest.approx(2/3)
    assert before.values[1,1]!=after.values[1,1]
    assert before.provenance_hashes!=after.provenance_hashes


def test_versioned_panel_uses_declared_schedule_and_missing_b04_status(tmp_path):
    from admission_fixtures import measurement_fixture
    from sentinel_gl.features import MultiModalPanel
    ctx, record = measurement_fixture(tmp_path)
    panel = FeatureExtractionPipeline(2, admission=ctx,
        channel_schema=OPTIONAL_CANDIDATE_SCHEMA).extract_panel(
            "ANALYTICAL-SITE", "candidate", "2023-01-01", optical_observations=[record])
    assert panel.values.shape == (2, len(OPTIONAL_CANDIDATE_SCHEMA.channels))
    idx = panel.channel_names.index("four_band_water_candidate_area_km2")
    assert not panel.mask[1, idx] and np.isnan(panel.values[1, idx])
    assert panel.statuses[1, idx] == "UNAVAILABLE_MISSING_B04"
    assert panel.channel_schema_id == OPTIONAL_CANDIDATE_SCHEMA.schema_id
    assert panel.channel_schema_version == OPTIONAL_CANDIDATE_SCHEMA.version
    assert panel.channel_schema_hash == OPTIONAL_CANDIDATE_SCHEMA.schema_hash

    with pytest.raises(ValueError, match="explicit per-cell statuses"):
        MultiModalPanel("fixture", "missing-status", panel.start_date, panel.end_date,
            panel.dates, panel.values, panel.mask, None, channel_names=panel.channel_names,
            channel_schema_id=panel.channel_schema_id,
            channel_schema_version=panel.channel_schema_version,
            channel_schema_hash=panel.channel_schema_hash)



def test_feature_extraction_runner_and_panel_archive(tmp_path):
    """Unsupported research inputs are rejected before any output is written."""
    from runners.run_extract_features import run_feature_extraction
    with pytest.raises(ValueError, match="BLOCKED"):
        run_feature_extraction(output_panels_path=tmp_path/"panels.npz")
    assert not list(tmp_path.iterdir())



def test_cli_runner_feature_extraction(tmp_path):
    root=Path(__file__).resolve().parents[2]
    res=subprocess.run([sys.executable,'-B',str(root/'source/runners/run_extract_features.py'),'--output-panels',str(tmp_path/'panels.npz')],capture_output=True,text=True)
    assert res.returncode==2 and 'BLOCKED' in res.stderr
    assert not list(tmp_path.iterdir())
