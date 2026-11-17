"""WP05 source-derived development feature tests; no values admit research use."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio

from sentinel_gl.admission import AdmissionError
from sentinel_gl import wp05_optical_features as wp05


ROOT = Path(__file__).resolve().parents[2]
WP04_BUNDLE = ROOT / "project/architecture/wp04_local_2026-10-14/run_003/bundle.v2.json"


@pytest.fixture(scope="module")
def current_features():
    context = wp05._verified_inputs(ROOT, WP04_BUNDLE)
    document = wp05.build_feature_bundle(ROOT, WP04_BUNDLE)
    return document, context


def _raster(record, layer):
    row = next(x for x in record["lineage"]["decoded_assets"] if x["layer"] == layer)
    path = ROOT / row["path"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform, ds.nodata


def _independent_stats(values, support):
    observed = np.asarray(values[support], dtype=np.float64)
    return {"count": int(observed.size), "mean": float(observed.mean()),
            "median": float(np.median(observed)), "minimum": float(observed.min()),
            "maximum": float(observed.max()), "std": float(observed.std(ddof=0))}


def test_genuine_source_arrays_independently_recompute_qa_indices_and_roi_area(current_features):
    document, _context = current_features
    assert document["research_admissible"] is False
    assert document["record_count"] == 2
    assert len({(r["site_id"], r["product_id"], r["support_id"]) for r in document["records"]}) == 2
    for record in document["records"]:
        assert record["site_exposure"] == "DEVELOPMENT_EXPOSED"
        assert record["scope_membership"]["scientific_source_memberships"] == []
        assert record["sampling_unit"]["physical_unit_id"] is None
        assert record["time_axes"]["historical_availability"]["status"] == "UNKNOWN"
        assert record["time_axes"]["product_generation"]["at"].startswith("2025-")

        b03_raw, transform, _ = _raster(record, "b03_reflectance_20m")
        b08_raw, _, _ = _raster(record, "b08_reflectance_20m")
        b11_raw, _, _ = _raster(record, "b11_reflectance_20m")
        b03, b08, b11 = (x.astype(np.float64) for x in (b03_raw, b08_raw, b11_raw))
        scl, _, scl_nodata = _raster(record, "scl_raw_20m_classes")
        scl_valid, _, _ = _raster(record, "scl_valid_mask")
        clear, _, _ = _raster(record, "scl_clear_4_5_6_mask")
        analysis, _, _ = _raster(record, "analysis_valid_mask")
        candidate, _, _ = _raster(record, "candidate_water_mask")
        b03_valid, _, _ = _raster(record, "b03_valid_mask")
        b08_valid, _, _ = _raster(record, "b08_valid_mask")
        b11_valid, _, _ = _raster(record, "b11_valid_mask")

        ndwi_denominator = b03 + b08
        mndwi_denominator = b03 + b11
        ndwi_valid = (b03_valid == 1) & (b08_valid == 1) & (ndwi_denominator != 0)
        mndwi_valid = (b03_valid == 1) & (b11_valid == 1) & (mndwi_denominator != 0)
        ndwi = np.full(b03.shape, np.nan)
        mndwi = np.full(b03.shape, np.nan)
        ndwi[ndwi_valid] = ((b03[ndwi_valid] - b08[ndwi_valid]) /
                            ndwi_denominator[ndwi_valid])
        mndwi[mndwi_valid] = ((b03[mndwi_valid] - b11[mndwi_valid]) /
                              mndwi_denominator[mndwi_valid])
        analysis_expected = ndwi_valid & mndwi_valid & (scl_valid == 1)
        qa = analysis_expected & (clear == 1)
        candidate_expected = qa & (ndwi > 0) & (mndwi > 0)
        assert np.array_equal(analysis == 1, analysis_expected)
        assert np.array_equal(candidate == 1, candidate_expected)
        assert int(qa.sum()) == record["support"]["qa_population_count"]
        assert int(candidate_expected.sum()) == record["support"]["candidate_population_count"]
        assert int(np.count_nonzero(scl_valid == 1)) == record["scl"]["valid_denominator_count"]

        classes = {r["class_id"]: r for r in record["scl"]["class_counts_and_fractions"]}
        denominator = int(np.count_nonzero(scl_valid == 1))
        for code in range(1, 12):
            count = int(np.count_nonzero((scl == code) & (scl_valid == 1)))
            assert classes[code]["count"] == count
            assert classes[code]["fraction"] == pytest.approx(count / denominator)

        area = abs(transform.a * transform.e - transform.b * transform.d)
        assert record["grid"]["pixel_area_m2"] == pytest.approx(area)
        assert record["support"]["candidate_area_m2"] == pytest.approx(int(candidate_expected.sum()) * area)
        by_channel = {c["channel_id"]: c for c in record["channels"]}
        expectations = {
            "B03": (b03, b03_valid == 1), "B08": (b08, b08_valid == 1),
            "B11": (b11, b11_valid == 1), "NDWI": (ndwi, ndwi_valid),
            "MNDWI_NDSI": (mndwi, mndwi_valid),
        }
        for channel, (values, value_valid) in expectations.items():
            direct = _independent_stats(values, value_valid & qa)
            summary = by_channel[channel]["summary"]["qa_supported"]
            assert summary["valid_count"] == direct["count"]
            assert summary["mean"] == pytest.approx(direct["mean"], abs=1e-10)
            assert summary["median"] == pytest.approx(direct["median"], abs=1e-10)
            assert summary["minimum"] == pytest.approx(direct["minimum"], abs=1e-10)
            assert summary["maximum"] == pytest.approx(direct["maximum"], abs=1e-10)
            assert summary["population_std"] == pytest.approx(direct["std"], abs=1e-10)
        assert by_channel["MNDWI_NDSI"]["aliases"] == ["MNDWI", "NDSI"]
        assert scl_nodata == 0

    south = next(r for r in document["records"] if r["site_id"] == "PILOT-SOUTH-LHONAK")
    south_b03, _, _ = _raster(south, "b03_reflectance_20m")
    south_valid, _, _ = _raster(south, "b03_valid_mask")
    assert np.any(south_b03[south_valid == 1] < 0)


def test_constructed_analytical_fixture_preserves_negative_unclipped_and_empty_semantics():
    origin = "CONSTRUCTED_ANALYTICAL_FIXTURE_ONLY"
    assert origin == "CONSTRUCTED_ANALYTICAL_FIXTURE_ONLY"
    green = np.array([-0.10, 0.10, 0.20], dtype=np.float64)
    nir = np.array([0.20, -0.10, 0.21], dtype=np.float64)
    den = green + nir
    valid = np.isfinite(den) & (den != 0)
    ratio = np.full(green.shape, np.nan)
    ratio[valid] = (green[valid] - nir[valid]) / den[valid]
    assert ratio[0] == pytest.approx(-3.0)
    assert np.isnan(ratio[1])
    assert ratio[0] < -1.0

    negative = wp05.summarize_values(green, np.array([True, False, False]), 1)
    assert negative["status"] == "AVAILABLE"
    assert negative["minimum"] == pytest.approx(-0.10)
    empty = wp05.summarize_values(green, np.zeros(3, dtype=bool), 0)
    assert empty["status"] == "EMPTY_POPULATION"
    assert empty["mean"] is None and empty["median"] is None
    assert empty["coverage_fraction"] is None


def test_constructed_candidate_area_distinguishes_observed_zero_empty_positive_and_invalid_counts():
    origin = "CONSTRUCTED_ANALYTICAL_FIXTURE_ONLY"
    assert origin == "CONSTRUCTED_ANALYTICAL_FIXTURE_ONLY"
    # A deliberately non-square, rotated affine makes a fixed 400 m² shortcut fail.
    a, b, d, e = 3.0, 1.0, 2.0, -4.0
    determinant = abs(a * e - b * d)
    assert determinant == 14.0

    observed_zero = wp05._candidate_area(qa_count=7, candidate_count=0,
                                         pixel_area_m2=determinant)
    assert observed_zero == (0.0, "AVAILABLE_WITHIN_ROI")

    absent = wp05._candidate_area(qa_count=0, candidate_count=0,
                                  pixel_area_m2=determinant)
    assert absent == (None, "EMPTY_POPULATION")
    no_candidates = wp05._channel(
        "fixture", [], "dimensionless", np.array([1., 2., 3.]), np.ones(3, dtype=bool),
        np.ones(3, dtype=bool), np.zeros(3, dtype=bool))
    assert no_candidates["summary"]["candidate_conditioned"] == {
        "status": "EMPTY_POPULATION", "valid_count": 0, "denominator_count": 0,
        "coverage_fraction": None, "mean": None, "median": None,
        "minimum": None, "maximum": None, "population_std": None,
    }

    positive = wp05._candidate_area(qa_count=5, candidate_count=3,
                                    pixel_area_m2=determinant)
    independently_calculated = 3 * abs(3.0 * -4.0 - 1.0 * 2.0)
    assert positive == (independently_calculated, "AVAILABLE_WITHIN_ROI")
    assert positive[0] == 42.0

    with pytest.raises(AdmissionError, match="candidate population count exceeds its QA support"):
        wp05._candidate_area(qa_count=2, candidate_count=3, pixel_area_m2=determinant)
    with pytest.raises(AdmissionError, match="nonnegative integer"):
        wp05._candidate_area(qa_count=-1, candidate_count=0, pixel_area_m2=determinant)


def test_schema_and_join_validation_reject_source_grid_and_channel_corruption(current_features):
    document, context = current_features
    bad = deepcopy(document)
    bad["records"][0]["product_id"] = "orphan-product"
    with pytest.raises(AdmissionError, match="product/site/time/exposure join"):
        wp05.validate_feature_bundle_document(bad, context)

    bad = deepcopy(document)
    bad["records"][0]["grid"]["pixel_area_m2"] += 1
    with pytest.raises(AdmissionError, match="affine determinant"):
        wp05.validate_feature_bundle_document(bad, context)

    bad = deepcopy(document)
    bad["records"][0]["channels"].append(deepcopy(bad["records"][0]["channels"][0]))
    with pytest.raises(AdmissionError, match="missing, duplicated or orphaned"):
        wp05.validate_feature_bundle_document(bad, context)

    bad = deepcopy(document)
    bad["records"][0]["channels"] = [c for c in bad["records"][0]["channels"]
                                        if c["channel_id"] != "B04"]
    with pytest.raises(AdmissionError, match="feature channels are missing, duplicated or orphaned"):
        wp05.validate_feature_bundle_document(bad, context)

    bad = deepcopy(document)
    bad["record_count"] += 1
    with pytest.raises(AdmissionError,
                       match="feature bundle counts differ from its source and record memberships"):
        wp05.validate_feature_bundle_document(bad, context)

    removed = deepcopy(document)
    removed["records"] = removed["records"][:1]
    removed["record_count"] = 1
    with pytest.raises(AdmissionError, match="membership is duplicated, omitted or orphaned"):
        wp05.validate_feature_bundle_document(removed, context)


@pytest.mark.parametrize("mutation,message", [
    ("qa", "independent current-source recomputation"),
    ("index", "independent current-source recomputation"),
    ("decoded_hash", "independent current-source recomputation"),
    ("offset", "independent current-source recomputation"),
    ("mask", "independent current-source recomputation"),
])
def test_independent_replay_rejects_output_mutations(current_features, mutation, message):
    original, _context = current_features
    changed = deepcopy(original)
    record = changed["records"][0]
    if mutation == "qa":
        record["support"]["qa_population_count"] += 1
    elif mutation == "index":
        next(c for c in record["channels"] if c["channel_id"] == "NDWI")["summary"]["qa_supported"]["mean"] += 0.01
    elif mutation == "decoded_hash":
        record["lineage"]["decoded_assets"][0]["sha256"] = "0" * 64
    elif mutation == "offset":
        next(c for c in record["channels"] if c["channel_id"] == "B03")["calibration"]["offset"] = 0.0
    elif mutation == "mask":
        record["candidate_rule"]["mask_sha256"] = "f" * 64
    with pytest.raises(AdmissionError, match=message):
        wp05.verify_feature_bundle(ROOT, WP04_BUNDLE, changed)


@pytest.mark.parametrize("mutation", ["unit", "offset"])
def test_constructed_calibration_metadata_corruption_fails_closed(mutation):
    tags = {"units": "dimensionless_surface_reflectance",
            "calibration_status": "MATCHED_COG_STAC_AND_NATIVE_XML",
            "native_scale": "0.0001", "cog_scale": "0.0001",
            "native_offset": "-0.1", "cog_offset": "-0.1"}
    if mutation == "unit":
        tags["units"] = "digital_number"
    else:
        tags["cog_offset"] = "0.1"
    with pytest.raises(AdmissionError, match="units/calibration|differs from native source calibration"):
        wp05._check_calibration_tags(tags, 0.0001, -0.1, "B03")
