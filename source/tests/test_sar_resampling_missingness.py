import numpy as np

from sentinel_gl.ingestion.sentinel1_calibration import bilinear_resample_with_valid_weights


def test_partial_support_omits_nan_neighbor_and_renormalizes_valid_weights():
    source = np.array([[1.0, np.nan], [3.0, 5.0]])
    declared = np.array([[True, False], [True, True]])
    result, coverage, valid = bilinear_resample_with_valid_weights(
        source, declared, np.array([[0.5]]), np.array([[0.5]]))
    assert valid[0, 0]
    assert coverage[0, 0] == 0.75
    assert result[0, 0] == 3.0


def test_all_invalid_neighbors_remain_missing_with_zero_coverage():
    source = np.array([[np.nan, 2.0], [4.0, 6.0]])
    declared = np.zeros((2, 2), dtype=bool)
    result, coverage, valid = bilinear_resample_with_valid_weights(
        source, declared, np.array([[0.5]]), np.array([[0.5]]))
    assert not valid[0, 0]
    assert coverage[0, 0] == 0.0
    assert np.isnan(result[0, 0])


def test_exact_integer_pixel_ignores_zero_weight_nan_neighbors():
    source = np.array([[2.0, np.nan], [np.nan, np.nan]])
    declared = np.ones((2, 2), dtype=bool)
    result, coverage, valid = bilinear_resample_with_valid_weights(
        source, declared, np.array([[0.0]]), np.array([[0.0]]))
    assert valid[0, 0]
    assert coverage[0, 0] == 1.0
    assert result[0, 0] == 2.0


def test_full_valid_bilinear_arithmetic_is_unchanged():
    source = np.array([[1.0, 3.0], [5.0, 7.0]])
    result, coverage, valid = bilinear_resample_with_valid_weights(
        source, np.ones((2, 2), dtype=bool), np.array([[0.5]]), np.array([[0.5]]))
    assert valid[0, 0]
    assert coverage[0, 0] == 1.0
    assert result[0, 0] == 4.0


def test_validity_tracks_finite_float32_power_representation():
    source = np.array([[1.0e39, 2.0], [3.0, 4.0]])
    result, coverage, valid = bilinear_resample_with_valid_weights(
        source, np.ones((2, 2), dtype=bool), np.array([[0.0]]), np.array([[0.0]]))
    assert coverage[0, 0] == 1.0
    assert np.isnan(result[0, 0])
    assert not valid[0, 0]
    assert np.array_equal(valid, np.isfinite(result) & (coverage > 0))


def test_outside_source_pixel_center_support_is_missing_without_extrapolation():
    source = np.array([[1.0, 3.0], [5.0, 7.0]])
    result, coverage, valid = bilinear_resample_with_valid_weights(
        source, np.ones((2, 2), dtype=bool), np.array([[-0.01, 0.5]]), np.array([[0.5, 1.01]]))
    assert not valid.any()
    assert (coverage == 0).all()
    assert np.isnan(result).all()
