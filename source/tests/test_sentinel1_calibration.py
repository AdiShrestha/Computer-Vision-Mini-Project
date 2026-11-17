"""Numerical and schema checks for explicitly constructed SAR fixtures only."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_gl.ingestion.sentinel1_calibration import (
    CalibrationError,
    calibrate_dn,
    calibrate_native_sar_fixture,
    sample_line_lut,
    sample_surface_lut,
    validate_native_sar_metadata,
)


FLAGS = {
    "input_radiometry": "RAW_DN",
    "sigma0_calibration_applied": False,
    "thermal_noise_correction_applied": False,
    "incidence_normalization_applied": False,
    "terrain_correction_applied": False,
}


def _surface(values, *, block="IW1", line=0, pixels=(0, 1, 2), max_line_gap=1, max_sample_gap=2):
    return {"vectors": [{"block_id": block, "line": line, "pixels": list(pixels), "values": list(values)}],
            "max_line_gap": max_line_gap, "max_sample_gap": max_sample_gap}


def _fixture(dn=None, *, line_offset=0, sample_offset=0, blocks=None,
             calibration=None, noise_range=None, noise_azimuth=None, flags=None,
             noise_support=None):
    dn = np.asarray([[10, 2, 1]] if dn is None else dn)
    blocks = np.full(dn.shape, "IW1", dtype=object) if blocks is None else np.asarray(blocks, dtype=object)
    return calibrate_dn(
        dn,
        calibration or _surface([2, 2, 2]),
        line_offset=line_offset,
        sample_offset=sample_offset,
        block_ids=blocks,
        processing_flags=FLAGS if flags is None else flags,
        noise_range_lut=noise_range or _surface([4, 4, 4]),
        noise_azimuth_lut=noise_azimuth or {"vectors": [{"block_id": "IW1", "line": 0, "value": 2}]},
        noise_support=noise_support or {"max_line_gap": 1, "max_sample_gap": 2, "max_azimuth_line_gap": 1},
        origin="fixture",
    )


def test_constructed_scalar_reference_values_keep_signed_power_and_null_db():
    result = _fixture()

    # Independent scalar arithmetic: DN^2/A^2, N/A^2, (DN^2-N)/A^2.
    np.testing.assert_allclose(result.raw_sigma0, [[25.0, 1.0, 0.25]])
    np.testing.assert_allclose(result.noise_equivalent_sigma0, [[2.0, 2.0, 2.0]])
    np.testing.assert_allclose(result.signed_denoised_sigma0, [[23.0, -1.0, -1.75]])
    assert result.db_status.tolist() == [["ESTIMATED", "UNAVAILABLE_NONPOSITIVE_POWER", "UNAVAILABLE_NONPOSITIVE_POWER"]]
    assert result.denoised_db[0, 0] == pytest.approx(10 * np.log10(23))
    assert np.isnan(result.denoised_db[0, 1:]).all()
    assert result.origin == "fixture" and result.research_admissible is False


def test_surface_lut_nonconstant_bilinear_interpolation_uses_absolute_coordinates():
    lut = {"vectors": [
        {"block_id": "IW1", "line": 10, "pixels": [20, 22], "values": [2.0, 4.0]},
        {"block_id": "IW1", "line": 12, "pixels": [20, 22], "values": [4.0, 8.0]},
    ]}
    sampled = sample_surface_lut(lut["vectors"], np.array([11]), np.array([21]),
        np.array([["IW1"]], dtype=object), max_line_gap=2, max_sample_gap=2)

    # The four surrounding calibration values are 2, 4, 4, 8; linear in each
    # declared coordinate gives the independently calculated midpoint 4.5.
    assert sampled.valid.tolist() == [[True]]
    assert sampled.values[0, 0] == pytest.approx(4.5)


def test_cropped_absolute_coordinates_match_the_same_full_image_window():
    calibration = {"vectors": [
        {"block_id": "IW1", "line": 10, "pixels": [20, 21, 22], "values": [2, 3, 4]},
        {"block_id": "IW1", "line": 11, "pixels": [20, 21, 22], "values": [4, 5, 6]},
    ], "max_line_gap": 1, "max_sample_gap": 1}
    full = np.array([[9, 12, 20], [8, 15, 24]], dtype=np.uint16)
    full_blocks = np.full(full.shape, "IW1", dtype=object)
    full_result = calibrate_dn(full, calibration, line_offset=10, sample_offset=20,
        block_ids=full_blocks, processing_flags=FLAGS, origin="fixture")
    crop_result = calibrate_dn(full[1:, 1:], calibration, line_offset=11, sample_offset=21,
        block_ids=full_blocks[1:, 1:], processing_flags=FLAGS, origin="fixture")

    np.testing.assert_allclose(crop_result.raw_sigma0, full_result.raw_sigma0[1:, 1:])


def test_lut_does_not_extrapolate_or_cross_block_boundaries():
    vectors = [
        {"block_id": "IW1", "line": 5, "pixels": [10, 12], "values": [2, 2]},
        {"block_id": "IW1", "line": 7, "pixels": [10, 12], "values": [4, 4]},
        {"block_id": "IW2", "line": 5, "pixels": [10, 12], "values": [8, 8]},
    ]
    lines = np.array([4, 6, 6, 8])
    samples = np.array([11])
    blocks = np.array([["IW1"], ["IW1"], ["IW2"], ["IW1"]], dtype=object)
    sampled = sample_surface_lut(vectors, lines, samples, blocks, max_line_gap=2, max_sample_gap=2)

    assert sampled.valid[:, 0].tolist() == [False, True, False, False]
    assert sampled.values[1, 0] == pytest.approx(3.0)


def test_line_lut_interpolation_is_support_bounded_and_block_specific():
    vectors = [
        {"block_id": "IW1", "line": 10, "value": 2.0},
        {"block_id": "IW1", "line": 12, "value": 4.0},
        {"block_id": "IW2", "line": 10, "value": 8.0},
    ]
    sampled = sample_line_lut(vectors, np.array([9, 11, 11]), np.array(["IW1", "IW1", "IW2"], dtype=object),
                              max_line_gap=2)
    assert sampled.valid.tolist() == [False, True, False]
    assert sampled.values[1] == pytest.approx(3.0)


@pytest.mark.parametrize("bad_vectors", [
    [{"block_id": "IW1", "line": 0, "pixels": [0, 0], "values": [1, 2]}],
    [{"block_id": "IW1", "line": 0, "pixels": [0, 1], "values": [1]}],
    [{"block_id": "IW1", "line": 0, "pixels": [-1], "values": [1]}],
    [{"block_id": "IW1", "line": 0, "pixels": [0], "values": [float("nan")]}],
])
def test_malformed_calibration_vectors_rejected(bad_vectors):
    with pytest.raises(CalibrationError):
        sample_surface_lut(bad_vectors, np.array([0]), np.array([0]),
                           np.array([["IW1"]], dtype=object), max_line_gap=1, max_sample_gap=1)


def test_nonfinite_or_overflowed_float64_dn_square_rejected():
    with pytest.raises(CalibrationError, match="overflowed"):
        _fixture([[1e308]], calibration=_surface([2], pixels=(0,)))


def test_missing_noise_is_explicit_and_does_not_create_denoised_values():
    dn = np.array([[10, 2, 1]], dtype=np.uint16)
    result = calibrate_dn(dn, _surface([2, 2, 2]), line_offset=0, sample_offset=0,
        block_ids=np.full(dn.shape, "IW1", dtype=object), processing_flags=FLAGS, origin="fixture")
    np.testing.assert_allclose(result.raw_sigma0, [[25, 1, 0.25]])
    assert result.noise_equivalent_sigma0 is None
    assert result.signed_denoised_sigma0 is None
    assert set(result.db_status.flat) == {"UNAVAILABLE_NOISE_LUT"}


@pytest.mark.parametrize("changed", [
    {"input_radiometry": "SIGMA0"},
    {"sigma0_calibration_applied": True},
    {"thermal_noise_correction_applied": True},
    {"incidence_normalization_applied": True},
    {"terrain_correction_applied": True},
])
def test_previously_corrected_or_unknown_dn_state_rejected(changed):
    flags = dict(FLAGS, **changed)
    with pytest.raises(CalibrationError, match="prior radiometric"):
        _fixture(flags=flags)


def test_native_metadata_role_identity_digests_flags_and_offsets_are_checked():
    import hashlib

    artifact_ids = sorted(["measurement", "calibration", "noise", "annotation", "manifest"])
    roles = {role: {"artifact_id": role, "sha256": hashlib.sha256(role.encode()).hexdigest()}
             for role in artifact_ids}
    metadata = {"schema_version": 1, "product_id": "measurement", "product_type": "GRD",
        "mode": "IW", "polarization": "VV", "ipf_version": "fixture-IPF-1", "subswath": "IW1",
        "roles": roles, "rows": 3, "columns": 2, "dtype": "uint16", "line_offset": 8,
        "sample_offset": 20, "processing_flags": FLAGS, "noise_status": "SUPPORTED_FIXTURE_POLICY"}
    manifest = {"artifacts": {name: {"artifact_id": name, "sha256": roles[name]["sha256"],
        "origin": "fixture"} for name in roles}}

    assert validate_native_sar_metadata(metadata, measurement_id="measurement-record",
        product_id="measurement", artifact_manifest=manifest, expected_origin="fixture")["ipf_version"] == "fixture-IPF-1"
    bad = {**metadata, "processing_flags": dict(FLAGS, sigma0_calibration_applied=True)}
    with pytest.raises(CalibrationError, match="input must be raw DN"):
        validate_native_sar_metadata(bad, product_id="measurement", artifact_manifest=manifest)
    bad = {**metadata, "roles": {**roles, "noise": dict(roles["noise"], sha256="0" * 64)}}
    with pytest.raises(CalibrationError, match="digest"):
        validate_native_sar_metadata(bad, product_id="measurement", artifact_manifest=manifest)


def _native_fixture(noise_status="SUPPORTED_FIXTURE_POLICY"):
    import hashlib
    roles = {role: {"artifact_id": role,
                    "sha256": hashlib.sha256(role.encode()).hexdigest()}
             for role in ("measurement", "calibration", "noise", "annotation", "manifest")}
    metadata = {"schema_version": 1, "product_id": "measurement", "product_type": "GRD",
        "mode": "IW", "polarization": "VV", "ipf_version": "fixture-IPF-1", "subswath": "IW1",
        "roles": roles, "rows": 1, "columns": 3, "dtype": "uint16", "line_offset": 0,
        "sample_offset": 0, "processing_flags": FLAGS, "noise_status": noise_status}
    manifest = {"artifacts": {name: {"artifact_id": name, "sha256": ref["sha256"],
        "origin": "fixture"} for name, ref in roles.items()}}
    return metadata, manifest


def test_native_metadata_noise_status_drives_fixture_calibration_and_nulls():
    metadata, manifest = _native_fixture()
    dn = np.array([[10, 2, 1]], dtype=np.uint16)
    result = calibrate_native_sar_fixture(metadata, dn, _surface([2, 2, 2]),
        block_ids=np.full(dn.shape, "IW1", dtype=object), artifact_manifest=manifest,
        noise_range_lut=_surface([4, 4, 4]),
        noise_azimuth_lut={"vectors": [{"block_id": "IW1", "line": 0, "value": 2}]},
        noise_support={"max_line_gap": 1, "max_sample_gap": 2, "max_azimuth_line_gap": 1})
    np.testing.assert_allclose(result.raw_sigma0, [[25, 1, .25]])
    np.testing.assert_allclose(result.signed_denoised_sigma0, [[23, -1, -1.75]])
    assert result.noise_status == "SUPPORTED_FIXTURE_POLICY"
    assert result.calibration_status == "CALIBRATED_RAW_SIGMA0"
    assert result.research_admissible is False

    metadata, manifest = _native_fixture("UNAVAILABLE_MISSING")
    result = calibrate_native_sar_fixture(metadata, dn, _surface([2, 2, 2]),
        block_ids=np.full(dn.shape, "IW1", dtype=object), artifact_manifest=manifest)
    assert result.noise_equivalent_sigma0 is None and result.signed_denoised_sigma0 is None
    assert result.noise_status == "UNAVAILABLE_MISSING"
    assert set(result.db_status.flat) == {"UNAVAILABLE_NOISE_MISSING"}
    with pytest.raises(CalibrationError, match="cannot override"):
        calibrate_native_sar_fixture(metadata, dn, _surface([2, 2, 2]),
            block_ids=np.full(dn.shape, "IW1", dtype=object), artifact_manifest=manifest,
            noise_range_lut=_surface([4, 4, 4]))


def test_native_metadata_array_dimensions_and_corrected_noise_conflict_are_rejected():
    metadata, manifest = _native_fixture()
    with pytest.raises(CalibrationError, match="dimensions/dtype"):
        calibrate_native_sar_fixture(metadata, np.ones((2, 2), dtype=np.uint16),
            _surface([2, 2, 2]), block_ids=np.full((2, 2), "IW1", dtype=object),
            artifact_manifest=manifest)
    metadata["noise_status"] = "ALREADY_CORRECTED"
    with pytest.raises(CalibrationError, match="conflicts"):
        validate_native_sar_metadata(metadata, product_id="measurement", artifact_manifest=manifest)
