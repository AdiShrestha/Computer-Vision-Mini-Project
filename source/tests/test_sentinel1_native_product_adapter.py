import numpy as np
import pytest

from runners.run_wp03_native_sar import manifest_ipf_identity
from sentinel_gl.ingestion.sentinel1_calibration import (
    CalibrationError,
    calibrate_native_noise_window,
    calibrate_native_sigma0_window,
    parse_native_sentinel1_product,
    strict_power_to_db,
)


def _product_xml(polarization="VV"):
    annotation = f"""<product><missionId>S1A</missionId><productType>GRD</productType><mode>IW</mode>
      <polarisation>{polarization}</polarisation><numberOfLines>4</numberOfLines><numberOfSamples>4</numberOfSamples>
      <startTime>2023-09-01T12:14:08.715352</startTime><stopTime>2023-09-01T12:14:33.714796</stopTime>
      <processingInformation><orbitDataFileUsed>true</orbitDataFileUsed><thermalNoiseCorrectionPerformed>false</thermalNoiseCorrectionPerformed></processingInformation>
      <geolocationGridPoint><line>0</line><pixel>0</pixel><latitude>27</latitude><longitude>88</longitude><height>0</height></geolocationGridPoint></product>""".encode()
    calibration = f"""<calibration><missionId>S1A</missionId><productType>GRD</productType><mode>IW</mode>
      <polarisation>{polarization}</polarisation><calibrationVector><line>0</line><pixel count="2">0 2</pixel>
      <sigmaNought count="2">2 4</sigmaNought></calibrationVector>
      <calibrationVector><line>2</line><pixel count="2">0 2</pixel><sigmaNought count="2">4 8</sigmaNought></calibrationVector></calibration>""".encode()
    noise = f"""<noise><missionId>S1A</missionId><productType>GRD</productType><mode>IW</mode>
      <polarisation>{polarization}</polarisation><noiseRangeVector><line>0</line><pixel count="2">0 2</pixel>
      <noiseRangeLut count="2">0 1</noiseRangeLut></noiseRangeVector>
      <noiseRangeVector><line>2</line><pixel count="3">0 1 2</pixel><noiseRangeLut count="3">0 1 2</noiseRangeLut></noiseRangeVector>
      <noiseAzimuthVector><swath>IW1</swath><firstAzimuthLine>0</firstAzimuthLine><lastAzimuthLine>3</lastAzimuthLine>
      <firstRangeSample>0</firstRangeSample><lastRangeSample>2</lastRangeSample><line count="2">0 2</line>
      <noiseAzimuthLut count="2">1 2</noiseAzimuthLut></noiseAzimuthVector></noise>""".encode()
    return annotation, calibration, noise


def test_parser_retains_variable_noise_sample_grids_and_source_flags():
    parsed = parse_native_sentinel1_product(*_product_xml())
    assert parsed["thermal_noise_correction_applied"] is False
    assert parsed["processing_flags"]["orbitDataFileUsed"] == "true"
    assert parsed["geolocation_grid_point_count"] == 1
    assert [len(row[1]) for row in parsed["noise_range_vectors"]] == [2, 3]
    assert parsed["noise_azimuth_blocks"][0]["swath"] == "IW1"


def test_parser_rejects_cross_polarization_xml_join():
    annotation, calibration, noise = _product_xml()
    with pytest.raises(CalibrationError, match="identities differ"):
        parse_native_sentinel1_product(annotation, calibration, _product_xml("VH")[2])


def test_native_sigma0_window_uses_absolute_coordinates_and_no_extrapolation():
    vectors = parse_native_sentinel1_product(*_product_xml())["calibration_vectors"]
    dn = np.array([[4, 8, 12], [6, 10, 20]], dtype=np.uint16)
    sigma, valid = calibrate_native_sigma0_window(dn, vectors, line_offset=0, sample_offset=0)
    assert valid.all()
    np.testing.assert_allclose(sigma[0], [4, 64 / 9, 9], rtol=1e-6)
    np.testing.assert_allclose(sigma[1], [4, 100 / 20.25, 400 / 36], rtol=1e-6)
    shifted, shifted_valid = calibrate_native_sigma0_window(dn[:1, :2], vectors,
                                                            line_offset=2, sample_offset=2)
    assert shifted_valid[0, 0]
    assert not shifted_valid[0, 1]


def test_decibels_are_null_for_zero_negative_or_nonfinite_power():
    values = np.array([[1.0, 0.0, -1.0, np.inf]])
    db, positive = strict_power_to_db(values, np.ones(values.shape, dtype=bool))
    assert positive.tolist() == [[True, False, False, False]]
    assert db[0, 0] == pytest.approx(0.0)
    assert np.isnan(db[0, 1:]).all()


def test_manifest_accepts_consistent_repeated_ipf_scopes_and_ignores_other_software():
    xml = b'<manifest><software name="Sentinel-1 IPF" version="003.61"/><software name="Other" version="bad"/><software name="Sentinel-1 IPF" version="003.61"/></manifest>'
    assert manifest_ipf_identity(xml) == ("003.61", 2)


@pytest.mark.parametrize("xml, message", [
    (b'<manifest><software name="Sentinel-1 IPF" version="003.61"/><software name="Sentinel-1 IPF" version="003.62"/></manifest>', "conflicting"),
    (b'<manifest><software name="Sentinel-1 IPF" version=""/></manifest>', "malformed"),
    (b'<manifest><software name="Sentinel-1 IPF" version="61"/></manifest>', "malformed"),
    (b'<manifest><software name="Other" version="003.61"/></manifest>', "no Sentinel-1 IPF"),
])
def test_manifest_rejects_invalid_or_conflicting_ipf_declarations(xml, message):
    with pytest.raises(ValueError, match=message):
        manifest_ipf_identity(xml)


def test_actual_shape_noise_product_arithmetic_retains_signed_power_and_nesz():
    parsed = parse_native_sentinel1_product(*_product_xml())
    dn = np.ones((2, 3), dtype=np.uint16)
    raw, raw_valid, nesz, signed, noise_valid = calibrate_native_noise_window(
        dn, parsed["calibration_vectors"], parsed["noise_range_vectors"],
        parsed["noise_azimuth_blocks"], line_offset=0, sample_offset=0,
        thermal_noise_correction_applied=parsed["thermal_noise_correction_applied"])
    assert raw_valid.all() and noise_valid.all()
    np.testing.assert_allclose(raw[0], [0.25, 1 / 9, 1 / 16], rtol=1e-6)
    np.testing.assert_allclose(nesz[0], [0, 0.5 / 9, 1 / 16], rtol=1e-6)
    assert signed[0, 0] == pytest.approx(0.25)
    assert signed[1, 1] < 0
    db, positive = strict_power_to_db(signed, noise_valid)
    assert positive[1, 1] == 0 and np.isnan(db[1, 1])
