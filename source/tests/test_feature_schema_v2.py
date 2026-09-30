"""
Test Suite for Feature Schema v2.
Validates physical metadata, channel ranges, and hash stability.
"""

import pytest
import numpy as np
from dataclasses import replace
from source.data.schemas.feature_schema_v2 import FeatureSchemaV2, FeatureChannel


EXPECTED_CHANNEL_IDS = {
    "s1_vv_backscatter",
    "s1_vh_backscatter",
    "s2_ndwi",
    "s2_mndwi",
    "s2_lake_area",
    "modis_lst_day",
    "modis_lst_night",
    "era5_2m_temp",
    "era5_total_precip",
    "era5_freezing_level",
    "topo_elevation",
    "topo_slope_surrounding",
}


def valid_matrix(schema):
    values = {
        "s1_vv_backscatter": np.array([-15.0, -12.0, -18.0]),
        "s1_vh_backscatter": np.array([-22.0, -20.0, -25.0]),
        "s2_ndwi": np.array([0.45, 0.50, 0.42]),
        "s2_mndwi": np.array([0.60, 0.65, 0.58]),
        "s2_lake_area": np.array([1.2, 1.25, 1.22]),
        "modis_lst_day": np.array([275.0, 278.0, 273.0]),
        "modis_lst_night": np.array([260.0, 262.0, 258.0]),
        "era5_2m_temp": np.array([270.0, 272.0, 269.0]),
        "era5_total_precip": np.array([0.01, 0.00, 0.05]),
        "era5_freezing_level": np.array([4500.0, 4600.0, 4400.0]),
        "topo_elevation": np.array([5200.0, 5200.0, 5200.0]),
        "topo_slope_surrounding": np.array([25.0, 25.0, 25.0]),
    }
    values.update(
        {
            channel.missingness_indicator_id: np.ones(3, dtype=bool)
            for channel in schema.channels
        }
    )
    return values


def test_default_schema_construction():
    schema = FeatureSchemaV2.default_multisensor_schema()
    assert {channel.channel_id for channel in schema.channels} == EXPECTED_CHANNEL_IDS
    assert schema.channel_count == len(schema.channels)
    assert all(channel.temporal_cadence != "static" for channel in schema.temporal_channels)
    assert all(channel.temporal_cadence == "static" for channel in schema.static_channels)
    assert all(
        channel.transform_fit_scope == "training_split_only"
        for channel in schema.channels
        if channel.requires_fitted_transform
    )


def test_schema_hash_deterministic():
    schema1 = FeatureSchemaV2.default_multisensor_schema()
    schema2 = FeatureSchemaV2.default_multisensor_schema()
    h1 = schema1.compute_schema_hash()
    h2 = schema2.compute_schema_hash()
    assert h1 == h2
    assert len(h1) == 64
    schema2.schema_version = "2.1"
    assert schema2.compute_schema_hash() != h1


def test_matrix_validation_valid():
    schema = FeatureSchemaV2.default_multisensor_schema()
    errors = schema.validate_matrix(valid_matrix(schema))
    assert len(errors) == 0


def test_matrix_validation_out_of_range():
    schema = FeatureSchemaV2.default_multisensor_schema()
    invalid_matrix = {
        "s1_vv_backscatter": np.array([50.0]),  # Invalid backscatter
        "s1_vv_backscatter__valid": np.array([True]),
    }
    errors = schema.validate_matrix(invalid_matrix)
    assert any("out of physical range" in e for e in errors)
    assert any("Missing required channel" in e for e in errors)


def test_feature_channel_rejects_invalid_physical_metadata():
    channel = FeatureSchemaV2.default_multisensor_schema().channels[0]
    with pytest.raises(ValueError, match="finite increasing"):
        replace(channel, valid_range=(5.0, -35.0))
    with pytest.raises(ValueError, match="Unsupported missing_value_strategy"):
        replace(channel, missing_value_strategy="guess_from_neighbors")
    with pytest.raises(ValueError, match="static cadence"):
        replace(channel, temporal_cadence="static")


def test_matrix_validation_enforces_shapes_and_known_keys():
    schema = FeatureSchemaV2.default_multisensor_schema()
    matrix = valid_matrix(schema)
    matrix["s1_vh_backscatter"] = np.array([-22.0, -20.0])
    matrix["s1_vh_backscatter__valid"] = np.ones(2, dtype=bool)
    matrix["legacy_channel_13"] = np.zeros(3)
    errors = schema.validate_matrix(matrix)
    assert any("does not match" in error for error in errors)
    assert any("Unexpected matrix keys" in error for error in errors)


def test_matrix_validation_preserves_missingness_observability():
    schema = FeatureSchemaV2.default_multisensor_schema()
    matrix = valid_matrix(schema)
    matrix["s2_ndwi"][1] = np.nan
    matrix["s2_ndwi__valid"][1] = False
    assert schema.validate_matrix(matrix) == []

    matrix["s2_ndwi__valid"][1] = True
    errors = schema.validate_matrix(matrix)
    assert any("marks non-finite values as valid" in error for error in errors)

    matrix["s2_ndwi"][1] = np.inf
    matrix["s2_ndwi__valid"][1] = False
    errors = schema.validate_matrix(matrix)
    assert any("must encode invalid entries as NaN" in error for error in errors)


def test_matrix_validation_returns_errors_for_nonnumeric_values():
    schema = FeatureSchemaV2.default_multisensor_schema()
    matrix = valid_matrix(schema)
    matrix["s1_vv_backscatter"] = np.array(["not-a-number"])
    matrix["s1_vv_backscatter__valid"] = np.array([True])
    errors = schema.validate_matrix(matrix)
    assert any("must contain numeric values" in error for error in errors)
