"""
Test Suite for Fail-Closed Acquisition Behavior.
Ensures network, authentication, and provider errors raise explicit exceptions
and never return generated or synthetic fallback values.
"""

import importlib
import pytest
from unittest.mock import patch
from source.data.acquisition.common import (
    AcquisitionBlockedError,
    AuthenticationError,
    NetworkReachabilityError,
    check_endpoint_reachability,
)
from source.data.acquisition.sentinel1 import Sentinel1AcquisitionAdapter
from source.data.acquisition.sentinel2 import Sentinel2AcquisitionAdapter
from source.data.acquisition.modis import ModisLSTAcquisitionAdapter
from source.data.acquisition.era5 import ERA5AcquisitionAdapter


def test_endpoint_unreachable_raises_network_error():
    with patch("source.data.acquisition.common.socket.create_connection", side_effect=OSError("Network down")):
        assert check_endpoint_reachability("invalid.domain.xyz", 443) is False
        adapter = Sentinel1AcquisitionAdapter()
        with pytest.raises(NetworkReachabilityError) as excinfo:
            adapter.initialize()
        assert "BLOCKED — HUMAN ACTION REQUIRED" in str(excinfo.value)


def test_s1_auth_failure_blocks():
    with patch("source.data.acquisition.sentinel1.verify_fail_closed_preconditions", return_value=None):
        with patch.dict("sys.modules", {"ee": None}):
            adapter = Sentinel1AcquisitionAdapter()
            with pytest.raises(AuthenticationError) as excinfo:
                adapter.initialize()
            assert "Authentic provider observations are required" in str(excinfo.value)


def test_s2_query_failure_blocks():
    with patch("source.data.acquisition.sentinel2.verify_fail_closed_preconditions", return_value=None):
        adapter = Sentinel2AcquisitionAdapter()
        adapter._ee = None
        with patch.object(adapter, "initialize", side_effect=AuthenticationError("No credentials")):
            with pytest.raises(AuthenticationError):
                adapter.acquire_lake_series("LK_01", {"type": "Point", "coordinates": [88.0, 27.5]}, "2020-01-01", "2020-12-31")


def test_era5_missing_credentials_blocks():
    with patch("source.data.acquisition.era5.verify_fail_closed_preconditions", return_value=None):
        from pathlib import Path
        adapter = ERA5AcquisitionAdapter(key_file=Path("/tmp/nonexistent_cdsapi_key"))
        with pytest.raises(AuthenticationError) as excinfo:
            adapter.initialize()
        assert "Please create .cdsapirc" in str(excinfo.value)


@pytest.mark.parametrize(
    "module_name",
    [
        "acquire_sentinel1",
        "acquire_sentinel2",
        "acquire_modis",
        "acquire_era5",
        "acquire_itslive",
        "acquire_landsat",
    ],
)
def test_legacy_entry_points_block_without_emitting_observations(module_name):
    module = importlib.import_module(f"source.data.acquisition.{module_name}")
    with pytest.raises(AcquisitionBlockedError) as excinfo:
        module.acquire(lake_id="LK_01")
    assert "BLOCKED — HUMAN ACTION REQUIRED" in str(excinfo.value)
