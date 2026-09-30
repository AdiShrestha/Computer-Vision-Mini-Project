"""
Sentinel-GL Acquisition Package.
All modules enforce fail-closed external acquisition per Factory v2.2.
"""

from .common import (
    AcquisitionError,
    AcquisitionBlockedError,
    AuthenticationError,
    NetworkReachabilityError,
    RateLimitError,
    check_endpoint_reachability,
)
from .sentinel1 import Sentinel1AcquisitionAdapter
from .sentinel2 import Sentinel2AcquisitionAdapter
from .modis import ModisLSTAcquisitionAdapter
from .era5 import ERA5AcquisitionAdapter
from .topography import TopographyAcquisitionAdapter

__all__ = [
    "AcquisitionError",
    "AcquisitionBlockedError",
    "AuthenticationError",
    "NetworkReachabilityError",
    "RateLimitError",
    "check_endpoint_reachability",
    "Sentinel1AcquisitionAdapter",
    "Sentinel2AcquisitionAdapter",
    "ModisLSTAcquisitionAdapter",
    "ERA5AcquisitionAdapter",
    "TopographyAcquisitionAdapter",
]
