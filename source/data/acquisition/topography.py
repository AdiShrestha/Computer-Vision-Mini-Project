"""
Sentinel-GL — Topographic (SRTM / NASADEM) Acquisition Adapter.
Fail-closed implementation. Authentic provider observations are required.
"""

from typing import Dict, Any, Optional
import logging
from .common import verify_fail_closed_preconditions, AuthenticationError, AcquisitionBlockedError

logger = logging.getLogger("sentinel_gl.acquisition.topography")

GEE_HOST = "earthengine.googleapis.com"
DEM_COLLECTION_ID = "NASA/NASADEM_HGT/001"


class TopographyAcquisitionAdapter:
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id
        self._ee = None

    def initialize(self) -> None:
        """Initialize Earth Engine connection."""
        verify_fail_closed_preconditions("Google Earth Engine", GEE_HOST)
        try:
            import ee
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
            self._ee = ee
        except Exception as e:
            raise AuthenticationError(
                f"Failed to initialize Earth Engine for Topography acquisition: {e}. "
                f"BLOCKED — HUMAN ACTION REQUIRED. Authentic provider observations are required."
            )

    def acquire_lake_topography(
        self,
        lake_id: str,
        geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Acquire authentic elevation and slope metrics from NASADEM.
        Raises AcquisitionBlockedError on failure; never emits manufactured observations.
        """
        if self._ee is None:
            self.initialize()

        try:
            ee = self._ee
            roi = ee.Geometry.Polygon(geometry["coordinates"]) if geometry.get("type") == "Polygon" else ee.Geometry.Point(geometry["coordinates"])
            dem = ee.Image(DEM_COLLECTION_ID).select(["elevation"])
            stats = dem.reduceRegion(
                reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True),
                geometry=roi,
                scale=30,
            ).getInfo()
            return {
                "lake_id": lake_id,
                "collection": DEM_COLLECTION_ID,
                "stats": stats,
                "status": "SUCCESS",
            }
        except Exception as e:
            raise AcquisitionBlockedError(
                f"Topography acquisition failed for lake {lake_id}: {e}. "
                f"BLOCKED — HUMAN ACTION REQUIRED."
            )
