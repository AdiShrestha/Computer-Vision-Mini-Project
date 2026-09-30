"""
Sentinel-GL — MODIS Land Surface Temperature (LST) Acquisition Adapter.
Fail-closed implementation. Authentic provider observations are required.
"""

from typing import Dict, Any, Optional
import logging
from .common import verify_fail_closed_preconditions, AuthenticationError, AcquisitionBlockedError

logger = logging.getLogger("sentinel_gl.acquisition.modis")

GEE_HOST = "earthengine.googleapis.com"
COLLECTION_ID = "MODIS/061/MOD11A1"


class ModisLSTAcquisitionAdapter:
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
                f"Failed to initialize Earth Engine for MODIS LST acquisition: {e}. "
                f"BLOCKED — HUMAN ACTION REQUIRED. Authentic provider observations are required."
            )

    def acquire_lake_series(
        self,
        lake_id: str,
        geometry: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Acquire authentic MODIS LST observations.
        Raises AcquisitionBlockedError on failure; never emits manufactured observations.
        """
        if self._ee is None:
            self.initialize()

        try:
            ee = self._ee
            roi = ee.Geometry.Polygon(geometry["coordinates"]) if geometry.get("type") == "Polygon" else ee.Geometry.Point(geometry["coordinates"])
            modis_col = (
                ee.ImageCollection(COLLECTION_ID)
                .filterBounds(roi)
                .filterDate(start_date, end_date)
            )
            count = modis_col.size().getInfo()
            return {
                "lake_id": lake_id,
                "collection": COLLECTION_ID,
                "scene_count": count,
                "start_date": start_date,
                "end_date": end_date,
                "status": "SUCCESS" if count > 0 else "NO_OBSERVATIONS",
            }
        except Exception as e:
            raise AcquisitionBlockedError(
                f"MODIS query failed for lake {lake_id}: {e}. "
                f"BLOCKED — HUMAN ACTION REQUIRED."
            )
