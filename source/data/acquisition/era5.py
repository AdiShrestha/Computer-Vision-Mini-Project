"""
Sentinel-GL — ERA5 / ERA5-Land Reanalysis Acquisition Adapter.
Fail-closed implementation. Authentic provider observations are required.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from .common import verify_fail_closed_preconditions, AuthenticationError, AcquisitionBlockedError

logger = logging.getLogger("sentinel_gl.acquisition.era5")

CDS_HOST = "cds.climate.copernicus.eu"


class ERA5AcquisitionAdapter:
    def __init__(self, key_file: Optional[Path] = None):
        self.key_file = key_file or Path.home() / ".cdsapirc"
        self._client = None

    def initialize(self) -> None:
        """Initialize CDS API client."""
        verify_fail_closed_preconditions("Copernicus CDS", CDS_HOST)
        if not self.key_file.exists():
            raise AuthenticationError(
                f"CDS credentials file '{self.key_file}' not found. "
                f"BLOCKED — HUMAN ACTION REQUIRED. Please create .cdsapirc with valid API key."
            )
        try:
            import cdsapi
            self._client = cdsapi.Client()
        except Exception as e:
            raise AuthenticationError(
                f"Failed to instantiate CDS API client: {e}. "
                f"BLOCKED — HUMAN ACTION REQUIRED. Authentic provider observations are required."
            )

    def request_lake_climate_data(
        self,
        lake_id: str,
        bbox: List[float],
        year: int,
        variables: List[str],
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Request authentic ERA5-Land reanalysis records from Copernicus Climate Data Store.
        Raises AcquisitionBlockedError on failure; never manufactures climate time series.
        """
        if self._client is None:
            self.initialize()

        try:
            request_params = {
                "format": "netcdf",
                "variable": variables,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                "area": bbox,  # North, West, South, East
            }
            self._client.retrieve("reanalysis-era5-land", request_params, str(output_path))
            return {
                "lake_id": lake_id,
                "product": "reanalysis-era5-land",
                "year": year,
                "output_file": str(output_path),
                "status": "SUCCESS",
            }
        except Exception as e:
            raise AcquisitionBlockedError(
                f"ERA5 acquisition request failed for lake {lake_id}: {e}. "
                f"BLOCKED — HUMAN ACTION REQUIRED."
            )
