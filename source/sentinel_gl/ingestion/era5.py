"""ERA5 atmospheric reanalysis ingestion and provenance."""
from __future__ import annotations
import datetime
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from .provenance import AcquisitionRecord, compute_bytes_sha256

OPEN_METEO_ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"


class ERA5Ingestor:
    """Ingest ECMWF ERA5 reanalysis data at lake centroid coordinates."""

    def __init__(self, user_agent: str = "Sentinel-GL-Research/1.0", timeout: int = 25):
        self.user_agent = user_agent
        self.timeout = timeout

    def query_series(
        self,
        lake_id: str,
        centroid_lat: float,
        centroid_lon: float,
        start_date: str,
        end_date: str,
    ) -> List[AcquisitionRecord]:
        """Fetch verified ECMWF ERA5 hourly records and produce daily provenance records."""
        params = {
            "latitude": f"{centroid_lat:.5f}",
            "longitude": f"{centroid_lon:.5f}",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation",
        }
        url = f"{OPEN_METEO_ERA5_URL}?{urllib.parse.urlencode(params)}"
        query_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as ex:
            return [
                AcquisitionRecord(
                    record_id=f"ERA5-QUERY-ERR-{lake_id}-{start_date}",
                    lake_id=lake_id,
                    modality="weather",
                    provider_name="ECMWF ERA5 Reanalysis",
                    product_identifier=url,
                    acquisition_timestamp=f"{start_date}T00:00:00Z",
                    query_timestamp=query_time,
                    roi_geometry={"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                    status="FAILED",
                    missingness_reason=f"PROVIDER_QUERY_ERROR: {ex}",
                )
            ]

        records: List[AcquisitionRecord] = []
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precips = hourly.get("precipitation", [])

        if not times:
            records.append(
                AcquisitionRecord(
                    record_id=f"ERA5-NODATA-{lake_id}-{start_date}-{end_date}",
                    lake_id=lake_id,
                    modality="weather",
                    provider_name="ECMWF ERA5 Reanalysis",
                    product_identifier="NO_DATA",
                    acquisition_timestamp=f"{start_date}T00:00:00Z",
                    query_timestamp=query_time,
                    roi_geometry={"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                    status="NO_DATA",
                    missingness_reason="EMPTY_ERA5_SERIES",
                )
            )
            return records

        # Group by date to produce daily reanalysis summaries
        daily_groups: Dict[str, Dict[str, list]] = {}
        for t_str, temp, prec in zip(times, temps, precips):
            day_str = t_str.split("T")[0]
            if day_str not in daily_groups:
                daily_groups[day_str] = {"temps": [], "precips": []}
            if temp is not None:
                daily_groups[day_str]["temps"].append(float(temp))
            if prec is not None:
                daily_groups[day_str]["precips"].append(float(prec))

        for day_str, vals in sorted(daily_groups.items()):
            t_list = vals["temps"]
            p_list = vals["precips"]
            t_mean = sum(t_list) / len(t_list) if t_list else float("nan")
            t_min = min(t_list) if t_list else float("nan")
            t_max = max(t_list) if t_list else float("nan")
            p_tot = sum(p_list) if p_list else float("nan")

            day_payload = {
                "date": day_str,
                "t2m_mean_c": t_mean,
                "t2m_min_c": t_min,
                "t2m_max_c": t_max,
                "precipitation_total_mm": p_tot,
                "elevation_m": data.get("elevation"),
            }
            day_bytes = json.dumps(day_payload, sort_keys=True).encode("utf-8")
            day_sha = compute_bytes_sha256(day_bytes)

            records.append(
                AcquisitionRecord(
                    record_id=f"ERA5-{lake_id}-{day_str}",
                    lake_id=lake_id,
                    modality="weather",
                    provider_name="ECMWF ERA5 Reanalysis",
                    product_identifier=f"ERA5_HOURLY_{day_str}",
                    acquisition_timestamp=f"{day_str}T12:00:00Z",
                    query_timestamp=query_time,
                    roi_geometry={"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                    status="CATALOGUED",
                    sha256=day_sha,
                    storage_bytes=len(day_bytes),
                    metadata=day_payload,
                )
            )

        return records
