"""Sentinel-1 C-band IW GRDH backscatter ingestion and provenance."""
from __future__ import annotations
import datetime
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from .provenance import AcquisitionRecord, compute_bytes_sha256

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


class Sentinel1Ingestor:
    """Ingest Sentinel-1 SAR IW GRDH acquisitions from Copernicus Data Space Ecosystem."""

    def __init__(self, user_agent: str = "Sentinel-GL-Research/1.0", timeout: int = 25):
        self.user_agent = user_agent
        self.timeout = timeout

    def query_catalog(
        self,
        lake_id: str,
        centroid_lat: float,
        centroid_lon: float,
        start_date: str,
        end_date: str,
        product_type: str = "GRD",
        max_records: int = 50,
    ) -> List[AcquisitionRecord]:
        """Query Copernicus Data Space catalog for genuine Sentinel-1 passes over lake centroid."""
        point_wkt = f"POINT({centroid_lon:.5f} {centroid_lat:.5f})"
        filter_parts = [
            "Collection/Name eq 'SENTINEL-1'",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{point_wkt}')",
            f"ContentDate/Start ge {start_date}T00:00:00.000Z",
            f"ContentDate/Start le {end_date}T23:59:59.000Z",
        ]
        if product_type:
            filter_parts.append(f"contains(Name, '{product_type}')")

        filter_expr = " and ".join(filter_parts)
        params = {
            "$filter": filter_expr,
            "$top": str(max_records),
            "$orderby": "ContentDate/Start asc",
        }
        url = f"{CDSE_ODATA_URL}?{urllib.parse.urlencode(params)}"
        query_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as ex:
            return [
                AcquisitionRecord(
                    record_id=f"S1-QUERY-ERR-{lake_id}-{start_date}",
                    lake_id=lake_id,
                    modality="sar",
                    provider_name="Copernicus Data Space Ecosystem",
                    product_identifier=url,
                    acquisition_timestamp=f"{start_date}T00:00:00Z",
                    query_timestamp=query_time,
                    roi_geometry={"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                    status="FAILED",
                    missingness_reason=f"PROVIDER_QUERY_ERROR: {ex}",
                )
            ]

        records: List[AcquisitionRecord] = []
        products = data.get("value", [])
        if not products:
            records.append(
                AcquisitionRecord(
                    record_id=f"S1-NODATA-{lake_id}-{start_date}-{end_date}",
                    lake_id=lake_id,
                    modality="sar",
                    provider_name="Copernicus Data Space Ecosystem",
                    product_identifier="NO_PRODUCTS_FOUND",
                    acquisition_timestamp=f"{start_date}T00:00:00Z",
                    query_timestamp=query_time,
                    roi_geometry={"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                    status="NO_DATA",
                    missingness_reason="NO_ACQUISITIONS_IN_WINDOW",
                )
            )
            return records

        for p in products:
            prod_id = p.get("Id", "")
            prod_name = p.get("Name", "")
            acq_time = p.get("ContentDate", {}).get("Start", "")
            orbit_direction = None
            polarization = None
            for attr in p.get("Attributes", []):
                if attr.get("Name") == "orbitDirection":
                    orbit_direction = attr.get("Value")
                elif attr.get("Name") == "polarisationChannels":
                    polarization = attr.get("Value")

            record_bytes = json.dumps(p, sort_keys=True).encode("utf-8")
            record_sha = compute_bytes_sha256(record_bytes)
            rec = AcquisitionRecord(
                record_id=f"S1-{prod_id[:16]}",
                lake_id=lake_id,
                modality="sar",
                provider_name="Copernicus Data Space Ecosystem",
                product_identifier=prod_name,
                acquisition_timestamp=acq_time,
                query_timestamp=query_time,
                roi_geometry={"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                status="CATALOGUED",
                sha256=record_sha,
                storage_bytes=len(record_bytes),
                metadata={
                    "cdse_product_id": prod_id,
                    "product_name": prod_name,
                    "polarization": polarization or "VV+VH",
                    "orbit_direction": orbit_direction or "UNKNOWN",
                    "terrain_correction": "Copernicus 30m DEM",
                },
            )
            records.append(rec)

        return records
