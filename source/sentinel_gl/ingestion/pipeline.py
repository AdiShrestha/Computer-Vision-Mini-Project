"""Pilot ingestion coordinator for South Lhonak (SGL-001) and Khangchung Tsho (SGL-002)."""
from __future__ import annotations
import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List
from .provenance import AcquisitionRecord
from .sentinel2 import Sentinel2Ingestor
from .sentinel1 import Sentinel1Ingestor
from .era5 import ERA5Ingestor


class PilotIngestionPipeline:
    """Coordinate real observational data ingestion for pilot cohort."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.s2_ingestor = Sentinel2Ingestor()
        self.s1_ingestor = Sentinel1Ingestor()
        self.era5_ingestor = ERA5Ingestor()

    def run_pilot(
        self,
        start_date: str = "2023-09-01",
        end_date: str = "2023-10-03",
    ) -> Dict[str, Any]:
        """Execute pilot observational acquisition for SGL-001 and SGL-002."""
        lake_registry_path = self.data_dir / "lake_registry.csv"
        if not lake_registry_path.exists():
            raise FileNotFoundError(f"Missing lake registry at {lake_registry_path}")

        lakes: List[Dict[str, str]] = []
        with open(lake_registry_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lakes.append(row)

        all_records: List[AcquisitionRecord] = []
        lake_summaries: Dict[str, Any] = {}

        for lake in lakes:
            lake_id = lake["lake_id"]
            name = lake["name"]
            lat = float(lake["centroid_lat"])
            lon = float(lake["centroid_lon"])

            # 1. Optical: Sentinel-2
            s2_recs = self.s2_ingestor.query_catalog(
                lake_id=lake_id,
                centroid_lat=lat,
                centroid_lon=lon,
                start_date=start_date,
                end_date=end_date,
            )
            # 2. SAR: Sentinel-1
            s1_recs = self.s1_ingestor.query_catalog(
                lake_id=lake_id,
                centroid_lat=lat,
                centroid_lon=lon,
                start_date=start_date,
                end_date=end_date,
            )
            # 3. Weather: ERA5
            era5_recs = self.era5_ingestor.query_series(
                lake_id=lake_id,
                centroid_lat=lat,
                centroid_lon=lon,
                start_date=start_date,
                end_date=end_date,
            )

            lake_records = s2_recs + s1_recs + era5_recs
            all_records.extend(lake_records)

            usable_s2 = [r for r in s2_recs if r.status in ("CATALOGUED", "HASH_VERIFIED")]
            usable_s1 = [r for r in s1_recs if r.status in ("CATALOGUED", "HASH_VERIFIED")]
            usable_era5 = [r for r in era5_recs if r.status in ("CATALOGUED", "HASH_VERIFIED")]

            lake_summaries[lake_id] = {
                "name": name,
                "role": lake["cohort_role"],
                "coordinates": [lat, lon],
                "elevation_m": int(lake["elevation_m"]),
                "observations": {
                    "sentinel2_passes": len(usable_s2),
                    "sentinel1_passes": len(usable_s1),
                    "era5_daily_records": len(usable_era5),
                },
                "products_catalogued": [r.product_identifier for r in lake_records if r.status == "CATALOGUED"],
                "total_metadata_bytes": sum(r.storage_bytes for r in lake_records),
            }

        total_storage_bytes = sum(r.storage_bytes for r in all_records)
        dossier = {
            "dossier_version": 1,
            "pilot_window": {
                "start_date": start_date,
                "end_date": end_date,
                "pre_event_cutoff": "2023-10-03T22:30:00Z",
            },
            "cohort_summary": lake_summaries,
            "data_authenticity": {
                "origin": "observational",
                "zero_synthetic_data_declaration": True,
                "providers": [
                    "Copernicus Data Space Ecosystem (Sentinel-1, Sentinel-2)",
                    "ECMWF Reanalysis v5 (ERA5)",
                ],
            },
            "storage_footprint_bytes": total_storage_bytes,
            "records_count": len(all_records),
            "record_provenance": [r.to_dict() for r in all_records],
        }

        dossier_path = self.data_dir / "pilot_dossier.json"
        with open(dossier_path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

        return dossier
