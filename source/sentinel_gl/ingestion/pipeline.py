"""Pilot execution remains blocked pending independently verified registry and measurement adapters."""
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
        from ..admission import blocked
        blocked("pilot ingestion from unverified registry")
