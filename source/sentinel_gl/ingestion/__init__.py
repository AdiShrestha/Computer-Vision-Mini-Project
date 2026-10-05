"""Observational remote-sensing and reanalysis data ingestion package."""
from .provenance import AcquisitionRecord, compute_bytes_sha256, compute_file_sha256
from .sentinel2 import Sentinel2Ingestor
from .sentinel1 import Sentinel1Ingestor
from .era5 import ERA5Ingestor
from .pipeline import PilotIngestionPipeline

__all__ = [
    "AcquisitionRecord",
    "compute_bytes_sha256",
    "compute_file_sha256",
    "Sentinel2Ingestor",
    "Sentinel1Ingestor",
    "ERA5Ingestor",
    "PilotIngestionPipeline",
]
