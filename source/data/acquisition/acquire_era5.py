"""Compatibility entry point for the retired ERA5 batch interface."""

from .common import AcquisitionBlockedError


def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",
            registry=None, config=None, output_dir=None):
    """Halt the legacy path so callers migrate to the verified adapter."""
    raise AcquisitionBlockedError(
        "The legacy ERA5 batch interface is disabled. Use "
        "ERA5AcquisitionAdapter with an authenticated CDS client. "
        "BLOCKED — HUMAN ACTION REQUIRED."
    )
