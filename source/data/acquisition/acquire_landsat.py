"""Fail-closed Landsat compatibility entry point."""

from .common import AcquisitionBlockedError


def acquire(lake_id=None, start_date="2016-01-01", end_date="2024-10-31",
            registry=None, config=None, output_dir=None):
    """Halt until a verified Landsat export adapter is supplied."""
    raise AcquisitionBlockedError(
        "No verified Landsat export adapter is configured. "
        "Authentic provider observations are required. "
        "BLOCKED — HUMAN ACTION REQUIRED."
    )
