"""Channel Registry mapping channel IDs to extraction modules and descriptions.

Defines CH-01 through CH-05, CH-07, CH-08.
Note: CH-06 (InSAR deformation) is handled separately by Contract C02-06.
"""
from typing import Dict, Tuple

CHANNEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    'CH-01': ('extract_extent', 'Lake Extent (Area)'),
    'CH-02': ('extract_spectral', 'Spectral / Turbidity'),
    'CH-03': ('extract_velocity', 'Glacier Velocity'),
    'CH-04': ('extract_temperature', 'Temperature Anomaly'),
    'CH-05': ('extract_sar', 'SAR Backscatter'),
    'CH-07': ('extract_sar', 'SAR Coherence'),
    'CH-08': ('extract_meteorological', 'Meteorological Context'),
}
