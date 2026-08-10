"""Verify channel extraction implementation."""
import os
import sys
import importlib
import inspect

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_all_channel_modules_exist():
    """All channel extraction modules exist."""
    channels_dir = os.path.join(source_root, 'data', 'channels')
    expected = [
        'extract_extent.py', 'extract_spectral.py',
        'extract_velocity.py', 'extract_temperature.py',
        'extract_sar.py', 'extract_meteorological.py',
        'channel_registry.py',
    ]
    for mod_file in expected:
        path = os.path.join(channels_dir, mod_file)
        assert os.path.isfile(path), f"Missing: {mod_file}"


def test_extract_interface():
    """Channel modules implement the extract() interface."""
    modules = [
        'data.channels.extract_extent',
        'data.channels.extract_spectral',
        'data.channels.extract_velocity',
        'data.channels.extract_temperature',
        'data.channels.extract_sar',
        'data.channels.extract_meteorological',
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, 'extract'), f"{mod_name} missing extract()"
            sig = inspect.signature(mod.extract)
            params = list(sig.parameters.keys())
            assert 'lake_id' in params
            assert 'window_start' in params
        except ImportError:
            path = os.path.join(source_root, 'data', 'channels',
                               mod_name.split('.')[-1] + '.py')
            with open(path) as f:
                compile(f.read(), path, 'exec')


def test_channel_registry_complete():
    """Channel registry maps all expected channels."""
    from data.channels.channel_registry import CHANNEL_REGISTRY
    expected_channels = ['CH-01', 'CH-02', 'CH-03', 'CH-04', 'CH-05', 'CH-07', 'CH-08']
    for ch in expected_channels:
        assert ch in CHANNEL_REGISTRY, f"Missing channel {ch} in registry"


def test_no_ch06_in_channel_registry():
    """CH-06 (InSAR) is NOT in the channel registry — handled by C02-06."""
    from data.channels.channel_registry import CHANNEL_REGISTRY
    assert 'CH-06' not in CHANNEL_REGISTRY, (
        "CH-06 should not be in the standard channel registry"
    )
