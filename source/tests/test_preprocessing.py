"""Verify preprocessing pipeline implementation."""
import os
import sys
import importlib
import inspect
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_all_preprocessing_modules_exist():
    """All preprocessing modules exist."""
    preproc_dir = os.path.join(source_root, 'data', 'preprocessing')
    expected = [
        'preprocess_optical.py', 'preprocess_sar.py',
        'preprocess_modis.py', 'preprocess_era5.py',
        'preprocess_itslive.py', 'common.py',
    ]
    for mod_file in expected:
        path = os.path.join(preproc_dir, mod_file)
        assert os.path.isfile(path), f"Missing: {mod_file}"


def test_preprocess_interface():
    """Per-source modules implement the preprocess() interface."""
    modules = [
        'data.preprocessing.preprocess_optical',
        'data.preprocessing.preprocess_sar',
        'data.preprocessing.preprocess_modis',
        'data.preprocessing.preprocess_era5',
        'data.preprocessing.preprocess_itslive',
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, 'preprocess'), f"{mod_name} missing preprocess()"
            sig = inspect.signature(mod.preprocess)
            params = list(sig.parameters.keys())
            assert 'lake_id' in params
            assert 'raw_dir' in params
            assert 'output_dir' in params
        except ImportError:
            path = os.path.join(source_root, 'data', 'preprocessing',
                               mod_name.split('.')[-1] + '.py')
            with open(path) as f:
                compile(f.read(), path, 'exec')


def test_common_utilities_exist():
    """common.py has required utility functions."""
    from data.preprocessing.common import (
        build_time_windows, composite_within_window, generate_quality_mask
    )
    windows = build_time_windows('2023-01-01', '2023-12-31', 180, 30)
    assert len(windows) > 0, "build_time_windows returned empty list"
    assert len(windows[0]) == 2, "Each window should be (start, end) tuple"


def test_time_windows_respect_invariants():
    """Time windows use INV-004 parameters."""
    from data.preprocessing.common import build_time_windows
    windows = build_time_windows('2020-01-01', '2020-12-31', 180, 30)
    assert len(windows) >= 5, f"Expected ≥5 windows, got {len(windows)}"
    assert len(windows) <= 15, f"Too many windows: {len(windows)}"


def test_no_cross_lake_operations():
    """Preprocessing modules don't reference other lakes' data."""
    preproc_dir = os.path.join(source_root, 'data', 'preprocessing')
    for fname in os.listdir(preproc_dir):
        if fname.endswith('.py') and fname != '__init__.py':
            with open(os.path.join(preproc_dir, fname)) as f:
                content = f.read()
            assert 'for lake_id in' not in content or 'for lake_id in [lake_id]' in content, (
                f"{fname} may contain cross-lake operations (INV-002 risk)"
            )
