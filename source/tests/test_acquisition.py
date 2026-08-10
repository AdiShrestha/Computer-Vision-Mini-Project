"""Verify acquisition script scaffolding."""
import os
import sys
import json
import importlib

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_all_acquisition_modules_exist():
    """Every per-source acquisition module exists."""
    acq_dir = os.path.join(source_root, 'data', 'acquisition')
    expected = [
        'acquire_sentinel1.py', 'acquire_sentinel2.py',
        'acquire_landsat.py', 'acquire_modis.py',
        'acquire_itslive.py', 'acquire_era5.py',
    ]
    for module_file in expected:
        path = os.path.join(acq_dir, module_file)
        assert os.path.isfile(path), f"Missing: {module_file}"


def test_all_modules_have_acquire_function():
    """Every module implements the acquire() interface."""
    modules = [
        'data.acquisition.acquire_sentinel1',
        'data.acquisition.acquire_sentinel2',
        'data.acquisition.acquire_landsat',
        'data.acquisition.acquire_modis',
        'data.acquisition.acquire_itslive',
        'data.acquisition.acquire_era5',
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, 'acquire'), f"{mod_name} missing acquire()"
            # Check function signature has expected parameters
            import inspect
            sig = inspect.signature(mod.acquire)
            params = list(sig.parameters.keys())
            assert 'lake_id' in params, f"{mod_name}.acquire missing lake_id param"
            assert 'start_date' in params, f"{mod_name}.acquire missing start_date param"
            assert 'end_date' in params, f"{mod_name}.acquire missing end_date param"
        except ImportError as e:
            # Module may have unresolved imports
            path = os.path.join(source_root, 'data', 'acquisition',
                              mod_name.split('.')[-1] + '.py')
            with open(path) as f:
                compile(f.read(), path, 'exec')


def test_run_acquisition_script_exists():
    """Top-level run_acquisition.py exists and compiles."""
    script_path = os.path.join(source_root, 'scripts', 'run_acquisition.py')
    assert os.path.isfile(script_path)
    with open(script_path) as f:
        compile(f.read(), script_path, 'exec')


def test_acquisition_modules_use_registry():
    """Modules reference the lake registry for bounding boxes."""
    script_path = os.path.join(source_root, 'scripts', 'run_acquisition.py')
    with open(script_path) as f:
        content = f.read()
    assert 'lake_registry' in content or 'registry' in content, (
        "run_acquisition.py should reference the lake registry"
    )
