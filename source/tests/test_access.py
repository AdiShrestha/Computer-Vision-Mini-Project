"""Verify data source access testing infrastructure exists."""
import os
import sys

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_verify_access_script_exists():
    """verify_access.py exists and is syntactically valid Python."""
    script_path = os.path.join(source_root, 'scripts', 'verify_access.py')
    assert os.path.isfile(script_path), "verify_access.py not found"
    # Check it's valid Python (compiles without error)
    with open(script_path) as f:
        compile(f.read(), script_path, 'exec')


def test_gee_import():
    """earthengine-api is importable."""
    import ee
    assert hasattr(ee, 'Initialize')


def test_cdsapi_import():
    """cdsapi is importable (ERA5 access)."""
    try:
        import cdsapi
        assert hasattr(cdsapi, 'Client')
    except ImportError:
        import pytest
        pytest.skip("cdsapi not installed — ERA5 access requires Human Action")


def test_acquisition_module_structure():
    """Acquisition module directory exists with __init__.py."""
    acq_dir = os.path.join(source_root, 'data', 'acquisition')
    assert os.path.isdir(acq_dir)
    assert os.path.isfile(os.path.join(acq_dir, '__init__.py'))
