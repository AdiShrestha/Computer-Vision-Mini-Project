"""Verify that the Python environment is correctly set up."""
import importlib
import sys
import os

def test_python_version():
    """Python >= 3.10 required per architecture.md §7."""
    assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version}"

def test_core_imports():
    """All core dependencies from architecture.md §7 are importable."""
    required = [
        'torch', 'numpy', 'pandas', 'xarray', 'zarr',
        'rasterio', 'ee',  # earthengine-api imports as 'ee'
        'sklearn', 'matplotlib', 'yaml',  # pyyaml imports as 'yaml'
    ]
    missing = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    assert not missing, f"Missing imports: {missing}"

def test_torch_cuda_determinism_available():
    """INV-012 requires CUDA determinism flags — verify they exist."""
    import torch
    # These attributes must exist even if CUDA is not available
    assert hasattr(torch.backends, 'cudnn')
    assert hasattr(torch.backends.cudnn, 'deterministic')
    assert hasattr(torch.backends.cudnn, 'benchmark')

def test_directory_structure():
    """Source directory structure matches architecture.md §11."""
    # Determine source root relative to this test file
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_dirs = [
        'config', 'config/experiment_configs',
        'data', 'data/registry', 'data/acquisition',
        'data/preprocessing', 'data/channels', 'data/insar',
        'models', 'models/encoder', 'models/anomaly', 'models/baseline',
        'evaluation', 'evaluation/protocols', 'evaluation/synthetic',
        'evaluation/visualization',
        'utils', 'tests', 'scripts',
    ]
    missing = [d for d in required_dirs
               if not os.path.isdir(os.path.join(source_root, d))]
    assert not missing, f"Missing directories: {missing}"

def test_requirements_file_exists():
    """requirements.txt exists at the repository root."""
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    assert os.path.isfile(os.path.join(repo_root, 'requirements.txt'))
