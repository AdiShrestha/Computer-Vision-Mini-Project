"""Verify configuration framework loads correctly and matches invariants."""
import os
import yaml

def test_default_config_loads():
    """default_config.yaml is valid YAML and loads without error."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict)
    assert 'project' in config

def test_invariant_parameters_present():
    """Every numeric invariant from invariants.md has a config entry."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # INV-003: Temporal extent
    assert config['temporal']['start_date'] == '2016-01-01'
    assert config['temporal']['end_date'] == '2024-10-31'
    # INV-004: Window size and stride
    assert config['temporal']['window_size_days'] == 180
    assert config['temporal']['stride_days'] == 30
    # INV-005: Masking ratio
    assert config['training']['masking_ratio'] == 0.5
    # INV-006: Smoothing
    assert config['anomaly']['smoothing_span'] == 5
    # INV-007: FP rate target
    assert config['evaluation']['fp_rate_target'] == 0.10
    # INV-008: Compute budget
    assert config['compute']['max_training_hours'] == 72
    # INV-009: Event date
    assert config['evaluation']['event_date'] == '2023-10-04'
    # INV-012: Seeds
    assert config['training']['seeds']['torch'] == 42
    assert config['training']['seeds']['numpy'] == 42
    assert config['training']['seeds']['synthetic_injection'] == 2023
    assert config['training']['seeds']['train_val_split'] == 7

def test_config_loader_import():
    """Config loader is importable and functional."""
    import sys
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from utils.config_loader import load_config
    config = load_config()
    assert config['temporal']['window_size_days'] == 180

def test_all_paths_defined():
    """All data paths from architecture.md §11 are present."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_path_keys = [
        'data_root', 'raw_data', 'processed_data',
        'features', 'embeddings', 'models', 'results',
        'lake_registry'
    ]
    for key in required_path_keys:
        assert key in config['paths'], f"Missing path key: {key}"
