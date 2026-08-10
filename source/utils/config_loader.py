"""Configuration loader for sentinel-gl.

Loads base default_config.yaml and optionally merges experiment-specific overrides.
"""
import os
import yaml
from typing import Optional, Dict, Any


def get_default_config_path() -> str:
    """Return absolute path to default_config.yaml."""
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(source_root, 'config', 'default_config.yaml')


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that required top-level and invariant sections exist."""
    required_sections = ['project', 'paths', 'temporal', 'training', 'anomaly', 'evaluation', 'compute']
    for sec in required_sections:
        if sec not in config:
            raise ValueError(f"Missing required configuration section: {sec}")

    # Validate specific invariant fields
    if 'start_date' not in config['temporal'] or 'end_date' not in config['temporal']:
        raise ValueError("Missing temporal start_date or end_date (INV-003)")
    if 'window_size_days' not in config['temporal'] or 'stride_days' not in config['temporal']:
        raise ValueError("Missing temporal window_size_days or stride_days (INV-004)")
    if 'masking_ratio' not in config['training']:
        raise ValueError("Missing training masking_ratio (INV-005)")
    if 'smoothing_span' not in config['anomaly']:
        raise ValueError("Missing anomaly smoothing_span (INV-006)")
    if 'fp_rate_target' not in config['evaluation']:
        raise ValueError("Missing evaluation fp_rate_target (INV-007)")
    if 'event_date' not in config['evaluation']:
        raise ValueError("Missing evaluation event_date (INV-009)")
    if 'seeds' not in config['training']:
        raise ValueError("Missing training seeds (INV-012)")

    return True


def load_config(experiment_name: Optional[str] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load default config and optionally merge an experiment override.

    Args:
        experiment_name: Name of YAML file in source/config/experiment_configs/ (with or without .yaml)
        config_path: Explicit path to default_config.yaml if overriding default location

    Returns:
        Dict[str, Any]: Loaded and validated configuration dict.
    """
    if config_path is None:
        config_path = get_default_config_path()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if experiment_name:
        if not experiment_name.endswith(('.yaml', '.yml')):
            experiment_filename = f"{experiment_name}.yaml"
        else:
            experiment_filename = experiment_name

        source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exp_path = os.path.join(source_root, 'config', 'experiment_configs', experiment_filename)

        if os.path.exists(exp_path):
            with open(exp_path, 'r', encoding='utf-8') as f:
                exp_config = yaml.safe_load(f)
            if exp_config and isinstance(exp_config, dict):
                config = _deep_merge(config, exp_config)
        else:
            raise FileNotFoundError(f"Experiment config file not found: {exp_path}")

    validate_config(config)
    return config
