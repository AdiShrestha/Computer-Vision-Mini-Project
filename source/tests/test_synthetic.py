"""Verify synthetic anomaly injection engine."""
import os
import sys
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_injector_module_exists():
    """SyntheticInjector module exists."""
    from evaluation.synthetic.injector import SyntheticInjector
    assert SyntheticInjector is not None


def test_injection_preserves_shape():
    """Synthetic injection preserves feature matrix shape (108, 15)."""
    from evaluation.synthetic.injector import SyntheticInjector
    injector = SyntheticInjector(seed=2023)
    features = np.random.rand(108, 15).astype(np.float32)
    modified, meta = injector.inject(features, anomaly_type=1, window_idx=50, channel_idx=0)
    assert modified.shape == features.shape


def test_injection_actually_modifies():
    """Synthetic injection modifies target channel values."""
    from evaluation.synthetic.injector import SyntheticInjector
    injector = SyntheticInjector(seed=2023)
    features = np.ones((108, 15), dtype=np.float32)
    modified, meta = injector.inject(features, anomaly_type=1, window_idx=50, channel_idx=0)
    assert not np.array_equal(modified, features)


def test_injection_seed_determinism():
    """INV-012: Same seed produces identical injections."""
    from evaluation.synthetic.injector import SyntheticInjector
    features = np.random.rand(108, 15).astype(np.float32)
    inj1 = SyntheticInjector(seed=2023)
    inj2 = SyntheticInjector(seed=2023)
    m1, _ = inj1.inject(features.copy(), 1, 50, 0)
    m2, _ = inj2.inject(features.copy(), 1, 50, 0)
    np.testing.assert_array_equal(m1, m2)


def test_anomaly_config_exists():
    """anomaly_config.json exists."""
    config_path = os.path.join(source_root, 'evaluation', 'synthetic', 'anomaly_config.json')
    assert os.path.isfile(config_path)
