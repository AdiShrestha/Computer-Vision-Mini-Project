"""Verify baseline detector."""
import os
import sys
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_baseline_module_exists():
    """Baseline module exists."""
    from models.baseline.extent_threshold import ExtentThresholdDetector
    assert ExtentThresholdDetector is not None


def test_baseline_produces_scores():
    """Baseline detector produces (108,) score array."""
    from models.baseline.extent_threshold import ExtentThresholdDetector
    detector = ExtentThresholdDetector(threshold=0.10)
    fake_area = np.random.rand(108) + 1.0  # area values > 0
    scores = detector.score(fake_area)
    assert scores.shape == (108,)


def test_baseline_detects_sudden_change():
    """Baseline detector flags sudden 50% extent change."""
    from models.baseline.extent_threshold import ExtentThresholdDetector
    detector = ExtentThresholdDetector(threshold=0.10)
    area = np.ones(108)
    area[50] = 1.5  # 50% spike
    scores = detector.score(area)
    assert scores[50] > 0.10  # Should exceed threshold
