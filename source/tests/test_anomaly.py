"""Verify anomaly scoring mechanisms."""
import os
import sys
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_score_a_module_exists():
    """Score-A ReconstructionScorer exists."""
    from models.anomaly.score_a import ReconstructionScorer
    assert ReconstructionScorer is not None


def test_score_b_module_exists():
    """Score-B EmbeddingDistanceScorer exists."""
    from models.anomaly.score_b import EmbeddingDistanceScorer
    assert EmbeddingDistanceScorer is not None


def test_score_c_module_exists():
    """Score-C CombinedScorer exists."""
    from models.anomaly.score_c import CombinedScorer
    assert CombinedScorer is not None


def test_smoothing_function():
    """EMA smoothing produces smoothed output array."""
    from models.anomaly.smoothing import ema_smooth
    scores = np.random.rand(50)
    smoothed = ema_smooth(scores, span=5)
    assert smoothed.shape == scores.shape
    # Smoothed sequence variance should be reduced or comparable
    assert smoothed.std() <= scores.std() + 0.05


def test_smoothing_span_matches_inv006():
    """INV-006: Default span is 5 windows."""
    import inspect
    from models.anomaly.smoothing import ema_smooth
    sig = inspect.signature(ema_smooth)
    assert sig.parameters['span'].default == 5
