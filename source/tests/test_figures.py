"""Verify result figures exist."""
import os
import sys

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
FIG_DIR = os.path.join(repo_root, 'results', 'figures')


def test_south_lhonak_timeline_exists():
    """South Lhonak anomaly timeline figure exists."""
    assert os.path.isfile(os.path.join(FIG_DIR, 'south_lhonak_anomaly_timeline.png'))


def test_scorer_comparison_exists():
    """Scorer comparison table figure exists."""
    assert os.path.isfile(os.path.join(FIG_DIR, 'scorer_comparison_table.png'))


def test_at_least_4_figures():
    """At least 4 PNG figures generated."""
    pngs = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')]
    assert len(pngs) >= 4, f"Only {len(pngs)} figures generated"
