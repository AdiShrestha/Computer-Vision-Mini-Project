"""Verify encoder architecture decision is documented."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
PROJECT_DIR = os.path.join(repo_root, 'project')


def test_decision_log_has_encoder_entry():
    """decision_log.md has an encoder architecture decision entry."""
    log_path = os.path.join(PROJECT_DIR, 'evolution', 'decision_log.md')
    assert os.path.isfile(log_path), "decision_log.md not found"
    with open(log_path) as f:
        content = f.read()
    assert 'encoder' in content.lower() or 'architecture' in content.lower(), (
        "decision_log.md missing encoder architecture entry"
    )


def test_decision_has_evidence():
    """Decision entry includes quantitative evidence."""
    log_path = os.path.join(PROJECT_DIR, 'evolution', 'decision_log.md')
    with open(log_path) as f:
        content = f.read()
    assert 'training' in content.lower() and ('window' in content.lower() or 'sample' in content.lower()), (
        "Decision lacks training corpus size evidence"
    )


def test_architecture_spec_exists():
    """Encoder architecture specification stub exists."""
    spec_path = os.path.join(source_root, 'models', 'encoder', 'architecture_spec.md')
    assert os.path.isfile(spec_path), "architecture_spec.md not found"
    with open(spec_path) as f:
        content = f.read()
    assert len(content) > 200, "Architecture spec seems too short"


def test_decision_references_compute_budget():
    """Decision considers compute constraints (INV-008)."""
    log_path = os.path.join(PROJECT_DIR, 'evolution', 'decision_log.md')
    with open(log_path) as f:
        content = f.read()
    assert 'compute' in content.lower() or 'vram' in content.lower() or 'INV-008' in content, (
        "Decision doesn't address compute budget constraint"
    )
