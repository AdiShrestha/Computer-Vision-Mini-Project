"""
Adversarial verification tests for Chunk 07 — Scientific Integrity Restoration & Real GEE Pipeline Rework.

Tests are added progressively by each contract (C07-00 through C07-05).
"""
import os
import json
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)


# ============================================================
# C07-00: INV-011 Correction Verification
# ============================================================

def test_inv011_no_ch06_reference():
    """INV-011 type 3 must not reference CH-06 after correction."""
    invariants_path = os.path.join(repo_root, 'project', 'invariants.md')
    with open(invariants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    inv011_start = content.find('## INV-011')
    inv011_end = content.find('## INV-012')
    inv011_text = content[inv011_start:inv011_end]
    assert 'CH-06' not in inv011_text, "INV-011 type 3 still references excluded CH-06"


def test_inv011_matches_decision_003():
    """INV-011 type 3 must reference CH-05 and +3 dB."""
    invariants_path = os.path.join(repo_root, 'project', 'invariants.md')
    with open(invariants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    inv011_start = content.find('## INV-011')
    inv011_end = content.find('## INV-012')
    inv011_text = content[inv011_start:inv011_end]
    assert 'CH-05' in inv011_text, "INV-011 type 3 must reference CH-05"
    assert '3 dB' in inv011_text or '+3 dB' in inv011_text, "INV-011 type 3 must specify +3 dB magnitude"


def test_inv011_revision_note_exists():
    """INV-011 must have a dated revision note for the C07-00 correction."""
    invariants_path = os.path.join(repo_root, 'project', 'invariants.md')
    with open(invariants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'C07-00' in content, "Missing C07-00 revision note in invariants.md"
    assert 'Decision 003' in content, "Revision note must reference Decision 003"
