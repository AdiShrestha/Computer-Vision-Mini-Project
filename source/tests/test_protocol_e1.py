"""
Unit test suite for Protocol E1 Resolution & F3 Verdict (Contract C08-08).
"""

import os
import json
import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from evaluation.protocols.protocol_e1 import resolve_protocol_e1_f3


def test_protocol_e1_f3_resolution():
    """Assert resolve_protocol_e1_f3 generates protocol_e1_real_data.json and appends Decision 006."""
    res = resolve_protocol_e1_f3()
    assert res['event_lake_id'] == 'SGL-001'
    assert 'f3_falsification_verdict' in res
    assert res['f3_falsification_verdict'] in ['SUCCESS', 'FAILURE', 'AMBIGUOUS_FAILURE']
    assert 'pre_event_flagged_percentage' in res

    artifact_path = PROJECT_ROOT / 'results' / 'evaluation' / 'protocol_e1_real_data.json'
    assert artifact_path.exists()

    decision_log_path = PROJECT_ROOT / 'project' / 'evolution' / 'decision_log.md'
    assert decision_log_path.exists()
    content = decision_log_path.read_text(encoding='utf-8')
    assert 'Decision 006 — Protocol E1 Falsification Resolution' in content
