"""Verify research question answers."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

RQ_DIR = os.path.join(repo_root, 'results', 'research_questions')


def test_rq1_answer_exists():
    """rq1_answer.md exists."""
    assert os.path.isfile(os.path.join(RQ_DIR, 'rq1_answer.md'))


def test_rq3_answer_exists():
    """rq3_answer.md exists."""
    assert os.path.isfile(os.path.join(RQ_DIR, 'rq3_answer.md'))


def test_evidence_summary_exists():
    """evidence_summary.json exists with valid verdicts."""
    path = os.path.join(RQ_DIR, 'evidence_summary.json')
    assert os.path.isfile(path)
    with open(path) as f:
        evidence = json.load(f)
    assert 'rq1' in evidence
    assert 'rq3' in evidence
    assert evidence['rq1']['verdict'] in ['positive', 'negative', 'mixed']


def test_rq1_has_verdict():
    """rq1_answer.md contains a verdict and detailed content."""
    with open(os.path.join(RQ_DIR, 'rq1_answer.md')) as f:
        content = f.read()
    assert 'Verdict' in content or 'verdict' in content
    assert len(content) > 500, "RQ1 answer too short"
