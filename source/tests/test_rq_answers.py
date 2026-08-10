"""Adversarial RQ verification tests (C04-R3)."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
RQ_DIR = os.path.join(repo_root, 'results', 'research_questions')
EVAL_SUMMARY = os.path.join(repo_root, 'results', 'evaluation', 'evaluation_summary.json')


def test_rq1_answer_exists():
    """rq1_answer.md exists."""
    assert os.path.isfile(os.path.join(RQ_DIR, 'rq1_answer.md'))


def test_rq3_answer_exists():
    """rq3_answer.md exists."""
    assert os.path.isfile(os.path.join(RQ_DIR, 'rq3_answer.md'))


def test_evidence_summary_matches_evaluation_summary():
    """ADVERSARIAL: Every number in evidence_summary must match evaluation_summary."""
    with open(os.path.join(RQ_DIR, 'evidence_summary.json')) as f:
        evidence = json.load(f)
    with open(EVAL_SUMMARY) as f:
        actual = json.load(f)

    for scorer in ['score_a', 'score_b', 'score_c']:
        actual_lt = actual['scorer_comparison'][scorer]['lead_time_days']
        evidence_lt = evidence['rq1']['lead_time_days'][scorer]
        assert evidence_lt == actual_lt, (
            f"FABRICATION: evidence says {scorer} lead_time={evidence_lt}, "
            f"actual is {actual_lt}"
        )

        actual_auc = actual['scorer_comparison'][scorer]['auc_roc']
        evidence_auc = evidence['rq1']['auc_roc'][scorer]
        assert abs(evidence_auc - actual_auc) < 0.001, (
            f"FABRICATION: evidence says {scorer} auc_roc={evidence_auc}, "
            f"actual is {actual_auc}"
        )


def test_evidence_has_source_file():
    """Evidence must trace to a source file."""
    with open(os.path.join(RQ_DIR, 'evidence_summary.json')) as f:
        evidence = json.load(f)
    assert 'source_file' in evidence['rq1'], "Missing source_file traceability"


def test_rq1_has_verdict():
    """rq1_answer.md contains a verdict and detailed content."""
    with open(os.path.join(RQ_DIR, 'rq1_answer.md')) as f:
        content = f.read()
    assert 'Verdict' in content or 'verdict' in content
    assert len(content) > 500, "RQ1 answer too short"
