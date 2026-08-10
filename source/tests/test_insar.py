"""Verify InSAR feasibility assessment."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

INSAR_DIR = os.path.join(source_root, 'data', 'insar')


def test_insar_module_exists():
    """InSAR feasibility module exists."""
    assert os.path.isfile(os.path.join(INSAR_DIR, 'insar_feasibility.py'))


def test_feasibility_report_exists():
    """Feasibility report JSON exists with a verdict."""
    report_path = os.path.join(INSAR_DIR, 'feasibility_report.json')
    assert os.path.isfile(report_path), "feasibility_report.json not found"
    with open(report_path) as f:
        report = json.load(f)
    assert 'overall_verdict' in report, "Missing overall_verdict"
    assert report['overall_verdict'] in ['FEASIBLE', 'MARGINAL', 'INFEASIBLE'], (
        f"Invalid verdict: {report['overall_verdict']}"
    )


def test_south_lhonak_assessed():
    """South Lhonak was included in the feasibility assessment."""
    report_path = os.path.join(INSAR_DIR, 'feasibility_report.json')
    with open(report_path) as f:
        report = json.load(f)
    lakes = report.get('lakes', report.get('per_lake', {}))
    sgl_001 = None
    if isinstance(lakes, list):
        sgl_001 = next((l for l in lakes if l.get('lake_id') == 'SGL-001'), None)
    elif isinstance(lakes, dict):
        sgl_001 = lakes.get('SGL-001')
    assert sgl_001 is not None, "SGL-001 not assessed"


def test_feasibility_has_evidence():
    """Feasibility report includes quantitative evidence."""
    report_path = os.path.join(INSAR_DIR, 'feasibility_report.json')
    with open(report_path) as f:
        report = json.load(f)
    assert ('evidence' in report or 'methodology' in report or 
            'slc_availability' in str(report)), (
        "Feasibility report lacks evidence"
    )
