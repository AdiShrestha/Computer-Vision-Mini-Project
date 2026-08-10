"""Verify data quality report and analysis."""
import os
import sys

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
RESULTS_DIR = os.path.join(repo_root, 'results', 'data_quality')


def test_data_quality_report_exists():
    """Data quality report markdown exists."""
    report_path = os.path.join(RESULTS_DIR, 'data_quality_report.md')
    assert os.path.isfile(report_path)
    with open(report_path) as f:
        content = f.read()
    assert len(content) > 1000, "Report seems too short"


def test_report_has_required_sections():
    """Report contains all required sections."""
    report_path = os.path.join(RESULTS_DIR, 'data_quality_report.md')
    with open(report_path) as f:
        content = f.read().lower()
    required = [
        'acquisition coverage', 'preprocessing', 'channel',
        'feature matrix', 'gaps', 'limitations', 'recommendation'
    ]
    for section in required:
        assert section in content, f"Missing section: {section}"


def test_data_quality_script_exists():
    """data_quality.py exists and compiles."""
    script_path = os.path.join(source_root, 'evaluation', 'data_quality.py')
    assert os.path.isfile(script_path)
    with open(script_path) as f:
        compile(f.read(), script_path, 'exec')


def test_south_lhonak_mentioned():
    """South Lhonak is specifically discussed in the report."""
    report_path = os.path.join(RESULTS_DIR, 'data_quality_report.md')
    with open(report_path) as f:
        content = f.read()
    assert 'South Lhonak' in content or 'SGL-001' in content
