"""Verify sanity check execution."""
import os
import sys

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

RESULTS_DIR = os.path.join(repo_root, 'results', 'sanity_check')


def test_sanity_check_module_exists():
    """Sanity check module exists."""
    path = os.path.join(source_root, 'evaluation', 'sanity_check.py')
    assert os.path.isfile(path)


def test_sanity_check_report_exists():
    """Sanity check report exists."""
    report_path = os.path.join(RESULTS_DIR, 'sanity_check_report.md')
    assert os.path.isfile(report_path)
    with open(report_path) as f:
        content = f.read()
    assert len(content) > 500, "Report too short"


def test_sanity_check_has_verdict():
    """Report contains a PASS or FAIL verdict."""
    report_path = os.path.join(RESULTS_DIR, 'sanity_check_report.md')
    with open(report_path) as f:
        content = f.read().upper()
    assert 'PASS' in content or 'FAIL' in content, "No verdict found"


def test_sanity_check_plots_exist():
    """At least one visualization plot was generated."""
    plot_files = [f for f in os.listdir(RESULTS_DIR)
                  if f.endswith('.png') or f.endswith('.jpg')]
    assert len(plot_files) >= 1, "No visualization plots found"
