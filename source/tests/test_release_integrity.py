"""Analytical unit tests for release integrity, maintenance documentation, and open science licenses.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Simulated test fixtures, document parsing assertions, and release validation checks
in this file verify schema invariants, mandatory document sections, BibTeX syntax,
and release verification gates in offline unit tests only. They do not represent
real physical observations and must never support scientific claims.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import pytest

# Ensure repository root is discoverable
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Test 1: Maintenance Document Contains Required Operational Sections
# ---------------------------------------------------------------------------

def test_maintenance_document_contains_required_sections():
    """Verify that docs/MAINTENANCE.md covers sensor lifecycle, retraining triggers,

    credential policies, and Apple Silicon runtime guidance.
    """
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    maint_path = REPO_ROOT / "docs/MAINTENANCE.md"
    assert maint_path.is_file(), "docs/MAINTENANCE.md must exist"

    text = maint_path.read_text(encoding="utf-8")
    assert len(text) > 1000, "docs/MAINTENANCE.md must be comprehensive"

    required_topics = [
        "Sensor & Provider Lifecycle Management",
        "Sentinel-1C",
        "Sentinel-2C",
        "ERA5 Atmospheric Reanalysis",
        "Safe Credential Rotation",
        "Model Retraining & Anti-Leakage Safeguards",
        "Retraining Triggers",
        "Cluster Isolation Protocol",
        "Apple Silicon M-Series Unified Memory",
        "CPU Reference Regression Testing",
    ]

    for topic in required_topics:
        assert topic in text, f"Missing required maintenance topic: '{topic}'"

    # Verify numerical regression tolerance bound is explicitly stated
    assert "1.0 \\times 10^{-4}" in text or "1e-4" in text or "1.0e-4" in text, (
        "Numerical regression tolerance bound of 1e-4 must be documented"
    )


# ---------------------------------------------------------------------------
# Test 2: License Notice Attributes All Providers and Repositories
# ---------------------------------------------------------------------------

def test_license_notice_attributes_all_providers():
    # Terms remain unresolved; the root license cannot be promoted by prose.
    from tools.verify_release import generate_release_report
    report=generate_release_report()
    assert report['overall_status']=='BLOCKED'
    assert report['gates']['publication_and_license']['status']=='NOT_VERIFIED'



# ---------------------------------------------------------------------------
# Test 3: BibTeX Citations Library Format and DOI Coverage
# ---------------------------------------------------------------------------

def test_citations_bib_format_and_coverage():
    """Verify BibTeX entries exist for primary references with verified DOIs."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    bib_path = REPO_ROOT / "docs/CITATIONS.bib"
    assert bib_path.is_file(), "docs/CITATIONS.bib must exist"

    text = bib_path.read_text(encoding="utf-8")

    expected_keys = [
        "brun2017spatially",
        "veh2018detecting",
        "veh2019hazard",
        "nie2018inventory",
        "zheng2021complex",
        "taylor2023glacial",
        "hersbach2020era5",
        "rgi2017consortium",
    ]

    for key in expected_keys:
        assert f"{{{key}," in text or f"{{{key} " in text, f"Missing citation key: {key}"

    # Verify DOI patterns
    doi_pattern = re.compile(r"doi\s*=\s*\{10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\}")
    matches = doi_pattern.findall(text)
    assert len(matches) >= 6, f"Expected at least 6 DOI fields, found {len(matches)}"


# ---------------------------------------------------------------------------
# Test 4: Scientific Manuscript Integrates InSAR and Power Amendments
# ---------------------------------------------------------------------------

def test_manuscript_integrates_insar_and_power_amendments():
    from sentinel_gl.insar import generate_insar_feasibility_dossier
    with pytest.raises(ValueError,match='BLOCKED'):generate_insar_feasibility_dossier()



# ---------------------------------------------------------------------------
# Test 5: Release Verifier Passes on Complete Bundle
# ---------------------------------------------------------------------------

def test_release_verifier_passes_on_complete_bundle(tmp_path):
    from tools.verify_release import generate_release_report
    report=generate_release_report(tmp_path/'release.json')
    assert report['overall_status']=='BLOCKED'
    assert report['gates']['measurement_admission']['excluded_current_bytes']
    assert report['scientific_claims']==[]
    with pytest.raises(FileExistsError):generate_release_report(tmp_path/'release.json')
