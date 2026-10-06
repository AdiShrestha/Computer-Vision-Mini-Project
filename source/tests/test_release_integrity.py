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
    """Verify attribution of ESA Copernicus, ECMWF, RGI/GLIMS, and repository terms."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    lic_path = REPO_ROOT / "docs/LICENSE_NOTICE.md"
    assert lic_path.is_file(), "docs/LICENSE_NOTICE.md must exist"

    text = lic_path.read_text(encoding="utf-8")

    # Provider requirements
    assert "European Space Agency (ESA)" in text or "Copernicus Sentinel" in text
    assert "European Centre for Medium-Range Weather Forecasts (ECMWF)" in text or "ERA5" in text
    assert "Randolph Glacier Inventory" in text or "RGI 6.0" in text
    assert "Global Land Ice Measurements from Space" in text or "GLIMS" in text
    assert "Apache License" in text or "Open Research Software License" in text

    # Mandatory attribution strings
    assert "Contains modified Copernicus Sentinel data" in text
    assert "Generated using Copernicus Climate Change Service information" in text
    assert "Non-Operational Warranty" in text or "AS IS" in text


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
    """Verify manuscript text contains InSAR decorrelation findings, small-N power

    disclosures, and retrospective task bounds.
    """
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    ms_path = REPO_ROOT / "docs/manuscript/manuscript.md"
    assert ms_path.is_file(), "docs/manuscript/manuscript.md must exist"

    text = ms_path.read_text(encoding="utf-8")

    # InSAR findings
    assert "InSAR SLC" in text or "Single Look Complex" in text
    assert "LAYOVER" in text and "SHADOW" in text
    assert "0.0176" in text or "gamma" in text
    assert "decorrelation" in text or "decorr" in text

    # Statistical power & small-N disclosures
    assert "power" in text.lower()
    assert "0.20" in text or "20%" in text
    assert "N \\ge 30" in text or "N >= 30" in text or "30 independent events" in text

    # Regional cohort integration
    assert "SGL-001" in text
    assert "SGL-003" in text
    assert "SGL-004" in text
    assert "SGL-005" in text

    # Conservative outcome reporting
    assert "FAIL_TO_REJECT" in text
    assert "non-operational" in text.lower()


# ---------------------------------------------------------------------------
# Test 5: Release Verifier Passes on Complete Bundle
# ---------------------------------------------------------------------------

def test_release_verifier_passes_on_complete_bundle():
    """Verify that tools/verify_release.py executes and passes all verification gates."""
    # FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    from tools.verify_release import generate_release_report

    out_file = REPO_ROOT / "docs/audit/wp12_release/test_release_report.json"
    report = generate_release_report(output_path=out_file)

    assert report["overall_status"] == "PASS", f"Release verifier failed: {report}"
    assert report["gates"]["reproduction_subsystem"]["status"] == "PASS"
    assert report["gates"]["document_completeness"]["status"] == "PASS"
    assert report["gates"]["credential_and_secret_hygiene"]["status"] == "PASS"
    assert report["gates"]["cohort_and_registry_invariants"]["status"] == "PASS"

    # Clean up test output file
    if out_file.is_file():
        out_file.unlink()
