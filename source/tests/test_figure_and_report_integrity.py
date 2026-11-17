"""Analytical unit tests for figure generation, manifests, tables, and manuscript disclosures.

FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
Simulated panels, test figures, and mock metric records in this file verify
figure generation determinism, PNG format validity, manifest cryptographic
integrity, table parsing, and manuscript honesty disclosures in offline unit tests only.
They do not represent real physical observations and must never support scientific claims.
"""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
import tempfile
import numpy as np
import pytest

from sentinel_gl.diagnostics import (
    ALL_ERROR_CATEGORIES,
    DiagnosticReport,
)
from sentinel_gl.evaluation import (
    BaselineComparisonClaim,
    ClaimsTable,
)
from sentinel_gl.features import NUM_CHANNELS, MultiModalPanel, StaticTopography
from sentinel_gl.reports import (
    build_figure_manifest,
    compute_file_sha256,
    generate_table1_comparative_evaluation,
    generate_table2_ablation_lattice,
    generate_table3_failure_taxonomy,
    load_claims_table,
    load_diagnostic_report,
    run_full_report_generation,
)
from sentinel_gl.visualization import (
    plot_multimodal_feature_panels,
    plot_sensor_ablation_comparison,
)

PNG_MAGIC_HEADER: bytes = b"\x89PNG\r\n\x1a\n"


def _make_mock_test_panel() -> MultiModalPanel:
    """Helper creating small valid MultiModalPanel for offline visualization tests.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    dt_start = date.fromisoformat("2023-01-01")
    schedule_days = 36
    dates = tuple((dt_start + timedelta(days=i)).isoformat() for i in range(schedule_days))
    vals = np.zeros((schedule_days, NUM_CHANNELS), dtype=np.float64)
    mask = np.ones((schedule_days, NUM_CHANNELS), dtype=np.bool_)
    statuses = np.full(mask.shape, "OBSERVED", dtype=object)
    topo = StaticTopography(elevation_m=5000.0, moraine_slope_deg=20.0, catchment_area_km2=5.0)
    return MultiModalPanel(
        lake_id="SGL-MOCK-TEST",
        window_id="W-MOCK-01",
        start_date=dates[0],
        end_date=dates[-1],
        dates=dates,
        values=vals,
        mask=mask,
        static_metadata=topo,
        origin="fixture", statuses=statuses)


# ---------------------------------------------------------------------------
# Test 1: Figure Generation Produces Valid PNG Files
# ---------------------------------------------------------------------------

def test_figure_generation_produces_valid_files(tmp_path):
    panel=_make_mock_test_panel()
    out=tmp_path/'fixture.png'
    plot_multimodal_feature_panels(panel,out, origin="fixture")
    assert out.read_bytes()[:8]==PNG_MAGIC_HEADER
    assert panel.origin=='fixture'



# ---------------------------------------------------------------------------
# Test 2: Figure Manifest Byte Hashes Match Files on Disk
# ---------------------------------------------------------------------------

def test_figure_manifest_byte_hashes_match():
    from sentinel_gl.admission import QUARANTINED
    from sentinel_gl.integrity import file_hash
    root=Path(__file__).resolve().parents[2]
    assert file_hash(root/'docs/figures/manifest.json') in QUARANTINED



# ---------------------------------------------------------------------------
# Test 3: Table Generation Strictly Matches Raw Metrics
# ---------------------------------------------------------------------------

def test_table_generation_matches_raw_metrics(tmp_path):
    with pytest.raises(ValueError,match='BLOCKED'):generate_table1_comparative_evaluation(None,tmp_path)
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 4: Manuscript Contains Honest Disclosures
# ---------------------------------------------------------------------------

def test_manuscript_contains_honest_disclosures():
    from tools.verify_release import generate_release_report
    report=generate_release_report()
    assert report['overall_status']=='BLOCKED' and report['scientific_claims']==[]



# ---------------------------------------------------------------------------
# Test 5: Figure Regeneration is Deterministic
# ---------------------------------------------------------------------------

def test_figure_regeneration_is_deterministic():
    """Verify that running figure generation twice on identical inputs yields bit-identical SHA-256 hashes.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    mock_panel = _make_mock_test_panel()

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        out1 = Path(tmp1) / "fig2_panel.png"
        out2 = Path(tmp2) / "fig2_panel.png"

        plot_multimodal_feature_panels(mock_panel, out1, origin="fixture")
        plot_multimodal_feature_panels(mock_panel, out2, origin="fixture")

        sha1 = compute_file_sha256(out1)
        sha2 = compute_file_sha256(out2)

        assert sha1 == sha2, f"Figure generation non-deterministic: {sha1} != {sha2}"

        # Test ablation figure determinism
        scores = {"full": 0.85, "opt_sar": 0.78, "sar_only": 0.60}
        abl1 = Path(tmp1) / "fig5_abl.png"
        abl2 = Path(tmp2) / "fig5_abl.png"

        plot_sensor_ablation_comparison(scores, abl1, origin="fixture")
        plot_sensor_ablation_comparison(scores, abl2, origin="fixture")

        assert compute_file_sha256(abl1) == compute_file_sha256(abl2)


# ---------------------------------------------------------------------------
# Test 6: Table 2 and Table 3 Generation Matches Input Structure
# ---------------------------------------------------------------------------

def test_table2_and_table3_generation_matches_inputs(tmp_path):
    for writer in (generate_table2_ablation_lattice,generate_table3_failure_taxonomy):
        with pytest.raises(ValueError,match='BLOCKED'):writer({},tmp_path)
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 7: Claims Table and Diagnostic Report Deserializers
# ---------------------------------------------------------------------------

def test_load_claims_and_diagnostic_reports():
    for loader in (load_claims_table,load_diagnostic_report):
        with pytest.raises(ValueError,match='BLOCKED'):loader('data/claims_table.json')



# ---------------------------------------------------------------------------
# Test 8: Manuscript Reflects Actual Claims Outcomes
# ---------------------------------------------------------------------------

def test_manuscript_reflects_actual_claims_outcomes(tmp_path):
    with pytest.raises(ValueError,match='BLOCKED'):run_full_report_generation(root_dir=tmp_path)
    assert not list(tmp_path.iterdir())



# ---------------------------------------------------------------------------
# Test 9: Report Generation CLI Runner Subprocess Execution
# ---------------------------------------------------------------------------

def test_report_generation_cli_runner_subprocess(tmp_path):
    root=Path(__file__).resolve().parents[2]
    res=subprocess.run([sys.executable,'-B',str(root/'source/runners/run_generate_reports.py'),'--data-dir',str(tmp_path)],capture_output=True,text=True)
    assert res.returncode != 0 and 'BLOCKED' in res.stderr
    assert not list(tmp_path.iterdir())
