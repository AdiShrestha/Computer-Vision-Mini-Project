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
from sentinel_gl.features import NUM_CHANNELS, WINDOW_DAYS, MultiModalPanel, StaticTopography
from sentinel_gl.reports import (
    build_figure_manifest,
    compute_file_sha256,
    generate_table1_comparative_evaluation,
    generate_table2_ablation_lattice,
    generate_table3_failure_taxonomy,
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
    dates = tuple((dt_start + timedelta(days=i)).isoformat() for i in range(WINDOW_DAYS))
    vals = np.zeros((WINDOW_DAYS, NUM_CHANNELS), dtype=np.float64)
    mask = np.ones((WINDOW_DAYS, NUM_CHANNELS), dtype=np.bool_)
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
    )


# ---------------------------------------------------------------------------
# Test 1: Figure Generation Produces Valid PNG Files
# ---------------------------------------------------------------------------

def test_figure_generation_produces_valid_files():
    """Verify that all 5 figure files exist and have valid PNG binary magic headers."""
    fig_dir = Path("docs/figures")
    assert fig_dir.exists(), f"Figure directory {fig_dir} must exist"

    expected_figures = [
        "fig1_observation_cadence.png",
        "fig2_feature_panels.png",
        "fig3_anomaly_lead_time.png",
        "fig4_control_episodes.png",
        "fig5_sensor_ablations.png",
    ]

    for fname in expected_figures:
        fpath = fig_dir / fname
        assert fpath.exists(), f"Expected figure {fpath} does not exist"
        assert fpath.stat().st_size > 1024, f"Figure {fpath} is suspiciously small ({fpath.stat().st_size} bytes)"

        with open(fpath, "rb") as f:
            header = f.read(8)
            assert header == PNG_MAGIC_HEADER, f"File {fpath} lacks valid PNG magic header"


# ---------------------------------------------------------------------------
# Test 2: Figure Manifest Byte Hashes Match Files on Disk
# ---------------------------------------------------------------------------

def test_figure_manifest_byte_hashes_match():
    """Verify that manifest.json accurately reflects SHA-256 byte digests of all figures."""
    manifest_path = Path("docs/figures/manifest.json")
    assert manifest_path.exists(), f"Figure manifest {manifest_path} must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "figures" in manifest
    assert len(manifest["figures"]) == 5

    for entry in manifest["figures"]:
        fname = entry["filename"]
        fpath = Path("docs/figures") / fname
        assert fpath.exists(), f"Figure {fname} listed in manifest does not exist on disk"

        recorded_sha = entry["sha256"]
        actual_sha = compute_file_sha256(fpath)
        assert recorded_sha == actual_sha, (
            f"Digest mismatch for {fname}: recorded {recorded_sha} != actual {actual_sha}"
        )


# ---------------------------------------------------------------------------
# Test 3: Table Generation Strictly Matches Raw Metrics
# ---------------------------------------------------------------------------

def test_table_generation_matches_raw_metrics():
    """Verify that generated markdown and CSV tables strictly match underlying metrics without fabrication.

    FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Mock ClaimsTable
        mock_claims = ClaimsTable(
            case_detections=[
                {"lake_id": "SGL-001", "event_id": "EVT-001", "lead_time_days": 42, "status": "DETECTED"}
            ],
            alert_burden={
                "status": "ESTIMATED",
                "lambda_alert": 0.8521,
                "poisson_ci_lower": 0.1234,
                "poisson_ci_upper": 2.4567,
            },
            baseline_comparisons=[
                BaselineComparisonClaim(
                    baseline_id="climatology",
                    n_paired_windows=10,
                    mean_delta_s=0.1500,
                    abs_statistic=0.1500,
                    unadjusted_p_value=0.0250,
                    adjusted_p_value=0.1000,
                    claim_finding="FAIL_TO_REJECT",
                )
            ],
            cluster_aggregation=None,
            alpha=0.05,
            status="COMPLETE",
            report_hash="test_hash_01",
        )

        md_p, csv_p = generate_table1_comparative_evaluation(mock_claims, out_dir)
        assert md_p.exists() and csv_p.exists()

        # Check CSV content
        with open(csv_p, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        # Row 1 is header, Row 2 is Case Detection
        assert reader[1][0] == "Case Detection"
        assert "42 days" in reader[1][2]
        assert reader[1][7] == "DETECTED"

        # Row 3 is Alert Burden
        assert reader[2][0] == "Alert Burden"
        assert "0.8521" in reader[2][2]
        assert "0.1234" in reader[2][3]
        assert "2.4567" in reader[2][4]

        # Row 4 is Baseline Contrast
        assert reader[3][0] == "Paired Contrast"
        assert "0.1500" in reader[3][2]
        assert "0.0250" in reader[3][5]
        assert "0.1000" in reader[3][6]
        assert reader[3][7] == "FAIL_TO_REJECT"


# ---------------------------------------------------------------------------
# Test 4: Manuscript Contains Honest Disclosures
# ---------------------------------------------------------------------------

def test_manuscript_contains_honest_disclosures():
    """Verify that manuscript text explicitly discloses population bounds,
    cloud obscuration limits, and non-operational research scope."""
    ms_path = Path("docs/manuscript/manuscript.md")
    assert ms_path.exists(), f"Manuscript file {ms_path} must exist"

    text = ms_path.read_text(encoding="utf-8").lower()

    # Required disclosures
    assert "population bounds" in text, "Manuscript must explicitly disclose population bounds"
    assert "cloud obscuration limits" in text or "cloud obscuration" in text, "Manuscript must disclose cloud obscuration limits"
    assert "non-operational research scope" in text or "not an operational civil defense warning system" in text, (
        "Manuscript must declare non-operational research scope"
    )

    # Primary glaciological citations
    assert "brun" in text, "Manuscript must cite Brun et al. (2017)"
    assert "veh" in text, "Manuscript must cite Veh et al. (2018/2019)"
    assert "nie" in text, "Manuscript must cite Nie et al. (2018)"
    assert "taylor" in text, "Manuscript must cite Taylor et al. (2023)"

    # Honest negative/inconclusive findings
    assert "fail_to_reject" in text, "Manuscript must disclose fail_to_reject outcomes"


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

        plot_multimodal_feature_panels(mock_panel, out1)
        plot_multimodal_feature_panels(mock_panel, out2)

        sha1 = compute_file_sha256(out1)
        sha2 = compute_file_sha256(out2)

        assert sha1 == sha2, f"Figure generation non-deterministic: {sha1} != {sha2}"

        # Test ablation figure determinism
        scores = {"full": 0.85, "opt_sar": 0.78, "sar_only": 0.60}
        abl1 = Path(tmp1) / "fig5_abl.png"
        abl2 = Path(tmp2) / "fig5_abl.png"

        plot_sensor_ablation_comparison(scores, abl1)
        plot_sensor_ablation_comparison(scores, abl2)

        assert compute_file_sha256(abl1) == compute_file_sha256(abl2)
