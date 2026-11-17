"""The actual WP04 consumer CLI must verify the retained offline draft."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_cli_verifies_generated_bundle_and_keeps_research_blocked():
    root = Path(__file__).resolve().parents[2]
    runner = root / "source/runners/run_wp04_manifest.py"
    bundle = root / "project/architecture/wp04_local_2026-10-14/run_003"
    result = subprocess.run([sys.executable, str(runner), "verify", "--root", str(root),
        "--bundle", str(bundle)], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["result"]["status"] == "PASS_INTEGRITY_LOCAL_DEVELOPMENT_DRAFT_ONLY"
    assert report["result"]["scope_roles"] == [
        "fitting", "development_validation", "threshold_calibration", "evaluation"]
    assert report["result"]["verified_source_and_receipt_files"] == 17
    assert report["result"]["verified_decoded_output_files"] == 50
    assert report["result"]["verified_historical_failed_inspections"] == 2
    assert report["result"]["verified_current_decode_products"] == 2
    assert report["result"]["research_status"] == "RESEARCH_NOT_READY"
    assert report["result"]["plan_executable"] is False
    assert report["result"]["release_status"] == "BLOCKED"
    assert report["runtime"]["network"] == "NOT_USED"


def test_cli_rejects_output_path_traversal_without_creating_output():
    root = Path(__file__).resolve().parents[2]
    runner = root / "source/runners/run_wp04_manifest.py"
    result = subprocess.run([sys.executable, str(runner), "build", "--root", str(root),
        "--output", "../../outside-wp04"], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stderr)["literal_error"] == "output path traversal is not permitted"
