"""
Adversarial verification tests for Chunk 07 — Scientific Integrity Restoration & Real GEE Pipeline Rework.

Tests are added progressively by each contract (C07-00 through C07-05).
"""
import os
import json
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)


# ============================================================
# C07-00: INV-011 Correction Verification
# ============================================================

def test_inv011_no_ch06_reference():
    """INV-011 type 3 must not reference CH-06 after correction."""
    invariants_path = os.path.join(repo_root, 'project', 'invariants.md')
    with open(invariants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    inv011_start = content.find('## INV-011')
    inv011_end = content.find('## INV-012')
    inv011_text = content[inv011_start:inv011_end]
    assert 'CH-06' not in inv011_text, "INV-011 type 3 still references excluded CH-06"


def test_inv011_matches_decision_003():
    """INV-011 type 3 must reference CH-05 and +3 dB."""
    invariants_path = os.path.join(repo_root, 'project', 'invariants.md')
    with open(invariants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    inv011_start = content.find('## INV-011')
    inv011_end = content.find('## INV-012')
    inv011_text = content[inv011_start:inv011_end]
    assert 'CH-05' in inv011_text, "INV-011 type 3 must reference CH-05"
    assert '3 dB' in inv011_text or '+3 dB' in inv011_text, "INV-011 type 3 must specify +3 dB magnitude"


def test_inv011_revision_note_exists():
    """INV-011 must have a dated revision note for the C07-00 correction."""
    invariants_path = os.path.join(repo_root, 'project', 'invariants.md')
    with open(invariants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'C07-00' in content, "Missing C07-00 revision note in invariants.md"
    assert 'Decision 003' in content, "Revision note must reference Decision 003"


# ============================================================
# C07-01: Real Data Acquisition — Sentinel-1 GRD Verification
# ============================================================

def test_sentinel1_all_lakes_present():
    """Every lake in the registry has a backscatter_timeseries.csv."""
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        csv_path = os.path.join(repo_root, 'data', 'raw', 'sentinel1', lake['id'], 'backscatter_timeseries.csv')
        assert os.path.exists(csv_path), f"Missing S1 data for {lake['id']}"


def test_sentinel1_date_range():
    """S1 data covers >=80% of 2016-01-01 to 2024-10-31 for every lake."""
    manifest_path = os.path.join(repo_root, 'data', 'raw', 'sentinel1', 'acquisition_manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    for lake_id, stats in manifest['per_lake_stats'].items():
        assert stats['coverage_pct'] >= 80.0, (
            f"{lake_id} S1 coverage {stats['coverage_pct']:.1f}% < 80%"
        )


def test_sentinel1_has_gaps():
    """S1 data has gaps (real data is not 100% complete)."""
    manifest_path = os.path.join(repo_root, 'data', 'raw', 'sentinel1', 'acquisition_manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    coverages = [s['coverage_pct'] for s in manifest['per_lake_stats'].values()]
    assert any(c < 100.0 for c in coverages), (
        "All lakes have 100% S1 coverage — suspicious for real data"
    )


def test_sentinel1_vv_vh_present():
    """S1 CSV files contain VV and VH columns."""
    import csv
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    lake_id = registry['lakes'][0]['id']
    csv_path = os.path.join(repo_root, 'data', 'raw', 'sentinel1', lake_id, 'backscatter_timeseries.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
    assert 'vv_lake_db' in headers, "Missing VV column"
    assert 'vh_lake_db' in headers, "Missing VH column"

