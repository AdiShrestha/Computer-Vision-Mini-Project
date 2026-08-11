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


# ============================================================
# C07-02: Real Data Acquisition — Sentinel-2 L2A Verification
# ============================================================

def test_sentinel2_all_lakes_present():
    """Every lake has an optical_timeseries.csv."""
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        csv_path = os.path.join(repo_root, 'data', 'raw', 'sentinel2', lake['id'], 'optical_timeseries.csv')
        assert os.path.exists(csv_path), f"Missing S2 data for {lake['id']}"


def test_sentinel2_cloud_fraction_column():
    """S2 CSVs contain cloud_fraction column with values > 0.5 somewhere."""
    import csv
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    lake_id = registry['lakes'][0]['id']
    csv_path = os.path.join(repo_root, 'data', 'raw', 'sentinel2', lake_id, 'optical_timeseries.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert 'cloud_fraction' in reader.fieldnames
        cloud_fracs = [float(row['cloud_fraction']) for row in reader
                       if row['cloud_fraction'] not in ('', 'nan', 'NaN')]
    assert any(cf > 0.5 for cf in cloud_fracs), (
        "No cloud fractions > 0.5 — suspicious for HKH optical data"
    )


def test_sentinel2_has_monsoon_gaps():
    """S2 data has NaN/missing values during monsoon months for >=60% of lakes."""
    manifest_path = os.path.join(repo_root, 'data', 'raw', 'sentinel2', 'acquisition_manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    lakes_with_monsoon_gaps = 0
    for lake_id, stats in manifest['per_lake_stats'].items():
        if stats['gap_rate_monsoon_jun_sep'] > 0.15:
            lakes_with_monsoon_gaps += 1
    frac = lakes_with_monsoon_gaps / len(manifest['per_lake_stats'])
    assert frac >= 0.60, (
        f"Only {frac*100:.0f}% of lakes have monsoon gaps > 15% — expected >=60% for real HKH data"
    )


def test_sentinel2_lake_area_plausible():
    """Lake areas are physically plausible (0.001–50 km²)."""
    import csv, math
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    lake_id = registry['lakes'][0]['id']
    csv_path = os.path.join(repo_root, 'data', 'raw', 'sentinel2', lake_id, 'optical_timeseries.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        areas = [float(row['lake_area_km2']) for row in reader
                 if row['lake_area_km2'] not in ('', 'nan', 'NaN')
                 and not math.isnan(float(row['lake_area_km2']))]
    assert len(areas) > 0, "No valid lake area measurements"
    assert all(0.001 <= a <= 50.0 for a in areas), (
        f"Lake area outside plausible range: min={min(areas)}, max={max(areas)}"
    )


