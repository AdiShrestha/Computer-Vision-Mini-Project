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


# ============================================================
# C07-03: Real Data Acquisition — Auxiliary Channels & CH-07 Removal Verification
# ============================================================

def test_itslive_coverage():
    """ITS_LIVE has >=5 annual observations per lake."""
    import csv
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        csv_path = os.path.join(repo_root, 'data', 'raw', 'itslive', lake['id'], 'velocity_timeseries.csv')
        assert os.path.exists(csv_path), f"Missing ITS_LIVE for {lake['id']}"
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 5, f"{lake['id']} has only {len(rows)} ITS_LIVE obs (need >=5)"


def test_modis_lst_coverage():
    """MODIS LST has >70% temporal coverage per lake."""
    manifest_path = os.path.join(repo_root, 'data', 'raw', 'auxiliary_acquisition_manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    avg_coverage = manifest['per_source_stats']['MODIS_LST']['coverage_pct_avg']
    assert avg_coverage >= 70.0, f"MODIS LST coverage {avg_coverage}% < 70%"


def test_era5_coverage():
    """ERA5 has >95% temporal coverage."""
    manifest_path = os.path.join(repo_root, 'data', 'raw', 'auxiliary_acquisition_manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    avg_coverage = manifest['per_source_stats']['ERA5']['coverage_pct_avg']
    assert avg_coverage >= 95.0, f"ERA5 coverage {avg_coverage}% < 95%"


def test_no_ch07_coherence_files():
    """No file or column labeled 'coherence' or 'CH-07' exists in raw data."""
    import glob
    coherence_files = glob.glob(os.path.join(repo_root, 'data', 'raw', '**', 'coherence*'), recursive=True)
    assert len(coherence_files) == 0, f"Found coherence files: {coherence_files}"
    coherence_files2 = glob.glob(os.path.join(repo_root, 'data', 'raw', '**', 'ch07*'), recursive=True)
    assert len(coherence_files2) == 0, f"Found CH-07 files: {coherence_files2}"

    manifest_path = os.path.join(repo_root, 'data', 'raw', 'auxiliary_acquisition_manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    assert 'CH-07' in manifest.get('channels_dropped', {}), (
        "CH-07 must be listed as dropped in manifest"
    )


# ============================================================
# C07-04: Feature Matrix Assembly & Reality Gate Verification
# ============================================================

def test_feature_matrices_13_channels():
    """All feature matrices have exactly 13 channels."""
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        npz_path = os.path.join(repo_root, 'data', 'features_real', lake['id'], 'feature_matrix.npz')
        assert os.path.exists(npz_path), f"Missing features for {lake['id']}"
        data = np.load(npz_path)
        assert data['features'].shape[1] == 13, (
            f"{lake['id']} has {data['features'].shape[1]} channels, expected 13"
        )


def test_feature_matrices_all_lakes():
    """All 20 lakes have feature matrices."""
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        assert os.path.exists(os.path.join(repo_root, 'data', 'features_real', lake['id'], 'feature_matrix.npz'))


def test_reality_gate_no_fail():
    """Reality Gate has no FAIL verdicts."""
    gate_path = os.path.join(repo_root, 'results', 'reality_gate', 'reality_gate_data.json')
    with open(gate_path, 'r', encoding='utf-8') as f:
        gate = json.load(f)
    for name, check in gate['checks'].items():
        assert check['verdict'] != 'FAIL', (
            f"Reality Gate FAIL on {name}: {check['reason']}"
        )


def test_normalization_training_only():
    """Normalization stats computed from training-role lakes only (INV-002)."""
    norm_path = os.path.join(repo_root, 'data', 'features_real', 'normalization_stats.json')
    with open(norm_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    training_ids = {l['id'] for l in registry['lakes'] if l['role'] == 'training'}
    computed_from = set(stats['computed_from'])
    assert computed_from == training_ids, (
        f"Normalization used non-training lakes: {computed_from - training_ids}"
    )


# ============================================================
# C07-05: TS-MAE Encoder Retraining on Real Features Verification
# ============================================================

def test_training_convergence():
    """Training loss decreased over epochs."""
    summary_path = os.path.join(repo_root, 'models', 'encoder', 'training_summary_real_data.json')
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    losses = summary['loss_history']
    assert len(losses) >= 5, f"Only {len(losses)} epochs recorded"
    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]} -> final={losses[-1]}"


def test_no_evaluation_lake_leakage():
    """Training used only training-role lake data (INV-002)."""
    summary_path = os.path.join(repo_root, 'models', 'encoder', 'training_summary_real_data.json')
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    training_ids = {l['id'] for l in registry['lakes'] if l['role'] == 'training'}
    used_ids = set(summary['training_lake_ids'])
    assert used_ids == training_ids, f"Leakage: training used {used_ids - training_ids}"


def test_embeddings_exist_all_lakes():
    """Embeddings extracted for all 20 lakes."""
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        emb_path = os.path.join(repo_root, 'data', 'embeddings', 'real_data', lake['id'], 'embeddings.npz')
        assert os.path.exists(emb_path), f"Missing embeddings for {lake['id']}"


def test_embeddings_nontrivial_variance():
    """Embeddings have nontrivial variance (not all zeros or constants)."""
    registry_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    for lake in registry['lakes']:
        emb_path = os.path.join(repo_root, 'data', 'embeddings', 'real_data', lake['id'], 'embeddings.npz')
        data = np.load(emb_path)
        embs = data['embeddings']
        assert embs.std() > 0.01, f"{lake['id']} embeddings have near-zero variance ({embs.std():.6f})"


def test_checkpoint_loadable():
    """Checkpoint can be loaded and model parameters exist."""
    import torch
    ckpt_path = os.path.join(repo_root, 'models', 'checkpoints', 'ts_mae_real_data.pt')
    assert os.path.exists(ckpt_path), "Checkpoint not found"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in ckpt
    assert ckpt['n_channels'] == 13





