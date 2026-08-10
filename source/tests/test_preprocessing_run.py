"""Verify full-scale preprocessing produced expected outputs."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
DATA_PROCESSED = os.path.join(repo_root, 'data', 'processed')


def test_preprocessing_summary_exists():
    """Preprocessing summary JSON exists."""
    summary_path = os.path.join(DATA_PROCESSED, 'preprocessing_summary.json')
    assert os.path.isfile(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    assert 'overall' in summary or 'per_source' in summary


def test_south_lhonak_preprocessed():
    """SGL-001 (South Lhonak) has preprocessed data."""
    sources = ['sentinel1_grd', 'sentinel2_l2a', 'landsat', 'modis_lst', 'itslive', 'era5']
    has_data = False
    for source in sources:
        lake_dir = os.path.join(DATA_PROCESSED, source, 'SGL-001')
        if os.path.isdir(lake_dir) and any(
            f for f in os.listdir(lake_dir)
            if f.endswith('.npz') or f.endswith('.nc') or f.endswith('.csv')
        ):
            has_data = True
            break
    assert has_data, "No preprocessed data for South Lhonak"


def test_preprocessing_script_exists():
    """run_preprocessing.py exists and compiles."""
    script_path = os.path.join(source_root, 'scripts', 'run_preprocessing.py')
    assert os.path.isfile(script_path)
    with open(script_path) as f:
        compile(f.read(), script_path, 'exec')


def test_at_least_80_percent_lakes():
    """At least 80% of lakes (16/20) have some preprocessed data."""
    lake_dirs_found = set()
    sources = ['sentinel1_grd', 'sentinel2_l2a', 'landsat', 'modis_lst', 'itslive', 'era5']
    for source in sources:
        source_dir = os.path.join(DATA_PROCESSED, source)
        if os.path.isdir(source_dir):
            for d in os.listdir(source_dir):
                if d.startswith('SGL-') and os.path.isdir(os.path.join(source_dir, d)):
                    lake_path = os.path.join(source_dir, d)
                    if any(f for f in os.listdir(lake_path) if not f.startswith('.')):
                        lake_dirs_found.add(d)
    assert len(lake_dirs_found) >= 16, (
        f"Only {len(lake_dirs_found)} lakes have preprocessed data, need >= 16"
    )
