"""Verify full-scale acquisition produced expected outputs."""
import os
import sys
import json

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

DATA_RAW = os.path.join(repo_root, 'data', 'raw')


def test_acquisition_summary_exists():
    """Acquisition summary JSON exists."""
    summary_path = os.path.join(DATA_RAW, 'acquisition_summary.json')
    assert os.path.isfile(summary_path), "acquisition_summary.json not found"
    with open(summary_path) as f:
        summary = json.load(f)
    assert 'overall' in summary or 'per_source' in summary


def test_at_least_one_source_has_data():
    """At least one data source directory contains downloaded files."""
    sources = ['sentinel1_grd', 'sentinel2_l2a', 'landsat', 'modis_lst', 'itslive', 'era5']
    has_data = False
    for source in sources:
        source_dir = os.path.join(DATA_RAW, source)
        if os.path.isdir(source_dir):
            for lake_dir in os.listdir(source_dir):
                lake_path = os.path.join(source_dir, lake_dir)
                if os.path.isdir(lake_path) and any(
                    f for f in os.listdir(lake_path) 
                    if not f.startswith('.')
                ):
                    has_data = True
                    break
    assert has_data, "No data downloaded from any source"


def test_south_lhonak_has_data():
    """SGL-001 (South Lhonak) has data from at least one source."""
    sources = ['sentinel1_grd', 'sentinel2_l2a', 'landsat', 'modis_lst', 'itslive', 'era5']
    has_data = False
    for source in sources:
        lake_dir = os.path.join(DATA_RAW, source, 'SGL-001')
        if os.path.isdir(lake_dir) and any(
            f for f in os.listdir(lake_dir) if not f.startswith('.')
        ):
            has_data = True
            break
    assert has_data, "No data for South Lhonak (SGL-001) from any source"


def test_manifests_have_hashes():
    """Manifest files contain SHA-256 hashes for downloaded files."""
    sources = ['sentinel1_grd', 'sentinel2_l2a', 'landsat', 'modis_lst', 'itslive', 'era5']
    for source in sources:
        manifest_path = os.path.join(DATA_RAW, source, 'manifest.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if isinstance(manifest, list) and len(manifest) > 0:
                file_entries = [e for e in manifest if isinstance(e, dict) and ('file_path' in e or 'path' in e)]
                if file_entries:
                    entry = file_entries[0]
                    assert 'sha256_hash' in entry or 'hash' in entry, (
                        f"Manifest for {source} missing hash field"
                    )
