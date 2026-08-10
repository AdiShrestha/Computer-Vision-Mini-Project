"""Top-level CLI runner for sentinel-gl channel extraction across lakes and windows.

Fulfills C02-05 requirements:
- Executes channel extraction modules across preprocessed data
- Assembles standardized 2D feature matrices: shape (n_windows, n_channels)
- Saves outputs to data/features/{lake_id}/feature_matrix.npz
- Generates data/features/feature_summary.json
"""
import os
import sys
import json
import argparse
import importlib
import numpy as np
from typing import Dict, Any, List

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from data.preprocessing.common import build_time_windows

CHANNEL_MODULES = [
    ('CH-01', 'data.channels.extract_extent'),
    ('CH-02', 'data.channels.extract_spectral'),
    ('CH-03', 'data.channels.extract_velocity'),
    ('CH-04', 'data.channels.extract_temperature'),
    ('CH-05', 'data.channels.extract_sar'),
    ('CH-07', 'data.channels.extract_sar'),
    ('CH-08', 'data.channels.extract_meteorological'),
]

COLUMN_NAMES = [
    'CH-01_area_km2',
    'CH-02_green_mean', 'CH-02_red_mean', 'CH-02_nir_mean', 'CH-02_turbidity_proxy',
    'CH-03_velocity_mean_m_yr', 'CH-03_velocity_max_m_yr',
    'CH-04_temp_anomaly_c',
    'CH-05_vv_mean_db', 'CH-05_vh_mean_db', 'CH-05_vv_vh_ratio',
    'CH-07_coherence',
    'CH-08_temp_anomaly_c', 'CH-08_precip_anomaly_mm', 'CH-08_snow_anomaly_mm'
]


def load_lake_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load canonical lake_registry.json."""
    reg_rel_path = config['paths']['lake_registry']
    repo_root = os.path.dirname(source_root)
    reg_path = os.path.join(repo_root, reg_rel_path)

    if not os.path.exists(reg_path):
        reg_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')

    with open(reg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_lake_features(lake_id: str, registry: Dict[str, Any], config: Dict[str, Any],
                          preprocessed_dir: str, features_dir: str, logger: Any) -> Dict[str, Any]:
    """Extract features and construct 2D feature matrix for one lake."""
    start_date = config.get('temporal', {}).get('start_date', '2016-01-01')
    end_date = config.get('temporal', {}).get('end_date', '2024-10-31')
    window_size = config.get('temporal', {}).get('window_size_days', 180)
    stride = config.get('temporal', {}).get('stride_days', 30)

    windows = build_time_windows(start_date, end_date, window_size, stride)
    n_windows = len(windows)
    n_channels = len(COLUMN_NAMES)

    features_matrix = np.full((n_windows, n_channels), np.nan, dtype=np.float32)
    quality_matrix = np.zeros((n_windows, n_channels), dtype=np.float32)
    window_dates = [w[0] for w in windows]

    for w_idx, (w_start, w_end) in enumerate(windows):
        col_offset = 0

        # CH-01: Extent
        try:
            m = importlib.import_module('data.channels.extract_extent')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry)
            features_matrix[w_idx, col_offset] = r.get('value', np.nan)
            quality_matrix[w_idx, col_offset] = r.get('quality', 0.0)
        except Exception:
            pass
        col_offset += 1

        # CH-02: Spectral (4 values)
        try:
            m = importlib.import_module('data.channels.extract_spectral')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry)
            val = r.get('value', {})
            q = r.get('quality', 0.0)
            if isinstance(val, dict):
                features_matrix[w_idx, col_offset] = val.get('green_mean', np.nan)
                features_matrix[w_idx, col_offset+1] = val.get('red_mean', np.nan)
                features_matrix[w_idx, col_offset+2] = val.get('nir_mean', np.nan)
                features_matrix[w_idx, col_offset+3] = val.get('turbidity_proxy', np.nan)
                quality_matrix[w_idx, col_offset:col_offset+4] = q
        except Exception:
            pass
        col_offset += 4

        # CH-03: Velocity (2 values)
        try:
            m = importlib.import_module('data.channels.extract_velocity')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry)
            val = r.get('value', {})
            q = r.get('quality', 0.0)
            if isinstance(val, dict):
                features_matrix[w_idx, col_offset] = val.get('velocity_mean_m_yr', np.nan)
                features_matrix[w_idx, col_offset+1] = val.get('velocity_max_m_yr', np.nan)
                quality_matrix[w_idx, col_offset:col_offset+2] = q
        except Exception:
            pass
        col_offset += 2

        # CH-04: Temperature
        try:
            m = importlib.import_module('data.channels.extract_temperature')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry)
            features_matrix[w_idx, col_offset] = r.get('value', np.nan)
            quality_matrix[w_idx, col_offset] = r.get('quality', 0.0)
        except Exception:
            pass
        col_offset += 1

        # CH-05: SAR Backscatter (3 values)
        try:
            m = importlib.import_module('data.channels.extract_sar')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry, channel_id="CH-05")
            val = r.get('value', {})
            q = r.get('quality', 0.0)
            if isinstance(val, dict):
                features_matrix[w_idx, col_offset] = val.get('vv_mean_db', np.nan)
                features_matrix[w_idx, col_offset+1] = val.get('vh_mean_db', np.nan)
                features_matrix[w_idx, col_offset+2] = val.get('vv_vh_ratio', np.nan)
                quality_matrix[w_idx, col_offset:col_offset+3] = q
        except Exception:
            pass
        col_offset += 3

        # CH-07: SAR Coherence
        try:
            m = importlib.import_module('data.channels.extract_sar')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry, channel_id="CH-07")
            features_matrix[w_idx, col_offset] = r.get('value', np.nan)
            quality_matrix[w_idx, col_offset] = r.get('quality', 0.0)
        except Exception:
            pass
        col_offset += 1

        # CH-08: Meteorological (3 values)
        try:
            m = importlib.import_module('data.channels.extract_meteorological')
            r = m.extract(lake_id, w_start, w_end, preprocessed_dir, config, registry)
            val = r.get('value', {})
            q = r.get('quality', 0.0)
            if isinstance(val, dict):
                features_matrix[w_idx, col_offset] = val.get('temp_anomaly_c', np.nan)
                features_matrix[w_idx, col_offset+1] = val.get('precip_anomaly_mm', np.nan)
                features_matrix[w_idx, col_offset+2] = val.get('snow_anomaly_mm', np.nan)
                quality_matrix[w_idx, col_offset:col_offset+3] = q
        except Exception:
            pass
        col_offset += 3

    lake_feat_dir = os.path.join(features_dir, lake_id)
    os.makedirs(lake_feat_dir, exist_ok=True)
    out_matrix_path = os.path.join(lake_feat_dir, 'feature_matrix.npz')

    np.savez_compressed(
        out_matrix_path,
        features=features_matrix,
        window_dates=np.array(window_dates),
        channel_names=np.array(COLUMN_NAMES),
        quality=quality_matrix,
        metadata={"lake_id": lake_id, "start_date": start_date, "end_date": end_date}
    )

    valid_cells = int(np.sum(~np.isnan(features_matrix)))
    total_cells = features_matrix.size
    completeness = float(valid_cells / total_cells) if total_cells > 0 else 0.0

    return {
        "lake_id": lake_id,
        "n_windows": n_windows,
        "n_channels": n_channels,
        "completeness": completeness,
        "out_path": out_matrix_path
    }


def main():
    parser = argparse.ArgumentParser(description="sentinel-gl Channel Extraction Runner")
    parser.add_argument("--lake", default="all", help="Lake ID (e.g., 'SGL-001' or 'all')")
    args = parser.parse_args()

    config = load_config()
    registry = load_lake_registry(config)
    logger = setup_logger("run_channel_extraction")

    repo_root = os.path.dirname(source_root)
    preprocessed_dir = os.path.join(repo_root, config['paths']['processed_data'])
    features_dir = os.path.join(repo_root, config['paths'].get('features', 'data/features'))
    os.makedirs(features_dir, exist_ok=True)

    lakes_to_run = [l['id'] for l in registry['lakes']] if args.lake.lower() == 'all' else [args.lake]

    logger.info(f"Starting channel extraction for {len(lakes_to_run)} lake(s)...")

    per_lake_stats = {}
    for l_id in lakes_to_run:
        res = extract_lake_features(l_id, registry, config, preprocessed_dir, features_dir, logger)
        per_lake_stats[l_id] = res

    total_matrices = len(per_lake_stats)
    mean_completeness = float(np.mean([r['completeness'] for r in per_lake_stats.values()]))

    summary = {
        "overall": {
            "total_feature_matrices": total_matrices,
            "n_channels_per_matrix": len(COLUMN_NAMES),
            "mean_completeness": mean_completeness,
            "channels": COLUMN_NAMES
        },
        "per_lake": per_lake_stats
    }

    summary_file = os.path.join(features_dir, 'feature_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Feature summary saved to: {summary_file}")
    print("\nChannel Extraction Batch Completed.")
    print(f"Total feature matrices: {total_matrices}, Channels: {len(COLUMN_NAMES)}, Mean completeness: {mean_completeness:.2%}")


if __name__ == '__main__':
    main()
