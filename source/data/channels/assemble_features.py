"""
Feature Matrix Assembly — Real Data Pipeline.

Assembles 13-channel feature matrices from raw satellite time series.
Missing values remain as NaN — no interpolation.
Normalization statistics computed from training-role lakes only (INV-002).

Output: data/features_real/{lake_id}/feature_matrix.npz
        data/features_real/normalization_stats.json
        data/features_real/channel_map.json
"""

import os
import sys
import json
import csv
import numpy as np
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_lake_registry():
    reg_path = PROJECT_ROOT / 'source' / 'data' / 'registry' / 'lake_registry.json'
    with open(reg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def date_range_daily(start_str, end_str):
    """Generate daily date index from start to end (inclusive)."""
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return dates


def load_csv_as_dict(csv_path, date_col='date'):
    """Load CSV, return dict of date -> row."""
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get(date_col, '').strip()
            if d:
                data[d] = row
    return data


def safe_float(val):
    """Convert to float, return NaN for invalid/missing."""
    if val is None or val == '' or str(val).lower() in ('nan', 'none', 'null'):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def assemble_lake_features(lake_id, dates, raw_root):
    """Assemble 13-channel feature matrix for one lake.

    Returns: np.ndarray of shape (T, 13) where T = len(dates).
    Missing values are NaN.
    """
    T = len(dates)
    C = 13
    matrix = np.full((T, C), np.nan, dtype=np.float64)

    # Load raw data
    s2 = load_csv_as_dict(raw_root / f'sentinel2/{lake_id}/optical_timeseries.csv')
    s1 = load_csv_as_dict(raw_root / f'sentinel1/{lake_id}/backscatter_timeseries.csv')
    itslive = load_csv_as_dict(raw_root / f'itslive/{lake_id}/velocity_timeseries.csv')
    modis = load_csv_as_dict(raw_root / f'modis/{lake_id}/lst_timeseries.csv')
    era5 = load_csv_as_dict(raw_root / f'era5/{lake_id}/meteorology_timeseries.csv')

    # Sort ITS_LIVE dates ascending for forward fill lookup
    itslive_dates_sorted = sorted(itslive.keys())

    for t, date_str in enumerate(dates):
        # CH-01: Lake area (Sentinel-2)
        if date_str in s2:
            matrix[t, 0] = safe_float(s2[date_str].get('lake_area_km2'))
        # CH-02a-d: Spectral (Sentinel-2)
        if date_str in s2:
            matrix[t, 1] = safe_float(s2[date_str].get('green_mean'))
            matrix[t, 2] = safe_float(s2[date_str].get('red_mean'))
            matrix[t, 3] = safe_float(s2[date_str].get('nir_mean'))
            matrix[t, 4] = safe_float(s2[date_str].get('ndwi_mean'))
        # CH-03a-b: Glacier velocity (ITS_LIVE) — annual, forward-fill
        # Find the most recent ITS_LIVE observation on or before this date
        for itslive_date in reversed(itslive_dates_sorted):
            if itslive_date <= date_str:
                matrix[t, 5] = safe_float(itslive[itslive_date].get('velocity_x_m_yr'))
                matrix[t, 6] = safe_float(itslive[itslive_date].get('velocity_y_m_yr'))
                break
        # CH-04: LST anomaly (MODIS)
        if date_str in modis:
            matrix[t, 7] = safe_float(modis[date_str].get('lst_anomaly_kelvin'))
        # CH-05a-c: SAR backscatter (Sentinel-1)
        if date_str in s1:
            matrix[t, 8] = safe_float(s1[date_str].get('vv_lake_db'))
            matrix[t, 9] = safe_float(s1[date_str].get('vh_lake_db'))
            matrix[t, 10] = safe_float(s1[date_str].get('vv_moraine_db'))
        # CH-08a-b: ERA5 meteorology
        if date_str in era5:
            matrix[t, 11] = safe_float(era5[date_str].get('temperature_2m_k'))
            matrix[t, 12] = safe_float(era5[date_str].get('total_precip_m_day'))

    return matrix


def compute_normalization_stats(feature_dir, registry, dates):
    """Compute z-score normalization stats from training-role lakes ONLY (INV-002)."""
    training_lakes = [l['id'] for l in registry['lakes'] if l['role'] == 'training']
    all_features = []
    for lake_id in training_lakes:
        npz_path = feature_dir / lake_id / 'feature_matrix.npz'
        if npz_path.exists():
            data = np.load(npz_path)
            all_features.append(data['features'])

    if not all_features:
        raise RuntimeError("No training lake features found for normalization")

    stacked = np.concatenate(all_features, axis=0)  # (total_days, 13)

    # Compute mean and std ignoring NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(stacked, axis=0)  # (13,)
        stds = np.nanstd(stacked, axis=0)   # (13,)

    # Avoid division by zero
    stds[stds < 1e-8] = 1.0

    stats = {
        'computed_from': training_lakes,
        'n_training_lakes': len(training_lakes),
        'channel_count': 13,
        'means': means.tolist(),
        'stds': stds.tolist()
    }
    return stats


def assemble_all():
    registry = load_lake_registry()
    raw_root = PROJECT_ROOT / 'data' / 'raw'
    feature_dir = PROJECT_ROOT / 'data' / 'features_real'
    feature_dir.mkdir(parents=True, exist_ok=True)

    dates = date_range_daily('2016-01-01', '2024-10-31')
    print(f"Temporal extent: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Channel map
    channel_map = {
        "0": "CH-01_lake_area_km2",
        "1": "CH-02a_green_reflectance",
        "2": "CH-02b_red_reflectance",
        "3": "CH-02c_nir_reflectance",
        "4": "CH-02d_ndwi",
        "5": "CH-03a_glacier_velocity_x",
        "6": "CH-03b_glacier_velocity_y",
        "7": "CH-04_lst_anomaly_k",
        "8": "CH-05a_vv_backscatter_lake_db",
        "9": "CH-05b_vh_backscatter_lake_db",
        "10": "CH-05c_vv_backscatter_moraine_db",
        "11": "CH-08a_temperature_2m_k",
        "12": "CH-08b_total_precip_m_day"
    }
    with open(feature_dir / 'channel_map.json', 'w', encoding='utf-8') as f:
        json.dump(channel_map, f, indent=2)

    # Assemble per lake
    for lake in registry['lakes']:
        lake_id = lake['id']
        lake_dir = feature_dir / lake_id
        lake_dir.mkdir(parents=True, exist_ok=True)

        matrix = assemble_lake_features(lake_id, dates, raw_root)
        np.savez_compressed(lake_dir / 'feature_matrix.npz', features=matrix, dates=dates)

    # Normalization stats (training lakes only — INV-002)
    stats = compute_normalization_stats(feature_dir, registry, dates)
    with open(feature_dir / 'normalization_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == '__main__':
    assemble_all()
