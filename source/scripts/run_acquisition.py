"""Top-level CLI runner for sentinel-gl full-scale data acquisition across sources and lakes.

Fulfills C02-01 requirements:
- Executes acquisition per-source, per-lake across 2016-2024 temporal extent
- Generates per-source data/raw/{source}/manifest.json
- Generates data/raw/acquisition_summary.json
"""
import os
import sys
import json
import argparse
import importlib
from typing import Dict, Any

# Ensure source root is in sys.path
source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.logging_utils import setup_logger, log_to_jsonl
from utils.hashing import hash_file

AVAILABLE_SOURCES = ['sentinel1', 'sentinel2', 'landsat', 'modis', 'itslive', 'era5']
SOURCE_DIR_MAP = {
    'sentinel1': 'sentinel1_grd',
    'sentinel2': 'sentinel2_l2a',
    'landsat': 'landsat',
    'modis': 'modis_lst',
    'itslive': 'itslive',
    'era5': 'era5'
}


def load_lake_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load canonical lake_registry.json referenced in config paths."""
    reg_rel_path = config['paths']['lake_registry']
    repo_root = os.path.dirname(source_root)
    reg_path = os.path.join(repo_root, reg_rel_path)

    if not os.path.exists(reg_path):
        reg_path = os.path.join(source_root, 'data', 'registry', 'lake_registry.json')

    with open(reg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_single_acquisition(source_name: str, lake_id: str, start_date: str, end_date: str,
                           registry: Dict[str, Any], config: Dict[str, Any], output_dir: str,
                           logger: Any) -> Dict[str, Any]:
    """Execute acquisition for a single source and lake."""
    mod_name = f"data.acquisition.acquire_{source_name}"
    try:
        module = importlib.import_module(mod_name)
    except Exception as e:
        logger.error(f"Failed to import acquisition module {mod_name}: {e}")
        return {
            "source": source_name,
            "lake_id": lake_id,
            "files": [],
            "errors": [{"date": start_date, "error": f"ImportError: {e}"}],
            "metadata": {"start_date": start_date, "end_date": end_date, "total_scenes": 0, "successful": 0, "failed": 1}
        }

    logger.info(f"Running acquisition: source={source_name}, lake={lake_id}, range={start_date} to {end_date}")
    result = module.acquire(lake_id, start_date, end_date, registry, config, output_dir)

    source_dir_name = SOURCE_DIR_MAP.get(source_name, source_name)
    manifest_dir = os.path.join(output_dir, source_dir_name)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_file = os.path.join(manifest_dir, 'manifest.json')

    # Maintain a JSON array manifest file per source
    manifest_data = []
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    manifest_data = loaded
                elif isinstance(loaded, dict):
                    manifest_data = [loaded]
        except Exception:
            manifest_data = []

    for f_info in result.get('files', []):
        entry = {
            "lake_id": lake_id,
            "date": f_info.get('date'),
            "file_path": f_info.get('path'),
            "sha256_hash": f_info.get('hash'),
            "size_bytes": f_info.get('size_bytes'),
            "source_api": source_name
        }
        manifest_data.append(entry)

    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="sentinel-gl Data Acquisition Runner")
    parser.add_argument("--lake", default="all", help="Lake ID (e.g., 'SGL-001' or 'all')")
    parser.add_argument("--source", default="all", help=f"Data source ({', '.join(AVAILABLE_SOURCES)} or 'all')")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: config start_date)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: config end_date)")
    parser.add_argument("--output-dir", default=None, help="Output root directory (default: data/raw)")
    args = parser.parse_args()

    config = load_config()
    registry = load_lake_registry(config)
    logger = setup_logger("run_acquisition")

    start_date = args.start or config['temporal']['start_date']
    end_date = args.end or config['temporal']['end_date']

    repo_root = os.path.dirname(source_root)
    output_dir = args.output_dir or os.path.join(repo_root, config['paths']['raw_data'])

    lakes_to_run = [l['id'] for l in registry['lakes']] if args.lake.lower() == 'all' else [args.lake]
    sources_to_run = AVAILABLE_SOURCES if args.source.lower() == 'all' else [args.source.lower()]

    logger.info(f"Starting acquisition batch for {len(lakes_to_run)} lake(s) across {len(sources_to_run)} source(s)...")

    results = []
    per_source_stats = {s: {"successful": 0, "failed": 0, "size_bytes": 0} for s in AVAILABLE_SOURCES}
    per_lake_stats = {l: {} for l in lakes_to_run}

    for l_id in lakes_to_run:
        for s_name in sources_to_run:
            res = run_single_acquisition(s_name, l_id, start_date, end_date, registry, config, output_dir, logger)
            results.append(res)

            succ = res.get('metadata', {}).get('successful', 0)
            fail = res.get('metadata', {}).get('failed', 0)
            bytes_total = sum(f.get('size_bytes', 0) for f in res.get('files', []))

            per_source_stats[s_name]["successful"] += succ
            per_source_stats[s_name]["failed"] += fail
            per_source_stats[s_name]["size_bytes"] += bytes_total

            per_lake_stats[l_id][s_name] = {
                "successful": succ,
                "failed": fail,
                "bytes": bytes_total
            }

    # Generate acquisition_summary.json
    total_files = sum(s["successful"] for s in per_source_stats.values())
    total_bytes = sum(s["size_bytes"] for s in per_source_stats.values())
    total_attempts = total_files + sum(s["failed"] for s in per_source_stats.values())
    success_rate = (total_files / total_attempts) if total_attempts > 0 else 1.0

    summary = {
        "overall": {
            "total_lakes": len(lakes_to_run),
            "total_files": total_files,
            "total_size_bytes": total_bytes,
            "success_rate": success_rate
        },
        "per_source": per_source_stats,
        "per_lake": per_lake_stats
    }

    summary_file = os.path.join(output_dir, 'acquisition_summary.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Acquisition summary saved to: {summary_file}")
    print("\nAcquisition Batch Completed.")
    print(f"Total files: {total_files}, Total size: {total_bytes} bytes, Success rate: {success_rate:.2%}")


if __name__ == '__main__':
    main()
