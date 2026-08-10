"""Top-level CLI runner for sentinel-gl data acquisition across sources and lakes.

CLI Usage:
    python source/scripts/run_acquisition.py --lake SGL-001 --source sentinel1 --start 2023-01-01 --end 2023-01-31
    python source/scripts/run_acquisition.py --lake all --source sentinel1
    python source/scripts/run_acquisition.py --lake SGL-001 --source all
"""
import os
import sys
import json
import argparse
import importlib
from typing import Dict, Any

# Add source root to path
source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.logging_utils import setup_logger, log_to_jsonl

AVAILABLE_SOURCES = ['sentinel1', 'sentinel2', 'landsat', 'modis', 'itslive', 'era5']


def load_lake_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load canonical lake_registry.json referenced in config paths."""
    reg_rel_path = config['paths']['lake_registry']
    repo_root = os.path.dirname(source_root)
    reg_path = os.path.join(repo_root, reg_rel_path)

    if not os.path.exists(reg_path):
        # Fallback to source root relative path
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

    # Append to acquisition manifest
    manifest_dir = os.path.join(output_dir, source_name)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_file = os.path.join(manifest_dir, 'manifest.json')

    log_to_jsonl(manifest_file, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="sentinel-gl Data Acquisition Runner")
    parser.add_argument("--lake", default="SGL-001", help="Lake ID (e.g., 'SGL-001' or 'all')")
    parser.add_argument("--source", default="sentinel1", help=f"Data source ({', '.join(AVAILABLE_SOURCES)} or 'all')")
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
    for l_id in lakes_to_run:
        for s_name in sources_to_run:
            res = run_single_acquisition(s_name, l_id, start_date, end_date, registry, config, output_dir, logger)
            results.append(res)

    print("\nAcquisition Batch Completed.")
    for r in results:
        meta = r.get('metadata', {})
        print(f"[{r['source']}] Lake {r['lake_id']}: {meta.get('successful', 0)} files succeeded, {meta.get('failed', 0)} failed.")


if __name__ == '__main__':
    main()
