"""Copernicus ERA5 reanalysis acquisition module via CDS API."""
import os
from typing import Dict, Any
from utils.hashing import hash_file


def acquire(lake_id: str, start_date: str, end_date: str,
            registry: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Download ERA5 temperature and precipitation reanalysis time series.

    Args:
        lake_id: Target lake ID.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        registry: Loaded lake registry dictionary.
        config: Loaded configuration dictionary.
        output_dir: Root output directory.

    Returns:
        Dict[str, Any]: Acquisition manifest record dictionary.
    """
    lakes = {l['id']: l for l in registry.get('lakes', [])}
    if lake_id not in lakes:
        raise ValueError(f"Lake ID {lake_id} not found in registry.")

    lake = lakes[lake_id]
    bbox = lake['bounding_box']

    manifest = {
        "source": "era5",
        "lake_id": lake_id,
        "files": [],
        "errors": [],
        "metadata": {
            "start_date": start_date,
            "end_date": end_date,
            "total_scenes": 0,
            "successful": 0,
            "failed": 0
        }
    }

    lake_dir = os.path.join(output_dir, 'era5', lake_id)
    os.makedirs(lake_dir, exist_ok=True)

    try:
        import cdsapi
        cds_file = os.path.expanduser('~/.cdsapirc')
        if not os.path.exists(cds_file):
            raise RuntimeError("Missing ~/.cdsapirc file for Copernicus CDS authentication.")

        c = cdsapi.Client()
        out_file = os.path.join(lake_dir, f"era5_{start_date}_{end_date}.nc")

        year = start_date.split('-')[0]
        month = start_date.split('-')[1]

        c.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature', 'total_precipitation'],
            'year': year,
            'month': month,
            'day': ['01', '15'],
            'time': '12:00',
            'area': [bbox['north'], bbox['west'], bbox['south'], bbox['east']],
            'format': 'netcdf',
        }, out_file)

        file_hash = hash_file(out_file)
        size_bytes = os.path.getsize(out_file)

        manifest['files'].append({
            "path": out_file,
            "date": start_date,
            "hash": file_hash,
            "size_bytes": size_bytes
        })
        manifest['metadata']['total_scenes'] = 1
        manifest['metadata']['successful'] = 1

    except Exception as e:
        manifest['errors'].append({
            "date": start_date,
            "error": str(e)
        })
        manifest['metadata']['failed'] += 1

    return manifest
