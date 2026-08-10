"""ITS_LIVE glacier velocity acquisition module."""
import os
import requests
from typing import Dict, Any
from utils.hashing import hash_file


def acquire(lake_id: str, start_date: str, end_date: str,
            registry: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Download ITS_LIVE velocity composite metadata for feeding glaciers.

    Args:
        lake_id: Target lake ID.
        start_date: Start date string.
        end_date: End date string.
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
        "source": "itslive",
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

    lake_dir = os.path.join(output_dir, 'itslive', lake_id)
    os.makedirs(lake_dir, exist_ok=True)

    try:
        url = "https://nsidc.org/apps/itslive-search/api/v1/search"
        params = {
            "bbox": f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
            "start": start_date,
            "end": end_date
        }

        # Handle retry loop
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    out_file = os.path.join(lake_dir, "velocity_search.json")
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(resp.text)

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
                    break
                else:
                    out_file = os.path.join(lake_dir, "query_metadata.json")
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(f'{{"status": {resp.status_code}, "bbox": "{params["bbox"]}"}}')
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
                    break
            except Exception as e:
                if attempt == 2:
                    raise e
                import time
                time.sleep(2 ** attempt)

    except Exception as e:
        manifest['errors'].append({
            "date": start_date,
            "error": str(e)
        })
        manifest['metadata']['failed'] += 1

    return manifest
