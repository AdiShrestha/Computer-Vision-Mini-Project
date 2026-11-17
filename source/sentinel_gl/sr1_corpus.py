"""Source-bound loading of the SR-1 Sentinel-2 corpus.

Every scene must be an ``observational`` row whose ROI array, receipt, STAC
item (Earth Search rows) or parent 2023 record (retained rows), and native
SAFE metadata receipts match their recorded SHA-256 digests. The loader
refuses constructed, unknown or mismatched inputs; it never substitutes data.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SITES = ("PILOT-SOUTH-LHONAK", "PILOT-KHANGCHUNG-TSHO")
SITE_LABELS = {"PILOT-SOUTH-LHONAK": "South Lhonak", "PILOT-KHANGCHUNG-TSHO": "Khangchung Tsho"}
# Event bracket in UTC: 3 October 2023 00:00 to 5 October 2023 00:00 Sikkim time (UTC+05:30).
EVENT_START_UTC = datetime.fromisoformat("2023-10-02T18:30:00+00:00")
EVENT_END_UTC = datetime.fromisoformat("2023-10-04T18:30:00+00:00")
ROI_SHAPE = (160, 160)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def role_for_time(dt: datetime) -> str:
    """Frozen retrospective split (protocol v4, unchanged by amendment PLOS-001)."""
    if dt.year in (2018, 2019, 2020, 2021):
        return "FIT"
    if dt.year == 2022:
        return "VAL"
    if dt.year == 2023 and dt.month <= 6:
        return "FIT"
    if dt.year == 2023 and 7 <= dt.month <= 9:
        return "VAL"
    if EVENT_START_UTC <= dt < EVENT_END_UTC:
        return "EVENT_UNCERTAIN"
    if dt >= EVENT_END_UTC:
        return "EVAL"
    return "OUT_OF_PROTOCOL_WINDOW"


def _verify_range_receipts(row: dict) -> None:
    for band in ("B03", "B08", "B11", "SCL"):
        receipts = row.get("range_receipts", {}).get(band, {}).get("range_receipts", [])
        if not receipts:
            raise ValueError(f"no byte-range receipt for {row['item_id']} {band}")
        for receipt in receipts:
            if receipt.get("http_status") != 206:
                raise ValueError(f"non-206 COG request in {row['item_id']} {band}")
            if not re.fullmatch(r"bytes \d+-\d+/\d+", receipt.get("content_range", "")):
                raise ValueError(f"malformed Content-Range in {row['item_id']} {band}")
            if receipt.get("body_status") != "COMPLETE_RESPONSE" or not receipt.get("sha256"):
                raise ValueError(f"unverified range payload in {row['item_id']} {band}")


def _verify_metadata_receipts(row: dict, item: dict, root: Path) -> None:
    for role in ("product_metadata", "granule_metadata"):
        receipt = row.get("metadata_receipts", {}).get(role)
        if not receipt or receipt.get("http_status") != 200 or receipt.get("body_status") != "COMPLETE_RESPONSE":
            raise ValueError(f"missing successful native metadata receipt for {row['item_id']} {role}")
        if receipt.get("url") != item.get("assets", {}).get(role, {}).get("href"):
            raise ValueError(f"native metadata receipt URL mismatch for {row['item_id']} {role}")
        path = root / receipt["path"]
        if digest(path) != receipt.get("sha256") or receipt.get("received_bytes") != path.stat().st_size:
            raise ValueError(f"native metadata file mismatch for {row['item_id']} {role}")


def item_path_for(row: dict, root: Path) -> Path:
    if row.get("item_file"):
        return root / row["item_file"]
    # Earlier attempts store item.json next to the provider receipt.
    return (root / row["receipt_file"]).parent / "item.json"


def load_manifest(manifest_path: Path, root: Path, *, parent_2023_manifest: Path | None = None) -> tuple[list[dict], dict]:
    """Load and verify every observation; returns records with arrays attached."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("origin") != "observational":
        raise ValueError("empirical analysis refuses a non-observational manifest")
    parent_path = parent_2023_manifest or root / "data/derived/expansion_2023_recentered/observations.v2.json"
    parents = {}
    if parent_path.exists():
        for row in json.loads(parent_path.read_text())["observations"]:
            if row.get("sensor") == "SENTINEL_2_OPTICAL":
                parents[row["observation_id"]] = row
    records = []
    for row in manifest.get("observations", []):
        if row.get("status") == "EXPORT_FAILED":
            records.append({**row, "role": role_for_time(parse_time(row["stac_datetime"]))})
            continue
        site, oid = row.get("site_id"), row.get("item_id")
        if site not in SITES:
            raise ValueError("manifest contains an undeclared site")
        if row.get("origin") != "observational":
            raise ValueError(f"constructed or unknown origin in empirical row {oid}")
        values_path, receipt_path = root / row["values_file"], root / row["receipt_file"]
        if digest(values_path) != row["values_sha256"] or digest(receipt_path) != row["receipt_sha256"]:
            raise ValueError(f"values or receipt hash mismatch for {oid}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("site_id") != site or receipt.get("values_sha256") != row["values_sha256"]:
            raise ValueError(f"receipt-to-row identity mismatch for {oid}")
        if parse_time(receipt["acquired_at"]) != parse_time(row["acquired_at"]):
            raise ValueError(f"manifest UTC time differs from receipt for {oid}")
        if row.get("provider") == "Earth Search C1 L2A":
            item_path = item_path_for(row, root)
            if digest(item_path) != row["item_sha256"]:
                raise ValueError(f"STAC item hash mismatch for {oid}")
            item = json.loads(item_path.read_text())
            if parse_time(item["properties"]["datetime"]) != parse_time(row["acquired_at"]):
                raise ValueError(f"STAC datetime differs from bound manifest time for {oid}")
            _verify_range_receipts(row)
            _verify_metadata_receipts(row, item, root)
        else:
            parent = parents.get(oid)
            if not parent:
                raise ValueError(f"2023 row does not bind to its source manifest observation {oid}")
            if (digest(root / parent["receipt_file"]) != row["parent_receipt_sha256"]
                    or digest(root / parent["values_file"]) != row["parent_values_sha256"]):
                raise ValueError(f"retained 2023 parent binding mismatch for {oid}")
            if parse_time(parent["acquired_at"]) != parse_time(row["acquired_at"]):
                raise ValueError(f"2023 source-manifest timestamp mismatch for {oid}")
            native = json.loads((root / parent["receipt_file"]).read_text()).get("native", {})
            if (row.get("processing_baseline") != native.get("baseline")
                    or row.get("parent_native_radiometry") != native.get("radiometry")
                    or parse_time(row.get("native_sensing_time")) != parse_time(native.get("native_sensing_time"))
                    or row.get("metadata_receipts") != native.get("receipts")):
                raise ValueError(f"2023 native source metadata differs from its parent receipt for {oid}")
            for metadata in row.get("metadata_receipts", {}).values():
                if digest(root / metadata["path"]) != metadata.get("sha256") or metadata.get("http_status") != 200:
                    raise ValueError(f"2023 native metadata receipt invalid for {oid}")
        with np.load(values_path, allow_pickle=False) as data:
            reflectance = data["reflectance"].astype(np.float64)
            valid = data["valid"].astype(bool)
            scl = data["scl"].astype(np.uint8)
            origin = str(data["origin"]) if "origin" in data else "observational"
        if origin != "observational":
            raise ValueError(f"array origin is not observational for {oid}")
        if reflectance.shape != (3, *ROI_SHAPE) or valid.shape != reflectance.shape or scl.shape != ROI_SHAPE:
            raise ValueError(f"input array dimensions mismatch for {oid}")
        dt = parse_time(row["acquired_at"])
        records.append({**row, "observation_id": f"{site}__{oid}", "dt": dt,
                        "role": role_for_time(dt), "reflectance": reflectance,
                        "valid": valid, "scl": scl})
    ids = [r["observation_id"] for r in records if "observation_id" in r]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate observation IDs in manifest")
    return records, manifest
