#!/usr/bin/env python3
"""Acquire and bind the authorized SR-1 multi-year optical study inputs.

This runner is deliberately separate from the earlier SR-1 execution. It uses
the approved Earth Search endpoint, score-blind temporal bins, HTTP 206 COG
windows, source-manifest timestamps for the retained 2023 scenes, and immutable
attempt directories. It does not make research-readiness claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds, from_origin
from pyproj import Transformer
from defusedxml import ElementTree as SafeET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.ingestion.expansion2023 import (  # noqa: E402
    Budget, fetch, grid_for, read_cog, sha, write,
)

SITES = {
    "PILOT-SOUTH-LHONAK": {"lon": 88.204000, "lat": 27.915000, "tile": "T45RXL"},
    "PILOT-KHANGCHUNG-TSHO": {"lon": 88.815000, "lat": 27.988333, "tile": "T45RXM"},
}
START = "2018-01-01T00:00:00Z"
END_EXCLUSIVE = "2023-11-16T00:00:00Z"
PRE_YEARS = tuple(range(2018, 2023))
ROLES = (("green", "B03"), ("nir", "B08"), ("swir16", "B11"), ("scl", "SCL"))
MAX_FAILURES = 2


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source acquisition time must carry a timezone")
    return parsed.astimezone(timezone.utc)


def canonical_utc(value: str) -> str:
    return utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def full_coverage(item: dict, grid: dict, expected_tile: str) -> bool:
    props = item.get("properties", {})
    tile = f"T{props.get('mgrs:utm_zone', '')}{props.get('mgrs:latitude_band', '')}{props.get('mgrs:grid_square', '')}"
    if tile != expected_tile or props.get("proj:epsg") != 32645:
        return False
    try:
        h, w = grid["shape"]
        t = rasterio.Affine(*grid["transform"])
        left, bottom, right, top = array_bounds(h, w, t)
        asset = item["assets"]["scl"]
        sh, sw = asset["proj:shape"]
        st = rasterio.Affine(*asset["proj:transform"][:6])
        sl, sb, sr, stp = array_bounds(sh, sw, st)
        return sl <= left and sb <= bottom and sr >= right and stp >= top
    except (KeyError, TypeError, ValueError):
        return False


def query_catalog(site: str, year: int, root: Path, budget: Budget) -> list[dict]:
    site_info = SITES[site]
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year + 1}-01-01T00:00:00Z" if year < 2023 else END_EXCLUSIVE
    body = {
        "collections": ["sentinel-2-c1-l2a"],
        "intersects": {"type": "Point", "coordinates": [site_info["lon"], site_info["lat"]]},
        "datetime": f"{start}/{end}", "limit": 100,
    }
    out = root / "catalogue" / site / str(year)
    out.mkdir(parents=True, exist_ok=False)
    features: list[dict] = []
    failures = 0
    for page in range(100):
        try:
            data, receipt = fetch(
                "https://earth-search.aws.element84.com/v1/search", budget,
                method="POST", payload=body, limit=20 * 1024 * 1024,
            )
            (out / f"page_{page:03d}.geojson").write_bytes(data)
            write(out / f"page_{page:03d}.receipt.json", receipt)
            response = json.loads(data)
            if not isinstance(response.get("features"), list):
                raise ValueError("STAC response has no features array")
            features.extend(response["features"])
            link = next((item for item in response.get("links", []) if item.get("rel") == "next"), None)
            if not link:
                break
            if link.get("href") != "https://earth-search.aws.element84.com/v1/search" or not isinstance(link.get("body"), dict):
                raise ValueError("STAC pagination link is outside approved endpoint or malformed")
            body = link["body"]
        except Exception:
            failures += 1
            if failures >= MAX_FAILURES:
                raise RuntimeError(f"two consecutive catalogue failures for {site}/{year}; stopping")
            continue
        failures = 0
    else:
        raise RuntimeError(f"catalogue pagination exceeded 100 pages for {site}/{year}")
    unique = {row["id"]: row for row in features if isinstance(row, dict) and isinstance(row.get("id"), str)}
    grid = load_grid(site, root)
    eligible = [row for row in unique.values() if full_coverage(row, grid, site_info["tile"])]
    eligible.sort(key=lambda row: (row.get("properties", {}).get("datetime", ""), row["id"]))
    write(out / "inventory.json", {
        "site_id": site, "year": year, "date_window": [start, end],
        "catalogue_items": len(unique), "full_roi_coverage_items": len(eligible),
        "selection_uses_scores": False,
        "eligible": [{"item_id": row["id"], "datetime": row["properties"].get("datetime"),
                      "cloud_cover": row["properties"].get("eo:cloud_cover"),
                      "processing_baseline": row["properties"].get("s2:processing_baseline"),
                      "item_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
                     for row in eligible],
    })
    return eligible


def load_grid(site: str, attempt_root: Path) -> dict:
    p = attempt_root / "grids" / site / "grid_definition.tif"
    if not p.is_file():
        raise FileNotFoundError(f"SR-1 grid is missing: {p}")
    with rasterio.open(p) as ds:
        if ds.crs.to_epsg() != 32645 or (ds.height, ds.width) != (160, 160):
            raise ValueError(f"SR-1 grid does not satisfy the declared 160x160 EPSG:32645 contract: {site}")
        return {"crs": str(ds.crs), "shape": [ds.height, ds.width],
                "transform": list(ds.transform)[:6], "grid_path": p.relative_to(ROOT).as_posix(),
                "grid_sha256": file_hash(p)}


def create_grids(attempt_root: Path) -> None:
    """Create immutable grids snapped to 20 m UTM pixels at the PI coordinates."""
    project = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    for site, info in SITES.items():
        x, y = project.transform(info["lon"], info["lat"])
        center_x = math.floor(x / 20.0 + 0.5) * 20.0
        center_y = math.floor(y / 20.0 + 0.5) * 20.0
        left, top = center_x - 80 * 20.0, center_y + 80 * 20.0
        target = attempt_root / "grids" / site / "grid_definition.tif"
        target.parent.mkdir(parents=True, exist_ok=False)
        with rasterio.open(target, "w", driver="GTiff", width=160, height=160,
                           count=1, dtype="uint8", crs="EPSG:32645",
                           transform=from_origin(left, top, 20, 20), nodata=0,
                           compress="deflate") as dst:
            dst.write(np.zeros((160, 160), dtype=np.uint8), 1)
            dst.update_tags(origin="PI_COORDINATE_SNAPPED_TO_20M_GRID", site_id=site,
                            center_lon=info["lon"], center_lat=info["lat"],
                            center_easting_m=x, center_northing_m=y,
                            center_pixel_easting_m=center_x, center_pixel_northing_m=center_y)


def seasonally_stratified(candidates: list[dict], year: int, bins: int = 12) -> list[dict]:
    """Choose lowest STAC cloud-cover item in each equal-duration seasonal bin."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    stop = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    buckets: list[list[dict]] = [[] for _ in range(bins)]
    duration = (stop - start).total_seconds()
    for item in candidates:
        instant = utc(item["properties"]["datetime"])
        index = min(bins - 1, int((instant - start).total_seconds() / duration * bins))
        buckets[index].append(item)
    chosen = []
    for index, bucket in enumerate(buckets):
        if bucket:
            chosen.append(min(bucket, key=lambda item: (
                float(item.get("properties", {}).get("eo:cloud_cover", 100.0)),
                item["properties"]["datetime"], item["id"])))
    return chosen


def extract_provider_scene(item: dict, site: str, root: Path, budget: Budget, ordinal: int) -> dict:
    site_root = root / "provider" / site / item["id"]
    site_root.mkdir(parents=True, exist_ok=False)
    (site_root / "item.json").write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
    grid = load_grid(site, root)
    native = read_native_metadata(item, site_root, budget)
    stac_time = canonical_utc(item["properties"]["datetime"])
    native_time = canonical_utc(native["native_sensing_time"])
    reflectance, valid_bands, range_receipts = [], [], {}
    for role, band in ROLES:
        raw, valid, details = read_cog(item["assets"][role], band, grid, budget, ordinal)
        expected = native["radiometry"].get(band)
        if band != "SCL":
            if not expected:
                raise ValueError(f"native BOA quantification/offset absent for {band}")
            native_values = raw.astype(np.float64) * float(expected["scale"]) + float(expected["offset"])
            reflectance.append(native_values)
        else:
            scl = raw.astype(np.uint8)
            scl_valid = valid
        valid_bands.append(valid)
        range_receipts[band] = details
    good = np.isin(scl, [4, 5, 6]) & scl_valid
    b03, b08, b11 = reflectance
    raw_values = np.stack([b03, b08, b11], axis=0).astype(np.float64)
    raw_values[:, ~good] = np.nan
    valid = np.broadcast_to(good, raw_values.shape).copy()
    record = {
        "schema_version": 1, "origin": "observational", "provider": "Earth Search C1 L2A",
        "site_id": site, "item_id": item["id"], "product_uri": native["product_uri"],
        "acquired_at": stac_time, "stac_datetime": stac_time,
        "native_sensing_time": native_time,
        "timestamp_join_status": "BOTH_SOURCE_TIMESTAMPS_RETAINED; STAC_ITEM_DATETIME_USED_FOR_ORDERING",
        "acquisition_end_at": item.get("properties", {}).get("end_datetime"),
        "acquisition_end_status": ("PROVIDER_INTERVAL_END_RECORDED" if item.get("properties", {}).get("end_datetime")
                                   else "NOT_PROVIDED_BY_STAC_OR_NATIVE_XML"),
        "processing_baseline": native["baseline"], "boa_radiometry": native["radiometry"],
        "grid": grid, "range_receipts": range_receipts,
        "item_sha256": file_hash(site_root / "item.json"),
    }
    target = root / "scenes" / site / f"{stac_time[:10]}__{item['id']}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, reflectance=raw_values, valid=valid, scl=scl,
                        origin=np.asarray("observational"))
    record.update({"values_file": target.relative_to(ROOT).as_posix(), "values_sha256": file_hash(target),
                   "valid_pixels": int(good.sum()), "roi_pixels": int(good.size)})
    receipt = site_root / "receipt.sr1.json"
    write(receipt, record)
    record["receipt_file"] = receipt.relative_to(ROOT).as_posix()
    record["receipt_sha256"] = file_hash(receipt)
    return record


def _text_unique(root, name: str) -> str:
    values = [node.text.strip() for node in root.iter()
              if node.tag.rsplit("}", 1)[-1] == name and node.text and node.text.strip()]
    if not values or len(set(values)) != 1:
        raise ValueError(f"native product metadata field {name} is absent or conflicting")
    return values[0]


def read_native_metadata(item: dict, folder: Path, budget: Budget) -> dict:
    """Read small native XML metadata; retain both native radiometry and request receipts."""
    xml = {}
    receipts = {}
    for role in ("product_metadata", "granule_metadata"):
        asset = item.get("assets", {}).get(role)
        if not asset or not isinstance(asset.get("href"), str):
            raise ValueError(f"C1 item lacks native {role}")
        data, receipt = fetch(asset["href"], budget, limit=4 * 1024 * 1024)
        checksum = asset.get("file:checksum")
        if isinstance(checksum, str) and checksum.startswith("1220"):
            actual = hashlib.sha256(data).hexdigest()
            if actual != checksum[4:]:
                raise ValueError(f"native {role} SHA-256 mismatch")
        path = folder / f"{role}.xml"
        path.write_bytes(data)
        xml[role] = SafeET.fromstring(data)
        receipts[role] = {**receipt, "path": path.relative_to(ROOT).as_posix(),
                          "sha256": file_hash(path), "expected_multihash": checksum}
    product, granule = xml["product_metadata"], xml["granule_metadata"]
    baseline = _text_unique(product, "PROCESSING_BASELINE")
    if baseline not in {"05.00", "05.09", "05.10", "05.11"} or baseline != item.get("properties", {}).get("s2:processing_baseline"):
        raise ValueError(f"unsupported/mismatched native processing baseline {baseline}")
    product_uri = _text_unique(product, "PRODUCT_URI")
    if product_uri != item.get("properties", {}).get("s2:product_uri"):
        raise ValueError("native product URI differs from STAC item")
    quant = float(_text_unique(product, "BOA_QUANTIFICATION_VALUE"))
    if not math.isfinite(quant) or quant <= 0:
        raise ValueError("invalid native BOA quantification value")
    bands = {}
    offsets = {}
    for node in product.iter():
        name = node.tag.rsplit("}", 1)[-1]
        if name == "Spectral_Information":
            physical = node.attrib.get("physicalBand", "").upper()
            match = re.fullmatch(r"B0?(\d{1,2})", physical)
            if match:
                bands[f"B{int(match.group(1)):02d}"] = node.attrib.get("bandId")
        elif name == "BOA_ADD_OFFSET":
            offsets[node.attrib.get("band_id", "")] = float(node.text)
    radiometry = {}
    for band in ("B03", "B08", "B11"):
        band_id = bands.get(band)
        if band_id is None or band_id not in offsets:
            raise ValueError(f"native BOA offset missing for {band}; no implicit zero fallback")
        radiometry[band] = {"quantification": quant, "offset_tag": offsets[band_id],
                            "scale": 1.0 / quant, "offset": offsets[band_id] / quant,
                            "offset_status": "EXPLICIT_NATIVE_XML_TAG"}
        stac_role = {"B03": "green", "B08": "nir", "B11": "swir16"}[band]
        declared = item["assets"][stac_role]["raster:bands"][0]
        if (abs(float(declared.get("scale", math.nan)) - radiometry[band]["scale"]) > 1e-12 or
                abs(float(declared.get("offset", math.nan)) - radiometry[band]["offset"]) > 1e-12):
            raise ValueError(f"native BOA quantification/offset differs from STAC COG metadata for {band}")
    return {"baseline": baseline, "product_uri": product_uri, "radiometry": radiometry,
            "native_sensing_time": _text_unique(granule, "SENSING_TIME"),
            "native_product_start_time": _text_unique(product, "PRODUCT_START_TIME"),
            "native_product_stop_time": _text_unique(product, "PRODUCT_STOP_TIME"),
            "metadata_receipts": receipts}


def crop_retained_2023(site: str, row: dict, root: Path) -> dict:
    receipt_path = ROOT / row["receipt_file"]
    values_path = ROOT / row["values_file"]
    if file_hash(receipt_path) != row["receipt_sha256"] or file_hash(values_path) != row["sha256"]:
        raise ValueError(f"retained 2023 parent receipt or source array hash mismatch: {row['observation_id']}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("acquired_at") != row["acquired_at"] or receipt.get("product_id") != row["product_id"]:
        raise ValueError("retained 2023 manifest fields do not match their receipt")
    with np.load(values_path, allow_pickle=False) as z:
        values = z["values"]
        scl = z["scl_classes"]
        observed = z["observed_valid"] if "observed_valid" in z else np.isfinite(values)
    parent_grid = row["grid"]
    target_grid = load_grid(site, root)
    pt = rasterio.Affine(*parent_grid["transform"])
    tt = rasterio.Affine(*target_grid["transform"])
    col = (tt.c - pt.c) / pt.a
    row_offset = (tt.f - pt.f) / pt.e
    if abs(col - round(col)) > 1e-9 or abs(row_offset - round(row_offset)) > 1e-9:
        raise ValueError("SR-1 160x160 grid is not exactly aligned to retained 499x499 source grid")
    c0, r0 = int(round(col)), int(round(row_offset))
    h, w = target_grid["shape"]
    if c0 < 0 or r0 < 0 or c0 + w > values.shape[2] or r0 + h > values.shape[1]:
        raise ValueError("SR-1 window extends outside retained 2023 source arrays")
    cube = values[:3, r0:r0+h, c0:c0+w].astype(np.float64)
    mask = observed[:3, r0:r0+h, c0:c0+w].astype(bool)
    scl_crop = scl[r0:r0+h, c0:c0+w].astype(np.uint8)
    good = np.isin(scl_crop, [4, 5, 6]) & mask.all(axis=0) & np.isfinite(cube).all(axis=0)
    cube[:, ~good] = np.nan
    target = root / "scenes" / site / f"{row['acquired_at'][:10]}__retained__{row['observation_id']}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, reflectance=cube, valid=np.broadcast_to(good, cube.shape).copy(),
                        scl=scl_crop, origin=np.asarray("observational"))
    out = {"schema_version": 1, "origin": "observational", "provider": "retained verified 2023 C1 ROI",
           "site_id": site, "item_id": row["observation_id"], "product_uri": row["product_id"],
           "acquired_at": canonical_utc(row["acquired_at"]), "acquisition_end_at": receipt.get("acquisition_end_at"),
           "acquisition_end_status": "NOT_PRESENT_IN_RETAINED_2023_MANIFEST_OR_RECEIPT",
           "processing_baseline": receipt.get("native", {}).get("baseline"),
           "parent_manifest_path": "data/derived/expansion_2023_recentered/observations.v2.json",
           "parent_manifest_observation": row, "parent_receipt_sha256": row["receipt_sha256"],
           "parent_values_sha256": row["sha256"], "grid": target_grid,
           "values_file": target.relative_to(ROOT).as_posix(), "values_sha256": file_hash(target),
           "valid_pixels": int(good.sum()), "roi_pixels": int(good.size)}
    rp = target.with_suffix(".receipt.json")
    write(rp, out)
    out["receipt_file"] = rp.relative_to(ROOT).as_posix()
    out["receipt_sha256"] = file_hash(rp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="attempt_002")
    ap.add_argument("--bins-per-year", type=int, default=12)
    args = ap.parse_args()
    if not re.fullmatch(r"attempt_[0-9]{3}", args.attempt):
        raise SystemExit("--attempt must be a new attempt_NNN directory")
    if not 1 <= args.bins_per_year <= 24:
        raise SystemExit("--bins-per-year must be in 1..24")
    root = ROOT / "data/raw/sr1_multiyear" / args.attempt
    if root.exists():
        raise SystemExit(f"immutable attempt already exists: {root}")
    root.mkdir(parents=True)
    create_grids(root)
    derived_root = ROOT / "data/derived/sr1_multiyear" / args.attempt
    derived_root.mkdir(parents=True)
    global_cap = 10_000_000_000
    prior_bytes = 0
    for prior_ledger in (ROOT / "data/raw/sr1_multiyear").glob("attempt_*/transfer_ledger.json"):
        if prior_ledger.parent != root:
            prior_bytes += int(json.loads(prior_ledger.read_text()).get("received_bytes", 0))
    unknown_reserve = 0
    interruption_note = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/INTERRUPTED_TRANSFER_ACCOUNTING.json"
    if interruption_note.exists():
        unknown_reserve = int(json.loads(interruption_note.read_text()).get("conservative_reserved_bytes", 0))
    remaining_cap = global_cap - prior_bytes - unknown_reserve
    if remaining_cap <= 0:
        raise SystemExit("cumulative SR-03 cap exhausted by recorded transfers and conservative reservations")
    budget = Budget(root / "transfer_ledger.json", cap=remaining_cap)

    # Catalogue query windows; all raw pages and actual transport receipts are retained.
    selected: dict[str, list[dict]] = {site: [] for site in SITES}
    inventory = []
    failure_count = 0
    for site in SITES:
        for year in PRE_YEARS:
            try:
                items = query_catalog(site, year, root, budget)
                chosen = seasonally_stratified(items, year, args.bins_per_year)
                inventory.append({"site_id": site, "year": year, "eligible": len(items),
                                  "selected": [item["id"] for item in chosen],
                                  "selection": "equal-duration temporal bins; lowest provider cloud cover then UTC/id",
                                  "bins": args.bins_per_year})
                selected[site].extend(chosen)
                failure_count = 0
            except Exception:
                failure_count += 1
                if failure_count >= MAX_FAILURES:
                    raise RuntimeError("two SR-03 failures; stopping and preserving all partial evidence")
                raise

    rows = []
    failures = []
    failed = 0
    ordinal = 0
    for site, items in selected.items():
        for item in items:
            try:
                rows.append(extract_provider_scene(item, site, root, budget, ordinal))
                ordinal += 1
                failed = 0
            except Exception as exc:
                failed += 1
                failure = {"status": "EXPORT_FAILED", "error_type": type(exc).__name__,
                           "error": str(exc), "site_id": site, "item_id": item.get("id"),
                           "stac_datetime": item.get("properties", {}).get("datetime"),
                           "at_utc": datetime.now(timezone.utc).isoformat(),
                           "score_blind_selection_retained": True}
                failures.append(failure)
                write(root / "failures.json", {"consecutive_failures": failed, "failed_attempts": failures})
                if failed >= MAX_FAILURES:
                    raise RuntimeError("two consecutive SR-03 decode/acquisition failures; stopping") from exc
                continue

    manifest_path = ROOT / "data/derived/expansion_2023_recentered/observations.v2.json"
    source = json.loads(manifest_path.read_text())
    retained2023 = []
    for site in SITES:
        optical = [r for r in source["observations"]
                   if r.get("site_id") == site and r.get("sensor") == "SENTINEL_2_OPTICAL"
                   and utc(r["acquired_at"]) < utc(END_EXCLUSIVE)]
        for row in optical:
            retained2023.append(crop_retained_2023(site, row, root))

    all_rows = rows + failures + retained2023
    all_rows.sort(key=lambda r: (r["site_id"], r["acquired_at"], r["item_id"]))
    output = {
        "schema_version": 1, "origin": "observational", "status": "ACQUIRED_WITH_SOURCE_RECEIPTS",
        "date_window": [START, END_EXCLUSIVE], "sites": list(SITES),
        "selection": {"pre_2023": "12 equal-duration bins per site-year; lowest STAC eo:cloud_cover; full ROI coverage; score blind",
                      "2023": "all retained source-manifest S2 observations; no re-download"},
        "catalogue_inventory": inventory, "observations": all_rows,
        "failed_scene_count": len(failures),
        "transfer_ledger": (root / "transfer_ledger.json").relative_to(ROOT).as_posix(),
        "transfer_cap_bytes": global_cap, "transfer_received_bytes": budget.received,
        "prior_attempt_received_bytes": prior_bytes,
        "conservative_unknown_transfer_reserve_bytes": unknown_reserve,
        "cumulative_received_and_reserved_bytes": prior_bytes + unknown_reserve + budget.received,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = derived_root / "observations.sr1.json"
    write(path, output)
    print(json.dumps({"status": output["status"], "observations": len(all_rows),
                      "new_provider_observations": len(rows), "retained_2023": len(retained2023),
                      "received_bytes": budget.received, "manifest": path.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
