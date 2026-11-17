#!/usr/bin/env python3
"""Re-acquire the South Lhonak ROI on a window centred on its inventory outline.

The attempt_006 South Lhonak window (centred on 27.915 N, 88.204 E) contained
only part of the lake. This runner builds a new immutable attempt in which:

* the South Lhonak 160 x 160 / 20 m window is centred on the centroid of the
  Hi-MAG 2017 outline GL088195E27912N, snapped to the 20 m UTM grid;
* the same 48 Earth Search items selected score-blind for 2018-2022 are read
  again by HTTP range requests on the new window (item JSON and native SAFE
  metadata are reused from attempt_006 after hash verification);
* the same 24 retained 2023 scenes are re-cropped from their verified
  499 x 499 parent arrays;
* Khangchung Tsho rows, whose window already contains its whole outline, are
  carried over unchanged by reference.

No scene is added, dropped or re-selected and no score is read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.ingestion.expansion2023 import Budget, read_cog, write  # noqa: E402
from sentinel_gl.sr1_corpus import digest  # noqa: E402
from sentinel_gl.sr1_geometry import (  # noqa: E402
    HI_MAG_IDS, HI_MAG_RECORD, HI_MAG_ZIP_SHA256, inventory_zone_map, load_hi_mag_outline,
    outline_centroid_lonlat, snapped_window,
)

SITE = "PILOT-SOUTH-LHONAK"
PARENT = "attempt_006"
HI_MAG_DIR = ROOT / "data/raw/external/hi_mag_zenodo_4275164"
HI_MAG_2017 = HI_MAG_DIR / "extracted/Hi-MAG database/Hi_MAG_database_2017"
ROLES = (("green", "B03"), ("nir", "B08"), ("swir16", "B11"), ("scl", "SCL"))
GLOBAL_CAP = 10_000_000_000


def write_grid(path: Path, grid: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=False)
    left, top = grid["transform"][2], grid["transform"][5]
    with rasterio.open(path, "w", driver="GTiff", width=160, height=160, count=1, dtype="uint8",
                       crs=grid["crs"], transform=from_origin(left, top, 20, 20), nodata=0,
                       compress="deflate") as dst:
        dst.write(np.zeros((160, 160), dtype=np.uint8), 1)
        dst.update_tags(origin="HI_MAG_2017_OUTLINE_CENTROID_SNAPPED_TO_20M_GRID", site_id=SITE,
                        hi_mag_id=HI_MAG_IDS[SITE], center_lon=grid["center_lon"], center_lat=grid["center_lat"])
    return {"crs": grid["crs"], "shape": [160, 160], "transform": grid["transform"],
            "grid_path": path.relative_to(ROOT).as_posix(), "grid_sha256": digest(path)}


def covers(item: dict, grid: dict) -> bool:
    asset = item["assets"]["scl"]
    sh, sw = asset["proj:shape"]
    st = rasterio.Affine(*asset["proj:transform"][:6])
    sl, sb, sr, stp = rasterio.transform.array_bounds(sh, sw, st)
    t = rasterio.Affine(*grid["transform"])
    left, bottom, right, top = rasterio.transform.array_bounds(160, 160, t)
    return item["properties"].get("proj:epsg") == 32645 and sl <= left and sb <= bottom and sr >= right and stp >= top


def reread_provider_scene(row: dict, grid: dict, root: Path, budget: Budget, ordinal: int) -> dict:
    old_item = (ROOT / row["receipt_file"]).parent / "item.json"
    if digest(old_item) != row["item_sha256"]:
        raise ValueError(f"retained STAC item hash mismatch: {row['item_id']}")
    item = json.loads(old_item.read_text())
    if not covers(item, grid):
        raise ValueError(f"item does not fully cover the re-centred window: {row['item_id']}")
    folder = root / "provider" / SITE / row["item_id"]
    folder.mkdir(parents=True, exist_ok=False)
    new_item = folder / "item.json"
    shutil.copyfile(old_item, new_item)
    reflectance, range_receipts = [], {}
    for role, band in ROLES:
        raw, valid, details = read_cog(item["assets"][role], band, grid, budget, ordinal)
        if band == "SCL":
            scl, scl_valid = raw.astype(np.uint8), valid
        else:
            native = row["boa_radiometry"][band]
            declared = item["assets"][role]["raster:bands"][0]
            if abs(float(declared["scale"]) - native["scale"]) > 1e-12 or abs(float(declared["offset"]) - native["offset"]) > 1e-12:
                raise ValueError(f"native radiometry differs from STAC metadata for {row['item_id']} {band}")
            reflectance.append(raw.astype(np.float64) * native["scale"] + native["offset"])
        range_receipts[band] = details
    good = np.isin(scl, [4, 5, 6]) & scl_valid
    cube = np.stack(reflectance, axis=0)
    cube[:, ~good] = np.nan
    target = root / "scenes" / SITE / f"{row['acquired_at'][:10]}__{row['item_id']}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, reflectance=cube, valid=np.broadcast_to(good, cube.shape).copy(),
                        scl=scl, origin=np.asarray("observational"))
    out = dict(row)
    out.update({"grid": grid, "range_receipts": range_receipts,
                "item_file": new_item.relative_to(ROOT).as_posix(), "item_sha256": digest(new_item),
                "values_file": target.relative_to(ROOT).as_posix(), "values_sha256": digest(target),
                "valid_pixels": int(good.sum()), "roi_pixels": int(good.size),
                "reacquired_from_parent_row": {"attempt": PARENT, "values_sha256": row["values_sha256"],
                                               "receipt_sha256": row["receipt_sha256"]}})
    for key in ("receipt_file", "receipt_sha256"):
        out.pop(key, None)
    receipt = folder / "receipt.sr1.recentred.json"
    write(receipt, out)
    out["receipt_file"] = receipt.relative_to(ROOT).as_posix()
    out["receipt_sha256"] = digest(receipt)
    return out


def recrop_2023(row: dict, grid: dict, root: Path) -> dict:
    parent = row["parent_manifest_observation"]
    if digest(ROOT / parent["values_file"]) != row["parent_values_sha256"]:
        raise ValueError(f"parent 2023 array hash mismatch: {row['item_id']}")
    with np.load(ROOT / parent["values_file"], allow_pickle=False) as z:
        values = z["values"]
        scl = z["scl_classes"]
        observed = z["observed_valid"] if "observed_valid" in z else np.isfinite(values)
    pt = rasterio.Affine(*parent["grid"]["transform"])
    tt = rasterio.Affine(*grid["transform"])
    col, rowoff = (tt.c - pt.c) / pt.a, (tt.f - pt.f) / pt.e
    if abs(col - round(col)) > 1e-9 or abs(rowoff - round(rowoff)) > 1e-9:
        raise ValueError("re-centred window is not aligned with the 2023 parent grid")
    c0, r0 = int(round(col)), int(round(rowoff))
    if c0 < 0 or r0 < 0 or c0 + 160 > values.shape[2] or r0 + 160 > values.shape[1]:
        raise ValueError("re-centred window extends outside the 2023 parent array")
    cube = values[:3, r0:r0 + 160, c0:c0 + 160].astype(np.float64)
    mask = observed[:3, r0:r0 + 160, c0:c0 + 160].astype(bool)
    scl_crop = scl[r0:r0 + 160, c0:c0 + 160].astype(np.uint8)
    good = np.isin(scl_crop, [4, 5, 6]) & mask.all(axis=0) & np.isfinite(cube).all(axis=0)
    cube[:, ~good] = np.nan
    target = root / "scenes" / SITE / f"{row['acquired_at'][:10]}__retained__{row['item_id']}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, reflectance=cube, valid=np.broadcast_to(good, cube.shape).copy(),
                        scl=scl_crop, origin=np.asarray("observational"))
    out = dict(row)
    out.update({"grid": grid, "values_file": target.relative_to(ROOT).as_posix(),
                "values_sha256": digest(target), "valid_pixels": int(good.sum()), "roi_pixels": int(good.size),
                "parent_crop_offset_rc": [r0, c0],
                "recropped_from_parent_row": {"attempt": PARENT, "values_sha256": row["values_sha256"]}})
    for key in ("receipt_file", "receipt_sha256"):
        out.pop(key, None)
    receipt = target.with_suffix(".receipt.json")
    write(receipt, out)
    out["receipt_file"] = receipt.relative_to(ROOT).as_posix()
    out["receipt_sha256"] = digest(receipt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="attempt_007")
    args = ap.parse_args()
    raw_root = ROOT / "data/raw/sr1_multiyear" / args.attempt
    derived_root = ROOT / "data/derived/sr1_multiyear" / args.attempt
    if raw_root.exists() or derived_root.exists():
        raise SystemExit(f"immutable attempt already exists: {args.attempt}")
    if digest(HI_MAG_DIR / "Hi-MAG_database.zip") != HI_MAG_ZIP_SHA256:
        raise SystemExit("Hi-MAG archive hash differs from the recorded value")
    parent_manifest_path = ROOT / "data/derived/sr1_multiyear" / PARENT / "observations.sr1.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    outline, attrs = load_hi_mag_outline(HI_MAG_2017, HI_MAG_IDS[SITE])
    lon, lat = outline_centroid_lonlat(outline)
    window = snapped_window(lon, lat)
    zone_check = inventory_zone_map(outline, window)
    if zone_check["outline_fraction_inside_window"] < 0.999:
        raise SystemExit("re-centred window still truncates the inventory outline")
    raw_root.mkdir(parents=True)
    derived_root.mkdir(parents=True)
    grid = write_grid(raw_root / "grids" / SITE / "grid_definition.tif", window)
    prior = sum(int(json.loads(p.read_text()).get("received_bytes", 0))
                for p in (ROOT / "data/raw/sr1_multiyear").glob("attempt_*/transfer_ledger.json") if p.parent != raw_root)
    reserve_note = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/INTERRUPTED_TRANSFER_ACCOUNTING.json"
    reserve = int(json.loads(reserve_note.read_text()).get("conservative_reserved_bytes", 0)) if reserve_note.exists() else 0
    budget = Budget(raw_root / "transfer_ledger.json", cap=GLOBAL_CAP - prior - reserve)
    rows, ordinal = [], 0
    for row in parent_manifest["observations"]:
        if row.get("site_id") != SITE:
            rows.append(dict(row))  # Khangchung Tsho: unchanged, bound by its original hashes
        elif row.get("provider") == "Earth Search C1 L2A":
            rows.append(reread_provider_scene(row, grid, raw_root, budget, ordinal))
            ordinal += 1
            print(json.dumps({"reread": row["item_id"], "received_bytes": budget.received}), flush=True)
        else:
            rows.append(recrop_2023(row, grid, raw_root))
    rows.sort(key=lambda r: (r["site_id"], r["acquired_at"], r["item_id"]))
    manifest = dict(parent_manifest)
    manifest.update({
        "status": "ACQUIRED_WITH_SOURCE_RECEIPTS_SOUTH_LHONAK_RECENTRED",
        "observations": rows, "parent_attempt": PARENT,
        "parent_manifest_sha256": digest(parent_manifest_path),
        "south_lhonak_window": {
            "rule": "Hi-MAG 2017 outline centroid snapped to the 20 m EPSG:32645 grid; 160 x 160 pixels",
            "hi_mag_record": HI_MAG_RECORD, "hi_mag_zip_sha256": HI_MAG_ZIP_SHA256,
            "hi_mag_id": HI_MAG_IDS[SITE], "hi_mag_attributes": attrs,
            "centroid_lon": lon, "centroid_lat": lat, "grid": grid,
            "outline_fraction_inside_window": zone_check["outline_fraction_inside_window"],
            "previous_grid": next(r["grid"] for r in parent_manifest["observations"]
                                  if r.get("site_id") == SITE and "grid" in r)},
        "khangchung_window": "unchanged from attempt_006 (inventory outline fully inside)",
        "transfer_ledger": (raw_root / "transfer_ledger.json").relative_to(ROOT).as_posix(),
        "transfer_received_bytes": budget.received, "prior_attempt_received_bytes": prior,
        "conservative_unknown_transfer_reserve_bytes": reserve,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    manifest.pop("native_metadata_receipt_enrichment", None)
    write(derived_root / "observations.sr1.json", manifest)
    print(json.dumps({"status": manifest["status"], "rows": len(rows), "received_bytes": budget.received}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
