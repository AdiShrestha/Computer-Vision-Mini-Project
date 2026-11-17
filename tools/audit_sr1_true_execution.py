#!/usr/bin/env python3
"""Independent source, split, and prediction arithmetic audit for the true SR-1 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from defusedxml import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp lacks UTC offset")
    return result.astimezone(timezone.utc)


def expected_role(value: str) -> str:
    dt = utc(value)
    if dt.year <= 2021 and dt.year >= 2018:
        return "FIT"
    if dt.year == 2022:
        return "VAL"
    if dt.year == 2023 and dt.month <= 6:
        return "FIT"
    if dt.year == 2023 and dt.month <= 9:
        return "VAL"
    if dt >= utc("2023-10-04T18:30:00Z"):
        return "EVAL"
    if dt >= utc("2023-10-02T18:30:00Z"):
        return "EVENT_UNCERTAIN"
    return "OUT_OF_PROTOCOL_WINDOW"


def check_close(actual, expected, label, errors, *, atol=2e-6):
    if actual is None or expected is None:
        if actual is not expected:
            errors.append(f"{label}: null/status mismatch actual={actual!r}, expected={expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=2e-6, abs_tol=atol):
        errors.append(f"{label}: arithmetic mismatch actual={actual}, expected={expected}")


def audit_native_radiometry(row, errors):
    oid=row['item_id'];receipts=row.get('metadata_receipts',{})
    try:
        product=ET.parse(str(ROOT/receipts['product_metadata']['path'])).getroot()
        granule=ET.parse(str(ROOT/receipts['granule_metadata']['path'])).getroot()
        def field(root,name):
            vals={n.text.strip() for n in root.iter() if n.tag.rsplit('}',1)[-1]==name and n.text and n.text.strip()}
            if len(vals)!=1:raise ValueError(f'expected one {name}, got {len(vals)}')
            return next(iter(vals))
        quant=float(field(product,'BOA_QUANTIFICATION_VALUE'))
        baseline=field(product,'PROCESSING_BASELINE')
        offsets={n.attrib.get('band_id'):float(n.text) for n in product.iter()
                 if n.tag.rsplit('}',1)[-1]=='BOA_ADD_OFFSET' and n.text}
        bands={}
        for n in product.iter():
            if n.tag.rsplit('}',1)[-1]!='Spectral_Information':continue
            match=re.fullmatch(r'B0?(\d{1,2})',n.attrib.get('physicalBand','').upper())
            if match:bands[f'B{int(match.group(1)):02d}']=n.attrib.get('bandId')
        if row.get('processing_baseline')!=baseline:errors.append(f'{oid}: processing baseline differs from native XML')
        if not math.isclose(float(row.get('native_boa_quantification_value',-1)),quant,rel_tol=0,abs_tol=1e-12):
            errors.append(f'{oid}: BOA quantification value differs from native XML')
        if utc(row.get('native_sensing_time'))!=utc(field(granule,'SENSING_TIME')):
            errors.append(f'{oid}: native sensing time differs from granule XML')
        if row.get('native_product_start_time')!=field(product,'PRODUCT_START_TIME') or row.get('native_product_stop_time')!=field(product,'PRODUCT_STOP_TIME'):
            errors.append(f'{oid}: native product interval differs from product XML')
        for band in ('B03','B08','B11'):
            band_id=bands.get(band)
            if band_id not in offsets:errors.append(f'{oid}: explicit native offset tag missing for {band}');continue
            actual=row.get('boa_radiometry',{}).get(band,{})
            tag=row.get('native_boa_offset_tags',{}).get(band,{})
            expected_offset=offsets[band_id]
            if not (math.isclose(float(actual.get('quantification',-1)),quant,rel_tol=0,abs_tol=1e-12)
                    and math.isclose(float(actual.get('offset_tag',math.nan)),expected_offset,rel_tol=0,abs_tol=1e-12)
                    and math.isclose(float(actual.get('scale',math.nan)),1/quant,rel_tol=0,abs_tol=1e-12)
                    and math.isclose(float(actual.get('offset',math.nan)),expected_offset/quant,rel_tol=0,abs_tol=1e-12)
                    and math.isclose(float(tag.get('boa_add_offset_tag',math.nan)),expected_offset,rel_tol=0,abs_tol=1e-12)):
                errors.append(f'{oid}: native radiometry or explicit offset arithmetic mismatch for {band}')
    except Exception as exc:
        errors.append(f'{oid}: cannot independently parse native radiometry metadata: {type(exc).__name__}: {exc}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="attempt_004")
    ap.add_argument("--analysis", default="analysis_001")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--source-only", action="store_true", help="stop after independent acquisition/source checks")
    args = ap.parse_args()
    manifest_path = ROOT / "data/derived/sr1_multiyear" / args.attempt / "observations.sr1.json"
    analysis_dir = manifest_path.parent / args.analysis
    manifest = json.loads(manifest_path.read_text())
    report_path = args.report.resolve()
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite audit report: {report_path}")
    errors, source_checks, arithmetic_checks = [], [], []
    source_by_id = {}
    old = json.loads((ROOT / "data/derived/expansion_2023_recentered/observations.v2.json").read_text())
    old_optical = {r["observation_id"]: r for r in old["observations"]
                   if r.get("sensor") == "SENTINEL_2_OPTICAL"}
    raw_root = ROOT / "data/raw/sr1_multiyear" / args.attempt
    # Reconstruct score-blind temporal-bin winners from the saved candidate inventory.
    inventory_rows=manifest.get("catalogue_inventory",[])
    if len(inventory_rows)!=10 or {(r.get('site_id'),r.get('year')) for r in inventory_rows}!={(s,y) for s in manifest.get('sites',[]) for y in range(2018,2023)}:
        errors.append('catalogue inventory does not cover the declared two sites and five pre-2023 years')
    selected_ids=set()
    for summary in inventory_rows:
        ipath=raw_root/'catalogue'/summary['site_id']/str(summary['year'])/'inventory.json'
        inv=json.loads(ipath.read_text())
        if inv.get('selection_uses_scores') is not False:
            errors.append(f"{summary['site_id']} {summary['year']}: selection was not declared score-blind")
        catalogue={}
        for page in sorted(ipath.parent.glob('page_*.geojson')):
            for item in json.loads(page.read_text()).get('features',[]):
                if isinstance(item,dict) and isinstance(item.get('id'),str):catalogue[item['id']]=item
        target_grid=raw_root/'grids'/summary['site_id']/'grid_definition.tif'
        with rasterio.open(target_grid) as grid_ds:
            left,bottom,right,top=grid_ds.bounds
        verified_candidates=[]
        expected_tile={'PILOT-SOUTH-LHONAK':'T45RXL','PILOT-KHANGCHUNG-TSHO':'T45RXM'}[summary['site_id']]
        for candidate in inv.get('eligible',[]):
            item=catalogue.get(candidate['item_id'])
            if not item:
                errors.append(f"{candidate['item_id']}: eligible item absent from retained raw catalogue pages")
                continue
            canonical=json.dumps(item,sort_keys=True,separators=(',',':')).encode()
            if hashlib.sha256(canonical).hexdigest()!=candidate.get('item_sha256'):
                errors.append(f"{candidate['item_id']}: catalogue item hash differs from inventory")
            prop=item.get('properties',{});tile=f"T{prop.get('mgrs:utm_zone','')}{prop.get('mgrs:latitude_band','')}{prop.get('mgrs:grid_square','')}"
            asset=item.get('assets',{}).get('scl',{});shape=asset.get('proj:shape');transform=asset.get('proj:transform')
            covered=False
            try:
                from rasterio.transform import array_bounds
                st=rasterio.Affine(*transform[:6]);sl,sb,sr,stp=array_bounds(shape[0],shape[1],st)
                covered=sl<=left and sb<=bottom and sr>=right and stp>=top
            except Exception:
                covered=False
            if tile!=expected_tile or prop.get('proj:epsg')!=32645 or not covered:
                errors.append(f"{candidate['item_id']}: candidate does not independently pass tile/CRS/full-window coverage")
            if candidate.get('datetime')!=prop.get('datetime') or candidate.get('cloud_cover')!=prop.get('eo:cloud_cover'):
                errors.append(f"{candidate['item_id']}: inventory time/cloud metadata differs from raw catalogue")
            verified_candidates.append(candidate)
        bins=int(summary.get('bins',0)); winners={}
        for candidate in verified_candidates:
            instant=utc(candidate['datetime']); year=int(summary['year'])
            start=datetime(year,1,1,tzinfo=timezone.utc); stop=datetime(year+1,1,1,tzinfo=timezone.utc)
            bucket=min(bins-1,int((instant-start).total_seconds()/(stop-start).total_seconds()*bins))
            cloud=candidate.get('cloud_cover')
            key=(float(cloud) if cloud is not None else 100.0,candidate['datetime'],candidate['item_id'])
            if bucket not in winners or key<winners[bucket][0]:winners[bucket]=(key,candidate)
        expected=[winners[b][1]['item_id'] for b in sorted(winners)]
        if expected!=summary.get('selected'):
            errors.append(f"{summary['site_id']} {summary['year']}: selected scene list differs from independent bin/cloud replay")
        if int(inv.get('full_roi_coverage_items',-1))!=len(inv.get('eligible',[])):
            errors.append(f"{summary['site_id']} {summary['year']}: eligible inventory count mismatch")
        for oid in summary.get('selected',[]):
            if oid in selected_ids:errors.append(f"{oid}: duplicate across score-blind site-year inventories")
            selected_ids.add(oid)
    ledger=json.loads((raw_root/'transfer_ledger.json').read_text())
    if manifest.get('transfer_received_bytes')!=ledger.get('received_bytes'):
        errors.append('manifest transfer byte count differs from transfer ledger')
    if manifest.get('cumulative_received_and_reserved_bytes',0)>manifest.get('transfer_cap_bytes',0):
        errors.append('cumulative received plus reserved transfer exceeds authorization cap')
    transformer=Transformer.from_crs('EPSG:4326','EPSG:32645',always_xy=True)
    for site,lon,lat in [('PILOT-SOUTH-LHONAK',88.204,27.915),('PILOT-KHANGCHUNG-TSHO',88.815,27.988333)]:
        x,y=transformer.transform(lon,lat)
        grid_path=raw_root/'grids'/site/'grid_definition.tif'
        with rasterio.open(grid_path) as ds:
            cx=(ds.bounds.left+ds.bounds.right)/2;cy=(ds.bounds.bottom+ds.bounds.top)/2
            if ds.crs.to_epsg()!=32645 or (ds.width,ds.height)!=(160,160) or ds.res!=(20.0,20.0):
                errors.append(f'{site}: ROI grid geometry differs from protocol')
            if abs(cx-round(x/20)*20)>1e-6 or abs(cy-round(y/20)*20)>1e-6:
                errors.append(f'{site}: ROI center is not nearest 20 m pixel center')
    for row in manifest["observations"]:
        if row.get("status") == "EXPORT_FAILED":
            continue
        oid = row["item_id"]
        if row.get("origin") != "observational":
            errors.append(f"{oid}: non-observational input entered source manifest")
        role = expected_role(row["acquired_at"])
        if row.get("acquired_at") != row.get("stac_datetime") and row.get("provider") == "Earth Search C1 L2A":
            errors.append(f"{oid}: provider timestamp binding differs from STAC timestamp")
        values = ROOT / row["values_file"]
        receipt_path = ROOT / row["receipt_file"]
        if sha(values) != row["values_sha256"] or sha(receipt_path) != row["receipt_sha256"]:
            errors.append(f"{oid}: source array or receipt SHA-256 mismatch")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("values_sha256") != row["values_sha256"] or receipt.get("item_id") != oid:
            errors.append(f"{oid}: receipt identity/value binding mismatch")
        if row.get("provider") == "Earth Search C1 L2A":
            item_path = raw_root / "provider" / row["site_id"] / oid / "item.json"
            if sha(item_path) != row["item_sha256"]:
                errors.append(f"{oid}: STAC item receipt mismatch")
            item = json.loads(item_path.read_text())
            if utc(item["properties"]["datetime"]) != utc(row["acquired_at"]):
                errors.append(f"{oid}: STAC UTC timestamp mismatch")
            for role in ("product_metadata", "granule_metadata"):
                meta = row.get("metadata_receipts", {}).get(role)
                if not meta or meta.get("url") != item.get("assets", {}).get(role, {}).get("href"):
                    errors.append(f"{oid}: native metadata source receipt missing/mismatched ({role})")
                    continue
                meta_path = ROOT / meta["path"]
                if (meta.get("http_status") != 200 or meta.get("body_status") != "COMPLETE_RESPONSE"
                        or sha(meta_path) != meta.get("sha256")
                        or meta.get("received_bytes") != meta_path.stat().st_size):
                    errors.append(f"{oid}: native metadata source receipt invalid ({role})")
            for band in ("B03", "B08", "B11", "SCL"):
                reqs = row.get("range_receipts", {}).get(band, {}).get("range_receipts", [])
                if not reqs:
                    errors.append(f"{oid}: {band} has no retained range reads")
                for req in reqs:
                    if req.get("http_status") != 206 or req.get("body_status") != "COMPLETE_RESPONSE":
                        errors.append(f"{oid}: {band} includes a non-206 or incomplete range response")
                    if not req.get("sha256") or not req.get("content_range", "").startswith("bytes "):
                        errors.append(f"{oid}: {band} range receipt lacks byte/hash evidence")
        else:
            parent = old_optical.get(oid)
            if not parent:
                errors.append(f"{oid}: 2023 row is absent from retained source manifest")
            else:
                if (sha(ROOT / parent["receipt_file"]) != row["parent_receipt_sha256"]
                        or sha(ROOT / parent["values_file"]) != row["parent_values_sha256"]):
                    errors.append(f"{oid}: 2023 parent receipt/value binding mismatch")
                if utc(parent["acquired_at"]) != utc(row["acquired_at"]):
                    errors.append(f"{oid}: 2023 acquired_at changed from source manifest")
                parent_receipt=json.loads((ROOT/parent['receipt_file']).read_text())
                native=parent_receipt.get('native',{})
                if (row.get('processing_baseline')!=native.get('baseline')
                        or row.get('parent_native_radiometry')!=native.get('radiometry')
                        or utc(row.get('native_sensing_time'))!=utc(native.get('native_sensing_time'))
                        or row.get('metadata_receipts')!=native.get('receipts')):
                    errors.append(f'{oid}: 2023 native metadata fields differ from source receipt')
                for metadata in row.get('metadata_receipts',{}).values():
                    metadata_path=ROOT/metadata['path']
                    if (sha(metadata_path)!=metadata.get('sha256') or metadata.get('http_status')!=200
                            or metadata.get('received_bytes')!=metadata_path.stat().st_size):
                        errors.append(f'{oid}: 2023 native XML receipt invalid')
        with np.load(values, allow_pickle=False) as data:
            refl = data["reflectance"].astype(np.float64)
            valid = data["valid"].astype(bool)
            scl = data["scl"]
        if refl.shape != (3, 160, 160) or valid.shape != refl.shape or scl.shape != (160, 160):
            errors.append(f"{oid}: ROI dimensions or channel layout differ")
        source_by_id[(row["site_id"], oid)] = {"role": role, "acquired_at": row["acquired_at"]}
        audit_native_radiometry(row,errors)
        source_checks.append({"item_id": oid, "provider": row.get("provider"), "role": role,
                              "source_receipt_sha256": row["receipt_sha256"]})

    if args.source_only:
        result={"status":"PASS" if not errors else "FAIL","manifest":str(manifest_path.relative_to(ROOT)),
            "manifest_sha256":sha(manifest_path),"source_rows_checked":len(source_checks),
            "arithmetic_rows_checked":0,"source_checks":source_checks,"arithmetic_checks":[],"errors":errors}
        report_path.parent.mkdir(parents=True,exist_ok=True)
        if report_path.exists():raise SystemExit(f"refusing to overwrite audit report {report_path}")
        report_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('status','source_rows_checked','errors')},indent=2))
        return 0 if not errors else 1

    predictions = json.loads((analysis_dir / "predictions.json").read_text())
    pred_index = {(p["observation_id"], int(p["seed"])): p for p in predictions}
    for replay_meta in json.loads((analysis_dir / "analysis_report.json").read_text())["arithmetic_replay_files"]:
        replay_path = ROOT / replay_meta["path"]
        if sha(replay_path) != replay_meta["sha256"]:
            errors.append(f"seed {replay_meta['seed']}: replay archive hash mismatch")
            continue
        with np.load(replay_path, allow_pickle=False) as z:
            ids = z["observation_ids"].astype(str).tolist()
            sites = z["site_ids"].astype(str).tolist()
            times = z["acquired_at"].astype(str).tolist()
            roles = z["roles"].astype(str).tolist()
            targets, observed = z["target"].astype(np.float64), z["observed"].astype(bool)
            zones, masked = z["zones"].astype(np.int8), z["masked"].astype(bool)
            output = z["prediction"].astype(np.float64)
            projected = z["projected"].astype(np.float64)
            fit_ids = set(z["fit_observation_ids"].astype(str).tolist())
        if len(ids) != len(set(ids)) or not (targets.shape == observed.shape == output.shape):
            errors.append(f"seed {replay_meta['seed']}: replay array shape/ID integrity failure")
            continue
        if len(zones) != len(ids) or projected.shape[:2] != zones.shape:
            errors.append(f"seed {replay_meta['seed']}: projected bank dimensions mismatch")
            continue
        for i, oid in enumerate(ids):
            if expected_role(times[i]) != roles[i]:
                errors.append(f"{oid}: replay role does not match UTC split")
            if roles[i] == "FIT" and oid not in fit_ids:
                errors.append(f"{oid}: FIT replay row omitted from fitted ID set")
            if roles[i] != "FIT" and oid in fit_ids:
                errors.append(f"{oid}: non-FIT replay row entered fitted ID set")
            valid_channels = observed[i] & np.isfinite(targets[i])
            patch_mse = np.full(zones.shape[1], np.nan, dtype=np.float64)
            for p in np.flatnonzero(masked[i]):
                channels = valid_channels[p]
                if channels.any():
                    patch_mse[p] = np.mean((output[i, p, channels] - targets[i, p, channels]) ** 2)
            target_mask = np.isin(zones[i], [0, 1]) & np.isfinite(patch_mse)
            background_mask = (zones[i] == 2) & np.isfinite(patch_mse)
            a_target = float(np.mean(patch_mse[target_mask])) if target_mask.any() else None
            a_background = float(np.mean(patch_mse[background_mask])) if background_mask.any() else None
            stored = pred_index.get((oid, int(replay_meta["seed"])))
            if stored is None:
                errors.append(f"{oid}: prediction summary missing for seed {replay_meta['seed']}")
                continue
            check_close(stored["score_a_zone0_1"], a_target, f"{oid} Score-A target", errors)
            check_close(stored.get("score_a_zone0_1_ln"), math.log(a_target) if a_target is not None and a_target > 0 else None,
                        f"{oid} log Score-A target", errors)
            check_close(stored["score_a_zone2"], a_background, f"{oid} Score-A background", errors)
            check_close(stored.get("score_a_zone2_ln"), math.log(a_background) if a_background is not None and a_background > 0 else None,
                        f"{oid} log Score-A background", errors)
            # Independently recompute location-matched Score B from retained projected vectors.
            current_doy = utc(times[i]).timetuple().tm_yday
            patch_scores = []
            for patch in np.flatnonzero(np.isin(zones[i], [0, 1])):
                distances = []
                for j, ref_id in enumerate(ids):
                    if ref_id == oid or sites[j] != sites[i] or ref_id not in fit_ids:
                        continue
                    delta = abs(current_doy - utc(times[j]).timetuple().tm_yday)
                    if min(delta, 366 - delta) <= 45:
                        distances.append(float(np.linalg.norm(projected[i, patch] - projected[j, patch])))
                if len(distances) >= 5:
                    patch_scores.append(float(np.mean(np.sort(distances)[:5])))
            b_value = float(np.mean(patch_scores)) if patch_scores else None
            check_close(stored["score_b_zone0_1"], b_value, f"{oid} Score-B", errors)
            arithmetic_checks.append({"observation_id": oid, "seed": int(replay_meta["seed"]),
                                     "score_a_target_replayed": a_target,
                                     "score_a_background_replayed": a_background,
                                     "score_b_replayed": b_value})
        seed_predictions = [p for p in predictions if int(p["seed"]) == int(replay_meta["seed"])]
        fit_pairs = [p for p in seed_predictions if p["role"] == "FIT"
                     and p["score_a_zone0_1"] is not None and p["score_a_zone0_1"] > 0
                     and p["score_b_zone0_1"] is not None]
        expected_c = {}
        if len(fit_pairs) >= 2:
            ln_a = np.log(np.asarray([p["score_a_zone0_1"] for p in fit_pairs], dtype=np.float64))
            b_values = np.asarray([p["score_b_zone0_1"] for p in fit_pairs], dtype=np.float64)
            med_a, mad_a = np.median(ln_a), np.median(np.abs(ln_a - np.median(ln_a)))
            med_b, mad_b = np.median(b_values), np.median(np.abs(b_values - np.median(b_values)))
            if mad_a > np.finfo(float).eps and mad_b > np.finfo(float).eps:
                for p in seed_predictions:
                    if p["score_a_zone0_1"] is not None and p["score_a_zone0_1"] > 0 and p["score_b_zone0_1"] is not None:
                        za = (math.log(p["score_a_zone0_1"]) - med_a) / (1.4826 * mad_a)
                        zb = (p["score_b_zone0_1"] - med_b) / (1.4826 * mad_b)
                        expected_c[p["observation_id"]] = 0.5 * za + 0.5 * zb
        for p in seed_predictions:
            check_close(p.get("score_c"), expected_c.get(p["observation_id"]),
                        f"{p['observation_id']} Score-C", errors)
    result = {"status": "PASS" if not errors else "FAIL", "manifest": str(manifest_path.relative_to(ROOT)),
              "manifest_sha256": sha(manifest_path), "source_rows_checked": len(source_checks),
              "arithmetic_rows_checked": len(arithmetic_checks), "source_checks": source_checks,
              "arithmetic_checks": arithmetic_checks, "errors": errors}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "source_rows_checked", "arithmetic_rows_checked", "errors")}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
