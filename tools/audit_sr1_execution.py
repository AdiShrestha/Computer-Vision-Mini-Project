"""Source-bind and independently recompute arithmetic in one fresh 160x160 run.

This audit does not certify the scientific estimands, lake geometry, or model.
It preserves an executable result ledger while explicitly reporting method
defects found by source inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SLICES = {
    "PILOT-SOUTH-LHONAK": (180, 340, 125, 285),
    "PILOT-KHANGCHUNG-TSHO": (170, 330, 170, 330),
}
DELTA = 0.002
FLOOR = 0.001


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(value: Any) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_root = (ROOT / args.artifact_root).resolve()
    output = (ROOT / args.output).resolve()
    for path in (artifact_root, output):
        if not path.is_relative_to(ROOT):
            raise ValueError("all paths must stay inside the workspace")
    result_path = artifact_root / "docs/tables/sr1/result_bindings.json"
    checkpoints = artifact_root / "data/derived/sr1_multiyear/checkpoints"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if output.exists():
        raise FileExistsError(output)

    manifest_path = ROOT / "data/derived/expansion_2023_recentered/observations.v2.json"
    registry_path = ROOT / "project/sr1_registry.json"
    focused_audit_path = ROOT / "data/derived/expansion_2023_recentered/focused_pipeline_audit.v2.json"
    bindings = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    registry = json.loads(registry_path.read_text())
    focused_audit = json.loads(focused_audit_path.read_text())
    manifest_by_name = {
        Path(row["values_file"]).name: row
        for row in manifest["observations"]
        if row.get("sensor") == "SENTINEL_2_OPTICAL"
    }
    observations = bindings.get("observations", [])
    issues: list[str] = []
    source_checks: list[dict[str, Any]] = []
    water_checks: list[dict[str, Any]] = []
    time_discrepancies: list[dict[str, Any]] = []
    contrast_errors: list[float] = []
    if len(observations) != 48:
        issues.append(f"expected 48 optical result rows, found {len(observations)}")
    if len({row.get("file_name") for row in observations}) != len(observations):
        issues.append("duplicate output file names")

    for record in observations:
        name = record["file_name"]
        source = manifest_by_name.get(name)
        if source is None:
            issues.append(f"no optical source manifest row for {name}")
            continue
        source_path = (ROOT / source["values_file"]).resolve()
        receipt_path = (ROOT / source["receipt_file"]).resolve()
        source_hash = sha(source_path)
        receipt_hash = sha(receipt_path)
        source_ok = source_hash == source["sha256"]
        receipt_ok = receipt_hash == source["receipt_sha256"]
        if not source_ok or not receipt_ok:
            issues.append(f"source or receipt binding mismatch for {name}")

        site = record["site_id"]
        if source["site_id"] != site:
            issues.append(f"site mismatch for {name}")
        row0, row1, col0, col1 = SLICES[site]
        zone_path = ROOT / registry["lakes"][site]["geomorphic_zones"]["zone_map_file"]
        with np.load(source_path, allow_pickle=False) as archive:
            raw = archive["values"]
            scl = archive["scl_classes"][row0:row1, col0:col1]
            if raw.shape != (5, 499, 499):
                issues.append(f"unexpected parent-array shape for {name}: {raw.shape}")
                continue
            b03 = np.maximum(raw[0, row0:row1, col0:col1], FLOOR)
            b08 = np.maximum(raw[1, row0:row1, col0:col1], FLOOR)
            b11 = np.maximum(raw[2, row0:row1, col0:col1], FLOOR)
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = np.clip((b03 - b08) / (b03 + b08 + DELTA), -1.0, 1.0).astype(np.float32)
                mndwi = np.clip((b03 - b11) / (b03 + b11 + DELTA), -1.0, 1.0).astype(np.float32)
            crop = np.stack([raw[0, row0:row1, col0:col1], raw[1, row0:row1, col0:col1],
                             raw[2, row0:row1, col0:col1], ndwi, mndwi])
            valid = np.isin(scl, [4, 5, 6]) & np.isfinite(crop).all(axis=0)
            clear_fraction = float(valid.sum() / valid.size)
            with np.load(zone_path, allow_pickle=False) as zone_archive:
                zones = zone_archive["zones"]
            core = zones == 0
            valid_qa = np.isin(scl, [4, 5, 6]) & np.isfinite(crop[0])
            core_support = float((valid_qa & core).sum() / max(1, int(core.sum())))
            if clear_fraction < 0.45 or core_support < 0.25:
                expected_area = float("nan")
            else:
                liquid = core & valid_qa & (
                    (scl == 6) |
                    ((crop[4] > 0.05) & (crop[3] > 0.0) & (crop[1] < 0.16) & (crop[0] > 0.02))
                )
                expected_area = float(int(liquid.sum()) * 400.0 / 1e6)
        observed_area = record.get("water_area_km2")
        areas_match = (
            math.isnan(expected_area) and (observed_area is None or (isinstance(observed_area, float) and math.isnan(observed_area)))
        ) or (
            math.isfinite(expected_area) and observed_area is not None and math.isfinite(float(observed_area))
            and abs(expected_area - float(observed_area)) <= 1e-12
        )
        if not areas_match:
            issues.append(f"independent water-rule arithmetic mismatch for {name}")
        source_checks.append({
            "file_name": name,
            "site_id": site,
            "source_path": source["values_file"],
            "source_sha256": source_hash,
            "receipt_sha256": receipt_hash,
            "source_binding_valid": source_ok,
            "receipt_binding_valid": receipt_ok,
        })
        water_checks.append({
            "file_name": name,
            "site_id": site,
            "clear_fraction_recomputed": clear_fraction,
            "core_support_fraction_recomputed": core_support,
            "reported_water_area_km2": finite_or_none(observed_area),
            "recomputed_water_area_km2": finite_or_none(expected_area),
            "arithmetic_matches": areas_match,
            "scientific_status": "EXPLORATORY_RULE_OUTPUT; NOT_VALIDATED_PHYSICAL_WATER_AREA",
        })
        manifest_dt = datetime.fromisoformat(source["acquired_at"].replace("Z", "+00:00"))
        run_dt = datetime.fromisoformat(record["acquired_at"].replace("Z", "+00:00"))
        same_date = manifest_dt.date() == run_dt.date()
        exact_time_matches = abs((manifest_dt - run_dt).total_seconds()) < 1.0
        if not same_date:
            issues.append(f"acquisition calendar-date mismatch for {name}")
        elif not exact_time_matches:
            time_discrepancies.append({
                "file_name": name,
                "manifest_acquired_at": source["acquired_at"],
                "run_acquired_at": record["acquired_at"],
                "difference_seconds": (run_dt - manifest_dt).total_seconds(),
            })
        if "score_target_zone" in record and "score_background_zone" in record:
            error = abs(float(record["zone_contrast"]) -
                        (float(record["score_target_zone"]) - float(record["score_background_zone"])))
            contrast_errors.append(error)
            if error > 1e-7:
                issues.append(f"zone contrast arithmetic mismatch for {name}")

    def median_for(site: str, role: str, field: str) -> float | None:
        values = [float(r[field]) for r in observations if r["site_id"] == site and r["role"] == role
                  and r.get(field) is not None and math.isfinite(float(r[field]))]
        return float(np.median(values)) if values else None

    roles = sorted({r.get("role") for r in observations})
    site_counts = {site: sum(r.get("site_id") == site for r in observations) for site in SLICES}
    seed_histories = bindings.get("training_histories", {})
    checkpoint_hashes = {}
    for seed in range(5):
        path = checkpoints / f"sr1_tmae_seed_{seed}.pt"
        if not path.is_file():
            issues.append(f"missing checkpoint for seed {seed}")
        else:
            checkpoint_hashes[str(seed)] = sha(path)

    findings = [
        "The index denominator is (a+b+delta) with a floor and clipping, not (a-b)/max(delta,a+b).",
        "The reported liquid-water rule is SCL==6 OR a conjunction of MNDWI>0.05, NDWI>0, B08<0.16, and B03>0.02; this is not NDWI>0.05 alone.",
        "Robust scaling and latent PCA use every pre-cutoff scene before the last-two-scenes-per-site validation split is declared; nominal validation scenes therefore enter preprocessing/reference state.",
        "E2 divides alert episodes by valid scene count, not eligible lake-year exposure.",
        "E3 returns 50.0 when support is unavailable; E4 returns numeric zero when zone support is absent.",
        "E5 is a fixed narrative string in the runner, not a computed incremental-value statistic.",
        "The bootstrap ignores record dates and block_days; intervals from fewer than three records are degenerate.",
        "CUSUM inputs are differently scaled and use different kappa/h settings across methods; a common calibrated alarm comparison is not established.",
        "Zone arrays are static hash-bound composites; independent lake-polygon validity and physical zone interpretation are not established.",
        "The independent source/crop audit found materially different stored index values and guarded-index recomputations; no clean 160x160-versus-499x499 factorial effect is identified.",
    ]
    report = {
        "schema_version": 1,
        "status": "PASS_SOURCE_BINDING_AND_ROW_ARITHMETIC_WITH_METHOD_AND_TIME_BLOCKERS" if not issues else "FAIL_SOURCE_BINDING_OR_ROW_ARITHMETIC",
        "scope": "FRESH_160X160_DEVELOPMENT_EXECUTION_AUDIT; NOT_SCIENTIFIC_VALIDATION_OR_PUBLICATION_CERTIFICATION",
        "artifact_root": str(artifact_root.relative_to(ROOT)),
        "result_bindings": str(result_path.relative_to(ROOT)),
        "result_bindings_sha256": sha(result_path),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha(manifest_path),
        "focused_source_crop_audit": str(focused_audit_path.relative_to(ROOT)),
        "focused_source_crop_audit_sha256": sha(focused_audit_path),
        "focused_source_crop_audit_status": focused_audit.get("audit_status"),
        "rows": len(observations),
        "rows_by_site": site_counts,
        "roles": roles,
        "training_seed_count": len(seed_histories),
        "parameter_count_reported": bindings.get("model_parameter_count"),
        "checkpoint_sha256": checkpoint_hashes,
        "source_and_receipt_rows_verified": len(source_checks),
        "source_time_discrepancy_count": len(time_discrepancies),
        "source_time_discrepancy_examples": time_discrepancies[:5],
        "independent_water_rule_arithmetic_rows": len(water_checks),
        "maximum_zone_contrast_arithmetic_error": max(contrast_errors, default=0.0),
        "recomputed_descriptive_medians": {
            site: {
                role: {
                    field: median_for(site, role, field)
                    for field in ("score_c", "water_area_km2", "zone_contrast")
                }
                for role in roles
            }
            for site in SLICES
        },
        "findings": findings,
        "limitations": [
            "Source manifest SHA bindings, row-to-source identities, the implemented water-rule arithmetic, and zone-contrast subtraction were checked; model training and latent embeddings were not independently recomputed.",
            "The run uses timestamps parsed from local filenames; those timestamps differ from source-manifest acquisition timestamps, even where the calendar date agrees. Event-time and delay calculations are not admitted until this source discrepancy is reconciled.",
            "Water-area arithmetic is reproduced as code output only. Static zone geometry, classification accuracy, pixel-to-lake correspondence, source truth and calibration remain unvalidated.",
            "Detection delay, false-alarm rate, percentile fallback, bootstrap intervals, E5 and alarm comparisons are not admitted as scientific estimands.",
            "The run is retrospective development work; the two ROIs are not independent event/control clusters and do not support a factorial ablation or operational warning result.",
        ],
        "source_and_receipt_bindings": source_checks,
        "water_rule_arithmetic": water_checks,
        "issues": issues,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_and_receipt_bindings", "water_rule_arithmetic"}}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
