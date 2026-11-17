"""Audit the quarantined 160x160 pipeline against retained source-bound observations.

This is an integrity audit, not an experiment runner. It never loads SR-1 result
bindings or checkpoints and never writes into their historical output locations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SLICES = {
    "PILOT-SOUTH-LHONAK": (180, 340, 125, 285),
    "PILOT-KHANGCHUNG-TSHO": (170, 330, 170, 330),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_member(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes repository: {value}")
    return path


def crop_transform(transform: list[float], row0: int, col0: int) -> list[float]:
    a, b, c, d, e, f = transform
    return [a, b, c + col0 * a + row0 * b, d, e, f + col0 * d + row0 * e]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/expansion_2023_recentered/focused_pipeline_audit.v1.json"),
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay inside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")

    manifest_path = ROOT / "data/derived/expansion_2023_recentered/observations.v2.json"
    registry_path = ROOT / "project/sr1_registry.json"
    domain_path = ROOT / "project/DOMAIN_READINESS_2023.json"
    manifest = json.loads(manifest_path.read_text())
    registry = json.loads(registry_path.read_text())
    domain = json.loads(domain_path.read_text())
    if sha256(manifest_path) != domain["dataset"]["sha256"]:
        raise ValueError("source manifest differs from domain readiness binding")

    optical = [row for row in manifest["observations"] if row["sensor"] == "SENTINEL_2_OPTICAL"]
    sar = [row for row in manifest["observations"] if row["sensor"] == "SENTINEL_1_IW_GRD"]
    issues: list[str] = []
    source_bindings: list[dict[str, Any]] = []
    scene_checks: list[dict[str, Any]] = []
    per_site: dict[str, Any] = {}

    if len(optical) != 48 or len(sar) != 2:
        issues.append(f"unexpected source counts: optical={len(optical)}, sar={len(sar)}")
    if len({row["observation_id"] for row in optical}) != len(optical):
        issues.append("duplicate optical observation identifiers")

    for site, (row0, row1, col0, col1) in SLICES.items():
        lake = registry["lakes"][site]
        rows = [row for row in optical if row["site_id"] == site]
        if len(rows) != 24:
            issues.append(f"{site}: expected 24 optical observations, found {len(rows)}")
        expected_grid = lake["grid"]
        derived_transforms = set()
        valid_crop_pixels = []
        guarded_bounds = [float("inf"), float("-inf")]
        stored_index_max_abs_error = [0.0, 0.0]
        stored_index_compared = 0

        for row in rows:
            values_path = resolve_member(row["values_file"])
            receipt_path = resolve_member(row["receipt_file"])
            values_hash = sha256(values_path)
            receipt_hash = sha256(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if values_hash != row["sha256"] or receipt_hash != row["receipt_sha256"]:
                issues.append(f"{row['observation_id']}: ROI or receipt hash mismatch")
            if row.get("origin") != "observational" or receipt.get("origin") != "observational":
                issues.append(f"{row['observation_id']}: non-observational origin")
            if receipt.get("observation_id") != row["observation_id"]:
                issues.append(f"{row['observation_id']}: receipt identity mismatch")
            if receipt.get("native", {}).get("baseline") != "05.09":
                issues.append(f"{row['observation_id']}: unexpected/missing processing baseline")
            for band in ("B03", "B08", "B11"):
                radiometry = receipt.get("native", {}).get("radiometry", {}).get(band, {})
                if radiometry != {"scale": 0.0001, "offset": -0.1}:
                    issues.append(f"{row['observation_id']}: unexpected {band} scale/offset")

            source_grid = row["grid"]
            derived = crop_transform(source_grid["transform"], row0, col0)
            derived_transforms.add(tuple(derived))
            if source_grid["crs"] != expected_grid["crs"] or derived != expected_grid["transform"]:
                issues.append(f"{row['observation_id']}: crop affine does not match registry")
            if tuple(source_grid["shape"]) != (499, 499):
                issues.append(f"{row['observation_id']}: parent grid is not 499x499")

            with np.load(values_path, allow_pickle=False) as archive:
                values = archive["values"]
                observed = archive["observed_valid"]
                scl = archive["scl_classes"]
                if values.shape != (5, 499, 499) or observed.shape != values.shape or scl.shape != (499, 499):
                    issues.append(f"{row['observation_id']}: source array shape mismatch")
                    continue
                if values.dtype != np.float32 or observed.dtype != np.bool_:
                    issues.append(f"{row['observation_id']}: unexpected source array dtype")
                if not np.isfinite(values[observed]).all() or not np.isnan(values[~observed]).all():
                    issues.append(f"{row['observation_id']}: observed/missing pixel semantics mismatch")
                crop = values[:, row0:row1, col0:col1]
                crop_obs = observed[:, row0:row1, col0:col1]
                crop_scl = scl[row0:row1, col0:col1]
                clear = np.isin(crop_scl, [4, 5, 6]) & crop_obs.all(axis=0)
                valid_crop_pixels.append(int(clear.sum()))

                # Independent guarded-index arithmetic from the retained three base bands.
                green = np.maximum(crop[0].astype(np.float64), 0.001)
                nir = np.maximum(crop[1].astype(np.float64), 0.001)
                swir1 = np.maximum(crop[2].astype(np.float64), 0.001)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndwi = (green - nir) / (green + nir + 0.002)
                    mndwi = (green - swir1) / (green + swir1 + 0.002)
                finite = np.isfinite(green) & np.isfinite(nir) & np.isfinite(swir1)
                if finite.any():
                    guarded_bounds[0] = min(guarded_bounds[0], float(ndwi[finite].min()))
                    guarded_bounds[1] = max(guarded_bounds[1], float(ndwi[finite].max()))
                    if np.any(np.abs(ndwi[finite]) > 1.0) or np.any(np.abs(mndwi[finite]) > 1.0):
                        issues.append(f"{row['observation_id']}: guarded index outside [-1,1]")
                    stored = crop[3:5].astype(np.float64)
                    stored_valid = crop_obs[3:5].all(axis=0) & np.isfinite(stored).all(axis=0) & finite
                    if stored_valid.any():
                        stored_index_compared += int(stored_valid.sum())
                        stored_index_max_abs_error[0] = max(
                            stored_index_max_abs_error[0], float(np.max(np.abs(stored[0][stored_valid] - ndwi[stored_valid])))
                        )
                        stored_index_max_abs_error[1] = max(
                            stored_index_max_abs_error[1], float(np.max(np.abs(stored[1][stored_valid] - mndwi[stored_valid])))
                        )
            source_bindings.extend([
                {"path": row["values_file"], "sha256": values_hash},
                {"path": row["receipt_file"], "sha256": receipt_hash},
            ])
            scene_checks.append({
                "observation_id": row["observation_id"],
                "site_id": site,
                "acquired_at": row["acquired_at"],
                "clear_crop_pixels": valid_crop_pixels[-1],
                "clear_crop_fraction": valid_crop_pixels[-1] / 25600.0,
            })

        zone_path = resolve_member(lake["geomorphic_zones"]["zone_map_file"])
        zone_hash = sha256(zone_path)
        if zone_hash != lake["geomorphic_zones"]["zone_map_sha256"]:
            issues.append(f"{site}: static zone map hash mismatch")
        with np.load(zone_path, allow_pickle=False) as zones_archive:
            zones = zones_archive["zones"]
            zone_origin = str(zones_archive["origin"].item())
            if zones.shape != (160, 160):
                issues.append(f"{site}: static zone map is not 160x160")
            zone_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(zones, return_counts=True))}
        per_site[site] = {
            "scene_count": len(rows),
            "source_grid_shape": [499, 499],
            "crop_shape": [row1 - row0, col1 - col0],
            "crop_transform_matches_registry": len(derived_transforms) == 1 and list(next(iter(derived_transforms))) == expected_grid["transform"],
            "crop_transform": list(next(iter(derived_transforms))) if derived_transforms else None,
            "zone_file_sha256_matches_registry": zone_hash == lake["geomorphic_zones"]["zone_map_sha256"],
            "zone_origin": zone_origin,
            "zone_pixel_counts": zone_counts,
            "minimum_clear_valid_crop_pixels": min(valid_crop_pixels) if valid_crop_pixels else None,
            "maximum_clear_valid_crop_pixels": max(valid_crop_pixels) if valid_crop_pixels else None,
            "guarded_ndwi_observed_min_max": guarded_bounds,
            "stored_index_vs_guarded_max_abs_error": stored_index_max_abs_error,
            "stored_index_valid_pixels_compared": stored_index_compared,
            "geometry_admission": "NOT_ESTABLISHED_BY_HASH_OR_CROP_AFFINE",
        }

    results = {
        "schema_version": 1,
        "audit_status": "PASS_SOURCE_BINDINGS_AND_CROP_ARITHMETIC_WITH_METHOD_BLOCKERS",
        "scope": "NON_DESTRUCTIVE_AUDIT; NO_SR1_BINDINGS_OR_CHECKPOINTS_LOADED; NO_HISTORICAL_OUTPUTS_OVERWRITTEN",
        "source_manifest_sha256": sha256(manifest_path),
        "source_manifest_observations": len(manifest["observations"]),
        "optical_observations_checked": len(optical),
        "sar_observations_checked_for_count_and_dates": len(sar),
        "sar_dates": sorted({row["acquired_at"][:10] for row in sar}),
        "source_and_receipt_bindings": len(source_bindings),
        "site_checks": per_site,
        "findings": [
            {
                "severity": "BLOCKING",
                "claim": "Physical water area using NDWI > 0.05",
                "evidence": "source/sentinel_gl/sr1_features.py instead uses SCL=6 OR a conjunction of MNDWI>0.05, NDWI>0, B08<0.16, and B03>0.02; it also marks low-clear scenes unavailable",
                "disposition": "Do not report as the requested NDWI-only physical-area estimand",
            },
            {
                "severity": "BLOCKING",
                "claim": "Denominator guard exactly as specified in the requested equation",
                "evidence": "source/sentinel_gl/sr1_features.py computes (a-b)/(a+b+delta), while the requested protocol specifies (a-b)/max(delta,a+b); these are different operators",
                "disposition": "Do not claim the requested equation was implemented without a separately reviewed protocol amendment and recomputation",
            },
            {
                "severity": "BLOCKING",
                "claim": "Independent chronological training/validation/reference partitions",
                "evidence": "source/runners/run_sr1_execution.py fits robust scaling on all pre-cutoff tokens before defining the last-two-scenes-per-site validation split; its PCA reference pool is also built from all pre-cutoff scenes",
                "disposition": "Validation observations contribute to preprocessing/reference state; no independent three-role evaluation claim",
            },
            {
                "severity": "BLOCKING",
                "claim": "Estimand E2 false alarms per eligible lake-year",
                "evidence": "source/sentinel_gl/sr1_eval.py returns episode_count / total_valid_scenes and does not use an eligible-window exposure ledger",
                "disposition": "Not a false-alarm rate per lake-year; keep unavailable",
            },
            {
                "severity": "BLOCKING",
                "claim": "Estimand E3 with unavailable status",
                "evidence": "compute_percentile_rank returns 50.0 when reference support or query is non-finite",
                "disposition": "Fallback value cannot support percentile inference; recompute with explicit null/status or omit",
            },
            {
                "severity": "BLOCKING",
                "claim": "Estimand E4 for scenes without target/background support",
                "evidence": "compute_zone_contrast returns numeric zero when either zone has no supported patches",
                "disposition": "Zero is confounded with a measured zero contrast; preserve as unavailable/status",
            },
            {
                "severity": "BLOCKING",
                "claim": "Estimand E5 incremental value",
                "evidence": "source/runners/run_sr1_execution.py assigns E5 a fixed narrative string rather than a computed comparison",
                "disposition": "Not implemented; omit",
            },
            {
                "severity": "BLOCKING",
                "claim": "Moving-block bootstrap confidence intervals",
                "evidence": "block_bootstrap_ci ignores record timestamps and block_days when choosing block_size; for fewer than three records it returns a degenerate interval",
                "disposition": "Reported intervals are not valid time-block bootstrap inference",
            },
            {
                "severity": "BLOCKING",
                "claim": "Universal common CUSUM evaluation across methods",
                "evidence": "run_sr1_execution.py uses different kappa/h settings for Score C versus area and tabular baselines, and rescales detector scores before updates",
                "disposition": "No common calibrated alarm comparison; omit alarm-performance claims",
            },
            {
                "severity": "BLOCKING",
                "claim": "Regional-versus-focused factorial ablation",
                "evidence": "the two runs change spatial support, feature processing, architecture, masking, normalization, and scoring simultaneously; no isolated factorial interventions are executed",
                "disposition": "Do not interpret pipeline comparison as a factorial ablation or causal mechanism test",
            },
            {
                "severity": "BLOCKING",
                "claim": "June Khangchung transient cause or validated lake-zone effect",
                "evidence": "the current zone maps are pre-cutoff observational composites but lake-polygon accuracy/zone validation is not established in DOMAIN_READINESS_2023.json",
                "disposition": "No physical or causal explanation; any values remain exploratory zone diagnostics",
            },
        ],
        "unresolved_research_scope": [
            "Static zone maps are hash-bound but not independently validated lake polygons.",
            "This audit verifies source identity and crop-grid arithmetic, not upstream independent source decoding, QA truth, or physical lake geometry.",
            "The SR-1 runner was not executed because it writes to fixed historical directories and contains unsupported/fallback estimand logic; its stored scores/checkpoints remain quarantined.",
            "No independent reviewer, source-rights review, publication-history review, venue-template confirmation, or license change is established by this local audit.",
        ],
        "source_code_sha256": {
            "source/runners/run_sr1_execution.py": sha256(ROOT / "source/runners/run_sr1_execution.py"),
            "source/sentinel_gl/sr1_features.py": sha256(ROOT / "source/sentinel_gl/sr1_features.py"),
            "source/sentinel_gl/sr1_eval.py": sha256(ROOT / "source/sentinel_gl/sr1_eval.py"),
            "source/sentinel_gl/sr1_baselines.py": sha256(ROOT / "source/sentinel_gl/sr1_baselines.py"),
            "source/sentinel_gl/sr1_model.py": sha256(ROOT / "source/sentinel_gl/sr1_model.py"),
        },
        "source_bindings": source_bindings,
        "scene_support_checks": scene_checks,
        "issues": issues,
    }
    results["audit_status"] = (
        "PASS_SOURCE_BINDINGS_AND_CROP_ARITHMETIC_WITH_METHOD_BLOCKERS" if not issues else "FAIL_SOURCE_BINDINGS_OR_CROP_ARITHMETIC"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k not in {"source_bindings", "scene_support_checks"}}, indent=2))


if __name__ == "__main__":
    main()
