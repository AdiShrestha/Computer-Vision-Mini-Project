#!/usr/bin/env python3
"""Independently recompute descriptive FIT/VAL summaries from retained SR-1 records.

This utility intentionally does not load model checkpoints or inspect EVAL rows.
It verifies report bindings and recomputes simple descriptive statistics only.
It is not an independent scientific review or a reproduction of training.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAPACITY = Path("data/derived/sr1_multiyear/attempt_006/sensitivity_001/capacity_report.json")
FACTORIAL = Path("data/derived/sr1_multiyear/attempt_006/sensitivity_002/factorial_report.json")
PRIMARY = Path("data/derived/sr1_multiyear/attempt_006/analysis_001/analysis_report.json")
PROTOCOLS = {
    "capacity": Path("project/sr1_capacity_sweep_protocol_v1.json"),
    "factorial": Path("project/sr1_factorial_ablation_protocol_v1.json"),
}
OUTPUT = Path("project/remediation/2026-10-26/sr1_true_implementation_001/SENSITIVITY_RECOMPUTATION_001.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty vector")
    return {
        "n_seed_runs": len(values),
        "values_by_seed": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def check_scope(report: dict[str, Any], primary: dict[str, Any]) -> None:
    expected_fit = set(primary["split"]["FIT_ids"])
    expected_val = set(primary["split"]["VAL_ids"])
    expected_eval = set(primary["split"]["EVAL_ids"])
    actual_fit = set(report["fit_ids"])
    actual_val = set(report["validation_ids"])
    if actual_fit != expected_fit or actual_val != expected_val:
        raise ValueError("Sensitivity report FIT/VAL membership differs from primary analysis")
    if actual_fit & actual_val or (actual_fit | actual_val) & expected_eval:
        raise ValueError("Sensitivity scopes overlap or include primary EVAL IDs")
    if report.get("evaluation_access") != "NO_EVAL_SCORING_OR_SUMMARY_PERFORMED":
        raise ValueError("Report does not bind a zero-EVAL access record")


def capacity_summary(report: dict[str, Any]) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = {32: [], 64: [], 128: []}
    for row in report["results"]:
        groups[int(row["d_model"])].append(row)
    groups[64] = report["width_64_reference"]
    output = {}
    for width in (32, 64, 128):
        rows = sorted(groups[width], key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(range(5)):
            raise ValueError(f"Width {width} does not have exactly seeds 0..4")
        values = [finite(row["best_val_masked_mse"], f"capacity width {width}") for row in rows]
        counts = {int(row["parameter_count"]) for row in rows}
        if len(counts) != 1:
            raise ValueError(f"Width {width} has inconsistent parameter counts")
        output[str(width)] = {
            "parameter_count": counts.pop(),
            "best_val_masked_mse": summarize(values),
            "stopping_status_counts": {
                status: sum(row["stopping_status"] == status for row in rows)
                for status in sorted({row["stopping_status"] for row in rows})
            } if all("stopping_status" in row for row in rows) else None,
            "epochs_completed": [int(row["epochs_completed"]) for row in rows]
            if all("epochs_completed" in row for row in rows) else None,
            "source_status": "NEW_CAPACITY_RUNS" if width != 64 else "RETAINED_PRIMARY_REFERENCE; SWEEP REPORT DOES NOT REPEAT STOPPING METADATA",
        }
    return output


def factorial_summary(report: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, bool, int], dict[int, dict[str, Any]]] = {}
    for row in report["results"]:
        key = (int(row["channels"]), bool(row["zones_enabled"]), int(row["context_scenes_max"]))
        seed = int(row["seed"])
        if seed in grouped.setdefault(key, {}):
            raise ValueError(f"Duplicate factorial seed {seed} for {key}")
        grouped[key][seed] = row
    if len(grouped) != 8 or any(sorted(rows) != list(range(5)) for rows in grouped.values()):
        raise ValueError("Factorial result rows are incomplete")

    cells: dict[str, Any] = {}
    for (channels, zones, context), rows in sorted(grouped.items()):
        key = f"channels_{channels}_zones_{'fit' if zones else 'constant2'}_context_{context}"
        ordered = [rows[seed] for seed in range(5)]
        counts = {int(row["parameter_count"]) for row in ordered}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent parameter counts in {key}")
        cells[key] = {
            "parameter_count": counts.pop(),
            "best_val_masked_mse": summarize([
                finite(row["best_val_masked_mse"], key) for row in ordered
            ]),
            "stopping_status_counts": {
                status: sum(row["stopping_status"] == status for row in ordered)
                for status in sorted({row["stopping_status"] for row in ordered})
            },
        }

    contrasts: dict[str, Any] = {}
    for channels in (3, 5):
        cell_values = {
            (zones, context): {
                seed: finite(grouped[(channels, zones, context)][seed]["best_val_masked_mse"], "factorial contrast")
                for seed in range(5)
            }
            for zones in (False, True)
            for context in (1, 8)
        }
        by_factor = {}
        # Paired seed deltas are descriptive optimizer-seed contrasts only.
        for label, pairs in {
            "zone_fit_minus_constant_context_1": ((True, 1), (False, 1)),
            "zone_fit_minus_constant_context_8": ((True, 8), (False, 8)),
            "context_8_minus_1_constant_zone": ((False, 8), (False, 1)),
            "context_8_minus_1_fit_zone": ((True, 8), (True, 1)),
        }.items():
            left, right = pairs
            deltas = [cell_values[left][seed] - cell_values[right][seed] for seed in range(5)]
            by_factor[label] = summarize(deltas)
        interaction = [
            (cell_values[(True, 8)][seed] - cell_values[(True, 1)][seed])
            - (cell_values[(False, 8)][seed] - cell_values[(False, 1)][seed])
            for seed in range(5)
        ]
        by_factor["zone_by_context_difference_in_differences"] = summarize(interaction)
        contrasts[str(channels)] = by_factor
    return {"cells": cells, "within_channel_paired_seed_deltas": contrasts}


def main() -> None:
    primary = read_json(ROOT / PRIMARY)
    capacity = read_json(ROOT / CAPACITY)
    factorial = read_json(ROOT / FACTORIAL)
    check_scope(capacity, primary)
    check_scope(factorial, primary)
    for name, path, report in (("capacity", CAPACITY, capacity), ("factorial", FACTORIAL, factorial)):
        if report["protocol_sha256"] != digest(ROOT / PROTOCOLS[name]):
            raise ValueError(f"{name} protocol hash does not match retained report")
        if report["manifest_sha256"] != primary["manifest_sha256"]:
            raise ValueError(f"{name} manifest does not match primary analysis")

    payload = {
        "schema_version": 1,
        "status": "PASS_DESCRIPTIVE_RECOMPUTATION_ONLY; NO_TRAINING_REPLAY; NO_INDEPENDENT_SCIENTIFIC_REVIEW",
        "scope": {
            "uses_only": "retained FIT/VAL report rows and primary manifest role IDs",
            "evaluation_rows_read": 0,
            "training_reproduced": False,
            "scientific_or_statistical_inference": False,
            "seed_scope": "paired optimizer-seed descriptions on a fixed corpus; seeds are not independent event or lake replicates",
            "channels_3_vs_5_comparison": "not computed across channel counts because target sets differ",
        },
        "inputs": {
            str(path): {"sha256": digest(ROOT / path)}
            for path in (CAPACITY, FACTORIAL, PRIMARY, *PROTOCOLS.values())
        },
        "fit_ids": sorted(capacity["fit_ids"]),
        "validation_ids": sorted(capacity["validation_ids"]),
        "capacity": capacity_summary(capacity),
        "factorial": factorial_summary(factorial),
    }
    out = ROOT / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": digest(out), "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
