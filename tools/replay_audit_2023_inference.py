"""Repeat inference from retained source pixels and checkpoints into a new epoch.

The original analysis directory is read-only to this script. The replay is an
execution reproducibility check, not independent scientific validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sentinel_gl.inference_v2 import run_inference_v2
from sentinel_gl.panel_v2 import load_panel_v2

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=Path("data/derived/expansion_2023_recentered/analysis_002"))
    parser.add_argument("--replay", type=Path, default=Path("data/derived/expansion_2023_recentered/analysis_002/independent_inference_replay_001"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/expansion_2023_recentered/analysis_002/inference_replay_audit.v1.json"))
    parser.add_argument("--compare-existing", action="store_true", help="compare a completed replay directory without rerunning inference")
    args = parser.parse_args()
    analysis = (ROOT / args.analysis).resolve()
    replay = (ROOT / args.replay).resolve()
    report_path = (ROOT / args.output).resolve()
    for path in (analysis, replay, report_path):
        if not path.is_relative_to(ROOT):
            raise ValueError("all paths must stay inside the workspace")
    if report_path.exists():
        raise FileExistsError("refusing to overwrite the audit report")
    if replay.exists() and not args.compare_existing:
        raise FileExistsError("refusing to overwrite the replay; use --compare-existing to audit an existing replay")
    if not replay.exists() and args.compare_existing:
        raise FileNotFoundError("--compare-existing requires an existing replay directory")

    manifest = ROOT / "data/derived/expansion_2023_recentered/observations.v2.json"
    panel = load_panel_v2(ROOT, manifest, mode="observational")
    if args.compare_existing:
        result = json.loads((replay / "analysis_summary.v2.json").read_text())
    else:
        result = run_inference_v2(panel, analysis / "training", replay)
    original_path = analysis / "inference/scores.v2.jsonl"
    replay_path = replay / "scores.v2.jsonl"
    original = [json.loads(line) for line in original_path.read_text().splitlines() if line]
    repeated = [json.loads(line) for line in replay_path.read_text().splitlines() if line]
    key = lambda row: (row["seed"], row["observation_id"], row["method"])
    original_map = {key(row): row for row in original}
    replay_map = {key(row): row for row in repeated}
    missing = sorted(set(original_map) - set(replay_map))
    extra = sorted(set(replay_map) - set(original_map))
    max_score_error = 0.0
    max_target_error = 0.0
    max_prediction_error = 0.0
    status_mismatches = []
    denominator_mismatches = []
    for record_key in sorted(set(original_map) & set(replay_map), key=lambda key: tuple("" if value is None else str(value) for value in key)):
        left, right = original_map[record_key], replay_map[record_key]
        if left.get("status") != right.get("status"):
            status_mismatches.append({"key": record_key, "original": left.get("status"), "replay": right.get("status")})
        if left.get("denominator") != right.get("denominator"):
            denominator_mismatches.append({"key": record_key, "original": left.get("denominator"), "replay": right.get("denominator")})
        a, b = left.get("score"), right.get("score")
        if a is not None and b is not None:
            max_score_error = max(max_score_error, abs(float(a) - float(b)))
        elif a != b:
            status_mismatches.append({"key": record_key, "original_score": a, "replay_score": b})
        if left["method"] == "tmae_score_a":
            lc = left.get("score_a_components", {}).get("quadrant_folds", [])
            rc = right.get("score_a_components", {}).get("quadrant_folds", [])
            for lfold, rfold in zip(lc, rc):
                for field, metric in (("target_values", "target"), ("predicted_values", "prediction")):
                    lv, rv = lfold.get(field, []), rfold.get(field, [])
                    for x, y in zip(lv, rv):
                        if x is not None and y is not None:
                            error = abs(float(x) - float(y))
                            if metric == "target":
                                max_target_error = max(max_target_error, error)
                            else:
                                max_prediction_error = max(max_prediction_error, error)

    passed = not (missing or extra or status_mismatches or denominator_mismatches) and max_score_error <= 1e-12
    report = {
        "schema_version": 1,
        "status": "PASS_DETERMINISTIC_INFERENCE_REPLAY" if passed else "FAIL_INFERENCE_REPLAY_MISMATCH",
        "scope": "REPLAYED_SOURCE_BOUND_INFERENCE_FROM_RETAINED_CHECKPOINTS; NOT_INDEPENDENT_MODEL_OR_SCIENTIFIC_VALIDATION",
        "analysis": str(analysis.relative_to(ROOT)),
        "replay": str(replay.relative_to(ROOT)),
        "source_manifest_sha256": panel.manifest_sha256,
        "original_scores_sha256": sha(original_path),
        "replayed_scores_sha256": sha(replay_path),
        "original_rows": len(original),
        "replayed_rows": len(repeated),
        "matched_rows": len(set(original_map) & set(replay_map)),
        "missing_keys": missing,
        "extra_keys": extra,
        "status_mismatches": status_mismatches,
        "denominator_mismatches": denominator_mismatches,
        "max_absolute_score_error": max_score_error,
        "max_absolute_score_a_target_error": max_target_error,
        "max_absolute_score_a_prediction_error": max_prediction_error,
        "seed_count": result.get("seed_count"),
        "analysis_status": result.get("status"),
        "limitations": [
            "Replay invokes the same inference implementation and is evidence of reproducibility, not an independent algorithm implementation.",
            "Source pixels and crop/lake geometry are not independently validated lake polygons.",
            "No event/control population inference, calibrated warning skill, or publication authorization is established.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
