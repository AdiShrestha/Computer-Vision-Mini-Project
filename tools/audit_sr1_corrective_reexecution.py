#!/usr/bin/env python3
"""Independent evaluation-loop replay for the corrected SR-1 FIT/VAL refits.

No training is repeated and no EVAL rows are scored or summarized. The replay
checks every frozen target-bearing VAL block against each retained checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source" / "runners"))
import analyze_sr1_true_pipeline as analysis  # noqa: E402
from replay_sr1_sensitivity_checkpoints import (  # noqa: E402
    load_checkpoint,
    prepare_rows,
    roles_and_token_rows,
    validation_loss,
)

BASE = Path("data/derived/sr1_multiyear/attempt_006")
RUN = BASE / "reexecution_001"
MANIFEST = BASE / "observations.sr1.json"
PRIMARY_REPORT = BASE / "analysis_001/analysis_report.json"
OUT = Path("project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_REEXECUTION_AUDIT_001.json")
TOL = 1e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / MANIFEST).read_text())
    run = json.loads((ROOT / RUN / "run_manifest.json").read_text())
    report = json.loads((ROOT / PRIMARY_REPORT).read_text())
    if run.get("status") != "PASS_COMPLETE_VAL" or run.get("jobs_expected") != 50:
        raise ValueError("corrected 50-fit execution is not complete and clean")
    if run.get("eval_ids_accessed") is not False:
        raise ValueError("execution manifest does not exclude EVAL")
    rows, _ = analysis.load_manifest(ROOT / MANIFEST)
    fit_ids, val_ids, usable = roles_and_token_rows(report, rows)
    results = []
    seen = set()
    for job in run["jobs"]:
        label = f"{job['family']}__{job['cell']}__seed_{job['seed']}"
        key = (job['family'], job['cell'], int(job['seed']))
        if key in seen:
            raise ValueError(f"duplicate job key {key}")
        seen.add(key)
        if job.get("status") != "PASS_COMPLETE_VAL":
            raise ValueError(f"retained fit is not clean: {label}")
        checkpoint_path = ROOT / job["checkpoint"]
        if sha(checkpoint_path) != job["checkpoint_sha256"]:
            raise ValueError(f"checkpoint digest mismatch: {label}")
        scaler_path = ROOT / job["scaler"]
        if sha(scaler_path) != job["scaler_sha256"]:
            raise ValueError(f"FIT scaler digest mismatch: {label}")
        payload = __import__('torch').load(checkpoint_path, map_location='cpu', weights_only=True)
        if payload.get("fit_observation_ids") != sorted(fit_ids):
            raise ValueError(f"FIT membership mismatch: {label}")
        if payload.get("validation_observation_ids") != sorted(val_ids):
            raise ValueError(f"VAL membership mismatch: {label}")
        if payload.get("parent_protocol_sha256") != run["parent_protocol_sha256"]:
            raise ValueError(f"protocol binding mismatch: {label}")
        if payload.get("trainer_sha256") != run["trainer_sha256"]:
            raise ValueError(f"trainer binding mismatch: {label}")
        model = load_checkpoint(Path(job["checkpoint"]), job["checkpoint_sha256"],
                                int(job["width"]), int(job["channels"]))
        val_rows = prepare_rows(usable, val_ids, Path(job["scaler"]),
                                int(job["channels"]), bool(job["zones"]))
        recomputed, finite_blocks, targets, nonfinite, no_target = validation_loss(
            model, val_rows, int(job["context"]), device="cpu")
        history = payload["history"]
        recorded = float(history["best_val_loss"])
        difference = None if recomputed is None else float(recomputed - recorded)
        expected_unsupported = history.get("validation_blocks_without_targets", [])
        actual_unsupported = [item["observation_ids"] for item in no_target]
        if actual_unsupported != expected_unsupported:
            raise ValueError(f"no-target validation block record mismatch: {label}")
        if nonfinite:
            status = "FAIL_NONFINITE_VAL_BLOCK"
        elif difference is None or abs(difference) > TOL:
            status = "FAIL_VAL_METRIC_MISMATCH"
        elif finite_blocks != history.get("validation_blocks_scored"):
            status = "FAIL_VAL_BLOCK_COUNT_MISMATCH"
        else:
            status = "PASS_FINITE_TARGET_BEARING_VAL; NO_TARGET_BLOCKS_EXPLICIT"
        results.append({"job": label, "family": job["family"], "cell": job["cell"],
                        "seed": int(job["seed"]), "channels": int(job["channels"]),
                        "zones_enabled": bool(job["zones"]), "context_scenes_max": int(job["context"]),
                        "checkpoint": job["checkpoint"], "checkpoint_sha256": job["checkpoint_sha256"],
                        "target_bearing_val_blocks": finite_blocks,
                        "no_target_val_blocks": actual_unsupported,
                        "nonfinite_val_blocks": nonfinite,
                        "valid_channel_targets": targets,
                        "recorded_best_val_masked_mse": recorded,
                        "replayed_mean_val_masked_mse": recomputed,
                        "difference": difference, "tolerance": TOL, "status": status})
        print(f"{len(results)}/50 {label} {status} Δ={difference}", flush=True)

    all_ok = len(results) == 50 and all(x["status"].startswith("PASS_") for x in results)
    output = {
        "schema_version": 1,
        "status": "PASS_ALL_50_CHECKPOINT_VAL_REPLAYS" if all_ok else "FAILURES_RETAINED_REVIEW_REQUIRED",
        "scope": {"unique_checkpoints": len(results), "eval_rows_scored": 0,
                  "training_reproduced": False, "independent_model_implementation": False,
                  "description": "Separate validation evaluation loop replay; checks complete finite losses on every target-bearing VAL block. No-target blocks remain explicit and unscored."},
        "inputs": {str(path): sha(ROOT / path) for path in (MANIFEST, PRIMARY_REPORT, RUN / "run_manifest.json")},
        "trainer_sha256": run["trainer_sha256"],
        "runner_sha256": run["script_sha256"],
        "results": results,
    }
    out = ROOT / OUT
    out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"path": OUT.as_posix(), "sha256": sha(out), "status": output["status"]}, indent=2))
    return 0 if all_ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
