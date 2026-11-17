#!/usr/bin/env python3
"""Independent arithmetic replay for corrected SR-1 score arrays."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/derived/sr1_multiyear/attempt_006/corrected_eval_001"
REPORT = ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_EVAL_AUDIT_001.json"
PREDICTIONS = json.loads((PACKAGE / "predictions.json").read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tol=2e-6):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def main() -> int:
    by_key = {(r["seed"], r["observation_id"]): r for r in PREDICTIONS}
    if len(by_key) != 705:
        raise ValueError(f"expected 705 seed-scene score rows, got {len(by_key)}")
    checked = {"score_a_zone0_1": 0, "score_a_zone2": 0,
               "score_b_zone0_1": 0, "score_c": 0}
    roles = {}
    replay_records = []
    for seed in range(5):
        path = PACKAGE / "replay" / f"seed_{seed}_score_replay.npz"
        with np.load(path, allow_pickle=False) as d:
            ids = d["observation_ids"].tolist()
            if len(ids) != 141 or len(set(ids)) != 141:
                raise ValueError(f"seed {seed}: replay does not contain 141 unique scenes")
            roles[seed] = {role: int(np.sum(d["roles"] == role)) for role in ("FIT", "VAL", "EVAL")}
            fit_ids = set(d["fit_observation_ids"].tolist())
            if len(fit_ids) != 117:
                raise ValueError("replay FIT bank is not 117 scenes")
            records = []
            for i, oid in enumerate(ids):
                row = {"i": i, "id": oid, "site": str(d["site_ids"][i]),
                       "time": str(d["acquired_at"][i]), "role": str(d["roles"][i]),
                       "target": d["target"][i], "observed": d["observed"][i].astype(bool),
                       "zones": d["zones"][i].astype(int), "masked": d["masked"][i].astype(bool),
                       "prediction": d["prediction"][i], "projected": d["projected"][i]}
                records.append(row)
            # Recompute Score A by first deriving every eligible patch MSE.
            # Score B is independently reconstructed from saved projected
            # latents and exact FIT scene dates, using same-patch +/-45 DOY kNN.
            vals = {}
            for row in records:
                patch_err = np.full(100, np.nan)
                for p in np.flatnonzero(row["masked"]):
                    good = row["observed"][p] & np.isfinite(row["target"][p])
                    if good.any():
                        patch_err[p] = float(np.mean((row["prediction"][p, good] - row["target"][p, good]) ** 2))
                z = row["zones"]
                corebuf = np.isin(z, (0, 1)) & np.isfinite(patch_err)
                background = (z == 2) & np.isfinite(patch_err)
                a01 = float(np.mean(patch_err[corebuf])) if corebuf.any() else None
                a2 = float(np.mean(patch_err[background])) if background.any() else None
                vals[row["id"]] = {"a01": a01, "a2": a2}
            for row in records:
                doy = datetime.fromisoformat(row["time"].replace("Z", "+00:00")).timetuple().tm_yday
                refs = []
                if np.isin(row["zones"], (0, 1)).any():
                    for ref in records:
                        if ref["id"] not in fit_ids or ref["site"] != row["site"] or ref["id"] == row["id"]:
                            continue
                        rdoy = datetime.fromisoformat(ref["time"].replace("Z", "+00:00")).timetuple().tm_yday
                        gap = abs(doy - rdoy)
                        if min(gap, 366 - gap) <= 45:
                            # Same patch locations are paired independently.
                            refs.append((ref, gap))
                patch_scores = []
                for p in np.flatnonzero(np.isin(row["zones"], (0, 1))):
                    ds = []
                    for ref, _ in refs:
                        rdoy = datetime.fromisoformat(ref["time"].replace("Z", "+00:00")).timetuple().tm_yday
                        gap = abs(doy - rdoy)
                        if min(gap, 366 - gap) <= 45:
                            ds.append(float(np.linalg.norm(row["projected"][p] - ref["projected"][p])))
                    if len(ds) >= 5:
                        patch_scores.append(float(np.mean(np.sort(ds)[:5])))
                b = float(np.mean(patch_scores)) if patch_scores else None
                vals[row["id"]]["b"] = b
            fit_pairs = [vals[r["id"]] for r in records if r["role"] == "FIT"
                         and vals[r["id"]]["a01"] is not None and vals[r["id"]]["a01"] > 0
                         and vals[r["id"]]["b"] is not None]
            ln_a = np.log(np.asarray([x["a01"] for x in fit_pairs], dtype=float))
            bfit = np.asarray([x["b"] for x in fit_pairs], dtype=float)
            med_a, mad_a = np.median(ln_a), np.median(np.abs(ln_a - np.median(ln_a)))
            med_b, mad_b = np.median(bfit), np.median(np.abs(bfit - np.median(bfit)))
            for row in records:
                key = (seed, row["id"])
                saved = by_key[key]
                expected = vals[row["id"]]
                for metric, value in [("score_a_zone0_1", expected["a01"]),
                                      ("score_a_zone2", expected["a2"]),
                                      ("score_b_zone0_1", expected["b"])]:
                    if not close(saved.get(metric), value):
                        raise ValueError(f"seed {seed} {row['id']} {metric} mismatch: {saved.get(metric)} != {value}")
                    checked[metric] += 1
                if expected["a01"] is not None and expected["a01"] > 0 and expected["b"] is not None and mad_a > 0 and mad_b > 0:
                    c = 0.5 * (math.log(expected["a01"]) - med_a) / (1.4826 * mad_a) + 0.5 * (expected["b"] - med_b) / (1.4826 * mad_b)
                else:
                    c = None
                if not close(saved.get("score_c"), c):
                    raise ValueError(f"seed {seed} {row['id']} Score C mismatch")
                checked["score_c"] += 1
                replay_records.append((seed, row["id"], row["role"]))
        if roles[seed] != {"FIT": 117, "VAL": 17, "EVAL": 7}:
            raise ValueError(f"seed {seed}: split membership mismatch {roles[seed]}")
    suite_path = PACKAGE / "baseline_suite.json"
    suite = json.loads(suite_path.read_text())
    model_path = ROOT / suite["model_path"]
    if sha(model_path) != suite["model_sha256"]:
        raise ValueError("saved baseline model bundle hash mismatch")
    if suite["thresholds"].get("val_q90_descriptive_only") is None:
        raise ValueError("descriptive VAL threshold record missing")
    for est in ("E1", "E2", "E3", "E5"):
        if json.loads((PACKAGE / "analysis_report.json").read_text())["estimands"][est]["value"] is not None:
            raise ValueError(f"unsupported {est} must remain null")
    result = {"schema_version": 1, "status": "PASS_ARITHMETIC_REPLAY_WITH_INFERENCE_GATES_OPEN",
        "protocol_freeze_sha256": sha(ROOT / "project/remediation/2026-10-26/sr1_true_implementation_001/CORRECTED_EVAL_PROTOCOL_FREEZE_001.json"),
        "analysis_report_sha256": sha(PACKAGE / "analysis_report.json"),
        "predictions_sha256": sha(PACKAGE / "predictions.json"),
        "baseline_suite_sha256": sha(suite_path), "baseline_model_sha256": sha(model_path),
        "replay_sha256": {f"seed_{seed}": sha(PACKAGE / "replay" / f"seed_{seed}_score_replay.npz") for seed in range(5)},
        "checked_score_values": checked, "score_rows": len(by_key), "roles_per_seed": roles,
        "baseline_scope": "The model artifact and declared FIT-only fit scope are hash-checked; baseline score implementations are not independently rerun by this arithmetic replay.",
        "estimands": {"E1": "NOT_ESTIMABLE", "E2": "NOT_ESTIMABLE", "E3": "NOT_ESTIMABLE", "E4": "CONDITIONAL_DESCRIPTIVE", "E5": "NOT_ESTIMABLE"},
        "limitations": ["Replay independently verifies stored primary A/B/C arithmetic from NPZ arrays and checkpoint outputs.",
                        "It does not independently retrain models or refit/replay classical baseline estimators.",
                        "EVAL scene count is seven; no EVAL-defined cut-point or inferential claim is supported."]}
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checked": checked, "report": REPORT.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
