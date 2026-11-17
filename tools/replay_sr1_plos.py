#!/usr/bin/env python3
"""Independent replay of the corrected SR-1 scores from retained checkpoints.

This tool does not import ``sentinel_gl.sr1_scoring`` or the analysis runner.
It rebuilds the eight-scene contexts, the checkerboard folds and Score A from
the stored standardized targets, re-runs every primary checkpoint and compares

* the cross-mask predictions with the stored replay arrays (exact match), and
* Score A (zone 0+1 and zone 2) with predictions.json (|difference| <= 1e-9).

It writes replay_report.json next to the scores and exits non-zero on any
mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.sr1_model import SR1TransformerMAE  # noqa: E402  (architecture only)

OUT = ROOT / "data/derived/sr1_multiyear/attempt_007/plos_001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_features(dt: datetime, prev: datetime | None) -> np.ndarray:
    import math
    doy = dt.timetuple().tm_yday
    angle = 2 * math.pi * (doy - 1) / 365.25
    gap = 0.0 if prev is None else max(0.0, (dt - prev).total_seconds() / 86400)
    return np.array([math.sin(angle), math.cos(angle), min(gap / 30, 5.0)], dtype=np.float32)


def main() -> int:
    torch.set_num_threads(8)
    run = json.loads((OUT / "training/run_manifest.json").read_text())
    preds = json.loads((OUT / "scores/predictions.json").read_text())
    stored_scores = {(r["seed"], r["observation_id"]): r for r in preds}
    acquired = {r["observation_id"]: r["acquired_at"] for r in preds}
    folds = np.array([((p // 10) + (p % 10)) % 2 for p in range(100)])
    report = {"tool": "tools/replay_sr1_plos.py", "tool_sha256": sha(Path(__file__)), "seeds": [], "status": "PASS"}
    for entry in sorted((e for e in run["jobs"] if e["family"] == "primary"), key=lambda e: e["seed"]):
        seed = entry["seed"]
        cp = ROOT / entry["checkpoint"]
        if sha(cp) != entry["checkpoint_sha256"]:
            raise SystemExit("checkpoint hash mismatch")
        rp = OUT / "scores/replay" / f"seed_{seed}.npz"
        with np.load(rp, allow_pickle=False) as z:
            ids = z["observation_ids"].tolist()
            target, cv, zones = z["target"], z["channel_valid"], z["zones"]
            stored_pred = z["pred_model"]
        model = SR1TransformerMAE(d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=1, num_patches=100)
        model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["state_dict"])
        model.eval()
        by_site: dict[str, list[int]] = {}
        for i, oid in enumerate(ids):
            by_site.setdefault(oid.split("__")[0], []).append(i)
        max_pred_diff, max_a_diff, n_checked = 0.0, 0.0, 0
        for site, idxs in by_site.items():
            idxs = sorted(idxs, key=lambda i: acquired[ids[i]])
            dts = [datetime.fromisoformat(acquired[ids[i]].replace("Z", "+00:00")) for i in idxs]
            tf = [time_features(dts[k], dts[k - 1] if k else None) for k in range(len(idxs))]
            for k, i in enumerate(idxs):
                ctx = list(range(max(0, k - 7), k + 1))
                gi = [idxs[c] for c in ctx]
                x = torch.tensor(np.nan_to_num(target[gi]), dtype=torch.float32).unsqueeze(0)
                obs_np = (cv[gi] & np.isfinite(target[gi])).any(-1)
                o = torch.tensor(obs_np).unsqueeze(0)
                zz = torch.tensor(zones[gi].astype(np.int64)).unsqueeze(0)
                tt = torch.tensor(np.stack([tf[c] for c in ctx])).unsqueeze(0)
                pred = np.full((100, target.shape[2]), np.nan)
                for f in (0, 1):
                    sel = obs_np[-1] & (folds == f)
                    if not sel.any():
                        continue
                    m = torch.zeros_like(o)
                    m[0, -1, torch.tensor(np.flatnonzero(sel))] = True
                    with torch.no_grad():
                        p, _ = model(x, o, m, zz, tt, modality_id=0)
                    pred[sel] = p[0, -1].numpy()[sel]
                both = np.isfinite(pred) | np.isfinite(stored_pred[i])
                if both.any():
                    diff = np.nanmax(np.abs(pred[both] - stored_pred[i][both]))
                    if not np.isfinite(diff) or np.isnan(pred[both]).any() != np.isnan(stored_pred[i][both]).any():
                        diff = np.inf
                    max_pred_diff = max(max_pred_diff, float(diff))
                errs = {}
                for group, zsel in (("score_a_zone0_1", np.isin(zones[i], [0, 1])), ("score_a_zone2", zones[i] == 2)):
                    vals = []
                    for p_ in np.flatnonzero(zsel & obs_np[-1]):
                        ch = cv[i, p_] & np.isfinite(target[i, p_])
                        if ch.any():
                            vals.append(float(np.mean((pred[p_, ch] - target[i, p_, ch]) ** 2)))
                    errs[group] = float(np.mean(vals)) if vals else None
                stored = stored_scores[(seed, ids[i])]
                for key, value in errs.items():
                    if (value is None) != (stored[key] is None):
                        max_a_diff = np.inf
                    elif value is not None:
                        max_a_diff = max(max_a_diff, abs(value - stored[key]))
                n_checked += 1
        ok = max_pred_diff == 0.0 and max_a_diff <= 1e-9
        report["seeds"].append({"seed": seed, "scenes": n_checked, "max_abs_prediction_difference": max_pred_diff,
                                "max_abs_score_a_difference": max_a_diff, "status": "PASS" if ok else "FAIL"})
        if not ok:
            report["status"] = "FAIL"
        print(json.dumps(report["seeds"][-1]), flush=True)
    (OUT / "scores/replay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
