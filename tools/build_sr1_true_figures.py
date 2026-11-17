#!/usr/bin/env python3
"""Build truthful figures from the immutable SR-1 multi-year execution outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITES = ["PILOT-SOUTH-LHONAK", "PILOT-KHANGCHUNG-TSHO"]
LABELS = {SITES[0]: "South Lhonak", SITES[1]: "Khangchung Tsho"}


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="attempt_004")
    ap.add_argument("--analysis", default="analysis_001")
    ap.add_argument("--output", type=Path, default=Path("docs/figures/sr1_true"))
    args = ap.parse_args()
    manifest_path = ROOT / "data/derived/sr1_multiyear" / args.attempt / "observations.sr1.json"
    analysis_dir = manifest_path.parent / args.analysis
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((analysis_dir / "analysis_report.json").read_text())
    predictions = json.loads((analysis_dir / "predictions.json").read_text())
    histories = json.loads((analysis_dir / "training_histories.json").read_text())
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})

    # Figure 1: retained observations and actual per-year support.
    counts = Counter((row["site_id"], row["acquired_at"][:4]) for row in manifest["observations"]
                     if row.get("origin") == "observational" and row.get("status") != "EXPORT_FAILED")
    years = [str(y) for y in range(2018, 2024)]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x = np.arange(len(years)); width = 0.36
    for k, site in enumerate(SITES):
        values = [counts[(site, year)] for year in years]
        ax.bar(x + (k - 0.5) * width, values, width, label=LABELS[site])
        for xpos, value in zip(x + (k - 0.5) * width, values):
            if value:
                ax.text(xpos, value, str(value), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, years); ax.set_ylabel("Retained optical acquisitions")
    ax.set_xlabel("Acquisition year (UTC)"); ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Source-bound observations; empty years remain visible")
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    save(fig, output / "fig1_observation_cadence")

    # Figure 2: actual FIT-only spectral candidate zones; these are not surveyed shorelines.
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.0))
    palette = matplotlib.colors.ListedColormap(["#2678b2", "#f2c14e", "#d8d8d8"])
    for ax, site in zip(axes, SITES):
        zone_path = ROOT / report["zones"][site]["path"]
        with np.load(zone_path, allow_pickle=False) as z:
            zones = z["zones"]
        ax.imshow(zones, cmap=palette, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(LABELS[site]); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("FIT-only spectral candidate zones (not independent geomorphic validation)", y=.96)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=8, label=label)
               for color, label in zip(["#2678b2", "#f2c14e", "#d8d8d8"],
                                       ["Core-water candidate", "≤500 m candidate buffer", "Background candidate"])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=8, columnspacing=1.0, handletextpad=.3)
    fig.subplots_adjust(left=.04, right=.98, top=.86, bottom=.20, wspace=.04)
    save(fig, output / "fig2_fit_only_candidate_zones")

    # Figure 3: measured native reflectance distribution; no clamping or fabricated values.
    sample = {"B03": [], "B08": [], "B11": []}
    for row in manifest["observations"]:
        if row.get("status") == "EXPORT_FAILED":
            continue
        with np.load(ROOT / row["values_file"], allow_pickle=False) as data:
            values, valid = data["reflectance"], data["valid"]
        for index, channel in enumerate(sample):
            sample[channel].append(values[index][valid[index] & np.isfinite(values[index])])
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0), constrained_layout=True)
    colors = ["#2678b2", "#3a923a", "#d9822b"]
    for ax, (channel, arrays), color in zip(axes, sample.items(), colors):
        values = np.concatenate(arrays) if arrays else np.asarray([])
        if values.size:
            ax.hist(values, bins=60, color=color, alpha=.82)
        ax.set_title(channel); ax.set_xlabel("Native BOA reflectance"); ax.set_ylabel("Pixel count")
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("All retained valid pixels; negative reflectance is preserved")
    save(fig, output / "fig3_reflectance_distributions")

    # Figure 4: retained per-seed validation traces.
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    for item in histories:
        history = item["history"]
        ax.plot(np.arange(1, len(history["val_loss_history"]) + 1), history["val_loss_history"],
                marker=".", markersize=3, linewidth=1, label=f"seed {history['seed']}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("VAL masked MSE")
    ax.set_title("Validation-selected training histories")
    ax.legend(frameon=False, ncol=5); ax.grid(alpha=.2)
    save(fig, output / "fig4_validation_loss_by_seed")

    # Figure 5: descriptive per-scene scores by frozen role; no performance claim is implied.
    fields = [("score_a_zone0_1", "Score A (Zone 0+1)"), ("score_b_zone0_1", "Score B (Zone 0+1)"),
              ("score_c", "Score C")]
    roles = ["FIT", "VAL", "EVENT_UNCERTAIN", "EVAL"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), constrained_layout=True)
    for ax, (field, label) in zip(axes, fields):
        for ri, role in enumerate(roles):
            vals = [p[field] for p in predictions if p["role"] == role and p[field] is not None and np.isfinite(p[field])]
            if vals:
                jitter = np.linspace(-.12, .12, len(vals)) if len(vals) > 1 else [0]
                ax.scatter(np.asarray(jitter) + ri, vals, s=12, alpha=.6)
        ax.set_xticks(range(len(roles)), ["FIT", "VAL", "Event\nuncertain", "EVAL"])
        ax.set_title(label); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Exploratory scores by manifest-defined role (points are scenes across seeds)")
    save(fig, output / "fig5_exploratory_scores_by_role")

    index = {"status": "GENERATED_FROM_OBSERVED_SR1_TRUE_RUN", "attempt": args.attempt,
             "analysis": args.analysis, "source_manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
             "figures": []}
    for stem in sorted(p.stem for p in output.glob("*.png")):
        index["figures"].append({"stem": stem, "png": (output / f"{stem}.png").relative_to(ROOT).as_posix(),
                                 "pdf": (output / f"{stem}.pdf").relative_to(ROOT).as_posix()})
    (output / "figure_manifest.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"status": index["status"], "figure_count": len(index["figures"]),
                      "output": output.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
