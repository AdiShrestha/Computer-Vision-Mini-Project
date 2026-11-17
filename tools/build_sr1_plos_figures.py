#!/usr/bin/env python3
"""Draw the five manuscript figures from the result packet and retained analysis rows.

Output: docs/figures/plos/Fig{1..5}.tif (LZW, 300 dpi, 7.4 in wide = 2,220 px,
as PLOS ONE requires) plus PNG previews and figure_manifest.json with input
and output SHA-256 digests. No value is drawn that is not in the inputs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sentinel_gl.sr1_corpus import SITE_LABELS, SITES, load_manifest  # noqa: E402

ATT = ROOT / "data/derived/sr1_multiyear/attempt_007"
OUT = ATT / "plos_001"
PACKET = ROOT / "docs/results/sr1_plos_packet.json"
FIGDIR = ROOT / "docs/figures/plos"
PARENT_MANIFEST = ROOT / "data/derived/sr1_multiyear/attempt_006/observations.sr1.json"
WIDTH_IN = 7.4
DPI = 300
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated categorical slots 1-6
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"
ROLE_COLOR = {"FIT": C[0], "VAL": C[1], "EVAL": C[2]}
EVENT = datetime(2023, 10, 3, 18, 30)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "lines.linewidth": 1.2, "savefig.dpi": DPI})


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, name, manifest):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / f"{name}.png"
    fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    img = Image.open(png).convert("RGB")
    if img.width > 2250:
        raise ValueError(f"{name} is {img.width} px wide; PLOS allows at most 2,250 px")
    tif = FIGDIR / f"{name}.tif"
    img.save(tif, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    manifest["figures"][name] = {"tif": tif.relative_to(ROOT).as_posix(), "tif_sha256": sha(tif),
                                 "png": png.relative_to(ROOT).as_posix(), "png_sha256": sha(png),
                                 "pixels": [img.width, img.height], "dpi": DPI}


def scene_medians(preds, field):
    grouped = defaultdict(list)
    for r in preds:
        v = r
        for k in field.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None and np.isfinite(v):
            grouped[r["observation_id"]].append(float(v))
    return {k: (float(np.median(v)), float(min(v)), float(max(v))) for k, v in grouped.items()}


def fig1(records, packet, manifest):
    fig = plt.figure(figsize=(WIDTH_IN, 6.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 1.0], hspace=0.55, wspace=0.18)
    parent = json.loads(PARENT_MANIFEST.read_text())
    for col, site in enumerate(SITES):
        ax = fig.add_subplot(gs[0, col])
        rows = [r for r in records if r["site_id"] == site and r["role"] == "FIT"]
        stack = []
        for r in rows:
            good = np.isin(r["scl"], [4, 5, 6]) & r["valid"].all(axis=0)
            v = np.where(good, r["reflectance"], np.nan)
            stack.append(v)
        comp = np.nanmedian(np.stack(stack), axis=0)  # (3, H, W): B03, B08, B11
        rgb = np.stack([comp[2], comp[1], comp[0]], axis=-1)  # SWIR, NIR, green false colour
        lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
        rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
        rgb = np.where(np.isfinite(rgb), rgb, 1.0)
        t = rows[0]["grid"]["transform"]
        left, top = t[2], t[5]
        extent = [left / 1000, (left + 3200) / 1000, (top - 3200) / 1000, top / 1000]
        ax.imshow(rgb, extent=extent, interpolation="nearest")
        with np.load(OUT / "zones" / f"{site}_inventory_zones.npz") as z:
            lake, zones = z["lake"], z["zones"]
        xs = extent[0] + (np.arange(160) + 0.5) * 0.02  # pixel centres (km)
        ys = extent[3] - (np.arange(160) + 0.5) * 0.02
        ax.contour(xs, ys, lake.astype(float), levels=[0.5], colors=[C[0]], linewidths=1.4)
        ax.contour(xs, ys, (zones <= 1).astype(float), levels=[0.5], colors=[C[3]], linewidths=1.0, linestyles="--")
        for k in range(1, 10):
            ax.axvline(extent[0] + k * 0.32, color="white", lw=0.3, alpha=0.5)
            ax.axhline(extent[2] + k * 0.32, color="white", lw=0.3, alpha=0.5)
        handles = [Line2D([], [], color=C[0], lw=1.4, label="Hi-MAG 2017 outline (zone 0)"),
                   Line2D([], [], color=C[3], lw=1.0, ls="--", label="500 m buffer (zone 1)"),
                   Patch(facecolor="white", edgecolor=MUTED, lw=0.5, label="No clear FIT pixel")]
        if site == SITES[0]:
            old = next(r["grid"]["transform"] for r in parent["observations"] if r.get("site_id") == site and "grid" in r)
            ax.add_patch(Rectangle((old[2] / 1000, (old[5] - 3200) / 1000), 3.2, 3.2, fill=False, ec=C[1], lw=1.2, ls=":"))
            handles.append(Line2D([], [], color=C[1], lw=1.2, ls=":", label="Earlier window (attempt 006)"))
            ax.set_xlim(min(extent[0], old[2] / 1000) - 0.05, max(extent[1], old[2] / 1000 + 3.2) + 0.05)
            ax.set_ylim(min(extent[2], (old[5] - 3200) / 1000) - 0.05, max(extent[3], old[5] / 1000) + 0.05)
        sx, sy = ax.get_xlim()[0] + 0.15, ax.get_ylim()[0] + 0.15
        ax.plot([sx, sx + 1.0], [sy, sy], color="white", lw=3, solid_capstyle="butt")
        ax.plot([sx, sx + 1.0], [sy, sy], color=INK, lw=1.5, solid_capstyle="butt")
        ax.text(sx + 0.5, sy + 0.1, "1 km", ha="center", va="bottom", fontsize=6.5,
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
        nx, ny = ax.get_xlim()[1] - 0.2, ax.get_ylim()[1] - 0.55
        ax.annotate("N", xy=(nx, ny + 0.4), xytext=(nx, ny), ha="center", fontsize=7, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0), bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
        ax.set_xlabel("Easting (km, UTM 45N)")
        if col == 0:
            ax.set_ylabel("Northing (km)")
        ax.set_title(SITE_LABELS[site], loc="left")
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.8)
        ax.tick_params(axis="both", labelsize=6)
        panel(ax, "AB"[col])
    for row, site in enumerate(SITES, start=1):
        ax = fig.add_subplot(gs[row, :])
        rows = sorted([r for r in records if r["site_id"] == site], key=lambda r: r["dt"])
        for role in ("FIT", "VAL", "EVAL"):
            rr = [r for r in rows if r["role"] == role]
            frac = [(np.isin(r["scl"], [4, 5, 6]) & r["valid"].all(axis=0)).mean() for r in rr]
            ax.scatter([r["dt"] for r in rr], frac, s=10, color=ROLE_COLOR[role], edgecolor="white", linewidth=0.4,
                       label=role, zorder=3)
        ax.axvline(EVENT, color=INK, lw=0.8, ls="--")
        ax.text(EVENT, 1.05, " 3-4 Oct 2023 outburst", fontsize=6, va="bottom", ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Clear fraction")
        ax.set_title(f"{SITE_LABELS[site]}: acquisitions by split ({len(rows)} scenes)", loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", color=GRID, lw=0.5)
        if row == 1:
            ax.legend(loc="lower right", ncol=3, frameon=True, edgecolor="none", framealpha=1.0,
                      bbox_to_anchor=(1.0, 1.02), borderaxespad=0.0)
        panel(ax, "CD"[row - 1])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07)
    save(fig, "Fig1", manifest)


def fig2(packet, manifest):
    tr = packet["training"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.95})
    ax = axes[0]
    prim = tr["primary_by_seed"]
    ax.scatter([p["seed"] for p in prim], [p["best_val_mse"] for p in prim], color=C[0], s=18, zorder=3,
               label="T-MAE, 150-epoch budget")
    ext = [p for p in prim if p["extended_budget"]]
    if ext:
        ax.scatter([p["seed"] for p in ext], [p["extended_budget"]["best_val_mse"] for p in ext], marker="D",
                   facecolor="white", edgecolor=C[0], s=16, zorder=3, label="Same seed, 450-epoch budget")
    ref = tr["validation_reference_mse_by_channels"]["5"]
    ax.axhline(ref["patch_seasonal_mean"], color=C[1], ls="--", lw=1.0, label="Per-patch seasonal FIT mean")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Selected VAL MSE (standardized units)")
    ax.set_xticks(range(len(prim)))
    ax.set_ylim(0, 0.75)
    ax.set_title("Primary configuration, 10 seeds", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.8, ncol=1)
    panel(ax, "A")
    ax = axes[1]
    cells = [c for c in tr["cells"] if c["family"] != "extended_budget"]
    order = sorted(cells, key=lambda c: (c["channels"], c["family"] != "primary", c["cell"]))
    labels, y = [], np.arange(len(order))
    for i, c in enumerate(order):
        col = C[0] if c["channels"] == 5 else C[2]
        sd = c["sample_sd_val_mse"] or 0.0
        ax.errorbar(c["mean_val_mse"], i, xerr=sd, fmt="o", color=col, ms=4, capsize=2, lw=1)
        zone = "inv. zones" if c["zones"] else "const. zone"
        name = f"5 ch, d={c['width']}" if c["family"] == "capacity" else f"{c['channels']} ch, {zone}, ctx {c['context']}"
        if c["family"] == "primary":
            name += " (primary)"
        labels.append(f"{name} (n={c['completed']})")
    for ch, col in ((5, C[0]), (3, C[2])):
        ax.axvline(tr["validation_reference_mse_by_channels"][str(ch)]["patch_seasonal_mean"], color=col, ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Selected VAL MSE, mean ± sample SD over seeds")
    ax.set_title("Capacity and factorial cells", loc="left")
    ax.legend(handles=[Line2D([], [], color=C[0], marker="o", lw=0, label="5 channels"),
                       Line2D([], [], color=C[2], marker="o", lw=0, label="3 channels"),
                       Line2D([], [], color=MUTED, ls="--", label="Seasonal mean, same targets")],
              frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=3)
    panel(ax, "B")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.3)
    save(fig, "Fig2", manifest)


PRED_LABELS = {"model": "T-MAE (cross-mask)", "model_visible_inputs_zeroed": "T-MAE, visible inputs set to 0",
               "model_all_context_hidden": "T-MAE, all context hidden", "fit_median": "FIT median (zero)",
               "patch_mean": "Per-patch FIT mean", "patch_seasonal_mean": "Per-patch seasonal FIT mean",
               "patch_seasonal_mean_plus_visible_offset": "Seasonal mean + visible offset",
               "calendar_site_mean": "Calendar-only site mean"}


def fig3(packet, manifest):
    rc = packet["reconstruction"]
    names = ["model", "model_visible_inputs_zeroed", "model_all_context_hidden", "patch_seasonal_mean_plus_visible_offset",
             "patch_seasonal_mean", "patch_mean", "calendar_site_mean", "fit_median"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.4), gridspec_kw={"wspace": 0.75, "width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    for k, (role, marker) in enumerate((("FIT", "o"), ("VAL", "s"), ("EVAL", "D"))):
        vals = [rc["by_role"][role]["pooled_mse"][n] for n in names]
        ax.scatter(vals, y + (k - 1) * 0.22, color=ROLE_COLOR[role], marker=marker, s=14, zorder=3,
                   label=f"{role} ({rc['by_role'][role]['scenes']} scenes)")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels([PRED_LABELS[n] for n in names], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Pooled MSE on hidden entries (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=3)
    ax.set_title("Reconstruction error by predictor", loc="left")
    panel(ax, "A")
    ax = axes[1]
    others = names[1:6]  # FIT-median and calendar-only differences are far larger; see panel A and Table 3
    for k, (role, marker) in enumerate((("VAL", "s"), ("EVAL", "D"))):
        for i, n in enumerate(others):
            d = rc["differences"][role][n]
            yy = i + (k - 0.5) * 0.3
            if d["status"] == "ESTIMATED":
                ax.errorbar(d["estimate"], yy, xerr=[[d["estimate"] - d["lower"]], [d["upper"] - d["estimate"]]],
                            fmt=marker, color=ROLE_COLOR[role], ms=3.5, capsize=2, lw=1)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([PRED_LABELS[n] for n in others], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("T-MAE minus comparator MSE (95% CI)")
    ax.legend(handles=[Line2D([], [], color=ROLE_COLOR[r], marker=m, lw=0, label=r) for r, m in (("VAL", "s"), ("EVAL", "D"))],
              frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Difference (negative: T-MAE lower)", loc="left")
    panel(ax, "B")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.25)
    save(fig, "Fig3", manifest)


def fig4(preds, manifest):
    fields = [("score_a_zone0_1", "Score A, zone 0+1", "log"), ("score_c", "Score C (robust z)", "linear"),
              ("page_cusum.statistic", "CUSUM on Score C", "linear"),
              ("water_candidate_area_km2", "Candidate water area (km²)", "linear")]
    fig, axes = plt.subplots(len(fields), 2, figsize=(WIDTH_IN, 7.2), sharex=True)
    meta = {r["observation_id"]: r for r in preds if r["seed"] == 0}
    for row, (field, label, scale) in enumerate(fields):
        med = scene_medians(preds, field)
        for col, site in enumerate(SITES):
            ax = axes[row, col]
            ids = sorted((oid for oid in med if meta[oid]["site_id"] == site), key=lambda o: meta[o]["acquired_at"])
            dts = [datetime.fromisoformat(meta[o]["acquired_at"].replace("Z", "+00:00")).replace(tzinfo=None) for o in ids]
            vals = np.array([med[o][0] for o in ids])
            lo = np.array([med[o][1] for o in ids])
            hi = np.array([med[o][2] for o in ids])
            seg = [0]
            for k in range(1, len(dts)):
                if (dts[k] - dts[k - 1]) > timedelta(days=60):
                    seg.append(k)
            seg.append(len(dts))
            for a, b in zip(seg[:-1], seg[1:]):
                ax.plot(dts[a:b], vals[a:b], color=MUTED, lw=0.6, zorder=1)
            for role in ("FIT", "VAL", "EVAL"):
                sel = [k for k, o in enumerate(ids) if meta[o]["role"] == role]
                if not sel:
                    continue
                ax.vlines([dts[k] for k in sel], lo[sel], hi[sel], color=ROLE_COLOR[role], lw=0.6, alpha=0.6, zorder=2)
                ax.scatter([dts[k] for k in sel], vals[sel], s=7, color=ROLE_COLOR[role], zorder=3,
                           label=role if (row == 0 and col == 0) else None)
            ax.axvline(EVENT, color=INK, lw=0.7, ls="--")
            if field == "page_cusum.statistic":
                ax.axhline(4.0, color=C[3], lw=0.9, ls="--")
                ax.text(dts[0], 4.0, " h = 4", va="bottom", fontsize=6)
            ax.set_yscale(scale)
            ax.grid(axis="y", color=GRID, lw=0.5)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(SITE_LABELS[site], loc="left")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            panel(ax, "ABCDEFGH"[row * 2 + col])
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left", fontsize=6)
    fig.text(0.5, 0.005, "Points: median over 10 seeds; vertical bars: seed range; lines break at gaps > 60 days; "
             "dashed vertical line: 3-4 October 2023 outburst", ha="center", fontsize=6, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.05, hspace=0.3, wspace=0.18)
    save(fig, "Fig4", manifest)


PC_LABELS = {"score_a_zone0_1": "Score A", "score_b_zone0_1": "Score B", "score_c": "Score C",
             "water_candidate_area_km2": "Candidate water area (decrease)", "seasonal_climatology_residual": "Seasonal residual",
             "spatial_persistence_residual": "Temporal persistence", "isolation_forest_score": "Isolation Forest",
             "one_class_svm_score": "One-class SVM", "missingness_only_score": "Missingness only"}


def fig5(packet, manifest):
    pc = packet["positive_control"]
    methods = list(pc["methods"])
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH_IN, 5.6), sharex=True, sharey=True)
    for k, m in enumerate(methods):
        ax = axes.flat[k]
        fr = [float(f) for f in pc["methods"][m]["by_fraction"]]
        d = pc["methods"][m]["by_fraction"]
        pert = [d[str(f)]["perturbed_detection_rate"] for f in fr]
        clean = [d[str(f)]["clean_detection_rate"] for f in fr]
        ax.plot(fr, pert, marker="o", ms=3.5, color=C[0], label="Perturbed scenes")
        ax.plot(fr, clean, marker="s", ms=3, color=MUTED, ls="--", label="Same scenes, unperturbed")
        ns = [d[str(f)]["scenes"] for f in fr]
        ax.set_title(f"{PC_LABELS[m]} (n={min(ns)}–{max(ns)})", loc="left", fontsize=7)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color=GRID, lw=0.5)
        if k % 3 == 0:
            ax.set_ylabel("Share above FIT threshold")
        if k >= 6:
            ax.set_xlabel("Fraction of lake pixels replaced")
        ax.set_xticks(fr)
        ax.set_xticklabels([f"{f:g}" for f in fr])
        panel(ax, "ABCDEFGHI"[k])
    axes.flat[0].legend(frameon=False, fontsize=5.8, loc="center right")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08, hspace=0.42, wspace=0.12)
    save(fig, "Fig5", manifest)


def main() -> int:
    packet = json.loads(PACKET.read_text())
    preds = json.loads((OUT / "scores/predictions.json").read_text())
    records, _ = load_manifest(ATT / "observations.sr1.json", ROOT)
    manifest = {"inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                           [PACKET, OUT / "scores/predictions.json", ATT / "observations.sr1.json",
                            OUT / "zones/zones_report.json"]},
                "tool_sha256": sha(Path(__file__)), "figures": {}}
    fig1(records, packet, manifest)
    fig2(packet, manifest)
    fig3(packet, manifest)
    fig4(preds, manifest)
    fig5(packet, manifest)
    (FIGDIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["pixels"] for k, v in manifest["figures"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
