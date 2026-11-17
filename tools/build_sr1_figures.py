#!/usr/bin/env python3
"""Build publication-quality figures for Milestone SR-1 and ICT-CEEL 2027.

Outputs:
  docs/figures/sr1/fig1_observation_timeline.{pdf,png}
  docs/figures/sr1/fig2_rois_and_geomorphic_zones.{pdf,png}
  docs/figures/sr1/fig3_tmae_training_convergence.{pdf,png}
  docs/figures/sr1/fig4_remediated_anomaly_trajectories.{pdf,png}
  docs/figures/sr1/fig5_zone_contrast_and_legacy_comparison.{pdf,png}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "docs/tables/sr1"
FIGURES_DIR = ROOT / "docs/figures/sr1"
CHECKPOINTS_DIR = ROOT / "data/derived/sr1_multiyear/checkpoints"
ROIS_DIR = ROOT / "data/derived/expansion_2023_recentered/rois"
REGISTRY_FILE = ROOT / "project/sr1_registry.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SL_COLOR = "#0D47A1"  # Deep Blue (South Lhonak)
KT_COLOR = "#00796B"  # Deep Teal (Khangchung Tsho)
EVENT_COLOR = "#D32F2F"  # Red for event bracket
SAR_COLOR = "#7B1FA2"  # Purple for SAR date


def load_result_bindings() -> Dict[str, Any]:
    return json.loads((TABLES_DIR / "result_bindings.json").read_text())


def load_training_histories() -> Dict[str, Any]:
    return json.loads((CHECKPOINTS_DIR / "training_histories.json").read_text())


def build_figure_1(bindings: Dict[str, Any]):
    """Figure 2 (and fig1 alias): Observation Support and Clear Fraction Timeline with Sikkim Local Time Event Bracket."""
    obs = bindings["observations"]
    sl_obs = sorted([o for o in obs if o["site_id"] == "PILOT-SOUTH-LHONAK"], key=lambda r: r["acquired_at"])
    kt_obs = sorted([o for o in obs if o["site_id"] == "PILOT-KHANGCHUNG-TSHO"], key=lambda r: r["acquired_at"])

    def parse_dates_cf(records):
        dts = [datetime.fromisoformat(r["acquired_at"]) for r in records]
        cfs = [r["clear_fraction"] * 100.0 for r in records]
        roles = [r["role"] for r in records]
        return dts, cfs, roles

    sl_dts, sl_cfs, sl_roles = parse_dates_cf(sl_obs)
    kt_dts, kt_cfs, kt_roles = parse_dates_cf(kt_obs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, sharey=True)

    # Event onset bracket: 2023-10-02 18:30 UTC to 2023-10-04 18:30 UTC
    t_event_start = datetime(2023, 10, 2, 18, 30, tzinfo=timezone.utc)
    t_event_end = datetime(2023, 10, 4, 18, 30, tzinfo=timezone.utc)
    t_sar = datetime(2023, 9, 1, 0, 0, tzinfo=timezone.utc)

    for ax, dts, cfs, site_name, col in [(ax1, sl_dts, sl_cfs, "South Lhonak Lake", SL_COLOR),
                                        (ax2, kt_dts, kt_cfs, "Khangchung Tsho", KT_COLOR)]:
        # Shaded event window
        ax.axvspan(t_event_start, t_event_end, color=EVENT_COLOR, alpha=0.25, label="Event Bracket (2-4 Oct)")
        ax.axvline(t_event_start, color=EVENT_COLOR, linestyle="--", linewidth=1.2)
        ax.axvline(t_sar, color=SAR_COLOR, linestyle=":", linewidth=1.2, label="Sentinel-1 SAR (1 Sep)")

        # Plot clear fraction line and points
        ax.plot(dts, cfs, color=col, alpha=0.4, linestyle="-", linewidth=1.0)
        sc = ax.scatter(dts, cfs, c=cfs, cmap="viridis", vmin=0, vmax=100, s=35, edgecolors=col, linewidths=0.8, zorder=4)

        ax.set_ylabel("Clear Fraction (%)")
        ax.set_title(site_name, loc="left", fontweight="bold", color=col)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(-5, 105)

    # Format x axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.set_xlabel("Acquisition Date (2023)")
    fig.autofmt_xdate()

    # Shared Legend positioned above top plot
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99), ncol=2, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # Save both Figure 2 and legacy Figure 1 filenames
    fig.savefig(FIGURES_DIR / "fig2_observation_timeline.pdf")
    fig.savefig(FIGURES_DIR / "fig2_observation_timeline.png")
    fig.savefig(FIGURES_DIR / "fig1_observation_timeline.pdf")
    fig.savefig(FIGURES_DIR / "fig1_observation_timeline.png")
    plt.close(fig)
    print("Saved fig2_observation_timeline.{pdf,png} (and fig1 alias)")


def build_figure_2():
    """Figure 2: Glacial Lake Focused ROIs (160x160 px, 20m) and Geomorphic Zone Partition."""
    registry = json.loads(REGISTRY_FILE.read_text())
    
    # Load zone maps
    sl_zones = np.load(ROOT / registry["lakes"]["PILOT-SOUTH-LHONAK"]["geomorphic_zones"]["zone_map_file"])["zones"]
    kt_zones = np.load(ROOT / registry["lakes"]["PILOT-KHANGCHUNG-TSHO"]["geomorphic_zones"]["zone_map_file"])["zones"]

    # Load pre and post South Lhonak rasters
    # Pre-event: 2023-09-26, Post-event: 2023-10-09
    sl_pre_path = ROIS_DIR / "PILOT-SOUTH-LHONAK__S2B_T45RXL_20230926T045046_L2A.npz"
    sl_post_path = ROIS_DIR / "PILOT-SOUTH-LHONAK__S2B_T45RXL_20231009T045202_L2A.npz"

    r_sl, c_sl = slice(180, 340), slice(125, 285)
    with np.load(sl_pre_path) as data:
        b03_pre = data["values"][0, r_sl, c_sl]
        b08_pre = data["values"][1, r_sl, c_sl]
        ndwi_pre = (b03_pre - b08_pre) / np.maximum(0.002, b03_pre + b08_pre)

    with np.load(sl_post_path) as data:
        b03_post = data["values"][0, r_sl, c_sl]
        b08_post = data["values"][1, r_sl, c_sl]
        ndwi_post = (b03_post - b08_post) / np.maximum(0.002, b03_post + b08_post)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))

    # Panel a: South Lhonak Pre-event NDWI
    im0 = axes[0].imshow(ndwi_pre, cmap="Blues", vmin=-0.4, vmax=0.6)
    axes[0].set_title("(a) SL Pre-Breach\n(26 Sep 2023)", fontsize=8.5, fontweight="bold")
    axes[0].axis("off")

    # Panel b: South Lhonak Post-event NDWI
    im1 = axes[1].imshow(ndwi_post, cmap="Blues", vmin=-0.4, vmax=0.6)
    axes[1].set_title("(b) SL Post-Breach\n(09 Oct 2023)", fontsize=8.5, fontweight="bold")
    axes[1].axis("off")

    # Discrete colormap for zones: 0: Water, 1: Moraine, 2: Background
    from matplotlib.colors import ListedColormap
    zone_cmap = ListedColormap(["#1E88E5", "#FFB300", "#78909C"])  # Blue, Amber, Slate

    # Panel c: South Lhonak Zone Partition
    axes[2].imshow(sl_zones, cmap=zone_cmap, vmin=0, vmax=2)
    axes[2].set_title("(c) SL Geomorphic\nZone Partition", fontsize=8.5, fontweight="bold")
    axes[2].axis("off")

    # Panel d: Khangchung Tsho Zone Partition
    axes[3].imshow(kt_zones, cmap=zone_cmap, vmin=0, vmax=2)
    axes[3].set_title("(d) KT Geomorphic\nZone Partition", fontsize=8.5, fontweight="bold")
    axes[3].axis("off")

    # Custom legend for zones
    legend_patches = [
        mpatches.Patch(color="#1E88E5", label="Zone 0: Core Water"),
        mpatches.Patch(color="#FFB300", label="Zone 1: Moraine Dam (≤500m)"),
        mpatches.Patch(color="#78909C", label="Zone 2: Background Terrain"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_rois_and_geomorphic_zones.pdf")
    fig.savefig(FIGURES_DIR / "fig1_rois_and_geomorphic_zones.png")
    fig.savefig(FIGURES_DIR / "fig2_rois_and_geomorphic_zones.pdf")
    fig.savefig(FIGURES_DIR / "fig2_rois_and_geomorphic_zones.png")
    plt.close(fig)
    print("Saved fig1_rois_and_geomorphic_zones.{pdf,png} (and fig2 alias)")


def build_figure_3():
    """Figure 3: 5-Seed Lean T-MAE Training Convergence (Cosine Decay & EMA)."""
    hist = load_training_histories()
    epochs = list(range(1, 151))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    seed_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C2185B"]

    for seed_idx, (seed_str, h_data) in enumerate(hist.items()):
        col = seed_colors[seed_idx % len(seed_colors)]
        ax1.plot(epochs, h_data["train_losses"], color=col, alpha=0.8, linewidth=1.2, label=f"Seed {seed_str}")
        ax2.plot(epochs, h_data["val_losses"], color=col, alpha=0.8, linewidth=1.2, label=f"Seed {seed_str}")

    ax1.set_xlabel("Optimization Epoch")
    ax1.set_ylabel("Training Masked MSE Loss")
    ax1.set_title("(a) Training Loss (CosineAnnealingLR)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.5)

    ax2.set_xlabel("Optimization Epoch")
    ax2.set_ylabel("Validation Masked MSE Loss")
    ax2.set_title("(b) Chronological Holdout Val Loss", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylim(0.0, 4.0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_tmae_training_convergence.pdf")
    fig.savefig(FIGURES_DIR / "fig3_tmae_training_convergence.png")
    plt.close(fig)
    print("Saved fig3_tmae_training_convergence.{pdf,png}")


def build_figure_4(bindings: Dict[str, Any]):
    """Figure 4: Remediated Multi-Detector Anomaly Trajectories."""
    obs = bindings["observations"]
    sl_obs = sorted([o for o in obs if o["site_id"] == "PILOT-SOUTH-LHONAK"], key=lambda r: r["acquired_at"])
    kt_obs = sorted([o for o in obs if o["site_id"] == "PILOT-KHANGCHUNG-TSHO"], key=lambda r: r["acquired_at"])

    sl_dts = [datetime.fromisoformat(r["acquired_at"]) for r in sl_obs]
    kt_dts = [datetime.fromisoformat(r["acquired_at"]) for r in kt_obs]

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.8), sharex=True)

    t_event_start = datetime(2023, 10, 2, 18, 30, tzinfo=timezone.utc)
    t_event_end = datetime(2023, 10, 4, 18, 30, tzinfo=timezone.utc)

    # 1. Lean T-MAE Score C
    ax = axes[0]
    ax.axvspan(t_event_start, t_event_end, color=EVENT_COLOR, alpha=0.25)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax.plot(sl_dts, [r["score_c"] for r in sl_obs], color=SL_COLOR, marker="o", label="South Lhonak")
    ax.plot(kt_dts, [r["score_c"] for r in kt_obs], color=KT_COLOR, marker="s", label="Khangchung Tsho")
    ax.set_ylabel("T-MAE Score C")
    ax.set_title("(a) Remediated Transformer Masked Autoencoder Score C", fontweight="bold", loc="left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    # 2. Physical Water Surface Area Baseline
    ax = axes[1]
    ax.axvspan(t_event_start, t_event_end, color=EVENT_COLOR, alpha=0.25)
    ax.plot(sl_dts, [r["water_area_km2"] for r in sl_obs], color=SL_COLOR, marker="o")
    ax.plot(kt_dts, [r["water_area_km2"] for r in kt_obs], color=KT_COLOR, marker="s")
    ax.set_ylabel("Water Area (km²)")
    ax.set_title("(b) Physical Water Surface Area Baseline", fontweight="bold", loc="left")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 3. Universal Page's CUSUM Detector
    ax = axes[2]
    ax.axvspan(t_event_start, t_event_end, color=EVENT_COLOR, alpha=0.25)
    ax.axhline(4.0, color="red", linestyle="--", linewidth=1.2, label="Alarm Threshold (h=4.0)")
    ax.plot(sl_dts, [r["cusum_tmae_c"] for r in sl_obs], color=SL_COLOR, marker="o", label="SL CUSUM")
    ax.plot(kt_dts, [r["cusum_tmae_c"] for r in kt_obs], color=KT_COLOR, marker="s", label="KT CUSUM")
    ax.set_ylabel("CUSUM Statistic")
    ax.set_title("(c) Universal Page's CUSUM Sequence Filter (T-MAE Score C)", fontweight="bold", loc="left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    # 4. Isolation Forest
    ax = axes[3]
    ax.axvspan(t_event_start, t_event_end, color=EVENT_COLOR, alpha=0.25)
    ax.plot(sl_dts, [r["iso_forest_score"] for r in sl_obs], color=SL_COLOR, marker="o")
    ax.plot(kt_dts, [r["iso_forest_score"] for r in kt_obs], color=KT_COLOR, marker="s")
    ax.set_ylabel("Isolation Forest")
    ax.set_title("(d) Classical Tabular Baseline (Isolation Forest)", fontweight="bold", loc="left")
    ax.grid(True, linestyle="--", alpha=0.4)

    axes[3].xaxis.set_major_locator(mdates.MonthLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[3].set_xlabel("Acquisition Date (2023)")
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_remediated_anomaly_trajectories.pdf")
    fig.savefig(FIGURES_DIR / "fig4_remediated_anomaly_trajectories.png")
    plt.close(fig)
    print("Saved fig4_remediated_anomaly_trajectories.{pdf,png}")


def build_figure_5(bindings: Dict[str, Any]):
    """Figure 5: Geomorphic Zone Contrast and Legacy Discrepancy Breakdown."""
    obs = bindings["observations"]
    sl_obs = sorted([o for o in obs if o["site_id"] == "PILOT-SOUTH-LHONAK"], key=lambda r: r["acquired_at"])
    kt_obs = sorted([o for o in obs if o["site_id"] == "PILOT-KHANGCHUNG-TSHO"], key=lambda r: r["acquired_at"])

    sl_dts = [datetime.fromisoformat(r["acquired_at"]) for r in sl_obs]
    kt_dts = [datetime.fromisoformat(r["acquired_at"]) for r in kt_obs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    t_event_start = datetime(2023, 10, 2, 18, 30, tzinfo=timezone.utc)
    t_event_end = datetime(2023, 10, 4, 18, 30, tzinfo=timezone.utc)

    # Panel a: Zone Contrast Timeline
    ax1.axvspan(t_event_start, t_event_end, color=EVENT_COLOR, alpha=0.25)
    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax1.plot(sl_dts, [r["zone_contrast"] for r in sl_obs], color=SL_COLOR, marker="o", label="South Lhonak")
    ax1.plot(kt_dts, [r["zone_contrast"] for r in kt_obs], color=KT_COLOR, marker="s", label="Khangchung Tsho")
    ax1.set_ylabel("Zone Contrast (Target - Background)")
    ax1.set_title("(a) Geomorphic Zone Contrast", fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # Panel b: Legacy vs Remediated Comparison Bar Chart
    leg = bindings["legacy_discrepancy_reproduction"]
    categories = ["SL Pre", "SL Post", "KT Pre", "KT Post"]
    legacy_vals = [
        leg["legacy_score_c_south_lhonak_pre"],
        leg["legacy_score_c_south_lhonak_post"],
        leg["legacy_score_c_khangchung_pre"],
        leg["legacy_score_c_khangchung_post"],
    ]
    remediated_vals = [
        leg["sr1_remediated_score_c_south_lhonak_pre"],
        leg["sr1_remediated_score_c_south_lhonak_post"],
        leg["sr1_remediated_score_c_khangchung_pre"],
        leg["sr1_remediated_score_c_khangchung_post"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width / 2, legacy_vals, width, label="Legacy 18-Oct", color="#E57373")
    ax2.bar(x + width / 2, remediated_vals, width, label="SR-1 Remediated", color="#1976D2")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Score C Median")
    ax2.set_title("(b) Legacy Discrepancy Explained", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_zone_contrast_and_legacy_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig5_zone_contrast_and_legacy_comparison.png")
    plt.close(fig)
    print("Saved fig5_zone_contrast_and_legacy_comparison.{pdf,png}")


def main():
    bindings = load_result_bindings()
    print("Building Milestone SR-1 Figures...")
    build_figure_1(bindings)
    build_figure_2()
    build_figure_3()
    build_figure_4(bindings)
    build_figure_5(bindings)
    print("All 5 figure families successfully generated in docs/figures/sr1/!")


if __name__ == "__main__":
    main()
