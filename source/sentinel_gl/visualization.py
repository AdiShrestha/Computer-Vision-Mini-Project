"""Deterministic scientific figure rendering engine for Sentinel-GL research artifacts.

Uses headless matplotlib backend ('Agg') with stripped metadata to guarantee
bit-identical reproducible PNG output. Generates the 5 primary research figures:
1. Fig 1: Observation Cadence & Monitoring Coverage (C_obs)
2. Fig 2: Multi-Modal Feature Panel Traces & Pre-Event Trajectory
3. Fig 3: Anomaly Score Trajectories & Bounded Lead Time (Delta t_lead)
4. Fig 4: Negative-Control Alert Episodes & Follow-Up Exposure (lambda_alert)
5. Fig 5: Sensor Ablation Lattice (2^N configurations)
"""
from __future__ import annotations
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .episodes import AlertEpisode, LeadTimeResult
from .features import FEATURE_CHANNELS, NUM_CHANNELS, WINDOW_DAYS, MultiModalPanel
from .predictions import ABLATION_CONFIGS


def _save_deterministic_fig(fig: plt.Figure, output_path: Union[str, Path], dpi: int = 150) -> str:
    """Save matplotlib figure with deterministic metadata to guarantee bit-identical PNG digests."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Date": None, "Software": "Matplotlib"},
    )
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 1: Observation Cadence & Monitoring Coverage (C_obs)
# ---------------------------------------------------------------------------

def plot_observation_cadence_and_coverage(
    dossier: Mapping[str, Any],
    output_path: Union[str, Path],
) -> str:
    """Plot observation arrivals across optical, SAR, and weather modalities showing monsoon cloud gaps."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, dpi=150)

    cohort = dossier.get("cohort_summary", {})
    records = dossier.get("record_provenance", [])

    # Group records by lake and modality
    # lake_id -> modality -> list of dates
    lake_obs: Dict[str, Dict[str, List[date]]] = {
        "SGL-001": {"optical": [], "sar": [], "weather": []},
        "SGL-002": {"optical": [], "sar": [], "weather": []},
    }

    for rec in records:
        l_id = rec.get("lake_id")
        mod = rec.get("modality")
        ts = rec.get("acquisition_timestamp") or rec.get("date") or ""
        if l_id in lake_obs and mod in lake_obs[l_id] and ts:
            d_obj = date.fromisoformat(ts.split("T")[0])
            lake_obs[l_id][mod].append(d_obj)

    # Plot for SGL-001 (South Lhonak) and SGL-002 (Control)
    lakes = [
        ("SGL-001", "South Lhonak (SGL-001, Event Case)", axes[0]),
        ("SGL-002", "Khangchung Tsho (SGL-002, Negative Control)", axes[1]),
    ]

    modality_colors = {
        "optical": "#1f77b4",   # Blue
        "sar": "#ff7f0e",       # Orange
        "weather": "#2ca02c",   # Green
    }
    y_positions = {"optical": 3, "sar": 2, "weather": 1}

    for l_id, title, ax in lakes:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        for mod, y_pos in y_positions.items():
            d_list = lake_obs[l_id][mod]
            if d_list:
                ax.scatter(
                    d_list,
                    [y_pos] * len(d_list),
                    color=modality_colors[mod],
                    s=45,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.5,
                    label=mod.upper() if l_id == "SGL-001" else "",
                )

        # Highlight pre-event cutoff if SGL-001
        if l_id == "SGL-001":
            cutoff_date = date.fromisoformat("2023-10-03")
            ax.axvline(cutoff_date, color="crimson", linestyle="--", linewidth=1.5, label="Pre-Event Cutoff")

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["ERA5 Reanalysis", "Sentinel-1 SAR", "Sentinel-2 MSI"], fontsize=9)
        ax.set_ylim(0.5, 3.8)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)
    axes[1].set_xlabel("Observation Date (September - October 2023)", fontsize=10, labelpad=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.suptitle("Figure 1: Authentic Observation Cadence & Multi-Modal Coverage (C_obs)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    return _save_deterministic_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2: Multi-Modal Feature Panel Traces
# ---------------------------------------------------------------------------

def plot_multimodal_feature_panels(
    panel: MultiModalPanel,
    output_path: Union[str, Path],
) -> str:
    """Plot 180-day time series across all 11 standardized channels distinguishing observed vs missing entries."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True, dpi=150)
    ax_flat = axes.flatten()

    dates = [date.fromisoformat(d.split("T")[0]) for d in panel.dates]

    channel_groups = {
        0: ("Lake Area (km²)", "#1f77b4"),
        1: ("NDWI Mean", "#17becf"),
        2: ("MNDWI Mean", "#1f77b4"),
        3: ("NDSI Mean", "#7f7f7f"),
        4: ("SAR VV (dB)", "#ff7f0e"),
        5: ("SAR VH (dB)", "#d62728"),
        6: ("SAR Cross-Ratio (dB)", "#e377c2"),
        7: ("SAR VV Variance", "#8c564b"),
        8: ("ERA5 2m Temp (°C)", "#2ca02c"),
        9: ("ERA5 Precip (mm)", "#bcbd22"),
        10: ("ERA5 Temp Anomaly (°C)", "#9467bd"),
    }

    for c_idx in range(NUM_CHANNELS):
        ax = ax_flat[c_idx]
        ch_name = FEATURE_CHANNELS[c_idx]
        title, col = channel_groups[c_idx]

        vals = panel.values[:, c_idx]
        mask = panel.mask[:, c_idx]

        # Plot observed points
        obs_idx = np.flatnonzero(mask)
        if obs_idx.size > 0:
            obs_dates = [dates[i] for i in obs_idx]
            obs_vals = vals[obs_idx]
            ax.plot(obs_dates, obs_vals, color=col, linewidth=1.2, linestyle="-", marker="o", markersize=3, alpha=0.9)

        # Plot missing regions in grey shading
        missing_idx = np.flatnonzero(~mask)
        if missing_idx.size > 0:
            for m_i in missing_idx:
                ax.axvspan(dates[m_i], dates[min(m_i + 1, len(dates) - 1)], color="#f0f0f0", alpha=0.5, linewidth=0)

        ax.set_title(f"c{c_idx}: {title}", fontsize=10, fontweight="bold", pad=4)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Empty 12th subplot for legend and metadata
    ax_unused = ax_flat[11]
    ax_unused.axis("off")
    ax_unused.text(
        0.1,
        0.5,
        f"Lake ID: {panel.lake_id}\nWindow: {panel.start_date} to {panel.end_date}\nObserved Count: {int(np.sum(panel.mask))}\nMissing Count: {int(np.sum(~panel.mask))}\n\nGrey bands: Masked Missingness",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9", edgecolor="#cccccc"),
    )

    for i in range(8, 11):
        ax_flat[i].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.suptitle(f"Figure 2: Multi-Modal Feature Panel Traces ({panel.lake_id})", fontsize=12, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.96, hspace=0.35, wspace=0.25)
    return _save_deterministic_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3: Anomaly Score Trajectories & Bounded Lead Time
# ---------------------------------------------------------------------------

def plot_anomaly_trajectories_and_lead_time(
    lead_time_result: LeadTimeResult,
    trajectory_data: Mapping[str, Sequence[Tuple[str, float]]],
    threshold: float,
    output_path: Union[str, Path],
) -> str:
    """Plot anomaly score trajectories comparing Sentinel-GL T-MAE against 4 baselines and lead time Delta t_lead."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    model_colors = {
        "tmae": ("#1f77b4", "Sentinel-GL (T-MAE)", 2.2),
        "climatology": ("#2ca02c", "Seasonal Climatology", 1.2),
        "area_trend": ("#ff7f0e", "Lake-Area Trend Heuristic", 1.2),
        "weather_only": ("#d62728", "Weather-Only Anomaly", 1.2),
        "rpca": ("#9467bd", "Robust PCA Baseline", 1.2),
    }

    for m_id, (col, label, lw) in model_colors.items():
        traj = trajectory_data.get(m_id, [])
        if traj:
            d_objs = [date.fromisoformat(item[0].split("T")[0]) for item in traj]
            scores = [item[1] for item in traj]
            ax.plot(d_objs, scores, label=label, color=col, linewidth=lw, marker="s" if m_id == "tmae" else "o", markersize=3.5)

    # Threshold horizontal line
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Calibrated Threshold θ ({threshold:.2f})")

    # Onset vertical line
    onset_d = date.fromisoformat(lead_time_result.onset_date.split("T")[0])
    ax.axvline(onset_d, color="crimson", linestyle="-", linewidth=2.0, label=f"Event Onset ({lead_time_result.onset_date})")

    # Lead time annotation
    if lead_time_result.status == "DETECTED" and lead_time_result.declaration_date:
        decl_d = date.fromisoformat(lead_time_result.declaration_date.split("T")[0])
        ax.axvline(decl_d, color="#1f77b4", linestyle=":", linewidth=1.5, label=f"Declaration ({lead_time_result.declaration_date})")
        # Draw double-headed arrow for lead time
        mid_y = threshold + 0.15
        ax.annotate(
            f"Δt_lead = {lead_time_result.lead_time_days} days",
            xy=(decl_d, mid_y),
            xytext=(onset_d, mid_y),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="crimson",
        )

    ax.set_ylabel("Standardized Anomaly Score S(W)", fontsize=10)
    ax.set_xlabel("Evaluation Decision Date", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    title_status = f"Status: {lead_time_result.status}"
    if lead_time_result.lead_time_days is not None:
        title_status += f" (Lead Time: {lead_time_result.lead_time_days}d)"
    fig.suptitle(f"Figure 3: Anomaly Trajectory & Lead Time ({lead_time_result.lake_id} — {title_status})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save_deterministic_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4: Negative-Control Alert Episodes & Exposure
# ---------------------------------------------------------------------------

def plot_control_episodes_and_exposure(
    episodes: Sequence[AlertEpisode],
    decisions: Sequence[Tuple[str, float, bool]],
    threshold: float,
    output_path: Union[str, Path],
) -> str:
    """Plot negative-control monitoring timeline illustrating sustained alarms, hysteresis, and episode collapse."""
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)

    dates = [date.fromisoformat(d[0].split("T")[0]) for d in decisions]
    scores = [d[1] for d in decisions]

    # Plot raw continuous score trajectory
    ax.plot(dates, scores, color="#2b5c8f", linewidth=1.5, marker="o", markersize=3.5, label="Decision Score S(t)")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Anomaly Threshold θ ({threshold:.2f})")

    # Highlight alert episodes with shaded boxes
    for ep in episodes:
        start_d = date.fromisoformat(ep.start_date.split("T")[0])
        end_d = date.fromisoformat(ep.end_date.split("T")[0])
        ax.axvspan(start_d, end_d, color="#d95f02", alpha=0.25, label="Alert Episode (q sustained, r hysteresis)")
        ax.text(
            start_d,
            ep.peak_score + 0.05,
            f"{ep.episode_id}\n(peak={ep.peak_score:.2f})",
            fontsize=8,
            fontweight="bold",
            color="#d95f02",
        )

    # Invariant disclosure text box
    ax.text(
        0.02,
        0.88,
        "Invariant: Multi-month sustained alarms collapse into exactly 1 episode.\nHysteresis r=2 & Refractory period=60d enforce calm recovery.",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffae6", edgecolor="#e6b800"),
    )

    ax.set_ylabel("Standardized Anomaly Score", fontsize=10)
    ax.set_xlabel("Monitoring Calendar Date", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle=":", alpha=0.5)

    # Unique legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle("Figure 4: Negative-Control Operational Alert Episodes & Exposure Accounting", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save_deterministic_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 5: Sensor Ablation Lattice
# ---------------------------------------------------------------------------

def plot_sensor_ablation_comparison(
    ablation_scores: Mapping[str, float],
    output_path: Union[str, Path],
) -> str:
    """Plot metric comparison across the 7 configurations of the 2^N sensor ablation lattice."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)

    configs = ["full", "opt_sar", "opt_era5", "sar_era5", "opt_only", "sar_only", "era5_only"]
    labels = [
        "Full (Opt+SAR+ERA5)",
        "Opt + SAR",
        "Opt + ERA5",
        "SAR + ERA5",
        "Opt Only",
        "SAR Only",
        "ERA5 Only",
    ]
    colors = ["#2ca02c", "#1f77b4", "#17becf", "#ff7f0e", "#9467bd", "#d62728", "#8c564b"]

    scores = [ablation_scores.get(c, 0.0) for c in configs]
    y_pos = np.arange(len(configs))

    bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()  # Top-down order

    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Reconstruction MSE / Relative Score Sensitivity", fontsize=10)
    ax.set_xlim(0, max(scores, default=1.0) * 1.25)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    fig.suptitle("Figure 5: 2^N Sensor Ablation Lattice Comparison", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save_deterministic_fig(fig, output_path)
