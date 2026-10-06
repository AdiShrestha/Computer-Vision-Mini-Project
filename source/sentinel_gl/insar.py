"""InSAR Single Look Complex (SLC) feasibility, geometric distortion, and coherence modeling engine.

Evaluates interferometric Synthetic Aperture Radar (InSAR) physical constraints in steep
High Mountain Asia glaciated terrain:
  1. Storage & memory budgeting: Sentinel-1 SLC product frame sizes vs laptop storage limits.
  2. Geometric distortion: Local incidence angle, radar layover, radar shadow, and foreshortening.
  3. Multi-temporal coherence decay: Geometric, temporal, and thermal decorrelation modeling.
  4. Feasibility dossier & technical report generation for Sentinel-GL.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Physical & Platform Constants
SENTINEL1_SLC_ZIP_BYTES: int = 4_200_000_000        # ~4.2 GB per SLC zip frame
SENTINEL1_SLC_UNCOMPRESSED_BYTES: int = 8_000_000_000 # ~8.0 GB uncompressed
MAX_LOCAL_STORAGE_BYTES: int = 114_000_000_000      # 114 GB available storage on M3 Air target
MAX_BURST_ROI_BYTES: int = 1_000_000_000            # 1.0 GB per burst-cropped lake ROI
CRITICAL_PERPENDICULAR_BASELINE_M: float = 5000.0   # Sentinel-1 nominal critical baseline
DEFAULT_SNR_DB: float = 15.0                        # Nominal thermal SNR for Sentinel-1 IW mode
PHASE_UNWRAPPING_COHERENCE_THRESHOLD: float = 0.30  # Minimum coherence for reliable phase unwrapping


def check_insar_storage_budget(
    lake_count: int,
    n_scenes_per_lake: int,
    burst_crop_mode: bool = False,
    available_storage_bytes: int = MAX_LOCAL_STORAGE_BYTES,
) -> Dict[str, Any]:
    """Audit InSAR SLC storage requirements against local storage allocation.

    Demonstrates that full-scene multi-temporal SLC downloads exceed laptop storage limits,
    establishing the necessity of provider-side burst cropping or GRD feature extraction.
    """
    unit_bytes = MAX_BURST_ROI_BYTES if burst_crop_mode else SENTINEL1_SLC_ZIP_BYTES
    total_required_bytes = lake_count * n_scenes_per_lake * unit_bytes

    is_feasible = total_required_bytes <= available_storage_bytes
    exceeded_bytes = max(0, total_required_bytes - available_storage_bytes)

    if burst_crop_mode:
        mode_desc = "Provider-side burst cropping / sub-setting (<= 1.0 GB per lake ROI)"
    else:
        mode_desc = "Full-scene Sentinel-1 SLC download (~4.2 GB zip per acquisition)"

    recommendation = (
        "Storage budget satisfied; local processing permitted."
        if is_feasible
        else (
            f"Full-scene SLC download requires {total_required_bytes / (1024**3):.1f} GB, "
            f"exceeding local budget {available_storage_bytes / (1024**3):.1f} GB by "
            f"{exceeded_bytes / (1024**3):.1f} GB. Provider-side burst cropping or GRD "
            "amplitude + spatial variance pipeline must be utilized instead."
        )
    )

    return {
        "is_feasible": is_feasible,
        "lake_count": lake_count,
        "n_scenes_per_lake": n_scenes_per_lake,
        "burst_crop_mode": burst_crop_mode,
        "mode_description": mode_desc,
        "unit_bytes": unit_bytes,
        "total_required_bytes": total_required_bytes,
        "available_storage_bytes": available_storage_bytes,
        "exceeded_bytes": exceeded_bytes,
        "recommendation": recommendation,
    }


def classify_geometric_distortion(
    slope_deg: float,
    aspect_deg: float,
    inc_angle_deg: float = 38.0,
    heading_deg: float = 348.0,
) -> Dict[str, Any]:
    """Model SAR local incidence angle and classify radar layover, shadow, and foreshortening.

    Parameters:
      slope_deg: Terrain slope angle (0 to 90 degrees).
      aspect_deg: Terrain aspect angle (0 to 360 degrees clockwise from North).
      inc_angle_deg: Nominal SAR incidence angle (default 38.0 deg for Sentinel-1 IW swath).
      heading_deg: Satellite flight heading (default 348.0 deg for Ascending pass).

    Returns:
      Dictionary with local incidence angle, distortion class, and distortion flag.
    """
    alpha = math.radians(slope_deg)
    theta_inc = math.radians(inc_angle_deg)
    beta = math.radians(aspect_deg)

    # Right-looking radar azimuth: heading + 90 degrees
    phi_look = math.radians((heading_deg + 90.0) % 360.0)

    # Angle between terrain aspect and radar look direction
    psi = phi_look - beta

    # Local incidence angle formula:
    # cos(theta_local) = cos(theta_inc) * cos(alpha) + sin(theta_inc) * sin(alpha) * cos(psi)
    cos_local = math.cos(theta_inc) * math.cos(alpha) + math.sin(theta_inc) * math.sin(alpha) * math.cos(psi)
    cos_local_clamped = max(-1.0, min(1.0, cos_local))
    theta_local_deg = math.degrees(math.acos(cos_local_clamped))

    # Distortion classification:
    # 1. Radar Layover: Steep slope facing the radar beam (alpha >= theta_inc and cos(psi) > 0)
    # 2. Radar Shadow: Backslope facing away from radar (theta_local >= 90 deg or cos(psi) < 0 with alpha >= 90 - theta_inc)
    # 3. Foreshortening: Slope facing radar with alpha < theta_inc
    # 4. Normal: Minimal distortion
    facing_radar = math.cos(psi) > 0.0

    if facing_radar and (slope_deg >= inc_angle_deg):
        distortion_class = "LAYOVER"
        is_distorted = True
        explanation = (
            f"Terrain slope ({slope_deg:.1f} deg) exceeds nominal incidence angle ({inc_angle_deg:.1f} deg) "
            "on radar-facing slope. Top of ridge returns signal before base, causing radar layover and phase inversion."
        )
    elif theta_local_deg >= 90.0 or (not facing_radar and slope_deg >= (90.0 - inc_angle_deg)):
        distortion_class = "SHADOW"
        is_distorted = True
        explanation = (
            f"Backslope facing away from radar beam (local incidence {theta_local_deg:.1f} deg >= 90 deg). "
            "Surface is obscured from radar line-of-sight, resulting in complete signal loss (radar shadow)."
        )
    elif facing_radar and slope_deg > 5.0:
        distortion_class = "FORESHORTENING"
        is_distorted = False
        explanation = (
            f"Terrain slope ({slope_deg:.1f} deg) faces radar, compressing slant-range pixel resolution (foreshortening)."
        )
    else:
        distortion_class = "NORMAL"
        is_distorted = False
        explanation = "Terrain geometry provides unobstructed radar illumination within operational limits."

    return {
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "nominal_inc_angle_deg": inc_angle_deg,
        "heading_deg": heading_deg,
        "local_incidence_deg": round(theta_local_deg, 2),
        "distortion_class": distortion_class,
        "is_distorted": is_distorted,
        "explanation": explanation,
    }


def model_insar_coherence(
    b_temp_days: float,
    b_perp_meters: float = 50.0,
    season: str = "monsoon",
    snr_db: float = DEFAULT_SNR_DB,
) -> Dict[str, Any]:
    """Model interferometric coherence decay over glacial lake moraines and terrain.

    Coherence model:
      gamma_total = gamma_geom * gamma_temp * gamma_thermal

    Parameters:
      b_temp_days: Temporal baseline in days (e.g. 12 days for Sentinel-1 single satellite).
      b_perp_meters: Perpendicular spatial baseline in meters.
      season: Environmental moisture regime ('dry_winter', 'transition', 'monsoon').
      snr_db: Signal-to-noise ratio in decibels.
    """
    # 1. Geometric decorrelation
    gamma_geom = max(0.0, 1.0 - (abs(b_perp_meters) / CRITICAL_PERPENDICULAR_BASELINE_M))

    # 2. Thermal decorrelation
    snr_linear = 10.0 ** (snr_db / 10.0)
    gamma_thermal = 1.0 / (1.0 + (1.0 / snr_linear))

    # 3. Temporal decorrelation time constant (tau_decorr in days)
    decorr_time_constants = {
        "dry_winter": 45.0,   # Frozen stable moraine / rock
        "transition": 15.0,   # Spring snowmelt / early thaw
        "monsoon": 3.0,       # Heavy rainfall, saturated glacial till, active surficial motion
    }
    tau_decorr = decorr_time_constants.get(season, 10.0)

    gamma_temp = math.exp(-b_temp_days / tau_decorr)
    gamma_total = gamma_geom * gamma_temp * gamma_thermal

    is_viable = gamma_total >= PHASE_UNWRAPPING_COHERENCE_THRESHOLD

    return {
        "b_temp_days": b_temp_days,
        "b_perp_meters": b_perp_meters,
        "season": season,
        "tau_decorr_days": tau_decorr,
        "gamma_geom": round(gamma_geom, 4),
        "gamma_thermal": round(gamma_thermal, 4),
        "gamma_temp": round(gamma_temp, 4),
        "gamma_total": round(gamma_total, 4),
        "coherence_threshold": PHASE_UNWRAPPING_COHERENCE_THRESHOLD,
        "is_phase_unwrapping_viable": is_viable,
        "implication": (
            "Coherence supports interferometric phase unwrapping."
            if is_viable
            else (
                f"Severe decorrelation (gamma={gamma_total:.4f} < {PHASE_UNWRAPPING_COHERENCE_THRESHOLD}). "
                "Phase noise precludes reliable interferometric deformation extraction; SAR GRD amplitude "
                "and spatial texture features must be prioritized."
            )
        ),
    }


def generate_insar_feasibility_dossier() -> Dict[str, Any]:
    """Generate complete computational InSAR feasibility dossier for High Mountain Asia."""
    # 1. Storage scenarios
    storage_full_scene = check_insar_storage_budget(lake_count=8, n_scenes_per_lake=30, burst_crop_mode=False)
    storage_burst_crop = check_insar_storage_budget(lake_count=8, n_scenes_per_lake=30, burst_crop_mode=True)

    # 2. Geometric distortion case studies
    # Steep headwall above South Lhonak: slope 45 deg, facing East (aspect 90 deg), Ascending pass (look 78 deg)
    geom_lhonak_headwall = classify_geometric_distortion(slope_deg=45.0, aspect_deg=80.0, inc_angle_deg=38.0, heading_deg=348.0)
    # Backslope moraine: slope 55 deg, facing West (aspect 260 deg)
    geom_lhonak_shadow = classify_geometric_distortion(slope_deg=55.0, aspect_deg=260.0, inc_angle_deg=38.0, heading_deg=348.0)
    # Gentle lake shore / outwash: slope 8 deg, aspect 120 deg
    geom_lake_shore = classify_geometric_distortion(slope_deg=8.0, aspect_deg=120.0, inc_angle_deg=38.0, heading_deg=348.0)

    # 3. Coherence decay scenarios (Sentinel-1 12-day repeat)
    coherence_winter = model_insar_coherence(b_temp_days=12.0, b_perp_meters=50.0, season="dry_winter")
    coherence_monsoon = model_insar_coherence(b_temp_days=12.0, b_perp_meters=50.0, season="monsoon")
    coherence_monsoon_6d = model_insar_coherence(b_temp_days=6.0, b_perp_meters=30.0, season="monsoon")

    dossier = {
        "dossier_title": "Sentinel-GL InSAR Feasibility & Decorrelation Dossier",
        "version": 1,
        "scope": "High Mountain Asia Glaciated Terrain & Moraine Dams",
        "storage_budget_analysis": {
            "available_local_storage_gb": MAX_LOCAL_STORAGE_BYTES / (1024**3),
            "full_scene_archive_scenario": storage_full_scene,
            "burst_cropped_scenario": storage_burst_crop,
        },
        "geometric_distortion_cases": {
            "steep_headwall_layover": geom_lhonak_headwall,
            "backslope_radar_shadow": geom_lhonak_shadow,
            "gentle_outwash_foreshortening": geom_lake_shore,
        },
        "coherence_decay_cases": {
            "winter_stable_12day": coherence_winter,
            "monsoon_moraine_12day": coherence_monsoon,
            "monsoon_moraine_6day": coherence_monsoon_6d,
        },
        "architectural_recommendation": {
            "selected_modality": "SAR GRD Linear Power + Spatial Texture",
            "justification": (
                "Due to pervasive monsoon decorrelation (gamma < 0.25) and radar layover/shadow "
                "on moraine headwalls, multi-temporal InSAR phase unwrapping cannot provide continuous "
                "pre-event monitoring. Calibrated dual-pol GRD amplitude (VV, VH, cross-ratio) combined "
                "with spatial variance provides unbroken, physically robust observational coverage."
            ),
        },
    }
    return dossier


def generate_insar_feasibility_report_markdown(dossier: Dict[str, Any]) -> str:
    """Render technical InSAR feasibility report as Markdown."""
    storage_fs = dossier["storage_budget_analysis"]["full_scene_archive_scenario"]
    storage_bc = dossier["storage_budget_analysis"]["burst_cropped_scenario"]
    geom_layover = dossier["geometric_distortion_cases"]["steep_headwall_layover"]
    geom_shadow = dossier["geometric_distortion_cases"]["backslope_radar_shadow"]
    coh_monsoon = dossier["coherence_decay_cases"]["monsoon_moraine_12day"]
    coh_winter = dossier["coherence_decay_cases"]["winter_stable_12day"]

    monsoon_gamma = coh_monsoon['gamma_total']
    monsoon_temp = coh_monsoon['gamma_temp']
    winter_gamma = coh_winter['gamma_total']

    return f"""# Technical Report: InSAR SLC Feasibility & Decorrelation Analysis in High Mountain Asia

**Report ID:** `REP-INSAR-FEASIBILITY-01`  
**Governing Work Package:** `WP11-COHORT-INSAR-01`  
**Target Terrain:** High Mountain Asia (Himalaya, Karakoram, Hindu Kush)  

---

## 1. Executive Summary

This report evaluates the physical feasibility of Sentinel-1 Single Look Complex (SLC) interferometric SAR (InSAR) for continuous glacial lake outburst flood (GLOF) monitoring. Analysis across storage budgets, radar geometric distortions, and multi-temporal interferometric coherence reveals severe physical limits in steep glaciated catchments:
1. **Storage Prohibitions:** Regional multi-temporal SLC full-scene archives exceed local workstation budgets (>1.0 TB vs 114 GB available), requiring burst-level sub-setting.
2. **Geometric Distortions:** Steep cirque headwalls (>= 38 deg) face acute radar layover, while opposing valley walls fall into radar shadow.
3. **Monsoon Coherence Collapse:** Heavy monsoon precipitation and moisture-saturated moraine till cause rapid temporal decorrelation (gamma_total = {monsoon_gamma:.4f} < 0.25), preventing reliable phase unwrapping during the peak hazard season (July-October).

Consequently, Sentinel-GL correctly selects **calibrated dual-polarization GRD backscatter amplitude and spatial texture** over fragile interferometric phase tracking.

---

## 2. Storage & Memory Budget Audit

Sentinel-1 SLC acquisitions contain full phase and amplitude at native single-look resolution, producing massive data volumes:

| InSAR Processing Scenario | Scene Unit Size | Total Archive Volume (8 lakes, 30 dates) | Budget Feasibility |
| :--- | :--- | :--- | :--- |
| **Full-Scene SLC Archive** | ~4.2 GB zip (~8.0 GB uncompressed) | {storage_fs['total_required_bytes'] / (1024**3):.1f} GB | **REJECTED (Exceeds 114 GB budget)** |
| **Burst-Cropped ROI** | <= 1.0 GB per lake ROI | {storage_bc['total_required_bytes'] / (1024**3):.1f} GB | **FEASIBLE (Within local limits)** |

---

## 3. SAR Geometric Distortion Modeling

In high-relief terrain, the local incidence angle theta_local governs radar reflectivity:
cos(theta_local) = cos(theta_inc) * cos(alpha) + sin(theta_inc) * sin(alpha) * cos(phi_look - beta)

### Modeled Case Studies:
- **Radar Layover (Steep Cirque Headwall):** Slope alpha = {geom_layover['slope_deg']:.1f} deg, Aspect beta = {geom_layover['aspect_deg']:.1f} deg. Result: `{geom_layover['distortion_class']}` (local incidence angle: {geom_layover['local_incidence_deg']:.1f} deg). The top of the slope reaches the radar antenna prior to the base, resulting in phase inversion and signal entanglement.
- **Radar Shadow (Opposing Valley Wall):** Slope alpha = {geom_shadow['slope_deg']:.1f} deg, Aspect beta = {geom_shadow['aspect_deg']:.1f} deg. Result: `{geom_shadow['distortion_class']}` (local incidence angle: {geom_shadow['local_incidence_deg']:.1f} deg). The slope is shielded from the radar beam line-of-sight, yielding complete signal void.

---

## 4. Multi-Temporal Coherence Decay Modeling

Interferometric coherence gamma_total = gamma_geom * gamma_temp * gamma_thermal reflects phase correlation across satellite passes:

gamma_temp = exp(-B_t / tau_decorr)

### Modeled Coherence Regimes (Sentinel-1 12-day repeat):
- **Winter Frozen Moraine:** tau_decorr = {coh_winter['tau_decorr_days']:.1f} days => gamma_total = {winter_gamma:.4f} (Viable for phase unwrapping).
- **Summer Monsoon Moraine (July-October):** Heavy rainfall (>50 mm/week) and glacial till saturation yield rapid decorrelation: tau_decorr = {coh_monsoon['tau_decorr_days']:.1f} days => gamma_temp = {monsoon_temp:.4f}, gamma_total = {monsoon_gamma:.4f}.

Because gamma_total < 0.25 falls far below the phase unwrapping threshold (0.30), interferometric phase coherence collapses during the South Lhonak pre-event window (August-October 2023).

---

## 5. Architectural Decision

Sentinel-GL rejects operational reliance on InSAR phase unwrapping. The project adopts calibrated Sentinel-1 GRD dual-polarization backscatter power (VV, VH), cross-ratio (VH/VV), and spatial texture variance, guaranteeing continuous observability through cloud and monsoon moisture.
"""


def export_insar_artifacts(output_dir: Path) -> Tuple[Path, Path]:
    """Export InSAR feasibility dossier JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dossier_path = output_dir / "insar_feasibility_dossier.json"
    report_path = output_dir / "insar_feasibility_report.md"

    dossier = generate_insar_feasibility_dossier()
    with open(dossier_path, "w", encoding="utf-8") as f:
        json.dump(dossier, f, indent=2)

    report_md = generate_insar_feasibility_report_markdown(dossier)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return dossier_path, report_path


def main() -> int:
    """CLI entrypoint for generating InSAR feasibility dossier and report."""
    parser = argparse.ArgumentParser(description="Sentinel-GL InSAR Feasibility Engine.")
    parser.add_argument("--generate", action="store_true", help="Generate InSAR dossier and report")
    parser.add_argument("--output-dir", type=str, default="docs/insar", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    d_path, r_path = export_insar_artifacts(out_dir)

    print("=" * 60)
    print("Sentinel-GL InSAR Feasibility & Decorrelation Engine")
    print("=" * 60)
    print(f"Dossier JSON written to:      {d_path}")
    print(f"Technical Report written to:  {r_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
