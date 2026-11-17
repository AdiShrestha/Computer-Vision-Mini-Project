"""Explicit assumed InSAR scenarios; no measured terrain, coherence or feasibility.

Defaults are retained historical hypotheses. They have not been empirically
validated and cannot determine sensor selection for the research system.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Historical scenario assumptions, not measured product sizes, hardware or site physics.
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
        "Assumed storage inequality satisfied; actual product sizes and resources remain unmeasured."
        if is_feasible
        else (
            f"Full-scene SLC download requires {total_required_bytes / (1024**3):.1f} GB, "
            f"exceeding local budget {available_storage_bytes / (1024**3):.1f} GB by "
            f"{exceeded_bytes / (1024**3):.1f} GB. Provider-side burst cropping or GRD "
            "amplitude alternatives require a separate measured feasibility assessment."
        )
    )

    return {
        "origin": "simulation", "research_admissible": False, "status": "ASSUMED_SCENARIO",
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
        "origin": "simulation", "research_admissible": False, "status": "ASSUMED_SCENARIO",
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
    if season not in decorr_time_constants:
        raise ValueError("unknown assumed season; no fallback decorrelation time")
    tau_decorr = decorr_time_constants[season]

    gamma_temp = math.exp(-b_temp_days / tau_decorr)
    gamma_total = gamma_geom * gamma_temp * gamma_thermal

    is_viable = gamma_total >= PHASE_UNWRAPPING_COHERENCE_THRESHOLD

    return {
        "origin": "simulation", "research_admissible": False, "status": "ASSUMED_SCENARIO",
        "assumptions": {"critical_perpendicular_baseline_m": CRITICAL_PERPENDICULAR_BASELINE_M, "snr_db": snr_db,
                        "decay_constants_days": decorr_time_constants, "evidence_status": "NOT_EMPIRICALLY_VALIDATED"},
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
        "implication": "The assumed model is above its scenario threshold." if is_viable else
                       "The assumed model is below its scenario threshold; no measured sensor feasibility conclusion follows.",
    }


def generate_insar_feasibility_dossier(*args, **kwargs):
    from .admission import blocked
    blocked("unverified cohort or empirical InSAR report")


def generate_insar_feasibility_report_markdown(*args, **kwargs):
    from .admission import blocked
    blocked("unverified cohort or empirical InSAR report")


def export_insar_artifacts(*args, **kwargs):
    from .admission import blocked
    blocked("unverified cohort or empirical InSAR report")


def main(*args, **kwargs):
    from .admission import blocked
    blocked("unverified cohort or empirical InSAR report")


if __name__ == "__main__":
    import sys
    sys.exit(main())
