"""Prediction ledger records, 2^N sensor ablation lattice, and paired differences.

Implements structured prediction generation, sensor ablation masks, window eligibility
checking, and paired difference estimand Delta S = S_model - S_baseline over identical
eligible supports.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np

from .features import FEATURE_CHANNELS, NUM_CHANNELS, MultiModalPanel


# ---------------------------------------------------------------------------
# Sensor Groups and Ablation Lattice
# ---------------------------------------------------------------------------

OPTICAL_CHANNELS: Tuple[int, ...] = (0, 1, 2, 3)
SAR_CHANNELS: Tuple[int, ...] = (4, 5, 6, 7)
ERA5_CHANNELS: Tuple[int, ...] = (8, 9, 10)

SENSOR_MODALITIES: Dict[str, Tuple[int, ...]] = {
    "optical": OPTICAL_CHANNELS,
    "sar": SAR_CHANNELS,
    "era5": ERA5_CHANNELS,
}

# 2^N named configurations (for N=3 sensors: Optical, SAR, ERA5)
# Excludes the trivial empty configuration where 0 sensors are active.
ABLATION_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "full": ("optical", "sar", "era5"),
    "opt_sar": ("optical", "sar"),
    "opt_era5": ("optical", "era5"),
    "sar_era5": ("sar", "era5"),
    "opt_only": ("optical",),
    "sar_only": ("sar",),
    "era5_only": ("era5",),
}


def get_active_channels(ablation_id: str) -> Set[int]:
    """Return the set of active channel indices for a given ablation configuration."""
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation_id '{ablation_id}'. Must be one of {list(ABLATION_CONFIGS.keys())}"
        )
    modalities = ABLATION_CONFIGS[ablation_id]
    active: Set[int] = set()
    for mod in modalities:
        active.update(SENSOR_MODALITIES[mod])
    return active


def apply_sensor_ablation(
    values: np.ndarray,
    mask: np.ndarray,
    ablation_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sensor ablation intervention by masking out inactive channels.

    Inactive channels have mask set to False and values set to NaN to strictly
    preserve explicit missingness invariants.
    """
    if values.shape != mask.shape:
        raise ValueError(f"values shape {values.shape} must match mask shape {mask.shape}")
    if values.shape[1] != NUM_CHANNELS:
        raise ValueError(f"expected {NUM_CHANNELS} channels, got {values.shape[1]}")

    active_channels = get_active_channels(ablation_id)
    all_channels = set(range(NUM_CHANNELS))
    inactive_channels = all_channels - active_channels

    ablated_values = values.copy()
    ablated_mask = mask.copy()

    for c in inactive_channels:
        ablated_mask[:, c] = False
        ablated_values[:, c] = np.nan

    return ablated_values, ablated_mask


def check_window_eligibility(
    mask: np.ndarray,
    ablation_id: str,
    min_obs_per_modality: int = 2,
) -> bool:
    """Check window eligibility: >= min_obs_per_modality valid observations for each active modality."""
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation_id '{ablation_id}'")

    active_modalities = ABLATION_CONFIGS[ablation_id]
    for mod in active_modalities:
        ch_indices = list(SENSOR_MODALITIES[mod])
        mod_mask = mask[:, ch_indices]
        # A day has a valid observation if any channel in that modality is valid
        valid_days = np.any(mod_mask, axis=1)
        if int(np.sum(valid_days)) < min_obs_per_modality:
            return False

    return True


# ---------------------------------------------------------------------------
# Structured Prediction Record
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PredictionRecord:
    """Structured decision record for closed-window evaluation."""
    sample_id: str
    lake_id: str
    decision_date: str
    window_start: str
    window_end: str
    model_id: str
    ablation_id: str
    score: float
    eligible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "lake_id": self.lake_id,
            "decision_date": self.decision_date,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "model_id": self.model_id,
            "ablation_id": self.ablation_id,
            "score": self.score,
            "eligible": self.eligible,
        }


def make_prediction_record(
    lake_id: str,
    decision_date: str,
    window_start: str,
    window_end: str,
    model_id: str,
    ablation_id: str,
    score: float,
    eligible: bool,
) -> PredictionRecord:
    """Construct PredictionRecord with deterministic cryptographic sample_id."""
    id_material = f"{lake_id}|{decision_date}|{model_id}|{ablation_id}"
    sample_id = hashlib.sha256(id_material.encode("utf-8")).hexdigest()

    return PredictionRecord(
        sample_id=sample_id,
        lake_id=lake_id,
        decision_date=decision_date,
        window_start=window_start,
        window_end=window_end,
        model_id=model_id,
        ablation_id=ablation_id,
        score=float(score),
        eligible=bool(eligible),
    )


# ---------------------------------------------------------------------------
# Paired Score Differences (Delta S)
# ---------------------------------------------------------------------------

def compute_paired_score_differences(
    model_records: Sequence[PredictionRecord],
    baseline_records: Sequence[PredictionRecord],
) -> List[Dict[str, Any]]:
    """Compute paired score difference Delta S = S_model - S_baseline over identical eligible windows.

    Raises ValueError if prediction supports (lake_id, decision_date, ablation_id) do not align.
    """
    model_dict: Dict[Tuple[str, str, str], PredictionRecord] = {}
    for r in model_records:
        key = (r.lake_id, r.decision_date, r.ablation_id)
        if key in model_dict:
            raise ValueError(f"Duplicate model record detected for key {key}")
        model_dict[key] = r

    baseline_dict: Dict[Tuple[str, str, str], PredictionRecord] = {}
    for r in baseline_records:
        key = (r.lake_id, r.decision_date, r.ablation_id)
        if key in baseline_dict:
            raise ValueError(f"Duplicate baseline record detected for key {key}")
        baseline_dict[key] = r

    if set(model_dict.keys()) != set(baseline_dict.keys()):
        diff_keys = set(model_dict.keys()) ^ set(baseline_dict.keys())
        raise ValueError(
            f"support mismatch: model and baseline prediction supports do not align ({len(diff_keys)} mismatched keys)"
        )

    paired_results: List[Dict[str, Any]] = []
    # Sort keys for deterministic output ordering
    sorted_keys = sorted(model_dict.keys())

    for key in sorted_keys:
        m = model_dict[key]
        b = baseline_dict[key]

        # Only evaluate paired difference over windows where BOTH are eligible
        if m.eligible and b.eligible:
            delta_s = float(m.score - b.score)
            paired_results.append({
                "lake_id": m.lake_id,
                "decision_date": m.decision_date,
                "ablation_id": m.ablation_id,
                "model_id": m.model_id,
                "baseline_id": b.model_id,
                "model_score": m.score,
                "baseline_score": b.score,
                "delta_s": delta_s,
            })

    return paired_results
