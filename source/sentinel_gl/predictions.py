"""Closed-window decision records and explicit sensor occlusion; raw-score contrasts are prohibited."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np

from .features import CHANNEL_MODALITIES, MultiModalPanel


# ---------------------------------------------------------------------------
# Sensor Groups and Ablation Lattice
# ---------------------------------------------------------------------------

# 2^N named configurations (for N=3 sensors: Optical, SAR, ERA5)
# The all-off cell is an explicit abstention.
ABLATION_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "all_off": (),
    "full": ("optical", "sar", "weather"),
    "opt_sar": ("optical", "sar"),
    "opt_era5": ("optical", "weather"),
    "sar_era5": ("sar", "weather"),
    "opt_only": ("optical",),
    "sar_only": ("sar",),
    "era5_only": ("weather",),
}


def get_active_channels(ablation_id: str, channel_names: Sequence[str]) -> Set[int]:
    """Resolve sensor ablation by declared channel name, never by fixed offsets."""
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation_id '{ablation_id}'. Must be one of {list(ABLATION_CONFIGS.keys())}"
        )
    modalities = ABLATION_CONFIGS[ablation_id]
    active: Set[int] = set()
    unknown = [name for name in channel_names if name not in CHANNEL_MODALITIES]
    if unknown:
        raise ValueError("unknown channel in ablation schema: " + ", ".join(unknown))
    for index, name in enumerate(channel_names):
        if CHANNEL_MODALITIES[name] in modalities:
            active.add(index)
    return active


def apply_sensor_ablation(
    values: np.ndarray,
    mask: np.ndarray,
    ablation_id: str,
    *, channel_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sensor ablation intervention by masking out inactive channels.

    Inactive channels have mask set to False and values set to NaN to strictly
    preserve explicit missingness invariants.
    """
    if values.shape != mask.shape:
        raise ValueError(f"values shape {values.shape} must match mask shape {mask.shape}")
    if values.ndim != 2 or values.shape[1] != len(channel_names):
        raise ValueError(f"expected {len(channel_names)} declared channels, got shape {values.shape}")

    active_channels = get_active_channels(ablation_id, channel_names)
    all_channels = set(range(len(channel_names)))
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
    min_obs_per_modality: int,
    *, channel_names: Sequence[str],
) -> bool:
    """Apply a caller-declared support rule to the active channel modalities."""
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation_id '{ablation_id}'")

    if type(min_obs_per_modality) is not int or min_obs_per_modality < 1:
        raise ValueError("an explicit positive support threshold is required")
    if np.asarray(mask).ndim != 2 or np.asarray(mask).shape[1] != len(channel_names):
        raise ValueError("mask channel dimension differs from its declared schema")
    active_modalities = ABLATION_CONFIGS[ablation_id]
    if not active_modalities:
        return False
    for mod in active_modalities:
        ch_indices = [index for index, name in enumerate(channel_names)
                      if CHANNEL_MODALITIES.get(name) == mod]
        if not ch_indices:
            return False
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
    score: Optional[float]
    eligible: bool

    def __post_init__(self):
        from datetime import date
        if type(self.eligible) is not bool:
            raise ValueError("eligibility must be boolean")
        if self.eligible:
            if type(self.score) not in (int, float) or not np.isfinite(self.score):
                raise ValueError("eligible score must be finite numeric data")
        elif self.score is not None:
            raise ValueError("ineligible score must remain null")
        start, end, decision = [date.fromisoformat(x) for x in (self.window_start, self.window_end, self.decision_date)]
        if start > end or end > decision:
            raise ValueError("window must be closed before decision")

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
            "status": "ESTIMATED" if self.eligible else "NOT_ESTIMABLE",
            "reason": None if self.eligible else "INELIGIBLE_WINDOW",
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
    id_material += f"|{window_start}|{window_end}"
    sample_id = hashlib.sha256(id_material.encode("utf-8")).hexdigest()

    return PredictionRecord(
        sample_id=sample_id,
        lake_id=lake_id,
        decision_date=decision_date,
        window_start=window_start,
        window_end=window_end,
        model_id=model_id,
        ablation_id=ablation_id,
        score=score,
        eligible=eligible,
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
    raise ValueError("raw cross-method score differences are not a performance estimand")
