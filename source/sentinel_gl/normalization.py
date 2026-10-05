"""Fitted-state and normalization contract for Sentinel-GL multi-modal features.

Computes feature mean and standard deviation strictly over observed elements
on training lakes only. Constant channels use unit scale. Transforms serialize
as immutable JSON/dict artifacts with verifiable provenance hashes.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from .features import FEATURE_CHANNELS, NUM_CHANNELS, MultiModalPanel


@dataclasses.dataclass(frozen=True)
class FittedNormalizer:
    """Immutable fitted normalization transform with full provenance."""
    channel_names: Tuple[str, ...]
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    constant_channels: Tuple[int, ...]
    training_lake_ids: Tuple[str, ...]
    temporal_cutoff: Optional[str]
    input_hashes: Mapping[str, str]
    state_hash: str

    def __post_init__(self):
        if len(self.mean) != NUM_CHANNELS or len(self.scale) != NUM_CHANNELS:
            raise ValueError(f"mean and scale must match channel count {NUM_CHANNELS}")
        if any(s <= 0 for s in self.scale):
            raise ValueError("all scales must be strictly positive")

    @classmethod
    def fit(
        cls,
        panels: Sequence[MultiModalPanel],
        training_lake_ids: Sequence[str],
        temporal_cutoff: Optional[str] = None,
    ) -> FittedNormalizer:
        """Fit normalization parameters strictly over training lakes prior to cutoff."""
        train_lakes_set = set(training_lake_ids)
        if not train_lakes_set:
            raise ValueError("training_lake_ids must be nonempty")

        valid_panels: List[MultiModalPanel] = []
        collected_hashes: Dict[str, str] = {}

        for p in panels:
            # Strictly isolate to training lakes
            if p.lake_id not in train_lakes_set:
                continue
            # Strictly isolate to pre-cutoff observation windows
            if temporal_cutoff is not None and p.end_date > temporal_cutoff:
                continue

            valid_panels.append(p)
            collected_hashes.update(p.provenance_hashes)

        if not valid_panels:
            raise ValueError(
                "No valid training panels available to fit normalizer (check lake IDs and cutoff)"
            )

        mean_list: List[float] = []
        scale_list: List[float] = []
        constant_channels: List[int] = []

        for c_idx in range(NUM_CHANNELS):
            obs_vals: List[np.ndarray] = []
            for p in valid_panels:
                col_vals = p.values[:, c_idx]
                col_mask = p.mask[:, c_idx]
                valid_entries = col_vals[col_mask]
                if valid_entries.size > 0:
                    obs_vals.append(valid_entries)

            if not obs_vals:
                raise ValueError(
                    f"Channel '{FEATURE_CHANNELS[c_idx]}' has zero observed elements in training set"
                )

            stacked = np.concatenate(obs_vals)
            mu = float(np.mean(stacked))
            sigma = float(np.std(stacked))

            # Constant channel handling: unit scale
            if sigma == 0.0 or not np.isfinite(sigma):
                scale = 1.0
                constant_channels.append(c_idx)
            else:
                scale = sigma

            mean_list.append(mu)
            scale_list.append(scale)

        state_repr = {
            "channel_names": list(FEATURE_CHANNELS),
            "mean": mean_list,
            "scale": scale_list,
            "constant_channels": constant_channels,
            "training_lake_ids": sorted(train_lakes_set),
            "temporal_cutoff": temporal_cutoff,
            "input_hashes": collected_hashes,
        }
        state_hash = hashlib.sha256(json.dumps(state_repr, sort_keys=True).encode("utf-8")).hexdigest()

        return cls(
            channel_names=tuple(FEATURE_CHANNELS),
            mean=tuple(mean_list),
            scale=tuple(scale_list),
            constant_channels=tuple(constant_channels),
            training_lake_ids=tuple(sorted(train_lakes_set)),
            temporal_cutoff=temporal_cutoff,
            input_hashes=MappingProxyType(collected_hashes),
            state_hash=state_hash,
        )

    def transform(
        self,
        values: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply z-score normalization: missing elements become 0.0 with mask=False."""
        val_arr = np.asarray(values, dtype=np.float64)
        mask_arr = np.asarray(mask, dtype=np.bool_)

        if val_arr.shape[-1] != NUM_CHANNELS:
            raise ValueError(f"Channel dimension must be {NUM_CHANNELS}, got {val_arr.shape[-1]}")

        mu = np.array(self.mean, dtype=np.float64)
        sigma = np.array(self.scale, dtype=np.float64)

        z = (val_arr - mu) / sigma
        # Unobserved elements are set to 0.0 for masked autoencoder consumption
        z_norm = np.where(mask_arr, z, 0.0).astype(np.float32)

        return z_norm, mask_arr.copy()

    def inverse_transform(
        self,
        values_norm: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reverse normalization: unobserved elements become NaN."""
        z = np.asarray(values_norm, dtype=np.float64)
        mu = np.array(self.mean, dtype=np.float64)
        sigma = np.array(self.scale, dtype=np.float64)

        orig = z * sigma + mu
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.bool_)
            orig = np.where(mask_arr, orig, np.nan)

        return orig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "constant_channels": list(self.constant_channels),
            "training_lake_ids": list(self.training_lake_ids),
            "temporal_cutoff": self.temporal_cutoff,
            "input_hashes": dict(self.input_hashes),
            "state_hash": self.state_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FittedNormalizer:
        return cls(
            channel_names=tuple(d["channel_names"]),
            mean=tuple(d["mean"]),
            scale=tuple(d["scale"]),
            constant_channels=tuple(d["constant_channels"]),
            training_lake_ids=tuple(d["training_lake_ids"]),
            temporal_cutoff=d.get("temporal_cutoff"),
            input_hashes=MappingProxyType(dict(d.get("input_hashes", {}))),
            state_hash=d["state_hash"],
        )

    @classmethod
    def from_json(cls, s: str) -> FittedNormalizer:
        return cls.from_dict(json.loads(s))
