"""Analytical fixture fitted states; observational fitting and loading remain blocked."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from .features import MultiModalPanel, FeatureNormalizer, FitScope, observed_array
from .integrity import digest, loads_strict


@dataclasses.dataclass(frozen=True)
class FittedNormalizer:
    """Immutable fitted normalization transform with full provenance."""
    channel_names: Tuple[str, ...]
    channel_schema_id: str
    channel_schema_hash: str
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    constant_channels: Tuple[int, ...]
    training_lake_ids: Tuple[str, ...]
    temporal_cutoff: Optional[str]
    input_hashes: Mapping[str, str]
    state_hash: str

    def __post_init__(self):
        from .contracts import instance_ids
        instance_ids(self.training_lake_ids)
        if any(type(v) not in (int, float) for v in (*self.mean, *self.scale)):
            raise ValueError("fitted numeric state rejects type confusion")
        n_channels = len(self.channel_names)
        if (not n_channels or len(set(self.channel_names)) != n_channels or
                any(type(c) is not int or not 0 <= c < n_channels for c in self.constant_channels) or
                len(set(self.constant_channels)) != len(self.constant_channels)):
            raise ValueError("invalid constant-channel membership")
        if len(self.mean) != n_channels or len(self.scale) != n_channels:
            raise ValueError("mean and scale must match the declared channel schema")
        if not self.channel_schema_id or not self.channel_schema_hash:
            raise ValueError("fitted transform requires a versioned channel schema identity")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("nonfinite fitted transform")
        if any(s <= 0 for s in self.scale):
            raise ValueError("all scales must be strictly positive")

    @classmethod
    def fit_fixture(
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

        if any(p.origin != 'fixture' for p in panels):
            raise ValueError("fixture fitting requires explicit fixture panels")
        schema_id = valid_panels[0].channel_schema_id
        schema_hash = valid_panels[0].channel_schema_hash
        channel_names = valid_panels[0].channel_names
        if any((p.channel_schema_id, p.channel_schema_hash, p.channel_names) !=
               (schema_id, schema_hash, channel_names) for p in valid_panels):
            raise ValueError("fit panels have different versioned channel schemas")
        # Use the single numerically stable core operator after explicit fixture partitioning.
        arrays = {}
        for lid in sorted({p.lake_id for p in valid_panels}):
            selected = [p for p in valid_panels if p.lake_id == lid]
            arrays[lid] = (np.concatenate([p.values for p in selected]), np.concatenate([p.mask for p in selected]))
        core = FeatureNormalizer(FitScope(tuple(sorted(arrays)), ('__fixture_holdout__',))).fit(arrays)
        mean_list, scale_list = list(core.state['mean']), list(core.state['scale'])
        constant_channels = list(core.state['constant_channels'])
        state_repr = {
            "channel_names": list(channel_names),
            "channel_schema_id": schema_id,
            "channel_schema_hash": schema_hash,
            "mean": mean_list,
            "scale": scale_list,
            "constant_channels": constant_channels,
            "training_lake_ids": sorted(train_lakes_set),
            "temporal_cutoff": temporal_cutoff,
            "input_hashes": collected_hashes,
        }
        state_hash = digest(state_repr)

        return cls(
            channel_names=tuple(channel_names),
            channel_schema_id=schema_id,
            channel_schema_hash=schema_hash,
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
        val_arr, mask_arr = observed_array(values, mask)

        if val_arr.shape[-1] != len(self.channel_names):
            raise ValueError(f"Channel dimension must be {len(self.channel_names)}, got {val_arr.shape[-1]}")

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
            "channel_schema_id": self.channel_schema_id,
            "channel_schema_hash": self.channel_schema_hash,
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
    def from_fixture_dict(cls, d: Mapping[str, Any]) -> FittedNormalizer:
        expected = {'channel_names', 'channel_schema_id', 'channel_schema_hash', 'mean', 'scale',
                    'constant_channels', 'training_lake_ids', 'temporal_cutoff', 'input_hashes', 'state_hash'}
        if set(d) != expected or d['state_hash'] != digest({k: v for k, v in d.items() if k != 'state_hash'}):
            raise ValueError("fitted state hash/schema mismatch")
        return cls(
            channel_names=tuple(d["channel_names"]),
            channel_schema_id=d["channel_schema_id"],
            channel_schema_hash=d["channel_schema_hash"],
            mean=tuple(d["mean"]),
            scale=tuple(d["scale"]),
            constant_channels=tuple(d["constant_channels"]),
            training_lake_ids=tuple(d["training_lake_ids"]),
            temporal_cutoff=d.get("temporal_cutoff"),
            input_hashes=MappingProxyType(dict(d.get("input_hashes", {}))),
            state_hash=d["state_hash"],
        )

    @classmethod
    def from_fixture_json(cls, s: str) -> FittedNormalizer:
        return cls.from_fixture_dict(loads_strict(s))

    @classmethod
    def fit(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research normalizer fitting")

    @classmethod
    def from_dict(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research fitted-state loading")

    @classmethod
    def from_json(cls, *args, **kwargs):
        from .admission import blocked
        blocked("research fitted-state loading")
