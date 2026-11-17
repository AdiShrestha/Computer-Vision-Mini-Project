"""Quarantined, local WP06 optimization diagnostic on retained panel attempt004.

This entry point is deliberately separate from the research training runner.
Its outputs are development-diagnostic optimization evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .multimodal import RaggedPanelDataset, collate_ragged_groups


POLICY = "ATTEMPT004_CHECKERBOARD_QUADRANT_POOL_V1"
TRANSFORM_VERSION = "WP06_FIT_ONLY_CHANNEL_STANDARDIZATION_V1"
CHANNEL_ORDER = ("B03", "B08", "B11", "NDWI", "MNDWI",
                 "SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR")
REFERENCE = datetime(2023, 9, 1, tzinfo=timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FittedTransform:
    """Immutable, fit-support-only state; monitor pixels are not an input."""
    version: str
    algorithm: str
    channel_order: tuple[str, ...]
    means: tuple[float | None, ...]
    scales: tuple[float | None, ...]
    counts: tuple[int, ...]
    fit_role: str
    fit_parity: int
    panel_membership_sha256: str
    fit_membership_sha256: str
    fit_source_values_sha256: str
    state_sha256: str

    def payload(self) -> dict[str, Any]:
        return {"version":self.version, "algorithm":self.algorithm,
            "channel_order":list(self.channel_order), "means":list(self.means),
            "scales":list(self.scales), "counts":list(self.counts),
            "fit_role":self.fit_role, "fit_parity":self.fit_parity,
            "panel_membership_sha256":self.panel_membership_sha256,
            "fit_membership_sha256":self.fit_membership_sha256,
            "fit_source_values_sha256":self.fit_source_values_sha256}

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "state_sha256":self.state_sha256}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FittedTransform":
        expected = {"version", "algorithm", "channel_order", "means", "scales", "counts",
                    "fit_role", "fit_parity", "panel_membership_sha256", "fit_membership_sha256",
                    "fit_source_values_sha256", "state_sha256"}
        if set(value) != expected:
            raise ValueError("fitted transform fields differ from the versioned contract")
        if value["version"] != TRANSFORM_VERSION or value["fit_role"] != "development-diagnostic-fit" or value["fit_parity"] != 0:
            raise ValueError("fitted transform role/version is not the declared parity-0 diagnostic fit")
        if tuple(value["channel_order"]) != CHANNEL_ORDER or not (len(value["means"]) == len(value["scales"]) == len(value["counts"]) == len(CHANNEL_ORDER)):
            raise ValueError("fitted transform channel membership or dimensions mismatch")
        if _canonical_sha256({k:v for k,v in value.items() if k != "state_sha256"}) != value["state_sha256"]:
            raise ValueError("fitted transform SHA-256 does not match its typed state")
        for mean, scale, count in zip(value["means"], value["scales"], value["counts"]):
            if type(count) is not int or count < 0:
                raise ValueError("fitted transform support counts must be nonnegative integers")
            if count == 0:
                if mean is not None or scale is not None:
                    raise ValueError("empty fit support must retain null mean and scale")
            elif (not isinstance(mean, (int, float)) or not math.isfinite(mean) or
                  not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
                raise ValueError("fitted transform moments must be finite with positive scale")
        return cls(value["version"], value["algorithm"], tuple(value["channel_order"]),
            tuple(value["means"]), tuple(value["scales"]), tuple(value["counts"]),
            value["fit_role"], value["fit_parity"], value["panel_membership_sha256"],
            value["fit_membership_sha256"], value["fit_source_values_sha256"], value["state_sha256"])


class DiagnosticMaskedMAE(nn.Module):
    """Two modality-specific projections and a masked, observed-only encoder."""
    def __init__(self):
        super().__init__()
        d = 128
        self.optical_projection = nn.Linear(10, d)
        self.sar_projection = nn.Linear(4, d)
        self.modality = nn.Embedding(2, d)
        self.time_projection = nn.Linear(4, d, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d, 4, 512, dropout=0.0, batch_first=True,
                                       norm_first=True, activation="gelu") for _ in range(4)])
        self.encoder_norm = nn.LayerNorm(d)
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d, 4, 256, dropout=0.0, batch_first=True,
                                       norm_first=True, activation="gelu") for _ in range(2)])
        self.decoder_norm = nn.LayerNorm(d)
        self.optical_head = nn.Linear(d, 5)
        self.sar_head = nn.Linear(d, 2)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, values: torch.Tensor, observed: torch.Tensor,
                modality: torch.Tensor, time_features: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # values/observed are B,T,7 in [optical 0:5, SAR 5:7] order.
        optical = torch.cat((values[..., :5], observed[..., :5].float()), dim=-1)
        sar = torch.cat((values[..., 5:], observed[..., 5:].float()), dim=-1)
        projected = torch.zeros((*values.shape[:2], 128), dtype=values.dtype, device=values.device)
        is_optical = modality == 0
        is_sar = modality == 1
        projected[is_optical] = self.optical_projection(optical[is_optical])
        projected[is_sar] = self.sar_projection(sar[is_sar])
        projected = projected + self.modality(modality) + self.time_projection(time_features)
        # Target values never enter the encoder. False key positions are ignored.
        target_token = target.any(dim=-1)
        key_padding = target_token | ~observed.any(dim=-1)
        encoded = projected
        for layer in self.encoder:
            encoded = layer(encoded, src_key_padding_mask=key_padding)
        encoded = self.encoder_norm(encoded)
        decoded = torch.where(target_token[..., None], self.mask_token.expand_as(encoded), encoded)
        for layer in self.decoder:
            decoded = layer(decoded)
        decoded = self.decoder_norm(decoded)
        prediction = torch.zeros_like(values)
        prediction[is_optical, :5] = self.optical_head(decoded[is_optical])
        prediction[is_sar, 5:] = self.sar_head(decoded[is_sar])
        return prediction


def masked_mse(prediction: torch.Tensor, target_values: torch.Tensor,
               target_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Independent supported-channel mean squared error; empty support is null."""
    count = int(target_mask.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    return ((prediction[target_mask] - target_values[target_mask]) ** 2).sum() / count, count


def _partition_mask(h: int, w: int, parity: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows // 16 + cols // 16) % 2 == parity)


def _time_features(group: dict) -> np.ndarray:
    acquisition = group["metadata"].get("acquisition", {})
    if acquisition.get("timezone_status") != "EXPLICIT_UTC" or not acquisition.get("at"):
        return np.zeros(4, dtype=np.float32)  # unknown SAR timezone remains unencoded
    stamp = datetime.fromisoformat(acquisition["at"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("explicit UTC acquisition unexpectedly lacks timezone")
    days = (stamp.astimezone(timezone.utc) - REFERENCE).total_seconds() / 86400.0
    year_days = 366.0 if stamp.year % 4 == 0 and (stamp.year % 100 != 0 or stamp.year % 400 == 0) else 365.0
    phase = 2 * math.pi * ((stamp.timetuple().tm_yday - 1) +
                           (stamp.hour * 3600 + stamp.minute * 60 + stamp.second) / 86400.0) / year_days
    return np.asarray([days / 365.25, math.sin(phase), math.cos(phase), 1.0], dtype=np.float32)


def _panel_membership_identity(samples: list[dict]) -> list[dict]:
    identity = []
    for sample in samples:
        for group in sample["groups"]:
            identity.append({"panel_id":sample["panel_id"], "site_id":sample["site_id"],
                "group_id":group["metadata"]["group_id"], "sensor":group["metadata"]["sensor"],
                "channel_ids":list(group["channel_ids"]), "shape":list(group["values"].shape),
                "support_id":group["metadata"].get("support_id")})
    return identity


def _load_view(root: Path, parity: int) -> dict:
    if parity not in (0, 1):
        raise ValueError("diagnostic tile parity must be zero (fit) or one (monitor)")
    dataset = RaggedPanelDataset(root, "data/derived/wp05_wp06_multimodal_attempt_004/panel.v1.json",
                                 mode="development-diagnostic")
    samples = [dataset[i] for i in range(len(dataset))]
    targets_by_sample: list[list[np.ndarray]] = []
    partition_by_group: list[list[np.ndarray]] = []
    for sample in samples:
        group_targets, group_partitions = [], []
        for gi, group in enumerate(sample["groups"]):
            c, h, w = group["values"].shape
            member = _partition_mask(h, w, parity)
            quadrant = gi % 4
            rows, cols = np.indices((h, w))
            qmask = ((rows >= (h // 2 if quadrant >= 2 else 0)) &
                     (rows < (h if quadrant >= 2 else h // 2)) &
                     (cols >= (w // 2 if quadrant % 2 else 0)) &
                     (cols < (w if quadrant % 2 else w // 2)))
            target = np.broadcast_to((member & qmask)[None, :, :], (c, h, w)).copy()
            target &= group["observed_valid"]
            group_targets.append(target)
            group_partitions.append(member)
        targets_by_sample.append(group_targets)
        partition_by_group.append(group_partitions)
    collated = collate_ragged_groups(samples, target_masks=targets_by_sample)
    input_paths = (root/"data/derived/wp05_wp06_multimodal_attempt_004/panel.v1.json",
                   root/"data/derived/wp05_wp06_multimodal_attempt_004/panel_arrays.npz")
    return {"role":"fit" if parity == 0 else "monitor", "parity":parity,
        "samples":samples, "targets_by_sample":targets_by_sample,
        "partition_by_group":partition_by_group, "collated":collated,
        "channel_order":CHANNEL_ORDER,
        "panel_membership_identity":_panel_membership_identity(samples),
        "panel_membership_sha256":_canonical_sha256(_panel_membership_identity(samples)),
        "input_sha256":{p.relative_to(root).as_posix():sha256_file(p) for p in input_paths}}


def fit_transform(view: dict) -> FittedTransform:
    """Fit typed channel moments from parity-0 valid context pixels only."""
    if view.get("role") != "fit" or view.get("parity") != 0:
        raise ValueError("fit_transform accepts only the declared parity-0 fitting view")
    if tuple(view.get("channel_order", ())) != CHANNEL_ORDER:
        raise ValueError("fit view channel membership differs from the transform contract")
    actual_identity=_panel_membership_identity(view.get("samples",[]))
    if actual_identity!=view.get("panel_membership_identity") or view.get("panel_membership_sha256") != _canonical_sha256(actual_identity):
        raise ValueError("fit view panel membership identity hash mismatch")
    means = np.zeros(len(CHANNEL_ORDER), dtype=np.float64)
    m2 = np.zeros(len(CHANNEL_ORDER), dtype=np.float64)
    counts = np.zeros(len(CHANNEL_ORDER), dtype=np.int64)
    fit_membership_hash = hashlib.sha256()
    fit_values_hash = hashlib.sha256()
    offsets = {"SENTINEL_2_OPTICAL":0, "SENTINEL_1_IW_GRD":5}
    for si, sample in enumerate(view["samples"]):
        for gi, group in enumerate(sample["groups"]):
            offset=offsets[group["metadata"]["sensor"]]
            target=view["targets_by_sample"][si][gi]
            fit_pixels=view["partition_by_group"][si][gi] & ~target.any(axis=0)
            fit_membership_hash.update(_canonical_sha256({"site_id":sample["site_id"],
                "group_id":group["metadata"]["group_id"],"parity":0,
                "target_exclusion":"all targeted optical channels at selected target pixel"}).encode())
            fit_membership_hash.update(np.ascontiguousarray(fit_pixels).tobytes())
            for ci in range(group["values"].shape[0]):
                channel=offset+ci
                observed=group["observed_valid"][ci] & fit_pixels
                values=np.asarray(group["values"][ci][observed],dtype=np.float64)
                fit_membership_hash.update(np.ascontiguousarray(observed).tobytes())
                fit_values_hash.update(np.asarray([sample["site_id"],group["metadata"]["group_id"],
                    CHANNEL_ORDER[channel]],dtype="U").tobytes())
                fit_values_hash.update(np.ascontiguousarray(group["values"][ci][observed],dtype=np.float32).tobytes())
                n=int(values.size)
                if n:
                    batch_mean=float(values.mean())
                    batch_m2=float(np.square(values-batch_mean).sum())
                    old=int(counts[channel]); total=old+n
                    delta=batch_mean-means[channel]
                    m2[channel]+=batch_m2+(delta*delta*old*n/total)
                    means[channel]+=delta*n/total; counts[channel]=total
    mean_values=[]; scale_values=[]
    for i,count in enumerate(counts):
        if count:
            mean_values.append(float(means[i]))
            scale_values.append(max(float(math.sqrt(m2[i]/count)),1e-6))
        else:
            mean_values.append(None); scale_values.append(None)
    payload={"version":TRANSFORM_VERSION,
        "algorithm":"float64 parallel-Welford population mean/std over valid parity-0 context pixels; scale floor 1e-6",
        "channel_order":list(CHANNEL_ORDER),"means":mean_values,"scales":scale_values,
        "counts":[int(x) for x in counts],"fit_role":"development-diagnostic-fit","fit_parity":0,
        "panel_membership_sha256":view["panel_membership_sha256"],
        "fit_membership_sha256":fit_membership_hash.hexdigest(),
        "fit_source_values_sha256":fit_values_hash.hexdigest()}
    transform=FittedTransform.from_dict({**payload,"state_sha256":_canonical_sha256(payload)})
    return transform


def apply_channel_values(values: np.ndarray, observed: np.ndarray,
                         transform: FittedTransform | dict, *, channel_offset: int = 0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen per-channel moments without estimating any new state."""
    state=FittedTransform.from_dict(transform.to_dict() if isinstance(transform,FittedTransform) else transform)
    raw=np.asarray(values,dtype=np.float64); mask=np.asarray(observed,dtype=bool)
    if (raw.shape!=mask.shape or raw.ndim<1 or type(channel_offset) is not int or
        channel_offset < 0 or channel_offset + raw.shape[0] > len(state.channel_order)):
        raise ValueError("transform input value/mask/channel shape mismatch")
    output=np.full(raw.shape,np.nan,dtype=np.float64); output_valid=mask & np.isfinite(raw)
    for local_channel,(mean,scale) in enumerate(zip(
        state.means[channel_offset:channel_offset+raw.shape[0]],
        state.scales[channel_offset:channel_offset+raw.shape[0]])):
        if mean is None or scale is None:
            output_valid[local_channel]=False
            continue
        valid=output_valid[local_channel]
        output[local_channel][valid]=(raw[local_channel][valid]-mean)/scale
    return output,output_valid


def apply_transform(view: dict, transform: FittedTransform | dict, *, application_role: str) -> tuple[list[dict], dict]:
    """Apply one verified frozen transform to fit or monitor; never fit here."""
    state=FittedTransform.from_dict(transform.to_dict() if isinstance(transform,FittedTransform) else transform)
    expected_role={"fit_application":"fit","monitor_application":"monitor"}.get(application_role)
    if expected_role is None or view.get("role")!=expected_role:
        raise ValueError("transform application role does not match fit/monitor view")
    expected_parity=0 if expected_role=="fit" else 1
    if view.get("parity")!=expected_parity:
        raise ValueError("transform application parity does not match the declared role")
    actual_identity=_panel_membership_identity(view.get("samples",[]))
    if actual_identity!=view.get("panel_membership_identity") or _canonical_sha256(actual_identity)!=view.get("panel_membership_sha256"):
        raise ValueError("view panel membership identity hash mismatch")
    if view.get("panel_membership_sha256")!=state.panel_membership_sha256:
        raise ValueError("transform source membership identity mismatch")
    if tuple(view.get("channel_order",()))!=state.channel_order:
        raise ValueError("transform channel membership mismatch")
    if state.state_sha256 != _canonical_sha256(state.payload()):
        raise ValueError("fitted transform changed after freezing")
    collated=view["collated"]; samples=view["samples"]
    offsets={"SENTINEL_2_OPTICAL":0,"SENTINEL_1_IW_GRD":5}
    prepared = []
    for si, sample in enumerate(samples):
        tokens, valids, targets, target_values, modalities, times, support = [], [], [], [], [], [], []
        for gi, group in enumerate(sample["groups"]):
            offset = offsets[group["metadata"]["sensor"]]
            source = group["values"]
            valid = group["observed_valid"]
            p = collated["padding_mask"].shape[-1]
            h, w = source.shape[-2:]
            context_visible = collated["context_visible"][si, gi, :source.shape[0], :h*w].reshape(source.shape)
            requested = collated["target_mask"][si, gi, :source.shape[0], :h*w].reshape(source.shape)
            normalized_source, transform_valid = apply_channel_values(
                source,valid,state,channel_offset=offset)
            member = view["partition_by_group"][si][gi]
            rows, cols = np.indices((h, w))
            for q in range(4):
                qmask_all = ((rows >= (h // 2 if q >= 2 else 0)) &
                             (rows < (h if q >= 2 else h // 2)) &
                             (cols >= (w // 2 if q % 2 else 0)) &
                             (cols < (w if q % 2 else w // 2)))
                spatial = qmask_all & member
                values7 = np.zeros(7, dtype=np.float32)
                valid7 = np.zeros(7, dtype=bool)
                target7 = np.zeros(7, dtype=bool)
                truth7 = np.zeros(7, dtype=np.float32)
                denominators = np.zeros(source.shape[0], dtype=int)
                untransformed = np.zeros(source.shape[0], dtype=bool)
                for ci in range(source.shape[0]):
                    channel = offset + ci
                    mean=state.means[channel]; scale=state.scales[channel]
                    if mean is None or scale is None:
                        untransformed[ci]=bool((requested[ci] | valid[ci]).any() & spatial.any())
                        continue
                    target_pixels = requested[ci] & transform_valid[ci] & spatial
                    visible_pixels = context_visible[ci] & transform_valid[ci] & spatial
                    if target_pixels.any():
                        target7[channel] = True
                        denominators[ci] = int(target_pixels.sum())
                        truth7[channel] = float(np.mean(normalized_source[ci][target_pixels]))
                    elif visible_pixels.any():
                        valid7[channel] = True
                        denominators[ci] = int(visible_pixels.sum())
                        values7[channel] = float(np.mean(normalized_source[ci][visible_pixels]))
                tokens.append(values7); valids.append(valid7); targets.append(target7); target_values.append(truth7)
                modalities.append(0 if offset == 0 else 1); times.append(_time_features(group))
                support.append({"group_id": group["metadata"]["group_id"], "quadrant": q,
                    "selected_target_quadrant": q == gi % 4,
                    "per_channel_source_pixel_denominators": denominators.tolist(),
                    "unavailable_fit_transform_channels": untransformed.tolist(),
                    "partition_pixel_count": int(spatial.sum()),
                    "source_shape": [h, w], "pooling": "VALID_PIXEL_MEAN_AFTER_DEPENDENCY_CLOSURE"})
        prepared.append({"values": np.asarray(tokens, dtype=np.float32),
            "observed": np.asarray(valids, dtype=bool), "target": np.asarray(targets, dtype=bool),
            "target_values": np.asarray(target_values, dtype=np.float32),
            "modality": np.asarray(modalities, dtype=np.int64),
            "time": np.asarray(times, dtype=np.float32), "support": support,
            "site_id": sample["site_id"], "panel_id": sample["panel_id"]})
    receipt={"application_role":application_role,"view_role":view["role"],"view_parity":view["parity"],
        "fit_transform_fitted_here":False,"fit_transform_sha256":state.state_sha256,
        "fit_source_values_sha256":state.fit_source_values_sha256,
        "fit_membership_sha256":state.fit_membership_sha256,
        "channel_order":list(state.channel_order),"panel_membership_sha256":state.panel_membership_sha256,
        "input_sha256":view["input_sha256"],"collate_status":collated["status"],
        "context_policy_version":collated["context_policy_version"],
        "target_count":collated["target_count"],"sample_metadata":collated["sample_metadata"],
        "token_support":[x["support"] for x in prepared]}
    return prepared, receipt


def _as_tensors(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(sample[key], device=device).unsqueeze(0) for key in
            ("values", "observed", "target", "target_values", "modality", "time")}


def _loss(model: DiagnosticMaskedMAE, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int, torch.Tensor]:
    prediction = model(batch["values"], batch["observed"], batch["modality"], batch["time"], batch["target"])
    loss, count = masked_mse(prediction, batch["target_values"], batch["target"])
    return loss, count, prediction


class CheckpointSelector:
    """Finite, denominator-supported monitor selection with explicit epoch rules."""
    def __init__(self, *, minimum_optimization_epochs: int,
                 earliest_checkpoint_epoch: int, patience: int, tolerance: float):
        if (type(minimum_optimization_epochs) is not int or minimum_optimization_epochs < 1 or
            type(earliest_checkpoint_epoch) is not int or earliest_checkpoint_epoch < minimum_optimization_epochs or
            type(patience) is not int or patience < 1 or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("checkpoint rule requires positive budgets/patience, earliest>=minimum and finite nonnegative tolerance")
        self.minimum_optimization_epochs=minimum_optimization_epochs
        self.earliest_checkpoint_epoch=earliest_checkpoint_epoch
        self.patience=patience; self.tolerance=float(tolerance)
        self.best_loss: float | None=None; self.best_epoch: int | None=None
        self.nonimprovement_count=0; self.eligible_count=0

    def observe(self, epoch: int, loss: float | None, denominator: int) -> dict:
        if type(epoch) is not int or epoch < 1 or type(denominator) is not int or denominator < 0:
            raise ValueError("checkpoint observation epoch/denominator is invalid")
        if epoch < self.earliest_checkpoint_epoch:
            return {"eligible":False,"selected":False,"reason":"BEFORE_EARLIEST_CHECKPOINT_EPOCH",
                    "nonimprovement_count":self.nonimprovement_count}
        if denominator == 0 or loss is None or not math.isfinite(float(loss)):
            return {"eligible":False,"selected":False,
                    "reason":"NO_FINITE_POSITIVE_DENOMINATOR_MONITOR_ESTIMATE",
                    "nonimprovement_count":self.nonimprovement_count}
        self.eligible_count+=1
        value=float(loss)
        if self.best_loss is None or value < self.best_loss-self.tolerance:
            self.best_loss=value; self.best_epoch=epoch; self.nonimprovement_count=0
            reason="FIRST_ELIGIBLE_FINITE_ESTIMATE" if self.eligible_count==1 else "IMPROVEMENT_BEYOND_TOLERANCE"
            selected=True
        else:
            # Exact/within-tolerance ties retain the earlier checkpoint and count
            # as a non-improvement. Missing/nonfinite estimates never consume patience.
            self.nonimprovement_count+=1; selected=False
            reason="TIE_WITHIN_TOLERANCE_RETAIN_EARLIER" if abs(value-self.best_loss)<=self.tolerance else "NO_IMPROVEMENT"
        stop=(epoch>=self.minimum_optimization_epochs and
              self.nonimprovement_count>=self.patience)
        return {"eligible":True,"selected":selected,"reason":reason,
            "nonimprovement_count":self.nonimprovement_count,"stop":stop}

    @property
    def selected_status(self) -> str:
        return "SELECTED_ELIGIBLE_FINITE_MONITOR_CHECKPOINT" if self.best_epoch is not None else "UNAVAILABLE_NO_ELIGIBLE_FINITE_MONITOR_ESTIMATE"


def _checkpoint_metadata(transform: FittedTransform, role: str, epoch: int | None) -> dict:
    return {"checkpoint_schema":"WP06_DIAGNOSTIC_CHECKPOINT_V2","role":role,"epoch":epoch,
        "fit_transform_sha256":transform.state_sha256,
        "fit_source_values_sha256":transform.fit_source_values_sha256,
        "fit_membership_sha256":transform.fit_membership_sha256,
        "panel_membership_sha256":transform.panel_membership_sha256,
        "channel_order":list(transform.channel_order),"transform_version":transform.version}


def _save_checkpoint(path: Path, model: DiagnosticMaskedMAE,
                     transform: FittedTransform, role: str, epoch: int | None) -> None:
    torch.save({"metadata":_checkpoint_metadata(transform,role,epoch),
        "model_state":model.state_dict()},path)


def _load_checkpoint(path: Path, model: DiagnosticMaskedMAE,
                     transform: FittedTransform) -> dict:
    bundle=torch.load(path,map_location=next(model.parameters()).device,weights_only=True)
    expected={"checkpoint_schema":"WP06_DIAGNOSTIC_CHECKPOINT_V2",
        "fit_transform_sha256":transform.state_sha256,
        "fit_source_values_sha256":transform.fit_source_values_sha256,
        "fit_membership_sha256":transform.fit_membership_sha256,
        "panel_membership_sha256":transform.panel_membership_sha256,
        "channel_order":list(transform.channel_order),"transform_version":transform.version}
    metadata=bundle.get("metadata",{})
    if any(metadata.get(key)!=value for key,value in expected.items()):
        raise ValueError("checkpoint transform/source/channel/membership identity mismatch")
    state=bundle.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no model state mapping")
    expected_state=model.state_dict()
    if set(state) != set(expected_state):
        missing=sorted(set(expected_state)-set(state)); extra=sorted(set(state)-set(expected_state))
        raise ValueError(f"checkpoint model keys differ (missing={missing}, extra={extra})")
    for key,expected_tensor in expected_state.items():
        value=state[key]
        if not isinstance(value,torch.Tensor):
            raise ValueError(f"checkpoint model value is not a tensor: {key}")
        if value.shape != expected_tensor.shape:
            raise ValueError(f"checkpoint model tensor shape mismatch: {key}")
        if value.dtype != expected_tensor.dtype:
            raise ValueError(f"checkpoint model tensor dtype mismatch: {key}")
    model.load_state_dict(dict(state),strict=True)
    return metadata


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _run_seed(seed: int, train: list[dict], monitor: list[dict], out: Path,
              config: dict, transform: FittedTransform, device: torch.device) -> dict:
    _set_seed(seed)
    model = DiagnosticMaskedMAE().to(device)
    initial_hash = _state_hash(model.state_dict())
    _save_checkpoint(out/"initial_checkpoint.pt",model,transform,"initial",None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    selector=CheckpointSelector(minimum_optimization_epochs=config["minimum_optimization_epochs"],
        earliest_checkpoint_epoch=config["earliest_checkpoint_epoch"],
        patience=config["patience"],tolerance=config["tolerance"])
    traces=[]
    started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train(); epoch_sum = 0.0; epoch_count = 0; grad_norms = []; steps = 0
        lr = config["learning_rate"] * min(1.0, epoch / config["warmup_epochs"])
        for group in optimizer.param_groups: group["lr"] = lr
        for sample in train:
            batch = _as_tensors(sample, device)
            optimizer.zero_grad(set_to_none=True)
            loss, count, _ = _loss(model, batch)
            if not count: continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            grad_norms.append(float(norm.detach().cpu()))
            optimizer.step(); steps += 1
            epoch_sum += float(loss.detach().cpu()) * count; epoch_count += count
        train_loss = epoch_sum / epoch_count if epoch_count else None
        model.eval(); monitor_sum = 0.0; monitor_count = 0
        with torch.no_grad():
            for sample in monitor:
                batch = _as_tensors(sample, device)
                loss, count, _ = _loss(model, batch)
                if count:
                    monitor_sum += float(loss.cpu()) * count; monitor_count += count
        monitor_loss = monitor_sum / monitor_count if monitor_count else None
        if monitor_loss is not None and not math.isfinite(monitor_loss):
            monitor_loss=None
        decision=selector.observe(epoch,monitor_loss,monitor_count)
        traces.append({"epoch": epoch, "training_masked_mse": train_loss,
            "training_loss_denominator": epoch_count, "monitor_masked_mse": monitor_loss,
            "monitor_loss_denominator": monitor_count, "learning_rate": lr,
            "gradient_norms_preclip": grad_norms, "optimizer_steps": steps,
            "checkpoint_decision":decision,
            "elapsed_seconds_cumulative": time.monotonic() - started})
        if decision["selected"]:
            _save_checkpoint(out/"selected_checkpoint.pt",model,transform,"selected",epoch)
        if decision.get("stop",False):
            break
    final_hash = _state_hash(model.state_dict())
    _save_checkpoint(out/"final_checkpoint.pt",model,transform,"final",len(traces))
    selected_path = out / "selected_checkpoint.pt"
    selected_hash = sha256_file(selected_path) if selected_path.exists() else None
    if selected_path.exists():
        selected_meta=_load_checkpoint(selected_path,model,transform)
        if selected_meta.get("epoch") != selector.best_epoch:
            raise RuntimeError("selected checkpoint epoch differs from selector state")
        model.eval(); replay_sum = 0.0; replay_count = 0
        with torch.no_grad():
            for sample in monitor:
                loss, count, _ = _loss(model, _as_tensors(sample, device))
                replay_sum += float(loss.cpu()) * count; replay_count += count
        replay_loss = replay_sum / replay_count if replay_count else None
    else:
        selected_meta=None; replay_loss=None
    return {"seed": seed, "initial_checkpoint_state_sha256": initial_hash,
        "initial_checkpoint_file_sha256":sha256_file(out/"initial_checkpoint.pt"),
        "selected_checkpoint_file_sha256": selected_hash, "final_checkpoint_state_sha256": final_hash,
        "final_checkpoint_file_sha256":sha256_file(out/"final_checkpoint.pt"),
        "selected_checkpoint_status":selector.selected_status,
        "selected_epoch": selector.best_epoch, "selected_monitor_mse": selector.best_loss,
        "selected_checkpoint_replay_monitor_mse": replay_loss,
        "checkpoint_replay_absolute_error": abs(replay_loss-selector.best_loss) if replay_loss is not None and selector.best_loss is not None else None,
        "replay_transform_sha256":transform.state_sha256 if selected_path.exists() else None,
        "fit_transform_sha256":transform.state_sha256,
        "earliest_checkpoint_epoch":selector.earliest_checkpoint_epoch,
        "minimum_optimization_epochs":selector.minimum_optimization_epochs,
        "selector_eligible_monitor_epochs":selector.eligible_count,
        "patience_nonimprovement_count":selector.nonimprovement_count,
        "epochs_executed": len(traces), "optimizer_steps": sum(x["optimizer_steps"] for x in traces),
        "parameter_updates_observed": initial_hash != final_hash, "trace": traces,
        "parameter_count": model.parameter_count, "elapsed_seconds": time.monotonic()-started}


def run(root: str | Path, output: str | Path, *, device_name: str = "cpu") -> dict:
    root, output = Path(root).resolve(strict=True), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    version = {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__,
               "executable": __import__("sys").executable}
    input_path = root / "data/derived/wp05_wp06_multimodal_attempt_004/panel.v1.json"
    array_path = root / "data/derived/wp05_wp06_multimodal_attempt_004/panel_arrays.npz"
    input_hashes = {str(p.relative_to(root)): sha256_file(p) for p in (input_path, array_path)}
    device = torch.device(device_name)
    fit_view=_load_view(root,0)
    transform=fit_transform(fit_view)
    transform_dict=transform.to_dict()
    transform_path=output/"fit_transform.json"
    transform_path.write_text(json.dumps(transform_dict,indent=2,sort_keys=True)+"\n")
    transform_file_sha=sha256_file(transform_path)
    fit_preparation_receipt={"status":"FROZEN_PARITY0_FIT_TRANSFORM",
        "fit_transform_sha256":transform.state_sha256,"fit_transform_file_sha256":transform_file_sha,
        "fit_role":transform.fit_role,"fit_parity":transform.fit_parity,
        "fit_source_values_sha256":transform.fit_source_values_sha256,
        "fit_membership_sha256":transform.fit_membership_sha256,
        "panel_membership_sha256":transform.panel_membership_sha256,
        "channel_order":list(transform.channel_order),"fit_channel_counts":list(transform.counts),
        "fit_input_sha256":fit_view["input_sha256"],
        "transform_fit_operation":"FIT_ONLY_PARITY0_VALID_CONTEXT; TARGET_PIXELS_EXCLUDED",
        "monitor_values_in_transform_state":False}
    (output/"fit_preparation_receipt.json").write_text(json.dumps(fit_preparation_receipt,indent=2,sort_keys=True)+"\n")
    fit,fit_application_receipt=apply_transform(fit_view,transform,application_role="fit_application")
    monitor_view=_load_view(root,1)
    monitor,monitor_application_receipt=apply_transform(monitor_view,transform,application_role="monitor_application")
    if monitor_application_receipt["fit_transform_sha256"]!=fit_preparation_receipt["fit_transform_sha256"]:
        raise RuntimeError("monitor application used a different frozen fit transform")
    (output/"fit_application_receipt.json").write_text(json.dumps(fit_application_receipt,indent=2,sort_keys=True)+"\n")
    (output/"monitor_application_receipt.json").write_text(json.dumps(monitor_application_receipt,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(output / "token_support.npz",
        fit_values=np.stack([x["values"] for x in fit]),
        fit_observed=np.stack([x["observed"] for x in fit]),
        fit_targets=np.stack([x["target"] for x in fit]),
        monitor_values=np.stack([x["values"] for x in monitor]),
        monitor_observed=np.stack([x["observed"] for x in monitor]),
        monitor_targets=np.stack([x["target"] for x in monitor]))
    pilot_protocol = {"optimizer_candidate": "AdamW", "learning_rate_candidate": 0.0003,
        "weight_decay_candidate": 0.0001, "warmup_epochs": 2, "gradient_clip": 1.0,
        "precision": "float32", "device": str(device), "pilot_epochs": 4,
        "pilot_seed": 917, "selection_rule": "measure fit/monitor loss, denominators and preclip gradients; no performance gate",
        "checkpoint_selection":{"minimum_optimization_epochs":4,"earliest_checkpoint_epoch":4,
            "max_epochs":4,"patience":3,"tolerance":0.0001,
            "ties":"within tolerance retain earlier checkpoint and count one eligible non-improvement",
            "empty_or_nonfinite":"no checkpoint and no patience increment; run to maximum if no eligible estimate"},
        "main_config_rule": "retain candidate if pilot monitor loss is finite and declines from first to last epoch with finite gradients; otherwise lower LR to 0.0001 and retain same bounded schedule"}
    (output / "pilot_protocol.json").write_text(json.dumps(pilot_protocol, indent=2, sort_keys=True)+"\n")
    pilot_config = {"learning_rate": pilot_protocol["learning_rate_candidate"],
        "weight_decay": pilot_protocol["weight_decay_candidate"], "warmup_epochs": 2,
        "gradient_clip": 1.0, "minimum_optimization_epochs": 4,
        "earliest_checkpoint_epoch":4,"max_epochs": 4,
        "patience": 3, "tolerance": 0.0001}
    (output/"pilot_prospective_config.json").write_text(json.dumps(pilot_config,indent=2,sort_keys=True)+"\n")
    pilot_dir = output / "pilot"; pilot_dir.mkdir()
    pilot = _run_seed(917, fit, monitor, pilot_dir, pilot_config, transform, device)
    pilot_trace = pilot["trace"]
    pilot_gradient_finite = all(math.isfinite(value) for row in pilot_trace
                                for value in row["gradient_norms_preclip"])
    pilot_monitor_declined = (len(pilot_trace) >= 2 and
        pilot_trace[-1]["monitor_masked_mse"] is not None and
        pilot_trace[0]["monitor_masked_mse"] is not None and
        pilot_trace[-1]["monitor_masked_mse"] < pilot_trace[0]["monitor_masked_mse"])
    learning_rate = pilot_protocol["learning_rate_candidate"] if pilot_gradient_finite and pilot_monitor_declined else 0.0001
    config = {"optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": 0.0001,
        "warmup_epochs": 2, "batch_size": 1, "gradient_clip": 1.0, "precision": "float32",
        "device": str(device), "minimum_optimization_epochs": 6,
        "earliest_checkpoint_epoch":6,"max_epochs": 12,"patience": 3,
        "tolerance": 0.0001, "monitor_selection": "lowest finite positive-denominator monitor masked MSE from earliest_checkpoint_epoch inclusive",
        "checkpoint_ties":"within tolerance retain earlier checkpoint; count one eligible non-improvement",
        "checkpoint_patience":"counts only finite eligible non-improvement epochs at/after minimum_optimization_epochs; missing/nonfinite/zero-denominator epochs do not increment",
        "no_eligible_checkpoint":"selected checkpoint null with UNAVAILABLE_NO_ELIGIBLE_FINITE_MONITOR_ESTIMATE; run to max_epochs",
        "stopping_rule":"never stop before minimum_optimization_epochs; after that stop on patience eligible finite non-improvements or at max_epochs",
        "training_selection": "all available diagnostic fit panels; no scientific role membership",
        "loss": "unweighted mean squared error over true-valid target channel summaries",
        "seeds": [0, 1, 2, 3, 4], "extended_budget_epochs": 3,
        "pilot": {"seed": 917, "epochs": 4, "role": "learning pilot completed before main-run settings were locked"},
        "pilot_basis": {"finite_gradient_norms": pilot_gradient_finite,
            "monitor_loss_declined": pilot_monitor_declined,
            "selected_learning_rate": learning_rate}}
    # This exact settings record is written after pilot and before any of the five seeds.
    config["fit_transform_sha256"]=transform.state_sha256
    config["fit_transform_file_sha256"]=transform_file_sha
    config["fit_source_values_sha256"]=transform.fit_source_values_sha256
    config["fit_membership_sha256"]=transform.fit_membership_sha256
    (output / "prospective_config.json").write_text(json.dumps(config, indent=2, sort_keys=True)+"\n")
    results = []
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"; seed_dir.mkdir()
        result = _run_seed(seed, fit, monitor, seed_dir, config, transform, device)
        results.append(result)
    extended = []
    # Bounded continuation from each final state; selected checkpoints remain untouched.
    for result in results:
        seed = result["seed"]
        model_ext = DiagnosticMaskedMAE().to(device)
        final_path = output / f"seed_{seed}" / "final_checkpoint.pt"
        extension_load_metadata=_load_checkpoint(final_path,model_ext,transform)
        optimizer_ext = torch.optim.AdamW(model_ext.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
        before = _state_hash(model_ext.state_dict())
        extension_trace = []
        extension_started=time.monotonic()
        for epoch in range(1, config["extended_budget_epochs"] + 1):
            model_ext.train(); total = 0.0; denominator = 0; grad_norms=[]; steps=0
            lr=config["learning_rate"]
            for group in optimizer_ext.param_groups:group["lr"]=lr
            for sample in fit:
                batch = _as_tensors(sample, device); optimizer_ext.zero_grad(set_to_none=True)
                loss, count, _ = _loss(model_ext, batch)
                if count:
                    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model_ext.parameters(), config["gradient_clip"])
                    grad_norms.append(float(norm.detach().cpu()))
                    optimizer_ext.step(); total += float(loss.detach().cpu()) * count; denominator += count
                    steps+=1
            extension_trace.append({"extension_epoch": epoch,
                "training_masked_mse": total / denominator if denominator else None,
                "loss_denominator": denominator,"learning_rate":lr,
                "gradient_norms_preclip":grad_norms,"optimizer_steps":steps,
                "elapsed_seconds_cumulative":time.monotonic()-extension_started})
        model_ext.eval(); total = 0.0; denominator = 0
        with torch.no_grad():
            for sample in monitor:
                loss, count, _ = _loss(model_ext, _as_tensors(sample, device))
                total += float(loss.cpu()) * count; denominator += count
        after = _state_hash(model_ext.state_dict())
        extension_path=output/f"seed_{seed}"/"extension_final_checkpoint.pt"
        _save_checkpoint(extension_path,model_ext,transform,"extended_final",config["extended_budget_epochs"])
        extended.append({"seed": seed, "starting_final_state_sha256": before,
            "extended_final_state_sha256": after, "parameter_updates_observed": before != after,
            "starting_checkpoint_fit_transform_sha256":extension_load_metadata["fit_transform_sha256"],
            "extension_checkpoint_file_sha256":sha256_file(extension_path),
            "fit_transform_sha256":transform.state_sha256,
            "monitor_masked_mse_after_extension": total / denominator if denominator else None,
            "monitor_loss_denominator": denominator, "trace": extension_trace,
            "selection_changed": False})
    model = DiagnosticMaskedMAE()
    summary = {"status": "LOCAL_DEVELOPMENT_DIAGNOSTIC_EXECUTION_COMPLETE_ARCHITECT_REVIEW_PENDING",
        "research_admissible": False, "plan_executable": False, "release_status": "BLOCKED",
        "input_sha256": input_hashes, "environment": version, "device": str(device),
        "tokenization_policy": POLICY, "fit_transform": transform.to_dict(),
        "fit_preparation_receipt":fit_preparation_receipt,
        "fit_application_receipt":fit_application_receipt,
        "monitor_application_receipt":monitor_application_receipt,
        "fit_support": [x["support"] for x in fit],
        "monitor_support": [x["support"] for x in monitor], "architecture": {
            "d_model": 128, "attention_heads_encoder_decoder": 4, "encoder_layers": 4,
            "decoder_layers": 2, "encoder_ffn_width": 512, "decoder_ffn_width": 256,
            "parameter_count": model.parameter_count, "parameters_from_sum": True,
            "dropout": 0.0, "normalization": "Pre-LayerNorm; final LayerNorm",
            "time_encoding": "elapsed years, annual phase sine/cosine, valid flag; UTC optical only; unknown SAR zero features"},
        "optimization_config": config, "pilot": pilot, "five_seed_runs": results,
        "fit_transform_hash_shared_by_all_runs":all(x["fit_transform_sha256"]==transform.state_sha256 for x in results),
        "checkpoint_selection_rule_consistent":all(x["selected_epoch"] is None or x["selected_epoch"]>=config["earliest_checkpoint_epoch"] for x in results),
        "extended_budget": extended, "scientific_claims": {"independent_validation": False,
            "temporal_learning": False, "forecasting": False, "anomaly_skill": False,
            "independent_lake_event_support": None, "status": "NOT_ESTIMATED"},
        "limitations": ["Two exposed sites and four asynchronous acquisition groups are diagnostic only.",
            "Monitor tiles are spatially disjoint but share scenes and are not independent validation.",
            "No physical sample membership, measurement admission, research training, scoring or evaluation occurred.",
            "Correct fit-only transform and checkpoint eligibility are engineering corrections pending independent Architect review."]}
    (output / "training_evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    output_hashes = {str(path.relative_to(output)): sha256_file(path) for path in output.rglob("*") if path.is_file()}
    final_input_hashes = {str(p.relative_to(root)): sha256_file(p) for p in (input_path, array_path)}
    if final_input_hashes != input_hashes:
        raise RuntimeError("source panel SHA-256 changed during local diagnostic training")
    manifest = {"inputs": input_hashes, "outputs_before_manifest": output_hashes,
        "input_recheck": "MATCH", "output_recheck": "PENDING_SELF_EXCLUDED_MANIFEST"}
    manifest_path = output / "sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    for rel, digest in output_hashes.items():
        if sha256_file(output / rel) != digest:
            raise RuntimeError(f"output SHA-256 mismatch: {rel}")
    manifest["output_recheck"] = "MATCH"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return summary
