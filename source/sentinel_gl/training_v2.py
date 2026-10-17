"""Version-2 temporal masked reconstruction for retrospective pre-breach analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn

from .panel_v2 import (
    CANONICAL_CHANNELS, PanelV2, _sha_json, causal_prefix,
    fit_channel_standardizer, pool_observation_quadrants, sha256_file,
)


TRAINING_V2_VERSION = "TEMPORAL_MASKED_RECONSTRUCTION_V2"
TIME_REFERENCE = datetime(2023, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TrainingConfigV2:
    d_model: int = 128
    n_heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 2
    encoder_ff: int = 512
    decoder_ff: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    gradient_clip: float = 1.0
    minimum_epochs: int = 12
    earliest_checkpoint_epoch: int = 12
    max_epochs: int = 80
    patience: int = 12
    tolerance: float = 1e-4
    dropout: float = 0.0

    def validate(self) -> None:
        integers = (self.d_model, self.n_heads, self.encoder_layers,
                    self.decoder_layers, self.encoder_ff, self.decoder_ff,
                    self.warmup_epochs, self.minimum_epochs,
                    self.earliest_checkpoint_epoch, self.max_epochs, self.patience)
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError("training architecture and budget fields must be positive integers")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.earliest_checkpoint_epoch < self.minimum_epochs:
            raise ValueError("checkpoint selection cannot start before the minimum fit budget")
        if self.max_epochs < self.minimum_epochs:
            raise ValueError("max_epochs cannot be below minimum_epochs")
        for name in ("learning_rate", "weight_decay", "gradient_clip", "tolerance"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(self.dropout) not in (float, int) or not math.isfinite(float(self.dropout)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")


class TemporalMaskedMAEV2(nn.Module):
    """Dynamic-length multimodal transformer with token and time identities.

    Values under target or natural-missing masks are zeroed by the caller and
    cannot enter the encoder. Optical dependency closure is applied to source
    pixels before quadrant pooling in panel_v2.
    """
    def __init__(self, config: TrainingConfigV2 | None = None):
        super().__init__()
        self.config = config or TrainingConfigV2()
        self.config.validate()
        d = self.config.d_model
        self.optical_projection = nn.Linear(10, d)
        self.sar_projection = nn.Linear(4, d)
        self.modality_embedding = nn.Embedding(2, d)
        self.quadrant_embedding = nn.Embedding(4, d)
        self.time_projection = nn.Linear(4, d, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d, self.config.n_heads, self.config.encoder_ff,
                dropout=self.config.dropout, batch_first=True, norm_first=True,
                activation="gelu") for _ in range(self.config.encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d)
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d, self.config.n_heads, self.config.decoder_ff,
                dropout=self.config.dropout, batch_first=True, norm_first=True,
                activation="gelu") for _ in range(self.config.decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d)
        self.optical_head = nn.Linear(d, 5)
        self.sar_head = nn.Linear(d, 2)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate(self, values, observed, modality, time_features, quadrant, target=None):
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] != 7:
            raise ValueError("v2 model values must have shape [batch, sequence>=2, 7]")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("v2 model inputs must be finite normalized float tensors")
        if observed.dtype != torch.bool or observed.shape != values.shape:
            raise ValueError("v2 observed mask must be boolean and match values")
        if torch.any(values[~observed] != 0):
            raise ValueError("missing and target-hidden model inputs must be zero behind false masks")
        shape = values.shape[:2]
        if modality.dtype not in (torch.int32, torch.int64) or modality.shape != shape or torch.any((modality < 0) | (modality > 1)):
            raise ValueError("v2 modality identities must be integer 0/1 per token")
        if quadrant.dtype not in (torch.int32, torch.int64) or quadrant.shape != shape or torch.any((quadrant < 0) | (quadrant > 3)):
            raise ValueError("v2 quadrant identities must be integer 0..3 per token")
        if time_features.shape != (*shape, 4) or not torch.isfinite(time_features).all():
            raise ValueError("v2 UTC time features must be finite with four features per token")
        if target is not None and (target.dtype != torch.bool or target.shape != values.shape):
            raise ValueError("v2 reconstruction target mask must be boolean and match values")

    def _project(self, values, observed, modality, time_features, quadrant):
        optical_features = torch.cat((values[..., :5], observed[..., :5].to(values.dtype)), dim=-1)
        sar_features = torch.cat((values[..., 5:], observed[..., 5:].to(values.dtype)), dim=-1)
        projected = torch.zeros((*values.shape[:2], self.config.d_model),
                                dtype=values.dtype, device=values.device)
        optical = modality == 0
        sar = modality == 1
        projected[optical] = self.optical_projection(optical_features[optical])
        projected[sar] = self.sar_projection(sar_features[sar])
        return projected + self._token_identity(modality, time_features, quadrant)

    def _token_identity(self, modality, time_features, quadrant):
        return (self.modality_embedding(modality) + self.quadrant_embedding(quadrant) +
                self.time_projection(time_features))

    def _decoder_inputs(self, encoded, target_tokens, modality, time_features, quadrant):
        target_representation = self.mask_token.expand_as(encoded) + self._token_identity(
            modality, time_features, quadrant)
        return torch.where(target_tokens[..., None], target_representation, encoded)

    def encode(self, values, observed, modality, time_features, quadrant,
               target_tokens=None):
        self._validate(values, observed, modality, time_features, quadrant)
        if target_tokens is None:
            target_tokens = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
        if target_tokens.dtype != torch.bool or target_tokens.shape != values.shape[:2]:
            raise ValueError("target_tokens must be a boolean mask over sequence positions")
        padding = target_tokens | ~observed.any(dim=-1)
        if torch.any((~padding).sum(dim=1) == 0):
            raise ValueError("v2 encoder requires at least one visible observed token")
        encoded = self._project(values, observed, modality, time_features, quadrant)
        for layer in self.encoder:
            encoded = layer(encoded, src_key_padding_mask=padding)
        return self.encoder_norm(encoded)

    def forward(self, values, observed, modality, time_features, quadrant, target):
        self._validate(values, observed, modality, time_features, quadrant, target)
        target_tokens = target.any(dim=-1)
        encoded = self.encode(values, observed, modality, time_features, quadrant,
                              target_tokens=target_tokens)
        decoded = self._decoder_inputs(encoded, target_tokens, modality, time_features, quadrant)
        padding = ~(target_tokens | observed.any(dim=-1))
        for layer in self.decoder:
            decoded = layer(decoded, src_key_padding_mask=padding)
        decoded = self.decoder_norm(decoded)
        optical = modality == 0
        sar = modality == 1
        prediction = torch.zeros_like(values)
        prediction[..., :5] = torch.where(optical[..., None],
            self.optical_head(decoded), torch.zeros_like(prediction[..., :5]))
        prediction[..., 5:] = torch.where(sar[..., None],
            self.sar_head(decoded), torch.zeros_like(prediction[..., 5:]))
        return prediction, encoded


def _time_features(acquired_at: str) -> np.ndarray:
    stamp = datetime.fromisoformat(acquired_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    elapsed_days = (stamp - TIME_REFERENCE).total_seconds() / 86400.0
    year_days = 366.0 if (stamp.year % 4 == 0 and (stamp.year % 100 != 0 or stamp.year % 400 == 0)) else 365.0
    fraction = ((stamp.timetuple().tm_yday - 1) +
                (stamp.hour * 3600 + stamp.minute * 60 + stamp.second + stamp.microsecond / 1e6) / 86400.0) / year_days
    phase = 2.0 * math.pi * fraction
    return np.asarray([elapsed_days / 365.25, math.sin(phase), math.cos(phase), 1.0], dtype=np.float32)


def _scope_records(panel: PanelV2, role: str) -> list[dict]:
    return [dict(row) for row in panel.observations if row["scope"]["role"] == role]


def _sequence_from_prefix(prefix: Sequence[Mapping[str, Any]], token_cache: Mapping[tuple[str, int], dict]):
    tokens = []
    owner_ids = []
    for record in prefix:
        for q in range(4):
            token = token_cache[(record["observation_id"], q)]
            tokens.append({**token, "observation_id": record["observation_id"]})
            owner_ids.append(record["observation_id"])
    if not tokens:
        raise ValueError("causal sequence has no tokens")
    values = np.stack([x["values"] for x in tokens]).astype(np.float32)
    observed = np.stack([x["observed"] for x in tokens]).astype(bool)
    modality = np.asarray([0 if x["sensor"] == "SENTINEL_2_OPTICAL" else 1 for x in tokens], dtype=np.int64)
    time = np.stack([_time_features(x["acquired_at"]) for x in tokens]).astype(np.float32)
    quadrant = np.asarray([x["quadrant"] for x in tokens], dtype=np.int64)
    return {"values": values, "observed": observed, "modality": modality,
            "time": time, "quadrant": quadrant, "owner_ids": owner_ids,
            "tokens": tokens}


def _fold_batch(sequence: Mapping[str, Any], query_id: str,
                device: torch.device) -> dict[str, torch.Tensor] | None:
    target_positions = [i for i, owner in enumerate(sequence["owner_ids"])
                        if owner == query_id]
    if len(target_positions) != 4:
        raise ValueError("causal sequence must contain exactly four query quadrants")
    eligible = [q for q, pos in enumerate(target_positions)
                if bool(sequence["tokens"][pos]["observed"].any())]
    if not eligible:
        return None
    base_values = sequence["values"]
    base_observed = sequence["observed"]
    batch_values = np.repeat(base_values[None, :, :], len(eligible), axis=0)
    batch_observed = np.repeat(base_observed[None, :, :], len(eligible), axis=0)
    target = np.zeros_like(batch_observed, dtype=bool)
    target_values = np.zeros_like(batch_values, dtype=np.float32)
    for bi, q in enumerate(eligible):
        pos = target_positions[q]
        support = sequence["tokens"][pos]["observed"]
        truth = sequence["tokens"][pos]["values"]
        target[bi, pos] = support
        target_values[bi, pos] = truth
        batch_observed[bi, pos] = False
        batch_values[bi, pos] = 0.0
    tensor = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype, device=device)
    return {
        "values": tensor(batch_values, torch.float32),
        "observed": tensor(batch_observed, torch.bool),
        "modality": tensor(np.repeat(sequence["modality"][None, :], len(eligible), axis=0), torch.long),
        "time": tensor(np.repeat(sequence["time"][None, :, :], len(eligible), axis=0), torch.float32),
        "quadrant": tensor(np.repeat(sequence["quadrant"][None, :], len(eligible), axis=0), torch.long),
        "target": tensor(target, torch.bool),
        "target_values": tensor(target_values, torch.float32),
        "folds": eligible,
        "target_positions": target_positions,
    }


def _masked_loss(model: TemporalMaskedMAEV2, batch: Mapping[str, torch.Tensor]):
    prediction, _ = model(batch["values"], batch["observed"], batch["modality"],
                          batch["time"], batch["quadrant"], batch["target"])
    mask = batch["target"]
    count = int(mask.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0, prediction
    loss = torch.square(prediction[mask] - batch["target_values"][mask]).sum() / count
    if not torch.isfinite(loss):
        raise ValueError("non-finite supported masked reconstruction loss")
    return loss, count, prediction


def _evaluate_examples(model, examples, device):
    model.eval()
    error_sum = 0.0
    denominator = 0
    with torch.no_grad():
        for batch in examples:
            loss, count, _ = _masked_loss(model, batch)
            if count:
                error_sum += float(loss.cpu()) * count
                denominator += count
    return (error_sum / denominator if denominator else None, denominator)


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8")); hasher.update(str(value.dtype).encode())
        hasher.update(np.asarray(value).tobytes())
    return hasher.hexdigest()


def _save_checkpoint(path: Path, model: TemporalMaskedMAEV2, metadata: Mapping[str, Any]) -> str:
    payload = {"checkpoint_version": 2, "metadata": dict(metadata),
               "model_state": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
               "model_state_sha256": _state_hash(model.state_dict())}
    torch.save(payload, path)
    return sha256_file(path)


def load_checkpoint_v2(path: str | Path, model: TemporalMaskedMAEV2, *,
                       expected_panel_sha256: str,
                       expected_transform_sha256: str) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict or checkpoint.get("checkpoint_version") != 2:
        raise ValueError("not a supported panel.v2 checkpoint")
    metadata = checkpoint.get("metadata")
    if type(metadata) is not dict or metadata.get("training_version") != TRAINING_V2_VERSION:
        raise ValueError("panel.v2 checkpoint metadata version mismatch")
    if metadata.get("panel_sha256") != expected_panel_sha256 or metadata.get("fit_transform_sha256") != expected_transform_sha256:
        raise ValueError("panel.v2 checkpoint source or fit-transform identity mismatch")
    actual = checkpoint.get("model_state")
    expected = model.state_dict()
    if type(actual) is not dict or set(actual) != set(expected):
        raise ValueError("panel.v2 checkpoint model keys differ")
    for key, target in expected.items():
        tensor = actual[key]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError(f"panel.v2 checkpoint tensor contract differs: {key}")
    if _state_hash(actual) != checkpoint.get("model_state_sha256"):
        raise ValueError("panel.v2 checkpoint tensor SHA-256 mismatch")
    model.load_state_dict(actual, strict=True)
    return metadata


def _make_examples(records, token_cache, *, parity, device):
    ordered = sorted(records, key=lambda x: (x["acquired_at"], x["site_id"], x["sensor"], x["product_id"]))
    examples = []
    for query in ordered:
        prefix = causal_prefix(records, query)
        sequence = _sequence_from_prefix(prefix, token_cache)
        batch = _fold_batch(sequence, query["observation_id"], device)
        if batch is not None:
            examples.append((query, batch))
    return examples


def _run_seed(seed: int, train_examples, monitor_examples, out: Path,
              panel: PanelV2, standardizer: Mapping[str, Any], config: TrainingConfigV2,
              *, device: torch.device) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = TemporalMaskedMAEV2(config).to(device)
    initial_state_hash = _state_hash(model.state_dict())
    common_meta = {"training_version": TRAINING_V2_VERSION, "seed": seed,
        "panel_sha256": panel.manifest_sha256, "fit_transform_sha256": standardizer["state_sha256"],
        "architecture": asdict(config), "parameter_count": model.parameter_count,
        "fit_source_observation_ids": standardizer["fit_observation_ids"],
        "development_monitor": "PARITY1_OF_THE_SAME_PRE_BREACH_FIT_SCENES"}
    initial_file_sha = _save_checkpoint(out / "initial_checkpoint.pt", model,
        common_meta | {"checkpoint_role": "INITIAL"})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    from .checkpoint_selection import CheckpointSelector
    selector = CheckpointSelector(minimum_optimization_epochs=config.minimum_epochs,
        earliest_checkpoint_epoch=config.earliest_checkpoint_epoch,
        patience=config.patience, tolerance=config.tolerance)
    trace = []
    started = time.monotonic()
    for epoch in range(1, config.max_epochs + 1):
        model.train(); fit_sum = 0.0; fit_count = 0; steps = 0; gradient_norms = []
        lr = config.learning_rate * min(1.0, epoch / config.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _, batch in train_examples:
            optimizer.zero_grad(set_to_none=True)
            loss, count, _ = _masked_loss(model, batch)
            if count == 0:
                continue
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            gradient_norms.append(float(grad.detach().cpu()))
            optimizer.step(); steps += 1
            fit_sum += float(loss.detach().cpu()) * count
            fit_count += count
        fit_loss = fit_sum / fit_count if fit_count else None
        monitor_loss, monitor_count = _evaluate_examples(
            model, [batch for _, batch in monitor_examples], device)
        decision = selector.observe(epoch, monitor_loss, monitor_count)
        trace.append({"epoch": epoch, "fit_masked_mse": fit_loss,
            "fit_target_channel_count": fit_count,
            "development_monitor_masked_mse": monitor_loss,
            "development_monitor_target_channel_count": monitor_count,
            "learning_rate": lr, "optimizer_steps": steps,
            "gradient_norms_preclip": gradient_norms,
            "checkpoint_decision": decision,
            "elapsed_seconds_cumulative": time.monotonic() - started})
        if decision["selected"]:
            _save_checkpoint(out / "selected_checkpoint.pt", model,
                common_meta | {"checkpoint_role": "SELECTED", "selected_epoch": epoch,
                              "selected_monitor_mse": monitor_loss})
        if decision.get("stop", False):
            break
    final_state_hash = _state_hash(model.state_dict())
    final_file_sha = _save_checkpoint(out / "final_checkpoint.pt", model,
        common_meta | {"checkpoint_role": "FINAL", "epochs_executed": len(trace)})
    selected_path = out / "selected_checkpoint.pt"
    selected_file_sha = sha256_file(selected_path) if selected_path.exists() else None
    selected_state_hash = None
    replay_loss = None
    replay_error = None
    selected_denominator = next((row["development_monitor_target_channel_count"]
        for row in trace if row["epoch"] == selector.best_epoch), None)
    if selected_path.exists():
        replay_model = TemporalMaskedMAEV2(config).to(device)
        selected_meta = load_checkpoint_v2(selected_path, replay_model,
            expected_panel_sha256=panel.manifest_sha256,
            expected_transform_sha256=standardizer["state_sha256"])
        selected_state_hash = _state_hash(replay_model.state_dict())
        replay_loss, replay_count = _evaluate_examples(
            replay_model, [batch for _, batch in monitor_examples], device)
        replay_error = (abs(replay_loss - selected_meta["selected_monitor_mse"])
                        if replay_loss is not None else None)
        if replay_count != selected_denominator:
            raise RuntimeError("selected checkpoint replay denominator differs from selection")
    return {
        "seed": seed, "initial_checkpoint_sha256": initial_file_sha,
        "initial_model_state_sha256": initial_state_hash,
        "final_checkpoint_sha256": final_file_sha,
        "final_model_state_sha256": final_state_hash,
        "selected_checkpoint_sha256": selected_file_sha,
        "selected_model_state_sha256": selected_state_hash,
        "selected_status": selector.selected_status,
        "selected_epoch": selector.best_epoch,
        "selected_monitor_mse": selector.best_loss,
        "selected_monitor_target_channel_count": selected_denominator,
        "selected_checkpoint_replay_mse": replay_loss,
        "selected_checkpoint_replay_absolute_error": replay_error,
        "epochs_executed": len(trace), "optimizer_steps": sum(x["optimizer_steps"] for x in trace),
        "parameter_updates_observed": initial_state_hash != final_state_hash,
        "parameter_count": model.parameter_count,
        "monitor_role": "DEVELOPMENT_PARITY1_SAME_SCENE_DEPENDENT",
        "trace": trace,
    }


def run_training_v2(panel: PanelV2, output: str | Path, *,
                    seeds: Sequence[int] = (0, 1, 2, 3, 4),
                    config: TrainingConfigV2 | None = None,
                    device_name: str = "cpu") -> dict:
    """Fit at least five deterministic seed models, selecting on parity-1 only."""
    selected_config = config or TrainingConfigV2()
    selected_config.validate()
    seeds = tuple(seeds)
    if len(seeds) < 5 or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("v2 training requires at least five distinct nonnegative integer seeds")
    if device_name != "cpu":
        raise ValueError("panel.v2 training currently accepts CPU only for deterministic local execution")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    output = Path(output)
    if not output.is_absolute():
        output = panel.root / output
    if not output.resolve(strict=False).is_relative_to(panel.root):
        raise ValueError("training output must remain inside workspace")
    output.mkdir(parents=True, exist_ok=False)
    standardizer = fit_channel_standardizer(panel)
    standardizer_path = output / "fit_transform.v2.json"
    standardizer_path.write_text(json.dumps(standardizer, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    standardizer_file_sha = sha256_file(standardizer_path)

    fit_records = _scope_records(panel, "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR")
    fit0_cache = {}
    fit1_cache = {}
    for record in fit_records:
        fit0 = pool_observation_quadrants(panel, record, standardizer, 0)
        fit1 = pool_observation_quadrants(panel, record, standardizer, 1)
        for token in fit0["tokens"]:
            fit0_cache[(record["observation_id"], token["quadrant"])] = {
                **token, "sensor": record["sensor"], "acquired_at": record["acquired_at"]}
        for token in fit1["tokens"]:
            fit1_cache[(record["observation_id"], token["quadrant"])] = {
                **token, "sensor": record["sensor"], "acquired_at": record["acquired_at"]}
    device = torch.device("cpu")
    train_examples = _make_examples(fit_records, fit0_cache, parity=0, device=device)
    monitor_examples = _make_examples(fit_records, fit1_cache, parity=1, device=device)
    if not train_examples:
        raise ValueError("no source-supported parity-0 fit reconstruction targets are available")
    config_payload = asdict(selected_config) | {
        "optimizer": "AdamW", "seed_set": list(seeds),
        "selection": "minimum finite parity-1 masked MSE on the same pre-breach scenes",
        "stopping": "eligible development nonimprovements only after minimum epoch; no convergence guarantee",
        "target": "mean of valid source pixels per channel and quadrant; four target-quadrant folds",
        "spatial_support": "16-pixel checkerboard parity-0 fit; parity-1 same-scene development monitor",
        "source_manifest_sha256": panel.manifest_sha256,
        "domain_protocol_sha256": panel.domain_protocol_sha256,
        "fit_transform_sha256": standardizer["state_sha256"],
    }
    (output / "prospective_config.v2.json").write_text(json.dumps(
        config_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    results = []
    for seed in seeds:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        result = _run_seed(seed, train_examples, monitor_examples, seed_dir,
                           panel, standardizer, selected_config, device=device)
        (seed_dir / "fit_monitor_history.v2.json").write_text(json.dumps({
            "training_version": TRAINING_V2_VERSION, "seed": seed,
            "fit_transform_sha256": standardizer["state_sha256"],
            "source_manifest_sha256": panel.manifest_sha256,
            "fit_role": "ALL_PRE_BREACH_PARITY0", "monitor_role": "ALL_PRE_BREACH_PARITY1_SAME_SCENE_DEPENDENT",
            "trace": result["trace"]}, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        summary = {key: value for key, value in result.items() if key != "trace"}
        results.append(summary)
    report = {
        "schema_version": 2, "training_version": TRAINING_V2_VERSION,
        "status": "FIT_EXECUTED_WITH_SOURCE_SUPPORT" if all(x["parameter_updates_observed"] for x in results)
                  else "FIT_SUPPORT_OR_PARAMETER_UPDATE_INCOMPLETE",
        "source_manifest_sha256": panel.manifest_sha256,
        "domain_protocol_sha256": panel.domain_protocol_sha256,
        "fit_transform_sha256": standardizer["state_sha256"],
        "fit_transform_file_sha256": standardizer_file_sha,
        "fit_observation_ids": standardizer["fit_observation_ids"],
        "fit_observation_count": len(fit_records),
        "train_examples_with_supported_targets": len(train_examples),
        "development_monitor_examples_with_supported_targets": len(monitor_examples),
        "seeds_are_training_replicates_not_independent_sample_units": list(seeds),
        "independent_physical_support_count": None,
        "independent_physical_support_status": "UNKNOWN",
        "model_configuration": config_payload,
        "seeds": results,
        "no_event_labels_used": True,
        "co_post_event_inputs_used_for_fit_or_checkpoint_selection": False,
        "analysis_status": "RETROSPECTIVE_DEVELOPMENT_ANALYSIS",
        "analysis_limitations": [
            "Every pre-breach acquisition up to the strict protocol cutoff contributes to fitted states; pre-period trajectories are development-exposed retrospective reconstructions.",
            "Each query context is causal in acquisition time, but the model and reference state use later pre-breach observations relative to early pre-period queries.",
            "Parity-1 monitoring is spatially held out within the same acquisitions and is not an independent temporal or physical sample.",
            "Physical sampling independence, operational availability, event ascertainment, and publication authorization are not established by this fit report.",
        ],
        "publication_authorized": False,
    }
    (output / "training_report.v2.json").write_text(json.dumps(report, indent=2,
        sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report

