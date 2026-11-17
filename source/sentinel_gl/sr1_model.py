"""SR-1 factorized temporal masked autoencoder (T-MAE) and its trainer.

Architecture (defaults): d_model=64, 4 heads, two spatial encoder layers, one
causal temporal encoder layer per patch location, one spatial decoder layer;
207,685 trainable parameters with five input channels.

Training masks a random 50% of the observed patches *in every scene* of a
chronological block and always leaves at least one observed patch visible
per scene (``sample_training_mask``). Validation masks every other observed
patch of each scene and the selection metric is the target-weighted mean
squared error (sum of squared errors over all VAL target entries divided by
their number). AdamW, cosine annealing and an exponential moving average of
the weights are used; the checkpoint is the EMA state at the epoch with the
lowest validation loss.

The trainer used for the retained pre-November checkpoints indexed only the
batch and time axes of the sampled mask and therefore hid every observed patch
of every scene; that version is archived outside the release tree and is
documented in docs/protocol/PRE_FIX_REGRESSION_RUN.txt.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


class SR1TransformerMAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        num_patches: int = 100,
        num_zones: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_patches = num_patches
        
        # Encoders for patch components
        self.channel_proj = nn.Linear(in_channels, d_model)
        self.patch_pos_embed = nn.Embedding(num_patches, d_model)
        self.zone_embed = nn.Embedding(num_zones, d_model)
        self.temporal_proj = nn.Linear(3, d_model)  # (sin_doy, cos_doy, delta_t)
        self.modality_embed = nn.Embedding(2, d_model)  # 0: optical, 1: SAR
        
        # Mask token for masked autoencoding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # The declared depth is factorized: all but the final encoder layer
        # attend within a scene across patch locations, while the final layer
        # attends across time for each fixed patch location.
        spatial_layers = max(1, num_encoder_layers - 1)
        temporal_layers = max(1, num_encoder_layers - spatial_layers)
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=spatial_layers)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.reconstruct_head = nn.Linear(d_model, in_channels)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,                # (B, N_patches, in_channels)
        observed_mask: torch.Tensor,    # (B, N_patches) - False for unobserved / cloudy
        masked_positions: torch.Tensor, # (B, N_patches) - True for MAE targets
        zone_ids: torch.Tensor,         # (B, N_patches) - zone ids
        time_feats: torch.Tensor,       # (B, 3) - temporal features
        modality_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass reconstructing masked patch tokens.
        
        Returns:
            reconstructed: (B, N_patches, in_channels)
            latent: (B, N_patches, d_model)
        """
        original_ndim = x.ndim
        if original_ndim == 3:
            x = x.unsqueeze(1)
            observed_mask = observed_mask.unsqueeze(1)
            masked_positions = masked_positions.unsqueeze(1)
            zone_ids = zone_ids.unsqueeze(1)
            time_feats = time_feats.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("x must have shape (B,T,N,C) or legacy (B,N,C)")
        B, T, N, C = x.shape
        if observed_mask.shape != (B, T, N) or masked_positions.shape != (B, T, N):
            raise ValueError("observation and target masks must match (B,T,N)")
        if zone_ids.shape != (B, T, N) or time_feats.shape != (B, T, 3):
            raise ValueError("zone and time tensors must match the scene sequence")
        device = x.device
        
        # Zero out unobserved NaNs before linear projection
        x_clean = torch.nan_to_num(x, 0.0)
        
        # Replace ONLY spectral channel features with mask_token at masked positions
        h_spec = self.channel_proj(x_clean.reshape(B * T, N, C)).reshape(B, T, N, self.d_model)
        mask_expand = self.mask_token.expand(B, T, N, -1)
        h_masked = torch.where(masked_positions.unsqueeze(-1), mask_expand, h_spec)
        
        # Add spatial patch position, geomorphic zone, temporal, and modality embeddings
        patch_ids = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
        t_embed = self.temporal_proj(time_feats).unsqueeze(2)  # (B,T,1,D)
        m_embed = self.modality_embed(torch.tensor(modality_id, device=device)).view(1, 1, -1)
        h_masked = (
            h_masked
            + self.patch_pos_embed(patch_ids)
            + self.zone_embed(zone_ids)
            + t_embed
            + m_embed
        )
        
        # Spatial attention is independent for each scene. Targets remain
        # visible only as learned mask tokens; cloud/missing patches are keys-masked.
        spatial_in = h_masked.reshape(B * T, N, self.d_model)
        spatial_valid = observed_mask.reshape(B * T, N)
        empty_scene = ~spatial_valid.any(dim=1)
        spatial_valid_safe = spatial_valid.clone()
        if empty_scene.any():
            spatial_valid_safe[empty_scene, 0] = True
            spatial_in = spatial_in.clone()
            spatial_in[empty_scene, 0] = 0
        spatial = self.spatial_encoder(spatial_in, src_key_padding_mask=~spatial_valid_safe)

        # Temporal attention runs over time separately at each fixed patch
        # location and is causal; future dates cannot influence earlier outputs.
        temporal_in = spatial.reshape(B, T, N, self.d_model).permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        temporal_valid = observed_mask.permute(0, 2, 1).reshape(B * N, T)
        # Encode only observed timestamps for each patch. Combining a causal
        # mask with left-padded key masking gives early missing queries no
        # permitted keys (all -inf attention logits), which can produce NaNs
        # that then contaminate later queries. Group equal validity patterns
        # so the encoder remains batched while every sequence contains only
        # real observations and keeps the original causal order.
        temporal = torch.zeros_like(temporal_in)
        pattern_rows: Dict[Tuple[int, ...], List[int]] = {}
        for row_index, validity in enumerate(temporal_valid):
            valid_indices = tuple(torch.nonzero(validity, as_tuple=False).flatten().tolist())
            if valid_indices:
                pattern_rows.setdefault(valid_indices, []).append(row_index)
        for valid_indices, row_indices in pattern_rows.items():
            time_index = torch.as_tensor(valid_indices, dtype=torch.long, device=device)
            batch_index = torch.as_tensor(row_indices, dtype=torch.long, device=device)
            compact = temporal_in.index_select(0, batch_index).index_select(1, time_index)
            causal_mask = torch.triu(
                torch.ones(len(valid_indices), len(valid_indices), dtype=torch.bool, device=device),
                diagonal=1,
            )
            encoded = self.temporal_encoder(compact, mask=causal_mask)
            temporal[batch_index[:, None], time_index[None, :], :] = encoded
        temporal = temporal.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)
        decoder_in = temporal.reshape(B * T, N, self.d_model)
        decoder_valid = spatial_valid_safe
        decoded = self.decoder(decoder_in, src_key_padding_mask=~decoder_valid)
        decoded = decoded.reshape(B, T, N, self.d_model)
        recon = self.reconstruct_head(decoded)
        if original_ndim == 3:
            return recon[:, 0], temporal[:, 0]
        return recon, temporal


def sample_training_mask(observed: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Sample a per-scene MAE target mask for a ``(B, T, N)`` observation mask.

    In every scene with at least two observed patches, ``round(ratio * n_obs)``
    observed patches (at least one, at most ``n_obs - 1``) are drawn uniformly
    without replacement from the global torch RNG. Scenes with fewer than two
    observed patches contribute no targets, so every scene keeps visible
    observed context. Unobserved patches are never targets.
    """
    if observed.ndim != 3 or observed.dtype != torch.bool:
        raise ValueError("observed must be a boolean (B, T, N) tensor")
    if not 0.0 < ratio < 1.0:
        raise ValueError("mask ratio must lie strictly between 0 and 1")
    masked = torch.zeros_like(observed)
    batch, steps, _ = observed.shape
    for b in range(batch):
        for t in range(steps):
            candidates = torch.nonzero(observed[b, t], as_tuple=False).flatten()
            n_obs = int(candidates.numel())
            if n_obs < 2:
                continue
            n_mask = min(n_obs - 1, max(1, int(round(ratio * n_obs))))
            chosen = candidates[torch.randperm(n_obs, device=observed.device)[:n_mask]]
            masked[b, t, chosen] = True
    return masked


def validation_mask(observed: torch.Tensor) -> torch.Tensor:
    """Deterministic VAL mask: every other observed patch (index order) per scene."""
    masked = torch.zeros_like(observed)
    for b in range(observed.shape[0]):
        for t in range(observed.shape[1]):
            valid_patch = torch.nonzero(observed[b, t], as_tuple=False).flatten()
            if valid_patch.numel():
                masked[b, t, valid_patch[::2]] = True
    return masked


def train_factorized_sr1_model(
    model: SR1TransformerMAE,
    records: List[Dict[str, Any]],
    *,
    fit_observation_ids: List[str],
    validation_observation_ids: List[str],
    max_epochs: int = 150,
    lr: float = 3e-4,
    min_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    max_context_scenes: int = 8,
    plateau_patience: int = 20,
    mask_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Train factorized spatial/temporal T-MAE with disjoint FIT and VAL IDs.

    Each optimizer example is a chronological block of at most
    ``max_context_scenes`` scenes; one block per optimizer step (no batching);
    temporal attention is causal and operates independently for each patch
    location. The validation observations are never included in optimizer
    batches or fitted preprocessing, and select the minimum masked validation
    loss checkpoint. Undefined channel targets are omitted via explicit masks.
    """
    if max_epochs < 1 or max_context_scenes < 1 or plateau_patience < 1:
        raise ValueError("training epochs and temporal context must be positive")
    if not fit_observation_ids or not validation_observation_ids:
        raise ValueError("non-empty FIT and VAL observation ID lists are required")
    fit_ids, val_ids = set(fit_observation_ids), set(validation_observation_ids)
    if fit_ids & val_ids:
        raise ValueError("FIT and VAL observation IDs overlap")
    by_id = {record.get("observation_id"): record for record in records}
    if len(by_id) != len(records) or None in by_id:
        raise ValueError("records must carry unique observation IDs")
    if (fit_ids | val_ids) - set(by_id):
        raise ValueError("split IDs are absent from supplied records")
    if any(by_id[oid].get("role") != "FIT" for oid in fit_ids):
        raise ValueError("optimizer input includes a scene not declared FIT")
    if any(by_id[oid].get("role") != "VAL" for oid in val_ids):
        raise ValueError("checkpoint-selection input includes a scene not declared VAL")
    if device.startswith("mps") and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")

    def make_blocks(ids: set[str]) -> List[List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for oid in ids:
            scene = by_id[oid]
            groups.setdefault(str(scene.get("site_id", "")), []).append(scene)
        blocks: List[List[Dict[str, Any]]] = []
        for site in sorted(groups):
            ordered = sorted(groups[site], key=lambda row: row["acquired_at"])
            for start in range(0, len(ordered), max_context_scenes):
                blocks.append(ordered[start:start + max_context_scenes])
        return blocks

    train_blocks = make_blocks(fit_ids)
    val_blocks = make_blocks(val_ids)
    if not train_blocks or not val_blocks:
        raise ValueError("split produced no temporal blocks")

    def tensors(block: List[Dict[str, Any]]):
        x_np = np.stack([np.asarray(row["std_features"], dtype=np.float32) for row in block])
        channel_valid_np = np.stack([np.asarray(row["observed"], dtype=bool) for row in block])
        if x_np.ndim != 3 or channel_valid_np.shape != x_np.shape:
            raise ValueError("each scene must provide aligned (patch,channel) values and masks")
        obs = channel_valid_np.any(axis=-1)
        zones = np.stack([np.asarray(row["zones"], dtype=np.int64) for row in block])
        times = np.stack([np.asarray(row["time_feats"], dtype=np.float32) for row in block])
        if zones.shape != obs.shape or times.shape != (len(block), 3):
            raise ValueError("zone/time metadata does not align with temporal block")
        return (
            torch.as_tensor(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(obs, dtype=torch.bool, device=device).unsqueeze(0),
            torch.as_tensor(channel_valid_np, dtype=torch.bool, device=device).unsqueeze(0),
            torch.as_tensor(zones, dtype=torch.long, device=device).unsqueeze(0),
            torch.as_tensor(times, dtype=torch.float32, device=device).unsqueeze(0),
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("mps"):
        torch.mps.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
    ema = EMA(model, decay=0.995)
    train_losses, val_losses = [], []
    val_blocks_without_targets: List[List[str]] = []
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    optimizer_steps = 0
    stopping_status = "MAX_EPOCH_BUDGET_REACHED"
    epochs_completed = 0

    for epoch in range(1, max_epochs + 1):
        epochs_completed = epoch
        model.train()
        total, batches = 0.0, 0
        order = np.random.permutation(len(train_blocks))
        for block_index in order:
            block = train_blocks[int(block_index)]
            x, observed, channel_valid, zones, times = tensors(block)
            masked = sample_training_mask(observed, mask_ratio)
            target_mask = masked.unsqueeze(-1) & channel_valid
            if not target_mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x, observed, masked, zones, times,
                                  modality_id=0)
            loss = (prediction[target_mask] - x[target_mask]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite masked loss; no zero-loss fallback is permitted")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            total += float(loss.detach().cpu())
            batches += 1
            optimizer_steps += 1
        if not batches:
            raise ValueError("FIT epoch has no supported masked targets")
        scheduler.step()
        train_losses.append(total / batches)

        original_state = copy.deepcopy(model.state_dict())
        ema.apply_to(model)
        model.eval()
        val_sse, val_entries, val_batches = 0.0, 0, 0
        block_entries: List[int] = []
        with torch.no_grad():
            for block in val_blocks:
                x, observed, channel_valid, zones, times = tensors(block)
                masked = validation_mask(observed)
                target_mask = masked.unsqueeze(-1) & channel_valid
                if not target_mask.any():
                    block_ids = [str(row["observation_id"]) for row in block]
                    if block_ids not in val_blocks_without_targets:
                        val_blocks_without_targets.append(block_ids)
                    continue
                prediction, _ = model(x, observed, masked, zones, times,
                                      modality_id=0)
                sse = (prediction[target_mask] - x[target_mask]).double().square().sum()
                if not torch.isfinite(sse):
                    block_ids = [str(row["observation_id"]) for row in block]
                    raise FloatingPointError(
                        "non-finite VAL masked loss; refusing to omit a validation block "
                        f"from checkpoint selection (observation_ids={block_ids})"
                    )
                n_entries = int(target_mask.sum())
                val_sse += float(sse.cpu())
                val_entries += n_entries
                block_entries.append(n_entries)
                val_batches += 1
        if not val_batches:
            raise ValueError("VAL split has no supported masked targets; checkpoint is unestimable")
        # Target-weighted: every VAL target entry has equal weight, so a block
        # holding one sparse scene cannot dominate checkpoint selection.
        mean_val = val_sse / val_entries
        val_losses.append(mean_val)
        if mean_val < best_val_loss:
            best_val_loss, best_epoch = mean_val, epoch
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(original_state)
        if epoch - best_epoch >= plateau_patience:
            stopping_status = "VALIDATION_PLATEAU"
            break

    if best_state is None:
        raise ValueError("no finite validation-selected checkpoint was produced")
    model.load_state_dict(best_state)
    return {
        "status": "COMPLETED_WITH_VALIDATION_SELECTED_CHECKPOINT",
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1], "final_val_loss": val_losses[-1],
        "train_loss_history": train_losses, "val_loss_history": val_losses,
        "optimizer_steps": optimizer_steps, "max_epochs": max_epochs,
        "epochs_completed": epochs_completed, "stopping_status": stopping_status,
        "plateau_patience": plateau_patience,
        "fit_observation_ids": sorted(fit_ids), "validation_observation_ids": sorted(val_ids),
        "fit_validation_disjoint": True, "max_context_scenes": max_context_scenes,
        "validation_block_count": len(val_blocks),
        "validation_blocks_scored": val_batches,
        "validation_blocks_without_targets": val_blocks_without_targets,
        "validation_target_entries": val_entries,
        "validation_target_entries_by_block": block_entries,
        "validation_loss_definition": "sum of squared errors over all VAL target entries / number of entries",
        "train_mask_ratio": mask_ratio,
        "train_mask_rule": "per scene: round(ratio*n_observed) observed patches, >=1 and <=n_observed-1; scenes with <2 observed patches give no targets",
        "train_block_count": len(train_blocks),
        "device": device, "seed": seed, "parameter_count": model.count_parameters(),
    }
