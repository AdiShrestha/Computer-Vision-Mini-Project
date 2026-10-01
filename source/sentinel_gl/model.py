"""Temporal masked autoencoder, derived from the legacy Sentinel-GL architecture.

This is a numerical building block, not a trained or validated GLOF model.
Input values are normalized; boolean observation masks are explicit inputs.
Full attention is permitted only INSIDE a trailing window that has already
closed. Callers must never interpret embeddings as available at window centres.
Default dimensions are engineering defaults, not selected research settings.
Legacy checkpoints are incompatible with the explicit observation-mask input.
"""

import torch
import torch.nn as nn
import math
import warnings
from typing import Tuple, Optional


def evaluation_masks(validity, partitions=2):
    """Value-independent cross masks, balanced by observed time positions.

    Each position is hidden exactly once. Fold capacities depend only on T,
    so each fold has equal visible counts across batch samples. At least two
    observed time positions are needed to retain real visible context.
    """
    if validity.dtype != torch.bool or validity.ndim != 3:
        raise ValueError("validity must be a boolean B,T,C tensor")
    batch, steps, _ = validity.shape
    if type(partitions) is not int or not 2 <= partitions <= steps:
        raise ValueError("partitions must be an integer between two and context length")
    masks = [torch.zeros(batch, steps, dtype=torch.bool, device=validity.device)
             for _ in range(partitions)]
    capacities = [len(range(part, steps, partitions)) for part in range(partitions)]
    for row in range(batch):
        observed = validity[row].any(dim=-1)
        indices = torch.nonzero(observed).flatten().tolist()
        if len(indices) < 2:
            raise ValueError("masked scoring needs at least two observed time positions")
        groups = [indices[part::partitions] for part in range(partitions)]
        missing = iter(torch.nonzero(~observed).flatten().tolist())
        for part, group in enumerate(groups):
            group.extend(next(missing) for _ in range(capacities[part]-len(group)))
            masks[part][row, group] = True
    return tuple(masks)


class PatchProjection(nn.Module):

    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(2 * n_channels, d_model)

    def forward(self, x: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat((x, validity.to(x.dtype)), dim=-1))


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.pos_embedding.size(1):
            raise ValueError("sequence exceeds configured positional capacity")
        return x + self.pos_embedding[:, :T, :]


class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TimeSeriesMAE(nn.Module):

    def __init__(
        self,
        n_channels: int = 15,
        n_windows: Optional[int] = None,
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_encoder_heads: int = 8,
        n_decoder_heads: int = 4,
        d_ff_encoder: int = 512,
        d_ff_decoder: int = 256,
        masking_ratio: float = 0.5,
        dropout: float = 0.1,
        *,
        max_time_steps: Optional[int] = None,
    ):
        super().__init__()
        if n_windows is not None and max_time_steps is not None:
            raise ValueError("specify max_time_steps only; n_windows is a deprecated alias")
        if n_windows is not None:
            warnings.warn("n_windows means time steps; use max_time_steps", DeprecationWarning, stacklevel=2)
        capacity = max_time_steps if max_time_steps is not None else n_windows if n_windows is not None else 108
        dimensions = (n_channels, capacity, d_model, n_encoder_layers,
                      n_decoder_layers, n_encoder_heads, n_decoder_heads,
                      d_ff_encoder, d_ff_decoder)
        if any(type(value) is not int or value < 1 for value in dimensions) or capacity < 2:
            raise ValueError("model dimensions must be positive integers; max_time_steps >= 2")
        if d_model % n_encoder_heads or d_model % n_decoder_heads:
            raise ValueError("model dimension must be divisible by both attention head counts")
        if type(masking_ratio) not in (int, float) or not math.isfinite(masking_ratio) or not 0 < masking_ratio < 1:
            raise ValueError("masking_ratio must be strictly between zero and one")
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.n_channels = n_channels
        self.max_time_steps = capacity
        self.d_model = d_model
        self.masking_ratio = masking_ratio
        self._configuration = dict(n_channels=n_channels, max_time_steps=capacity,
            d_model=d_model, n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers, n_encoder_heads=n_encoder_heads,
            n_decoder_heads=n_decoder_heads, d_ff_encoder=d_ff_encoder,
            d_ff_decoder=d_ff_decoder, masking_ratio=masking_ratio, dropout=dropout)

        # Input projection: values plus observation mask, 2C → d_model
        self.patch_projection = PatchProjection(n_channels, d_model)

        # Positional embeddings
        self.pos_embedding = LearnedPositionalEmbedding(capacity, d_model)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_encoder_heads, d_ff_encoder, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Mask token (learned, shared across all masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Decoder positional embeddings (separate from encoder's)
        self.decoder_pos_embedding = LearnedPositionalEmbedding(capacity, d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_decoder_heads, d_ff_decoder, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Reconstruction head: d_model → C
        self.reconstruction_head = nn.Linear(d_model, n_channels)

        # Initialize weights
        self._init_weights()

    @property
    def n_windows(self):
        """Deprecated read-only alias for maximum time steps, not context count."""
        return self.max_time_steps

    @property
    def configuration(self):
        """Resolved constructor defaults for prospective checkpoint metadata."""
        return self._configuration.copy()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _generate_mask(self, batch_size: int, seq_len: int,
                       device: torch.device, validity: torch.Tensor) -> torch.Tensor:
        if seq_len < 2:
            raise ValueError("masked reconstruction needs at least two time steps")
        n_masked = min(seq_len - 1, max(1, int(seq_len * self.masking_ratio)))

        # Generate random permutation indices per batch element
        # Then take the first n_masked as the masked positions
        noise = torch.rand(batch_size, seq_len, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        # Preserve one observed target and one observed visible time position.
        # No measurement values or event labels influence mask selection.
        for row in range(batch_size):
            order = ids_shuffle[row]
            observed = order[validity[row].any(dim=-1)[order]]
            if observed.numel() < 2:
                raise ValueError("masked training needs at least two observed time positions")
            target, visible = observed[0], observed[1]
            remainder = order[(order != target) & (order != visible)]
            chosen = torch.cat((target.reshape(1), remainder[:n_masked-1]))
            mask[row, chosen] = True

        return mask

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, validity: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project and add positional embeddings
        validity = self._validate_input(x, validity)
        if mask is not None:
            self._validate_mask(mask, x)
            if torch.any(((~mask).unsqueeze(-1) & validity).sum(dim=(1, 2)) == 0):
                raise ValueError("masked reconstruction requires observed visible context")
        x = self.patch_projection(x, validity)
        x = self.pos_embedding(x)

        if mask is not None:
            # Keep only unmasked (visible) patches for encoder
            # mask=True means MASKED, so we keep where mask=False
            B, T, D = x.shape

            # Number of visible patches (same for all batch elements)
            visible_bool = mask.logical_not()  # True where visible
            n_visible = visible_bool.sum(dim=1)[0].item()

            # Sort indices to preserve temporal order for visible patches
            ids_visible = torch.argsort(visible_bool.float(), dim=1, descending=True)[:, :n_visible]
            ids_visible = ids_visible.sort(dim=1).values

            # Gather visible patches
            x = torch.gather(x, 1, ids_visible.unsqueeze(-1).expand(-1, -1, D))

        # Run through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.encoder_norm(x)
        return x

    def decode(self, latent: torch.Tensor, mask: torch.Tensor,
               seq_len: int) -> torch.Tensor:
        B, _, D = latent.shape

        # Create full sequence: visible patches + mask tokens
        full_sequence = torch.zeros(B, seq_len, D, device=latent.device, dtype=latent.dtype)

        # Place visible (encoded) patches back in their original positions
        visible_bool = mask.logical_not()
        n_visible = visible_bool.sum(dim=1)[0].item()
        ids_visible = torch.argsort(visible_bool.float(), dim=1, descending=True)[:, :n_visible]
        ids_visible = ids_visible.sort(dim=1).values

        full_sequence.scatter_(1, ids_visible.unsqueeze(-1).expand(-1, -1, D), latent)

        # Place mask tokens in masked positions
        mask_tokens = self.mask_token.expand(B, seq_len, -1)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
        full_sequence = torch.where(mask_expanded, mask_tokens, full_sequence)

        # Add decoder positional embeddings
        full_sequence = self.decoder_pos_embedding(full_sequence)

        # Run through decoder layers
        for layer in self.decoder_layers:
            full_sequence = layer(full_sequence)

        full_sequence = self.decoder_norm(full_sequence)
        return full_sequence

    def _validate_input(self, x, validity):
        if x.ndim != 3 or x.shape[0] == 0 or not 2 <= x.shape[1] <= self.max_time_steps or x.shape[2] != self.n_channels:
            raise ValueError("input must be nonempty B,T,C within configured dimensions")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("model input must be finite floating point normalized values")
        if validity is None:
            raise ValueError("an explicit observation validity mask is required")
        if validity.dtype != torch.bool or validity.shape != x.shape or validity.device != x.device:
            raise ValueError("validity must be a boolean tensor matching input shape/device")
        if torch.any(x[~validity] != 0):
            raise ValueError("unobserved model inputs must be explicitly zero imputed in normalized space")
        if torch.any(validity.sum(dim=(1, 2)) == 0):
            raise ValueError("one or more contexts contain no observed input")
        return validity

    def _validate_mask(self, mask, x):
        if mask.dtype != torch.bool or mask.shape != x.shape[:2] or mask.device != x.device:
            raise ValueError("temporal mask must be boolean B,T on the input device")
        counts = (~mask).sum(dim=1)
        if torch.any(counts == 0) or not torch.all(counts == counts[0]):
            raise ValueError("each sample needs an equal positive number of visible time steps")

    def reconstruct(self, x, mask, validity=None):
        validity = self._validate_input(x, validity)
        self._validate_mask(mask, x)
        latent = self.encode(x, mask=mask, validity=validity)
        reconstruction = self.reconstruction_head(self.decode(latent, mask, x.shape[1]))
        return reconstruction, latent

    def forward(self, x, mask=None, validity=None):
        validity = self._validate_input(x, validity)
        if mask is None:
            mask = self._generate_mask(x.shape[0], x.shape[1], x.device, validity)
        reconstruction, latent = self.reconstruct(x, mask, validity)
        loss_mask = mask.unsqueeze(-1) & validity
        if torch.any(loss_mask.sum(dim=(1, 2)) == 0):
            raise ValueError("no observed masked targets: loss is not estimable")
        loss = ((reconstruction[loss_mask] - x[loss_mask]) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite reconstruction loss")
        return {'reconstruction': reconstruction, 'mask': mask, 'loss': loss,
                'latent': latent, 'target_count': int(loss_mask.sum().item())}

    def get_full_embeddings(self, x: torch.Tensor, validity: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode(x, mask=None, validity=validity)

    def get_pooled_embedding(self, x: torch.Tensor, validity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean over calendar positions, including explicitly missing positions.

        They carry observation masks and time positions, not implicit padding.
        Missingness can affect embeddings; baselines/ablations must test whether
        it explains apparent skill before any physical interpretation.
        """
        full_emb = self.get_full_embeddings(x, validity=validity)
        return full_emb.mean(dim=1)  # Mean pool over time

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
