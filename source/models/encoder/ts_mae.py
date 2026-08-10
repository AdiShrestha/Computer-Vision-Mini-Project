"""
Time-Series Masked Autoencoder (TS-MAE) — Path B Encoder.

Architecture Decision: Decision 002 (project/evolution/decision_log.md)
Specification: source/models/encoder/architecture_spec.md

This module implements the self-supervised encoder for multi-channel
glacial lake time series. The encoder learns representations by
reconstructing randomly masked temporal patches.

Input:  (batch_size, n_windows, n_channels) — e.g., (16, 108, 15)
Output: (batch_size, n_windows, d_model) — e.g., (16, 108, 128)

Invariant compliance:
    INV-005: masking_ratio=0.5 (configurable, verified in training loop)
    INV-012: Seed-controlled masking for reproducibility
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PatchProjection(nn.Module):
    """Project each time step's channel vector to the latent dimension.
    
    Input:  (B, T, C)  where C = n_channels (15)
    Output: (B, T, d_model) where d_model = 128
    """
    
    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(n_channels, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned 1D positional embeddings for temporal positions.
    
    Each of the T time windows gets a unique learned embedding vector
    added to its projected representation.
    """
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings. x shape: (B, T, d_model)."""
        T = x.size(1)
        return x + self.pos_embedding[:, :T, :]


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm architecture.
    
    Pre-norm (LayerNorm before attention/FFN) is more stable for training
    than post-norm, especially important for self-supervised objectives
    where gradients through the reconstruction path can be noisy.
    """
    
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
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block for reconstruction.
    
    Uses self-attention only (no cross-attention) — the decoder sees
    the full sequence (unmasked latents + mask tokens) and reconstructs
    the original channel values for masked positions.
    """
    
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
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TimeSeriesMAE(nn.Module):
    """Time-Series Masked Autoencoder.
    
    Architecture (from architecture_spec.md):
        Input (B, T, C=15)
            → Linear Patch Projection (15 → d_model=128)
            → Add Learned Positional Embeddings
            → Random 50% Temporal Masking (INV-005)
            → Transformer Encoder (4 layers, 8 heads, d_ff=512)
            → Latent Representations (B, T_unmasked, 128)
            → Re-insert Mask Tokens
            → Transformer Decoder (2 layers, 4 heads, d_ff=256)
            → Linear Reconstruction Head (128 → 15)
            → Reconstructed (B, T, C=15)
    
    Args:
        n_channels: Number of input channels (default: 15)
        n_windows: Number of time windows (default: 108)
        d_model: Latent dimension (default: 128)
        n_encoder_layers: Transformer encoder layers (default: 4)
        n_decoder_layers: Transformer decoder layers (default: 2)
        n_encoder_heads: Encoder attention heads (default: 8)
        n_decoder_heads: Decoder attention heads (default: 4)
        d_ff_encoder: Encoder FFN dimension (default: 512)
        d_ff_decoder: Decoder FFN dimension (default: 256)
        masking_ratio: Fraction of time steps to mask (default: 0.5, INV-005)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        n_channels: int = 15,
        n_windows: int = 108,
        d_model: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        n_encoder_heads: int = 8,
        n_decoder_heads: int = 4,
        d_ff_encoder: int = 512,
        d_ff_decoder: int = 256,
        masking_ratio: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_windows = n_windows
        self.d_model = d_model
        self.masking_ratio = masking_ratio
        
        # Input projection: C → d_model
        self.patch_projection = PatchProjection(n_channels, d_model)
        
        # Positional embeddings
        self.pos_embedding = LearnedPositionalEmbedding(n_windows, d_model)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_encoder_heads, d_ff_encoder, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # Mask token (learned, shared across all masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Decoder positional embeddings (separate from encoder's)
        self.decoder_pos_embedding = LearnedPositionalEmbedding(n_windows, d_model)
        
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
    
    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _generate_mask(self, batch_size: int, seq_len: int, 
                       device: torch.device) -> torch.Tensor:
        """Generate random binary mask for temporal masking.
        
        Returns:
            mask: (B, T) boolean tensor. True = MASKED (will be reconstructed).
        
        The number of masked positions is deterministic for a given seq_len
        and masking_ratio: floor(seq_len * masking_ratio).
        The WHICH positions are masked is random per sample in the batch.
        """
        n_masked = int(seq_len * self.masking_ratio)
        
        # Generate random permutation indices per batch element
        # Then take the first n_masked as the masked positions
        noise = torch.rand(batch_size, seq_len, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        # The first n_masked positions in the shuffled order are masked
        mask_indices = ids_shuffle[:, :n_masked]
        mask.scatter_(1, mask_indices, True)
        
        return mask
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input through the encoder (unmasked patches only during training).
        
        For inference (mask=None): encode all time steps.
        For training (mask provided): encode only unmasked time steps.
        
        Args:
            x: (B, T, C) input tensor
            mask: (B, T) boolean mask. True = masked (excluded from encoder).
                  None for inference (encode everything).
        
        Returns:
            latent: (B, T_visible, d_model) if mask provided, 
                    (B, T, d_model) if mask is None.
        """
        # Project and add positional embeddings
        x = self.patch_projection(x)
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
        """Reconstruct full sequence from encoder output + mask tokens.
        
        Args:
            latent: (B, T_visible, d_model) encoder output
            mask: (B, T) original mask
            seq_len: original sequence length T
        
        Returns:
            reconstructed: (B, T, d_model) full sequence (before recon head)
        """
        B, _, D = latent.shape
        
        # Create full sequence: visible patches + mask tokens
        full_sequence = torch.zeros(B, seq_len, D, device=latent.device)
        
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
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> dict:
        """Forward pass.
        
        Training mode (mask=None auto-generates):
            Returns dict with 'reconstruction', 'mask', 'loss'.
        
        Inference mode (call encode() directly):
            Use model.encode(x) for embedding extraction.
        
        Args:
            x: (B, T, C) input tensor
            mask: Optional pre-generated mask. If None, generates one.
        
        Returns:
            dict with:
                'reconstruction': (B, T, C) reconstructed input
                'mask': (B, T) boolean mask used
                'loss': scalar MSE loss over masked positions only
                'latent': (B, T_visible, d_model) encoder output
        """
        B, T, C = x.shape
        
        # Generate mask if not provided
        if mask is None:
            mask = self._generate_mask(B, T, x.device)
        
        # Encode (visible patches only)
        latent = self.encode(x, mask=mask)
        
        # Decode (full sequence reconstruction)
        decoded = self.decode(latent, mask, T)
        
        # Reconstruction head
        reconstruction = self.reconstruction_head(decoded)
        
        # Compute MSE loss over MASKED positions only
        # This is the core self-supervised objective:
        # the model is scored only on its ability to reconstruct
        # the parts it didn't see.
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstruction)
        masked_pred = reconstruction[mask_expanded]
        masked_target = x[mask_expanded]
        
        loss = nn.functional.mse_loss(masked_pred, masked_target)
        
        return {
            'reconstruction': reconstruction,
            'mask': mask,
            'loss': loss,
            'latent': latent,
        }
    
    def get_full_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract full-sequence embeddings for inference (no masking).
        
        This is the method used for embedding extraction in Chunk 03's
        downstream contracts. It runs the full sequence through the encoder
        without any masking.
        
        Args:
            x: (B, T, C) input tensor (already normalized)
        
        Returns:
            embeddings: (B, T, d_model) per-window embeddings
        """
        return self.encode(x, mask=None)
    
    def get_pooled_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a single global embedding per sample (mean-pooled).
        
        Used for anomaly scoring via embedding distance (Score-B).
        
        Args:
            x: (B, T, C) input tensor (already normalized)
        
        Returns:
            embedding: (B, d_model) global embedding
        """
        full_emb = self.get_full_embeddings(x)
        return full_emb.mean(dim=1)  # Mean pool over time
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
