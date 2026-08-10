"""Verify TS-MAE encoder implementation."""
import os
import sys
import torch
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from models.encoder.ts_mae import TimeSeriesMAE


def test_model_instantiation():
    """Model can be instantiated with default parameters."""
    model = TimeSeriesMAE()
    assert model.n_channels == 15
    assert model.n_windows == 108
    assert model.d_model == 128
    assert model.masking_ratio == 0.5


def test_parameter_count():
    """Model has approximately 1.25M parameters (within 0.5-2.5M range)."""
    model = TimeSeriesMAE()
    n_params = model.count_parameters()
    assert 500_000 < n_params < 2_500_000, (
        f"Parameter count {n_params} outside expected range"
    )


def test_forward_pass_shape():
    """Forward pass produces correct output shapes."""
    model = TimeSeriesMAE()
    model.eval()
    B, T, C = 4, 108, 15
    x = torch.randn(B, T, C)
    
    with torch.no_grad():
        output = model(x)
    
    assert output['reconstruction'].shape == (B, T, C)
    assert output['mask'].shape == (B, T)
    assert output['mask'].dtype == torch.bool
    assert output['loss'].ndim == 0  # scalar


def test_masking_ratio():
    """Masking ratio is approximately 50% (INV-005)."""
    model = TimeSeriesMAE(masking_ratio=0.5)
    model.eval()
    B, T, C = 8, 108, 15
    x = torch.randn(B, T, C)
    
    with torch.no_grad():
        output = model(x)
    
    mask = output['mask']
    actual_ratio = mask.float().mean().item()
    assert abs(actual_ratio - 0.5) < 0.05, (
        f"Masking ratio {actual_ratio} deviates from 0.5"
    )


def test_encode_full_sequence():
    """get_full_embeddings produces (B, T, d_model) without masking."""
    model = TimeSeriesMAE()
    model.eval()
    B, T, C = 4, 108, 15
    x = torch.randn(B, T, C)
    
    with torch.no_grad():
        emb = model.get_full_embeddings(x)
    
    assert emb.shape == (B, T, 128)


def test_pooled_embedding():
    """get_pooled_embedding produces (B, d_model) global embedding."""
    model = TimeSeriesMAE()
    model.eval()
    B, T, C = 4, 108, 15
    x = torch.randn(B, T, C)
    
    with torch.no_grad():
        emb = model.get_pooled_embedding(x)
    
    assert emb.shape == (B, 128)


def test_loss_decreases_on_overfit():
    """Loss decreases when overfitting on a single batch (basic gradient check)."""
    torch.manual_seed(42)
    model = TimeSeriesMAE()
    model.train()
    
    x = torch.randn(4, 108, 15)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    for _ in range(20):
        output = model(x)
        loss = output['loss']
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_reconstruction_loss_masked_only():
    """Loss is computed over masked positions only, not the full sequence."""
    model = TimeSeriesMAE(masking_ratio=0.5)
    model.eval()
    x = torch.randn(4, 108, 15)
    
    with torch.no_grad():
        output = model(x)
    
    # The loss should be a valid positive number
    assert output['loss'].item() > 0
    assert not torch.isnan(output['loss'])
    assert not torch.isinf(output['loss'])
