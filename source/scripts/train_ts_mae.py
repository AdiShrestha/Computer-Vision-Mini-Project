"""
TS-MAE Training Script — Real Data Version.

Trains the Temporal-Spatial Masked Autoencoder on real 13-channel features.
NaN-aware masking: pre-existing gaps always masked, loss only on valid positions.

Invariants:
    INV-002: Training data from training-role lakes only
    INV-004: Window size = 180, stride = 30
    INV-005: Masking ratio = 0.50 on valid positions
    INV-008: Must complete within 72 GPU-hours
    INV-012: Seeds pinned (42 for training)
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from models.encoder.ts_mae import TimeSeriesMAE

# INV-012: Pin all seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RealDataWindowDataset(Dataset):
    """Dataset yielding temporal windows from real feature matrices.

    NaN positions are tracked via a validity mask.
    """

    def __init__(self, feature_dir, lake_ids, norm_stats, window_size=180, stride=30):
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.validity_masks = []

        means = np.array(norm_stats['means'])
        stds = np.array(norm_stats['stds'])

        for lake_id in lake_ids:
            npz_path = feature_dir / lake_id / 'feature_matrix.npz'
            data = np.load(npz_path)
            features = data['features']  # (T, 13)

            # Create validity mask BEFORE normalization
            validity = ~np.isnan(features)  # True where valid

            # Z-score normalize (NaN becomes 0 after normalization)
            normed = (features - means) / stds
            normed = np.nan_to_num(normed, nan=0.0)

            # Extract windows
            T = features.shape[0]
            for start in range(0, T - window_size + 1, stride):
                window = normed[start:start + window_size]  # (180, 13)
                mask = validity[start:start + window_size]  # (180, 13)

                # Skip windows with >90% missing
                if mask.mean() < 0.10:
                    continue

                self.windows.append(window.astype(np.float32))
                self.validity_masks.append(mask.astype(np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.validity_masks[idx])
        )


def create_nan_aware_mask(validity_mask, masking_ratio=0.5):
    """Vectorized NaN-aware reconstruction mask creation.

    Args:
        validity_mask: (B, T, C) float tensor. 1.0 = valid, 0.0 = missing.
        masking_ratio: Fraction of VALID positions to mask for reconstruction.

    Returns:
        recon_mask: (B, T, C) float tensor. 1.0 = must reconstruct, 0.0 = visible or NaN.
    """
    rand = torch.rand_like(validity_mask)
    recon_mask = ((rand < masking_ratio) & (validity_mask > 0.5)).float()
    return recon_mask


def train_epoch(model, dataloader, optimizer, device, masking_ratio=0.5):
    """Train one epoch with model internal masking and NaN-aware loss mask."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for windows, validity in dataloader:
        windows = windows.to(device)      # (B, 180, 13)
        validity = validity.to(device)     # (B, 180, 13)

        # Model forward pass generates standard 50% temporal mask
        output = model(windows)
        reconstructed = output['reconstruction']  # (B, 180, 13)
        mask = output['mask']                      # (B, 180) bool: True = masked

        # Loss: MSE only on masked time steps AND valid non-NaN positions
        loss_mask = mask.unsqueeze(-1).float() * validity  # (B, 180, 13)
        n_loss_positions = loss_mask.sum()

        if n_loss_positions > 0:
            loss = ((reconstructed - windows) ** 2 * loss_mask).sum() / n_loss_positions
        else:
            loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def extract_embeddings(model, feature_dir, lake_ids, norm_stats, output_dir, device,
                       window_size=180, stride=30):
    """Extract embeddings for all lakes (training + evaluation)."""
    model.eval()
    means = np.array(norm_stats['means'])
    stds = np.array(norm_stats['stds'])

    output_dir.mkdir(parents=True, exist_ok=True)

    for lake_id in lake_ids:
        npz_path = feature_dir / lake_id / 'feature_matrix.npz'
        data = np.load(npz_path)
        features = data['features']
        dates = data['dates']

        normed = (features - means) / stds
        normed = np.nan_to_num(normed, nan=0.0)

        T = features.shape[0]
        window_list = []
        window_dates = []

        for start in range(0, T - window_size + 1, stride):
            window = normed[start:start + window_size]
            window_list.append(window.astype(np.float32))
            window_dates.append(str(dates[start + window_size // 2]))

        if not window_list:
            continue

        # Stack into batch (N_windows, 180, 13)
        batch_tensor = torch.from_numpy(np.array(window_list)).to(device)

        with torch.no_grad():
            # (N_windows, 180, 128) -> mean pool over T -> (N_windows, 128)
            emb_latents = model.encode(batch_tensor)  # (N_windows, 180, 128)
            embs = emb_latents.mean(dim=1).cpu().numpy()  # (N_windows, 128)

        lake_dir = output_dir / lake_id
        lake_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            lake_dir / 'embeddings.npz',
            embeddings=embs,
            dates=window_dates
        )
        print(f"  {lake_id}: {embs.shape[0]} windows, embedding shape {embs.shape}", flush=True)


def main():
    # Gate check
    gate_path = PROJECT_ROOT / 'results' / 'reality_gate' / 'reality_gate_data.json'
    if gate_path.exists():
        with open(gate_path, 'r', encoding='utf-8') as f:
            gate = json.load(f)
        if gate['overall_verdict'] == 'FAIL':
            print("ERROR: Reality Gate returned FAIL. Cannot train.")
            sys.exit(1)
        print(f"Reality Gate: {gate['overall_verdict']} — proceeding.")

    # Load registry
    reg_path = PROJECT_ROOT / 'source' / 'data' / 'registry' / 'lake_registry.json'
    with open(reg_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    feature_dir = PROJECT_ROOT / 'data' / 'features_real'
    norm_path = feature_dir / 'normalization_stats.json'
    with open(norm_path, 'r', encoding='utf-8') as f:
        norm_stats = json.load(f)

    # INV-002: Training data from training-role lakes ONLY
    training_lake_ids = [l['id'] for l in registry['lakes'] if l['role'] == 'training']
    all_lake_ids = [l['id'] for l in registry['lakes']]
    eval_lake_ids = [l['id'] for l in registry['lakes'] if l['role'] != 'training']
    print(f"Training lakes: {len(training_lake_ids)}")
    print(f"Evaluation lakes: {len(eval_lake_ids)} (excluded from training — INV-002)")

    # Dataset
    dataset = RealDataWindowDataset(
        feature_dir, training_lake_ids, norm_stats,
        window_size=180, stride=30  # INV-004
    )
    print(f"Training windows: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    # Device: use MPS on Apple Silicon
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Model — 13 channels instead of 15
    model = TimeSeriesMAE(
        n_channels=13,
        n_windows=180,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=2,
        n_encoder_heads=4,
        n_decoder_heads=4,
        masking_ratio=0.5
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    start_time = time.time()
    best_loss = float('inf')
    losses = []

    n_epochs = 25
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device, masking_ratio=0.5)
        scheduler.step()
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            ckpt_path = PROJECT_ROOT / 'models' / 'checkpoints' / 'ts_mae_real_data.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'n_channels': 13,
                'window_size': 180,
                'seed': SEED,
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{n_epochs}: loss={loss:.6f}  best={best_loss:.6f}  "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

        # Divergence check
        if np.isnan(loss) or np.isinf(loss):
            print(f"ERROR: Training diverged at epoch {epoch} (loss={loss})")
            sys.exit(1)

    total_time = time.time() - start_time

    # Training summary
    summary = {
        'training_version': 'real_data_v1',
        'n_channels': 13,
        'window_size': 180,
        'stride': 30,
        'masking_ratio': 0.5,
        'n_epochs': n_epochs,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'training_time_seconds': total_time,
        'training_time_hours': total_time / 3600,
        'device': str(device),
        'parameter_count': param_count,
        'n_training_lakes': len(training_lake_ids),
        'training_lake_ids': training_lake_ids,
        'n_training_windows': len(dataset),
        'seed': SEED,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'optimizer': 'AdamW',
        'weight_decay': 0.01,
        'scheduler': 'CosineAnnealingLR',
        'loss_history': losses
    }

    summary_path = PROJECT_ROOT / 'models' / 'encoder' / 'training_summary_real_data.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete: {n_epochs} epochs, {total_time/60:.1f} min")
    print(f"Best loss: {best_loss:.6f}")

    # Load best checkpoint for embedding extraction
    ckpt = torch.load(
        PROJECT_ROOT / 'models' / 'checkpoints' / 'ts_mae_real_data.pt',
        map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})")

    # Extract embeddings for ALL lakes (training + evaluation)
    print("\nExtracting embeddings...")
    embedding_dir = PROJECT_ROOT / 'data' / 'embeddings' / 'real_data'
    extract_embeddings(model, feature_dir, all_lake_ids, norm_stats, embedding_dir, device)
    print("Done.")


if __name__ == '__main__':
    main()
