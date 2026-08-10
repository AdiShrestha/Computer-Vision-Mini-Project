"""
TS-MAE Pretraining CLI Execution Script.

Orchestrates self-supervised pretraining of the TimeSeriesMAE model:
- Configures deterministic random seeds (INV-012)
- Creates leakage-safe training & validation data loaders (INV-002)
- Instantiates TS-MAE model and Trainer
- Trains for specified epochs with Cosine LR decay and early stopping
- Saves checkpoints to models/checkpoints/ and summaries to results/training/
"""

import os
import sys
import json
import argparse
import torch

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from utils.config_loader import load_config
from utils.reproducibility import set_seed
from utils.logging_utils import setup_logger
from models.encoder.ts_mae import TimeSeriesMAE
from data.loaders.lake_dataset import create_data_loaders
from models.training.trainer import Trainer, get_device


def main():
    parser = argparse.ArgumentParser(description="TS-MAE Self-Supervised Pretraining Runner")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    set_seed(42)  # INV-012
    logger = setup_logger("train_encoder")
    config = load_config()

    # Update config with CLI parameters
    if 'training' not in config:
        config['training'] = {}
    config['training']['epochs'] = args.epochs
    config['training']['lr'] = args.lr
    config['training']['batch_size'] = args.batch_size

    repo_root = os.path.dirname(source_root)
    features_dir = os.path.join(repo_root, config['paths']['features'])
    registry_path = os.path.join(repo_root, config['paths']['lake_registry'])
    output_dir = os.path.join(repo_root, 'results', 'training')
    checkpoints_dir = os.path.join(repo_root, 'models', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    logger.info(f"Target execution device: {device}")

    # Create DataLoaders
    train_loader, val_loader, norm_stats = create_data_loaders(
        features_dir=features_dir,
        registry_path=registry_path,
        batch_size=args.batch_size,
        val_fraction=0.2,
        val_split_seed=7,
        num_workers=0
    )

    # Instantiate Model
    model = TimeSeriesMAE(
        n_channels=15,
        n_windows=108,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=2,
        n_encoder_heads=8,
        n_decoder_heads=4,
        masking_ratio=config.get('training', {}).get('masking_ratio', 0.5)
    )

    logger.info(f"Model instantiated with {model.count_parameters():,} trainable parameters.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        norm_stats=norm_stats
    )

    logger.info(f"Starting training for {args.epochs} epochs...")
    fit_result = trainer.fit(n_epochs=args.epochs)

    # Save final checkpoint and best checkpoint in models/checkpoints/
    best_ckpt = os.path.join(trainer.checkpoint_dir, 'checkpoint_best.pt')
    final_ckpt = os.path.join(trainer.checkpoint_dir, 'checkpoint_latest.pt')

    target_best_ckpt = os.path.join(checkpoints_dir, 'ts_mae_best.pt')
    target_final_ckpt = os.path.join(checkpoints_dir, 'ts_mae_final.pt')

    if os.path.exists(best_ckpt):
        torch.save(torch.load(best_ckpt, weights_only=False, map_location='cpu'), target_best_ckpt)
    if os.path.exists(final_ckpt):
        torch.save(torch.load(final_ckpt, weights_only=False, map_location='cpu'), target_final_ckpt)

    history = fit_result['history']
    initial_loss = history[0]['train_loss'] if history else 0.0
    final_loss = history[-1]['train_loss'] if history else 0.0

    summary = {
        "model": "TimeSeriesMAE",
        "device": str(device),
        "epochs_total": len(history),
        "epochs_best": fit_result['best_epoch'],
        "train_loss_initial": float(initial_loss),
        "train_loss_final": float(final_loss),
        "val_loss_best": float(fit_result['best_val_loss']),
        "wall_time_seconds": float(sum(h['wall_time_s'] for h in history)),
        "convergence": "converged" if final_loss < initial_loss else "diverged",
        "seed": 42,
        "checkpoint_path": "models/checkpoints/ts_mae_best.pt"
    }

    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pretraining completed. Best Val Loss: {fit_result['best_val_loss']:.4f}. Summary saved to {summary_path}")
    print(f"\nTS-MAE Pretraining Complete. Final Train Loss: {final_loss:.4f}, Best Val Loss: {fit_result['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
