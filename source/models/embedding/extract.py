"""
Embedding Extraction Pipeline.

Extracts full temporal sequence embeddings and global pooled embeddings
for all 20 lakes (training + evaluation) using a trained TS-MAE checkpoint.

Maintains INV-002:
- Normalization statistics are loaded directly from the checkpoint
  (which were computed strictly from training-role lakes).
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional

source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from models.encoder.ts_mae import TimeSeriesMAE
from data.loaders.lake_dataset import GlacialLakeDataset, load_registry
from models.training.trainer import get_device


def extract_embeddings(
    checkpoint_path: str,
    features_dir: str,
    registry_path: str,
    output_dir: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Extract latent embeddings for all lakes using trained TS-MAE checkpoint.
    
    Args:
        checkpoint_path: Path to trained .pt checkpoint
        features_dir: Path to data/features/
        registry_path: Path to lake_registry.json
        output_dir: Path to output data/embeddings/ directory
        device: PyTorch device (optional, uses get_device() if None)
        
    Returns:
        Summary dictionary containing extraction metadata
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm_stats = checkpoint.get('norm_stats')
    if norm_stats is None:
        raise ValueError(f"Checkpoint at {checkpoint_path} missing 'norm_stats'")

    # Load model
    model = TimeSeriesMAE(
        n_channels=checkpoint.get('config', {}).get('model', {}).get('n_channels', 15),
        n_windows=checkpoint.get('config', {}).get('model', {}).get('n_windows', 108),
        d_model=checkpoint.get('config', {}).get('model', {}).get('d_model', 128),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    registry = load_registry(registry_path)
    all_lake_ids = [l['id'] for l in registry['lakes']]

    os.makedirs(output_dir, exist_ok=True)

    extracted_lakes = {}
    with torch.no_grad():
        for l_id in all_lake_ids:
            matrix_path = os.path.join(features_dir, l_id, 'feature_matrix.npz')
            if not os.path.exists(matrix_path):
                continue
                
            npz_data = np.load(matrix_path, allow_pickle=True)
            raw_features = npz_data['features'].astype(np.float32)
            window_dates = npz_data.get('window_dates', np.array([]))

            # Apply normalization stats from checkpoint (INV-002 safe)
            mean = norm_stats['mean']
            std = norm_stats['std']
            norm_features = (raw_features - mean) / std

            # Shape: (1, T, C)
            x_tensor = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)

            # Extract full temporal embeddings (1, T, d_model) and pooled (1, d_model)
            full_emb = model.get_full_embeddings(x_tensor).squeeze(0).cpu().numpy()
            pooled_emb = model.get_pooled_embedding(x_tensor).squeeze(0).cpu().numpy()

            lake_out_dir = os.path.join(output_dir, l_id)
            os.makedirs(lake_out_dir, exist_ok=True)
            out_file = os.path.join(lake_out_dir, 'embeddings.npz')

            np.savez_compressed(
                out_file,
                embeddings=full_emb,
                pooled_embedding=pooled_emb,
                window_dates=window_dates,
                lake_id=l_id,
            )

            extracted_lakes[l_id] = {
                "embeddings_shape": list(full_emb.shape),
                "pooled_shape": list(pooled_emb.shape),
                "out_file": out_file
            }

    summary = {
        "total_lakes_extracted": len(extracted_lakes),
        "checkpoint_used": checkpoint_path,
        "contributing_norm_lakes": norm_stats.get('contributing_lake_ids', []),
        "per_lake": extracted_lakes
    }

    summary_file = os.path.join(output_dir, 'embedding_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ckpt = os.path.join(repo_root, 'models', 'checkpoints', 'ts_mae_best.pt')
    feat = os.path.join(repo_root, 'data', 'features')
    reg = os.path.join(repo_root, 'source', 'data', 'registry', 'lake_registry.json')
    out = os.path.join(repo_root, 'data', 'embeddings')

    if os.path.exists(ckpt):
        extract_embeddings(ckpt, feat, reg, out)
        print("Embedding Extraction Pipeline Completed.")
