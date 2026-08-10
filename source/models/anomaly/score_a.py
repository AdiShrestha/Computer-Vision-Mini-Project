"""
Score-A: Reconstruction Error Scorer.

Computes per-window reconstruction MSE by passing NORMALIZED feature matrices
(using checkpoint's norm_stats) through the trained TS-MAE model.
Higher reconstruction error = higher anomaly score.

IMPORTANT: Features MUST be normalized using the checkpoint's norm_stats
before model inference. The model was trained on normalized features.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from models.encoder.ts_mae import TimeSeriesMAE
from models.training.trainer import get_device


class ReconstructionScorer:
    """Score-A: Reconstruction MSE Scorer with proper normalization."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        norm_stats: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize scorer.

        Args:
            checkpoint_path: Path to .pt checkpoint (loads model + norm_stats)
            model: Pre-loaded model (if provided, checkpoint_path is ignored for model)
            norm_stats: Pre-loaded norm_stats dict with 'mean' and 'std' keys
            device: Torch device
        """
        self.device = device or get_device()
        self.norm_mean = None
        self.norm_std = None

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Load model
            if model is not None:
                self.model = model.to(self.device)
            else:
                self.model = TimeSeriesMAE(
                    n_channels=ckpt.get('config', {}).get('model', {}).get('n_channels', 15),
                    n_windows=ckpt.get('config', {}).get('model', {}).get('n_windows', 108),
                    d_model=ckpt.get('config', {}).get('model', {}).get('d_model', 128),
                )
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.to(self.device)

            # Load norm_stats from checkpoint
            if norm_stats is not None:
                self.norm_mean = np.array(norm_stats['mean'], dtype=np.float32)
                self.norm_std = np.array(norm_stats['std'], dtype=np.float32)
            elif 'norm_stats' in ckpt:
                self.norm_mean = np.array(ckpt['norm_stats']['mean'], dtype=np.float32)
                self.norm_std = np.array(ckpt['norm_stats']['std'], dtype=np.float32)
        elif model is not None:
            self.model = model.to(self.device)
            if norm_stats is not None:
                self.norm_mean = np.array(norm_stats['mean'], dtype=np.float32)
                self.norm_std = np.array(norm_stats['std'], dtype=np.float32)
        else:
            # Default instantiation for smoke testing
            self.model = TimeSeriesMAE().to(self.device)

        self.model.eval()

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply checkpoint normalization to raw features.

        If norm_stats are not available, returns features unchanged.
        """
        if self.norm_mean is not None and self.norm_std is not None:
            return (features - self.norm_mean) / (self.norm_std + 1e-8)
        return features

    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute per-window reconstruction error.

        This method normalizes raw features using the checkpoint's
        norm_stats before model inference. MSE is computed in normalized space.

        Args:
            features: (T, C) raw feature matrix (NOT pre-normalized)

        Returns:
            scores: (T,) array of per-window MSE reconstruction error
        """
        features_arr = np.asarray(features, dtype=np.float32)
        T, C = features_arr.shape

        # Normalize features using training-set statistics (INV-002 compliant)
        normed = self._normalize(features_arr)

        x_tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mask_zero = torch.zeros((1, T), dtype=torch.bool, device=self.device)
            out = self.model(x_tensor, mask=mask_zero)
            recon = out['reconstruction'].squeeze(0).cpu().numpy()

        # Per-window MSE in normalized space
        window_mse = np.mean((normed - recon) ** 2, axis=-1)
        return window_mse.astype(np.float32)

    def get_embeddings(self, features: np.ndarray) -> np.ndarray:
        """Extract encoder embeddings from features (for Score-B/C E3 pipeline).

        Args:
            features: (T, C) raw feature matrix

        Returns:
            embeddings: (T, d_model) encoder embeddings
        """
        features_arr = np.asarray(features, dtype=np.float32)
        T, C = features_arr.shape

        normed = self._normalize(features_arr)
        x_tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model.encode(x_tensor)
            embeddings = out.squeeze(0).cpu().numpy()

        return embeddings.astype(np.float32)
