"""
Score-A: Reconstruction-based Anomaly Scorer.

Computes Mean Squared Error (MSE) between input features and TS-MAE reconstruction.
Follows INV-002 (Data Leakage Boundaries): normalize input features using training-set
norm_stats before model inference. MSE is computed in normalized space.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'source'))

from models.encoder.ts_mae import TimeSeriesMAE
from models.training.trainer import get_device


class ReconstructionScorer:
    """Reconstruction-based anomaly scorer (Score-A).

    Uses TS-MAE reconstruction MSE as anomaly signal. Higher MSE indicates higher anomaly.
    Normalizes inputs using training-set normalization statistics (mean, std).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[TimeSeriesMAE] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
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
                    n_channels=ckpt.get('config', {}).get('model', {}).get('n_channels', 13),
                    n_windows=ckpt.get('config', {}).get('model', {}).get('n_windows', 180),
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
            raise ValueError("Either checkpoint_path or model must be provided.")

        self.model.eval()

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using training-set statistics (INV-002 compliant)."""
        if self.norm_mean is None or self.norm_std is None:
            return features
        std = np.where(self.norm_std < 1e-6, 1.0, self.norm_std)
        return (features - self.norm_mean) / std

    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute reconstruction MSE scores.

        Args:
            features: (180, C) or (B, 180, C) raw feature matrix (NOT pre-normalized)

        Returns:
            scores: (B,) array of per-window MSE reconstruction errors
        """
        features_arr = np.asarray(features, dtype=np.float32)
        if features_arr.ndim == 2:
            features_arr = features_arr[np.newaxis, ...]

        normed = self._normalize(features_arr)
        x_tensor = torch.tensor(normed, dtype=torch.float32).to(self.device)
        B, T, C = x_tensor.shape

        with torch.no_grad():
            mask_zero = torch.zeros((B, T), dtype=torch.bool, device=self.device)
            out = self.model(x_tensor, mask=mask_zero)
            recon = out['reconstruction'].cpu().numpy()

        window_mse = np.mean((normed - recon) ** 2, axis=(1, 2))
        return window_mse.astype(np.float32)

    def get_embeddings(self, features: np.ndarray) -> np.ndarray:
        """Extract encoder embeddings from features (for Score-B/C pipeline).

        Args:
            features: (180, C) or (B, 180, C) raw feature matrix

        Returns:
            embeddings: (B, 128) mean-pooled window embeddings
        """
        features_arr = np.asarray(features, dtype=np.float32)
        if features_arr.ndim == 2:
            features_arr = features_arr[np.newaxis, ...]

        normed = self._normalize(features_arr)
        x_tensor = torch.tensor(normed, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            emb = self.model.encode(x_tensor).cpu().numpy()  # (B, T, 128)

        emb_pooled = emb.mean(axis=1)  # (B, 128)
        return emb_pooled.astype(np.float32)
