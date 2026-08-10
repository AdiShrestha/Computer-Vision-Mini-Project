"""
Score-A: Reconstruction Error Scorer.

Computes per-window reconstruction MSE by passing full feature matrices
(without temporal masking) through the trained TS-MAE model.
Higher reconstruction error = higher anomaly score.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from models.encoder.ts_mae import TimeSeriesMAE
from models.training.trainer import get_device


class ReconstructionScorer:
    """Score-A: Reconstruction MSE Scorer."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, model: Optional[nn.Module] = None, device: Optional[torch.device] = None):
        self.device = device or get_device()
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model = TimeSeriesMAE(
                n_channels=ckpt.get('config', {}).get('model', {}).get('n_channels', 15),
                n_windows=ckpt.get('config', {}).get('model', {}).get('n_windows', 108),
                d_model=ckpt.get('config', {}).get('model', {}).get('d_model', 128),
            )
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        else:
            # Default instantiation for smoke testing / modular use
            self.model = TimeSeriesMAE().to(self.device)
            self.model.eval()
            
    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute per-window reconstruction error.
        
        Args:
            features: (T, C) normalized feature matrix
            
        Returns:
            scores: (T,) array of per-window MSE reconstruction error
        """
        features_arr = np.asarray(features, dtype=np.float32)
        T, C = features_arr.shape
        
        x_tensor = torch.tensor(features_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass complete window (mask=None) for reconstruction
            # Model forward with no masking or zero mask
            mask_zero = torch.zeros((1, T), dtype=torch.bool, device=self.device)
            out = self.model(x_tensor, mask=mask_zero)
            recon = out['reconstruction'].squeeze(0).cpu().numpy()
            
        # Per-window MSE across channels C
        window_mse = np.mean((features_arr - recon) ** 2, axis=-1)
        return window_mse.astype(np.float32)
