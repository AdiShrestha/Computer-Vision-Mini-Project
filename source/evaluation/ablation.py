"""
Channel Ablation Engine (C05-02).

Performs zero-retraining channel masking ablation to quantify per-channel-group
contribution to anomaly detection performance.

Ablation approach: For each configuration, zero-out excluded channel columns
(in normalized space, after norm_stats applied) then re-run frozen encoder
forward pass to get ablated embeddings. Score with frozen Score-B and Score-C.

The encoder and Score-B density model remain FROZEN from Chunk 03/04 — no
retraining occurs. This is 'channel dropout at inference time.'

Invariant compliance:
    INV-002: Frozen encoder trained on training lakes only
    INV-008: Zero compute overhead (no retraining)
    INV-012: Same seeds as Chunk 04 evaluation
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
import torch

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from models.anomaly.score_a import ReconstructionScorer
from models.anomaly.score_b import EmbeddingDistanceScorer
from models.anomaly.score_c import CombinedScorer
from models.anomaly.smoothing import ema_smooth
from evaluation.protocols.e3_synthetic import run_e3_synthetic
from evaluation.synthetic.injector import SyntheticInjector


# ============================================================
# Channel Group Definitions
# ============================================================

# Full channel list (15 columns, from feature_matrix.npz channel_names)
CHANNEL_GROUPS = {
    'CH-01': [0],                  # Lake extent (1 col)
    'CH-02': [1, 2, 3, 4],        # Spectral/turbidity (4 cols)
    'CH-03': [5, 6],              # Glacier velocity (2 cols)
    'CH-04': [7],                 # Temperature anomaly (1 col)
    'CH-05': [8, 9, 10],          # SAR backscatter (3 cols)
    'CH-07': [11],                # SAR coherence (1 col)
    'CH-08': [12, 13, 14],        # Meteorological (3 cols)
}

# All 15 channel indices
ALL_CHANNELS = list(range(15))

# Ablation configurations: name -> columns to KEEP (all others zeroed)
ABLATION_CONFIGS = {
    'FULL_15CH': ALL_CHANNELS,
    'NO_CH01': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-01']],
    'NO_CH02': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-02']],
    'NO_CH03': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-03']],
    'NO_CH04': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-04']],
    'NO_CH05': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-05']],
    'NO_CH07': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-07']],
    'NO_CH08': [i for i in ALL_CHANNELS if i not in CHANNEL_GROUPS['CH-08']],
    'OPTICAL_ONLY': CHANNEL_GROUPS['CH-01'] + CHANNEL_GROUPS['CH-02'] + CHANNEL_GROUPS['CH-04'],
    'SAR_ONLY': CHANNEL_GROUPS['CH-05'] + CHANNEL_GROUPS['CH-07'],
    'DYNAMIC_ONLY': CHANNEL_GROUPS['CH-03'] + CHANNEL_GROUPS['CH-08'],
}


def apply_channel_mask(features: np.ndarray, keep_cols: List[int]) -> np.ndarray:
    """Zero out all columns NOT in keep_cols (in-place copy).

    Masking is applied AFTER normalization — the AblationExperiment normalizes
    first, then this function zeros the excluded columns. This correctly
    simulates 'these channels are absent' at the encoder's input.

    Args:
        features: (T, 15) raw or normalized feature matrix
        keep_cols: list of column indices to KEEP (all others set to 0.0)

    Returns:
        masked: (T, 15) array with excluded columns zeroed
    """
    masked = features.copy()
    all_cols = set(range(features.shape[1]))
    zero_cols = list(all_cols - set(keep_cols))
    if zero_cols:
        masked[:, zero_cols] = 0.0
    return masked


class AblationExperiment:
    """Runs a single ablation configuration.

    The encoder, Score-B density model, and Score-C alpha are FROZEN.
    Only the channel mask changes between configurations.
    """

    def __init__(
        self,
        score_a_inst: ReconstructionScorer,
        score_b_inst: EmbeddingDistanceScorer,
        score_c_inst: CombinedScorer,
        ckpt_path: str,
    ):
        self.score_a = score_a_inst
        self.score_b = score_b_inst
        self.score_c = score_c_inst
        self.ckpt_path = ckpt_path
        # Record checkpoint mtime for the 'no retraining' invariant
        self._ckpt_mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else None

    def verify_no_retraining(self) -> bool:
        """Confirm the checkpoint file was not modified since init (INV-008)."""
        if self._ckpt_mtime is None:
            return True
        return abs(os.path.getmtime(self.ckpt_path) - self._ckpt_mtime) < 1.0

    def run_config(
        self,
        config_name: str,
        keep_cols: List[int],
        features_map: Dict[str, np.ndarray],
        control_ids: List[str],
        output_dir: str,
    ) -> Dict:
        """Run a single ablation configuration and return metrics.

        Args:
            config_name: Name of this config (for output directory)
            keep_cols: Column indices to retain (others zeroed)
            features_map: {lake_id: (T, 15) raw features}
            control_ids: IDs of evaluation_control lakes
            output_dir: Base results output directory

        Returns:
            dict with auc_roc, auc_pr, detection_rate, threshold, keep_cols
        """
        config_out = os.path.join(output_dir, config_name)
        os.makedirs(config_out, exist_ok=True)

        # Step 1: For each lake, apply mask then compute ablated scores
        smoothed_scores = {}
        for lid, feat in features_map.items():
            # Apply channel mask AFTER normalization (Score-A normalizes internally)
            # We need to normalize first, then mask, then we can't use Score-A's
            # internal normalize+infer because it normalizes internally.
            # Instead we:
            #   (a) normalize manually using score_a's norm_stats
            #   (b) zero out excluded cols on normalized features
            #   (c) run encoder forward pass manually
            #   (d) use result for Score-B and Score-C

            feat_arr = np.asarray(feat, dtype=np.float32)
            normed = self.score_a._normalize(feat_arr)
            masked_normed = apply_channel_mask(normed, keep_cols)

            # Get ablated embeddings from frozen encoder
            ablated_emb = self._get_ablated_embeddings(masked_normed)

            # Score-B on ablated embeddings
            sb = self.score_b.score(ablated_emb)
            # Score-C: Score-A uses masked features; Score-B uses ablated embeddings
            sa = self._score_a_from_normalized(masked_normed)
            sc = self.score_c._min_max_normalize(
                self.score_c.alpha * self.score_c._min_max_normalize(sa)
                + (1.0 - self.score_c.alpha) * self.score_c._min_max_normalize(sb)
            )

            smoothed_scores[lid] = ema_smooth(sc, span=5)

        # Step 2: Derive threshold at 85th percentile of control scores
        all_ctrl_scores = np.concatenate([
            smoothed_scores[lid] for lid in control_ids if lid in smoothed_scores
        ])
        threshold = float(np.percentile(all_ctrl_scores, 85))

        # Step 3: Run E3 synthetic injection with this ablated scorer
        control_feats = {lid: features_map[lid] for lid in control_ids if lid in features_map}

        def ablated_scorer_fn(modified_features: np.ndarray) -> np.ndarray:
            feat_arr = np.asarray(modified_features, dtype=np.float32)
            normed = self.score_a._normalize(feat_arr)
            masked_normed = apply_channel_mask(normed, keep_cols)
            ablated_emb = self._get_ablated_embeddings(masked_normed)
            sb = self.score_b.score(ablated_emb)
            sa = self._score_a_from_normalized(masked_normed)
            sc = (
                self.score_c.alpha * self.score_c._min_max_normalize(sa)
                + (1.0 - self.score_c.alpha) * self.score_c._min_max_normalize(sb)
            )
            return ema_smooth(sc, span=5)

        e3_res = run_e3_synthetic(
            scorer_fn=ablated_scorer_fn,
            control_features=control_feats,
            injector=SyntheticInjector(seed=2023),
            threshold=threshold,
            output_dir=config_out,
        )

        result = {
            'config_name': config_name,
            'keep_cols': keep_cols,
            'n_active_channels': len(keep_cols),
            'threshold': threshold,
            'auc_roc': e3_res['auc_roc'],
            'auc_pr': e3_res['auc_pr'],
            'synthetic_detection_rate': e3_res['overall_detection_rate'],
            'false_positive_rate': float(np.mean([
                float(np.mean(smoothed_scores[lid] > threshold))
                for lid in control_ids if lid in smoothed_scores
            ])),
        }

        # Save per-config result
        with open(os.path.join(config_out, 'ablation_config_result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _get_ablated_embeddings(self, masked_normed: np.ndarray) -> np.ndarray:
        """Run frozen encoder on masked+normalized features to get ablated embeddings."""
        T, C = masked_normed.shape
        x_tensor = torch.tensor(masked_normed, dtype=torch.float32).unsqueeze(0).to(
            self.score_a.device
        )
        with torch.no_grad():
            emb = self.score_a.model.encode(x_tensor)
            return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def _score_a_from_normalized(self, masked_normed: np.ndarray) -> np.ndarray:
        """Compute Score-A reconstruction MSE from already-normalized+masked features."""
        T, C = masked_normed.shape
        x_tensor = torch.tensor(masked_normed, dtype=torch.float32).unsqueeze(0).to(
            self.score_a.device
        )
        with torch.no_grad():
            mask_zero = torch.zeros((1, T), dtype=torch.bool, device=self.score_a.device)
            out = self.score_a.model(x_tensor, mask=mask_zero)
            recon = out['reconstruction'].squeeze(0).cpu().numpy()
        window_mse = np.mean((masked_normed - recon) ** 2, axis=-1)
        return window_mse.astype(np.float32)
