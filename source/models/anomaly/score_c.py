"""
Score-C: Combined Scorer.

Linearly combines normalized Score-A (Reconstruction Error) and Score-B (Embedding Distance):
Score-C = alpha * norm(Score-A) + (1 - alpha) * norm(Score-B).
Alpha tuning is strictly performed on validation splits of training lakes (INV-002).
"""

import numpy as np
from typing import Dict, Any, Optional


class CombinedScorer:
    """Score-C: Combined Scorer (Alpha-weighted sum of Score-A and Score-B)."""
    
    def __init__(self, score_a_scorer=None, score_b_scorer=None, alpha: float = 0.5):
        self.score_a_scorer = score_a_scorer
        self.score_b_scorer = score_b_scorer
        self.alpha = float(alpha)
        
    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize score sequence to [0, 1]."""
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max - s_min < 1e-8:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
        
    def score(self, features: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute combined anomaly score.
        
        Args:
            features: (T, C) feature matrix
            embeddings: (T, d_model) embedding matrix
            
        Returns:
            scores: (T,) combined anomaly scores
        """
        if self.score_a_scorer is not None:
            s_a = self.score_a_scorer.score(features)
        else:
            s_a = np.mean(features ** 2, axis=-1)
            
        if self.score_b_scorer is not None:
            s_b = self.score_b_scorer.score(embeddings)
        else:
            s_b = np.linalg.norm(embeddings, axis=-1)
            
        norm_a = self._min_max_normalize(s_a)
        norm_b = self._min_max_normalize(s_b)
        
        combined = self.alpha * norm_a + (1.0 - self.alpha) * norm_b
        return combined.astype(np.float32)

    def tune_alpha(self, val_features: Dict[str, np.ndarray], val_embeddings: Dict[str, np.ndarray]) -> float:
        """Grid search optimal alpha in [0.0..1.0] on validation lakes (INV-002)."""
        best_alpha = 0.5
        best_variance = -1.0
        
        alphas = np.linspace(0.0, 1.0, 11)
        for a in alphas:
            self.alpha = float(a)
            all_scores = []
            for lid in val_features:
                if lid in val_embeddings:
                    sc = self.score(val_features[lid], val_embeddings[lid])
                    all_scores.extend(sc.tolist())
                    
            var = float(np.var(all_scores)) if all_scores else 0.0
            if var > best_variance:
                best_variance = var
                best_alpha = float(a)
                
        self.alpha = best_alpha
        return best_alpha
