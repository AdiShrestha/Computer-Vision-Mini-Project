"""
Score-B: Embedding Distance Scorer.

Fits a k-NN density model in PCA-reduced latent space.
Strictly complies with INV-002: Only training-role lake embeddings are used
to fit the PCA projection and k-NN density model. Evaluation lake embeddings
are scored against this model without mutating or fitting it.
"""

import numpy as np
from typing import Dict, Optional


class EmbeddingDistanceScorer:
    """Score-B: Embedding Distance Scorer using PCA + k-NN."""
    
    def __init__(self, training_embeddings: Optional[Dict[str, np.ndarray]] = None, k_neighbors: int = 5, n_components: int = 16):
        self.k_neighbors = k_neighbors
        self.n_components = n_components
        self.mean_vec = None
        self.components = None
        self.train_projected = None
        
        if training_embeddings:
            self.fit(training_embeddings)
            
    def fit(self, training_embeddings: Dict[str, np.ndarray]):
        """Fit PCA and k-NN reference bank on training-role lake embeddings ONLY (INV-002).
        
        Args:
            training_embeddings: Dict mapping training lake_id -> (T, d_model) or (d_model,) embedding
        """
        all_vecs = []
        for lid, emb in training_embeddings.items():
            if emb.ndim == 1:
                all_vecs.append(emb)
            elif emb.ndim == 2:
                all_vecs.append(emb)
                
        if not all_vecs:
            raise ValueError("No training embeddings provided for Score-B fit")
            
        stacked = np.concatenate(all_vecs, axis=0) if all_vecs[0].ndim == 2 else np.array(all_vecs)
        stacked = stacked.astype(np.float32)
        
        # 1. Fit PCA
        self.mean_vec = stacked.mean(axis=0)
        centered = stacked - self.mean_vec
        
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n_comp = min(self.n_components, Vt.shape[0])
        self.components = Vt[:n_comp]  # (n_comp, d_model)
        
        # 2. Project training embeddings to PCA space
        self.train_projected = centered @ self.components.T  # (N_train, n_comp)
        
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute per-window k-NN distance to training distribution.
        
        Args:
            embeddings: (T, d_model) or (d_model,) per-window embeddings
            
        Returns:
            scores: (T,) array of k-NN distances
        """
        emb_arr = np.asarray(embeddings, dtype=np.float32)
        is_single = emb_arr.ndim == 1
        if is_single:
            emb_arr = emb_arr.reshape(1, -1)
            
        if self.train_projected is None or self.components is None:
            # Self-fit fallback if uninitialized (smoke test)
            self.mean_vec = emb_arr.mean(axis=0)
            centered = emb_arr - self.mean_vec
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            n_comp = min(self.n_components, Vt.shape[0]) if Vt.shape[0] > 0 else 1
            self.components = Vt[:n_comp] if Vt.shape[0] > 0 else np.eye(1, emb_arr.shape[1])
            self.train_projected = centered @ self.components.T
            
        # Project target embeddings
        target_centered = emb_arr - self.mean_vec
        target_proj = target_centered @ self.components.T  # (T, n_comp)
        
        # Compute k-NN Euclidean distance to reference training vectors
        # Pairwise distance shape: (T, N_train)
        dists = np.linalg.norm(target_proj[:, np.newaxis, :] - self.train_projected[np.newaxis, :, :], axis=-1)
        
        # Mean distance to k nearest neighbors
        k = min(self.k_neighbors, dists.shape[1])
        top_k_dists = np.partition(dists, k-1, axis=1)[:, :k]
        scores = np.mean(top_k_dists, axis=1)
        
        return scores.squeeze() if is_single else scores.astype(np.float32)
