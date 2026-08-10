"""
Leakage-safe data loader for glacial lake feature matrices.

This module implements the data loading pipeline with triple-redundant
INV-002 enforcement for training, while allowing evaluation-role lakes
for inference-only operations (embedding extraction).

Invariant compliance:
    INV-002: Training/evaluation data leakage boundary
    INV-012: Deterministic seeds (train/val split seed=7)
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Evaluation roles that must NEVER appear in training data
EVALUATION_ROLES = frozenset({'evaluation_event', 'evaluation_control'})
TRAINING_ROLE = 'training'


def load_registry(registry_path: str) -> dict:
    """Load and return the lake registry."""
    with open(registry_path) as f:
        return json.load(f)


def get_lakes_by_role(registry: dict) -> Dict[str, List[str]]:
    """Partition lake IDs by role.
    
    Returns:
        dict mapping role -> list of lake IDs
    """
    by_role = {}
    for lake in registry['lakes']:
        role = lake['role']
        if role not in by_role:
            by_role[role] = []
        by_role[role].append(lake['id'])
    return by_role


class GlacialLakeDataset(Dataset):
    """Dataset for glacial lake time series feature matrices.
    
    Each sample is one lake's full temporal feature matrix: (T, C).
    
    INV-002 ENFORCEMENT (training mode only):
        - Constructor ASSERTS that no evaluation-role lake is in the list
        - Normalization stats computed from training-role lakes only
        - Every __getitem__ call verifies the lake ID
    
    For inference (allow_evaluation=True):
        - Evaluation-role lakes ARE permitted (no gradient updates)
        - Normalization stats must be provided externally (from training data)
    
    Args:
        features_dir: Path to data/features/
        lake_ids: Explicit list of lake IDs to include
        registry: Loaded lake registry dict
        normalize: Whether to apply z-score normalization
        norm_stats: Pre-computed normalization stats. Required if 
                    allow_evaluation=True and normalize=True.
        allow_evaluation: If False (default), rejects evaluation-role lakes.
                          If True, permits all roles (for inference only).
    """
    
    def __init__(
        self,
        features_dir: str,
        lake_ids: List[str],
        registry: dict,
        normalize: bool = True,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        allow_evaluation: bool = False,
    ):
        self.features_dir = features_dir
        self.normalize = normalize
        self.allow_evaluation = allow_evaluation
        
        # Build role lookup
        self.roles_by_id = {lake['id']: lake['role'] for lake in registry['lakes']}
        
        if not allow_evaluation:
            # === INV-002 CHECK #1: Reject evaluation lakes at construction ===
            for lid in lake_ids:
                role = self.roles_by_id.get(lid, 'unknown')
                if role in EVALUATION_ROLES:
                    raise ValueError(
                        f"INV-002 VIOLATION: Lake {lid} has role '{role}' and "
                        f"must not be included in a training dataset. "
                        f"Evaluation roles {EVALUATION_ROLES} are forbidden."
                    )
        
        self.lake_ids = list(lake_ids)
        
        # Load all feature matrices
        self.data = {}
        for lid in self.lake_ids:
            matrix_path = os.path.join(features_dir, lid, 'feature_matrix.npz')
            if os.path.exists(matrix_path):
                loaded = np.load(matrix_path, allow_pickle=True)
                self.data[lid] = {
                    'features': loaded['features'].astype(np.float32),
                    'quality': loaded['quality'].astype(np.float32),
                }
            else:
                logger.warning(f"Feature matrix not found for {lid}, skipping")
        
        # Update lake_ids to only include lakes with data
        self.lake_ids = [lid for lid in self.lake_ids if lid in self.data]
        
        # Compute or load normalization statistics
        if normalize:
            if norm_stats is not None:
                self.norm_mean = norm_stats['mean']
                self.norm_std = norm_stats['std']
                self.norm_lake_ids = norm_stats.get('contributing_lake_ids', [])
            elif allow_evaluation:
                raise ValueError(
                    "INV-002: norm_stats must be provided when "
                    "allow_evaluation=True to ensure normalization uses "
                    "training-role statistics only."
                )
            else:
                self._compute_norm_stats()
        else:
            self.norm_mean = None
            self.norm_std = None
            self.norm_lake_ids = []
    
    def _compute_norm_stats(self):
        """Compute per-channel mean and std from this dataset's data ONLY.
        
        INV-002: These statistics must come from training-role lakes only.
        The contributing lake IDs are recorded for audit.
        """
        all_features = []
        contributing_ids = []
        
        for lid in self.lake_ids:
            # === INV-002 CHECK #2: Verify role before including in stats ===
            role = self.roles_by_id.get(lid, 'unknown')
            assert role not in EVALUATION_ROLES, (
                f"INV-002 VIOLATION: Attempting to compute normalization "
                f"statistics using lake {lid} (role={role})"
            )
            all_features.append(self.data[lid]['features'])
            contributing_ids.append(lid)
        
        if not all_features:
            raise ValueError("No feature data available for normalization")
        
        stacked = np.concatenate(all_features, axis=0)  # (N*T, C)
        self.norm_mean = stacked.mean(axis=0).astype(np.float32)  # (C,)
        self.norm_std = stacked.std(axis=0).astype(np.float32)    # (C,)
        
        # Prevent division by zero for constant channels
        self.norm_std = np.where(self.norm_std < 1e-8, 1.0, self.norm_std)
        
        self.norm_lake_ids = contributing_ids
        
        logger.info(
            f"Normalization stats computed from {len(contributing_ids)} lakes: "
            f"{contributing_ids}"
        )
    
    def get_norm_stats(self) -> Dict[str, np.ndarray]:
        """Return normalization statistics for reuse by other datasets."""
        return {
            'mean': self.norm_mean,
            'std': self.norm_std,
            'contributing_lake_ids': self.norm_lake_ids,
        }
    
    def __len__(self) -> int:
        return len(self.lake_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one lake's feature matrix as a tensor."""
        lid = self.lake_ids[idx]
        
        if not self.allow_evaluation:
            # === INV-002 CHECK #3: Runtime assertion per access ===
            role = self.roles_by_id.get(lid, 'unknown')
            assert role not in EVALUATION_ROLES, (
                f"INV-002 RUNTIME VIOLATION: Accessed lake {lid} (role={role}) "
                f"from a training dataset"
            )
        
        features = self.data[lid]['features'].copy()
        quality = self.data[lid]['quality'].copy()
        
        if self.normalize and self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'quality': torch.tensor(quality, dtype=torch.float32),
            'lake_id': lid,
        }


def create_data_loaders(
    features_dir: str,
    registry_path: str,
    batch_size: int = 16,
    val_fraction: float = 0.2,
    val_split_seed: int = 7,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Create train and validation data loaders with INV-002 enforcement.
    
    The train/validation split is ONLY on training-role lakes.
    Evaluation-role lakes are NEVER included in either split.
    
    Args:
        features_dir: Path to data/features/
        registry_path: Path to lake_registry.json
        batch_size: Batch size (default 16)
        val_fraction: Fraction of training lakes for validation (default 0.2)
        val_split_seed: Random seed for split (default 7, INV-012)
        num_workers: DataLoader workers (default 0 for MPS compatibility)
    
    Returns:
        (train_loader, val_loader, norm_stats)
    """
    registry = load_registry(registry_path)
    lakes_by_role = get_lakes_by_role(registry)
    
    training_lake_ids = lakes_by_role.get(TRAINING_ROLE, [])
    
    if not training_lake_ids:
        raise ValueError("No training-role lakes found in registry")
    
    # Log evaluation lakes that are explicitly excluded
    eval_lake_ids = []
    for role in EVALUATION_ROLES:
        eval_lake_ids.extend(lakes_by_role.get(role, []))
    
    logger.info(
        f"INV-002: {len(training_lake_ids)} training lakes, "
        f"{len(eval_lake_ids)} evaluation lakes EXCLUDED: {eval_lake_ids}"
    )
    
    # Split training lakes into train/val subsets
    rng = random.Random(val_split_seed)
    shuffled = list(training_lake_ids)
    rng.shuffle(shuffled)
    
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    
    logger.info(f"Train/val split (seed={val_split_seed}): "
                f"{len(train_ids)} train, {len(val_ids)} val")
    logger.info(f"  Train lakes: {train_ids}")
    logger.info(f"  Val lakes: {val_ids}")
    
    # Create training dataset (computes normalization stats)
    train_dataset = GlacialLakeDataset(
        features_dir=features_dir,
        lake_ids=train_ids,
        registry=registry,
        normalize=True,
        allow_evaluation=False,
    )
    
    norm_stats = train_dataset.get_norm_stats()
    
    # Create validation dataset with SAME normalization stats
    val_dataset = GlacialLakeDataset(
        features_dir=features_dir,
        lake_ids=val_ids,
        registry=registry,
        normalize=True,
        norm_stats=norm_stats,
        allow_evaluation=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    return train_loader, val_loader, norm_stats


def create_inference_loader(
    features_dir: str,
    registry_path: str,
    norm_stats: Dict[str, np.ndarray],
    lake_ids: Optional[List[str]] = None,
    batch_size: int = 16,
) -> DataLoader:
    """Create a data loader for inference (embedding extraction).
    
    Unlike training loaders, this CAN include evaluation-role lakes
    because inference doesn't update model weights. Normalization stats
    must come from training-role data (provided externally).
    
    Args:
        features_dir: Path to data/features/
        registry_path: Path to lake_registry.json
        norm_stats: Normalization stats from training (must be from training lakes)
        lake_ids: Optional specific lake IDs. If None, loads ALL lakes.
        batch_size: Batch size
    
    Returns:
        DataLoader for inference
    """
    registry = load_registry(registry_path)
    
    if lake_ids is None:
        lake_ids = [lake['id'] for lake in registry['lakes']]
    
    dataset = GlacialLakeDataset(
        features_dir=features_dir,
        lake_ids=lake_ids,
        registry=registry,
        normalize=True,
        norm_stats=norm_stats,
        allow_evaluation=True,  # Safe: inference only, no gradient updates
    )
    
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
