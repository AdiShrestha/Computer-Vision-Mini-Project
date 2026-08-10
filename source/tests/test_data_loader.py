"""Verify data loader with INV-002 leakage boundary enforcement."""
import os
import sys
import json
import pytest
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from data.loaders.lake_dataset import (
    GlacialLakeDataset, create_data_loaders, load_registry,
    get_lakes_by_role, EVALUATION_ROLES, TRAINING_ROLE,
    create_inference_loader,
)

FEATURES_DIR = os.path.join(repo_root, 'data', 'features')
REGISTRY_PATH = os.path.join(repo_root, 'source', 'data', 'registry', 'lake_registry.json')


def test_registry_role_partition():
    """Registry has correct role partition: 15 training, 5 evaluation."""
    registry = load_registry(REGISTRY_PATH)
    by_role = get_lakes_by_role(registry)
    
    assert len(by_role.get('training', [])) == 15
    eval_count = len(by_role.get('evaluation_event', [])) + \
                 len(by_role.get('evaluation_control', []))
    assert eval_count == 5


def test_inv002_rejects_evaluation_lakes_in_training():
    """INV-002: Training dataset construction rejects evaluation-role lakes."""
    registry = load_registry(REGISTRY_PATH)
    by_role = get_lakes_by_role(registry)
    
    eval_ids = by_role.get('evaluation_event', []) + \
               by_role.get('evaluation_control', [])
    
    for eval_id in eval_ids:
        with pytest.raises(ValueError, match="INV-002"):
            GlacialLakeDataset(
                features_dir=FEATURES_DIR,
                lake_ids=[eval_id],
                registry=registry,
                normalize=False,
                allow_evaluation=False,
            )


def test_inv002_training_loader_excludes_evaluation():
    """INV-002: create_data_loaders never includes evaluation lakes."""
    train_loader, val_loader, norm_stats = create_data_loaders(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
    )
    
    registry = load_registry(REGISTRY_PATH)
    by_role = get_lakes_by_role(registry)
    eval_ids = set(
        by_role.get('evaluation_event', []) +
        by_role.get('evaluation_control', [])
    )
    
    # Check training loader
    for batch in train_loader:
        for lid in batch['lake_id']:
            assert lid not in eval_ids, f"Evaluation lake {lid} in training batch"
    
    # Check validation loader
    for batch in val_loader:
        for lid in batch['lake_id']:
            assert lid not in eval_ids, f"Evaluation lake {lid} in validation batch"


def test_normalization_stats_from_training_only():
    """INV-002: Normalization stats come from training-role lakes only."""
    _, _, norm_stats = create_data_loaders(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
    )
    
    registry = load_registry(REGISTRY_PATH)
    by_role = get_lakes_by_role(registry)
    eval_ids = set(
        by_role.get('evaluation_event', []) +
        by_role.get('evaluation_control', [])
    )
    
    contributing = set(norm_stats['contributing_lake_ids'])
    leak = contributing & eval_ids
    assert len(leak) == 0, (
        f"INV-002 VIOLATION: Normalization uses evaluation lakes: {leak}"
    )


def test_train_val_split_seed_determinism():
    """INV-012: Train/val split with seed=7 is deterministic."""
    _, _, stats1 = create_data_loaders(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
        val_split_seed=7,
    )
    _, _, stats2 = create_data_loaders(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
        val_split_seed=7,
    )
    
    assert stats1['contributing_lake_ids'] == stats2['contributing_lake_ids']
    np.testing.assert_array_equal(stats1['mean'], stats2['mean'])
    np.testing.assert_array_equal(stats1['std'], stats2['std'])


def test_inference_loader_includes_all_lakes():
    """Inference loader can include evaluation lakes (for embedding extraction)."""
    _, _, norm_stats = create_data_loaders(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
    )
    
    inf_loader = create_inference_loader(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
        norm_stats=norm_stats,
    )
    
    all_ids = set()
    for batch in inf_loader:
        for lid in batch['lake_id']:
            all_ids.add(lid)
    
    # Should include evaluation lakes
    assert 'SGL-001' in all_ids, "Inference loader missing South Lhonak"
    assert len(all_ids) == 20, f"Expected 20 lakes, got {len(all_ids)}"


def test_feature_tensor_shape():
    """Feature tensors have expected shape (T, C)."""
    train_loader, _, _ = create_data_loaders(
        features_dir=FEATURES_DIR,
        registry_path=REGISTRY_PATH,
    )
    
    batch = next(iter(train_loader))
    features = batch['features']
    
    assert features.ndim == 3  # (B, T, C)
    assert features.shape[2] == 15  # 15 channels
    assert features.shape[1] > 100  # ~108 windows


def test_inference_loader_requires_norm_stats():
    """INV-002: Inference loader requires externally-provided norm stats."""
    with pytest.raises(ValueError, match="norm_stats"):
        registry = load_registry(REGISTRY_PATH)
        GlacialLakeDataset(
            features_dir=FEATURES_DIR,
            lake_ids=['SGL-001'],
            registry=registry,
            normalize=True,
            norm_stats=None,
            allow_evaluation=True,
        )
