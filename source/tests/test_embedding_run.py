"""Verify full-scale embedding extraction."""
import os
import sys
import json
import numpy as np

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

EMBEDDINGS_DIR = os.path.join(repo_root, 'data', 'embeddings')


def test_embedding_summary_exists():
    """Embedding summary JSON exists."""
    path = os.path.join(EMBEDDINGS_DIR, 'embedding_summary.json')
    assert os.path.isfile(path)


def test_all_lakes_have_embeddings():
    """All 20 lakes have embedding files."""
    lake_dirs = [d for d in os.listdir(EMBEDDINGS_DIR)
                 if d.startswith('SGL-') and
                 os.path.isfile(os.path.join(EMBEDDINGS_DIR, d, 'embeddings.npz'))]
    assert len(lake_dirs) == 20, f"Only {len(lake_dirs)} lakes have embeddings"


def test_south_lhonak_embedding_shape():
    """South Lhonak embeddings have correct shape."""
    path = os.path.join(EMBEDDINGS_DIR, 'SGL-001', 'embeddings.npz')
    assert os.path.isfile(path)
    data = np.load(path, allow_pickle=True)
    emb = data['embeddings']
    assert emb.shape[1] == 128, f"Expected d_model=128, got {emb.shape[1]}"
    assert emb.shape[0] > 100, f"Only {emb.shape[0]} windows"


def test_embeddings_not_collapsed():
    """Embeddings are not all identical (basic non-collapse check)."""
    path = os.path.join(EMBEDDINGS_DIR, 'SGL-001', 'embeddings.npz')
    data = np.load(path, allow_pickle=True)
    emb = data['embeddings']
    # Check variance across time windows
    variance = emb.var(axis=0).mean()
    assert variance > 1e-6, f"Embeddings appear collapsed (var={variance})"
