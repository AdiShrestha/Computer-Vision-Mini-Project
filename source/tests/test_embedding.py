"""Verify embedding extraction pipeline."""
import os
import sys
import importlib

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_embedding_module_exists():
    """Embedding extraction module exists."""
    from models.embedding.extract import extract_embeddings
    assert callable(extract_embeddings)


def test_extract_function_signature():
    """extract_embeddings has expected parameters."""
    import inspect
    from models.embedding.extract import extract_embeddings
    sig = inspect.signature(extract_embeddings)
    params = list(sig.parameters.keys())
    assert 'checkpoint_path' in params


def test_embedding_module_compiles():
    """Embedding module compiles without errors."""
    path = os.path.join(source_root, 'models', 'embedding', 'extract.py')
    with open(path) as f:
        compile(f.read(), path, 'exec')
