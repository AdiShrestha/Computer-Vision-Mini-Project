"""Verify core utilities work correctly."""
import os
import sys
import json
import tempfile
import hashlib

# Ensure source root is importable
source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def test_set_seed_determinism():
    """set_seed produces identical random sequences across calls."""
    from utils.reproducibility import set_seed
    import numpy as np
    import torch
    
    set_seed(42)
    a_np = np.random.rand(10)
    a_torch = torch.rand(10)
    
    set_seed(42)
    b_np = np.random.rand(10)
    b_torch = torch.rand(10)
    
    assert (a_np == b_np).all(), "NumPy not deterministic after set_seed"
    assert torch.equal(a_torch, b_torch), "PyTorch not deterministic after set_seed"


def test_hash_file():
    """hash_file produces correct SHA-256 for known content."""
    from utils.hashing import hash_file
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
        f.write(b"sentinel-gl test content")
        tmp_path = f.name
    
    try:
        result = hash_file(tmp_path)
        expected = hashlib.sha256(b"sentinel-gl test content").hexdigest()
        assert result == expected, f"Hash mismatch: {result} != {expected}"
    finally:
        os.unlink(tmp_path)


def test_hash_directory():
    """hash_directory returns sorted, deterministic hashes."""
    from utils.hashing import hash_directory
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files in non-alphabetical order
        for name, content in [('b.txt', b'bbb'), ('a.txt', b'aaa')]:
            with open(os.path.join(tmpdir, name), 'wb') as f:
                f.write(content)
        
        result = hash_directory(tmpdir)
        assert list(result.keys()) == ['a.txt', 'b.txt'], "Not sorted"
        assert result['a.txt'] == hashlib.sha256(b'aaa').hexdigest()


def test_verify_hash():
    """verify_hash returns True for matching, False for non-matching."""
    from utils.hashing import hash_file, verify_hash
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
        f.write(b"test")
        tmp_path = f.name
    
    try:
        correct_hash = hash_file(tmp_path)
        assert verify_hash(tmp_path, correct_hash) is True
        assert verify_hash(tmp_path, "wrong_hash") is False
    finally:
        os.unlink(tmp_path)


def test_setup_logger():
    """setup_logger creates a logger that writes to console and file."""
    from utils.logging_utils import setup_logger
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger('test_logger', log_dir=tmpdir)
        logger.info("Test message")
        
        log_file = os.path.join(tmpdir, 'test_logger.log')
        assert os.path.exists(log_file), "Log file not created"
        with open(log_file) as f:
            content = f.read()
        assert "Test message" in content


def test_log_to_jsonl():
    """log_to_jsonl appends valid JSON lines with timestamps."""
    from utils.logging_utils import log_to_jsonl
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        tmp_path = f.name
    
    try:
        log_to_jsonl(tmp_path, {"event": "test", "value": 42})
        log_to_jsonl(tmp_path, {"event": "test2", "value": 99})
        
        with open(tmp_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        entry1 = json.loads(lines[0])
        assert entry1['event'] == 'test'
        assert entry1['value'] == 42
        assert 'timestamp' in entry1, "Missing automatic timestamp"
    finally:
        os.unlink(tmp_path)
