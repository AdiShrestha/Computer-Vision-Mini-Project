"""Content-addressable file and directory hashing utilities.

Used for manifest verification and frozen-file checks.
"""
import os
import hashlib
from typing import Dict


def hash_file(path: str, algorithm: str = 'sha256') -> str:
    """Compute hex digest of a file's content in binary chunks.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm name (default: 'sha256').

    Returns:
        str: Hex digest string.
    """
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(directory_path: str, algorithm: str = 'sha256') -> Dict[str, str]:
    """Compute relative_path -> hash map for all files in a directory, sorted.

    Args:
        directory_path: Path to root directory.
        algorithm: Hash algorithm name.

    Returns:
        Dict[str, str]: Map of relative file paths to their hex digests.
    """
    results = {}
    for root, _, files in os.walk(directory_path):
        for fname in sorted(files):
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, directory_path)
            results[rel_path] = hash_file(full_path, algorithm=algorithm)
    return dict(sorted(results.items()))


def verify_hash(path: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """Verify if a file matches an expected hash string.

    Args:
        path: Path to file.
        expected_hash: Expected hex digest string.
        algorithm: Hash algorithm name.

    Returns:
        bool: True if hash matches, False otherwise.
    """
    if not os.path.isfile(path):
        return False
    actual_hash = hash_file(path, algorithm=algorithm)
    return actual_hash.lower() == expected_hash.lower()
