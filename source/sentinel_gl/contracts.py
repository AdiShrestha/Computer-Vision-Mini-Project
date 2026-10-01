"""Small NumPy-only input contracts; these do not authenticate data lineage."""
from collections.abc import Mapping, Set
import numpy as np


def instance_ids(values):
    """Nonempty ordered IDs; a bare string is never an observation sequence."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise ValueError("instance IDs must be an ordered collection, not a string, mapping or set")
    try:
        ids = tuple(values)
    except TypeError as error:
        raise ValueError("instance IDs must be an ordered collection") from error
    if not ids or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError("instance IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate instance IDs")
    return ids


def finite_vector(values):
    arr = np.asarray(values)
    if arr.dtype.kind not in "iuf":
        raise ValueError("scores must be real numeric values, not strings or booleans")
    arr = arr.astype(float)
    if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all():
        raise ValueError("scores must be a finite nonempty one-dimensional vector")
    return arr


def finite_matrix(values):
    arr = np.asarray(values)
    if arr.dtype.kind not in "iuf" or arr.ndim != 2 or not arr.size:
        raise ValueError("embeddings must be nonempty real numeric matrices")
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        raise ValueError("embeddings must be finite")
    return arr
