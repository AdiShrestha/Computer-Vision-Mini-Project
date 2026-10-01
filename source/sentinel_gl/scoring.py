"""Frozen scoring operators with no evaluation-time fitting or score synthesis."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .features import FitScope
from .model import evaluation_masks
from .integrity import digest
from .contracts import finite_vector, finite_matrix


class EmbeddingDistanceScorer:
    def __init__(self, scope: FitScope, k_neighbors: int = 5, n_components: int = 16):
        if type(k_neighbors) is not int or k_neighbors < 1 or type(n_components) is not int or n_components < 1:
            raise ValueError("k_neighbors and n_components must be positive integers")
        self._scope, self._k_neighbors, self._n_components = scope, k_neighbors, n_components
        self._pca = self._bank = None
        self._reference = None
        self._fit_lake_ids = ()

    scope = property(lambda self: self._scope)
    k_neighbors = property(lambda self: self._k_neighbors)
    n_components = property(lambda self: self._n_components)
    fit_lake_ids = property(lambda self: self._fit_lake_ids)
    pca = property(lambda self: deepcopy(self._pca))
    bank = property(lambda self: deepcopy(self._bank))

    @property
    def state_hash(self):
        if self._bank is None:
            raise RuntimeError("density scorer has not been fitted")
        return digest({"fit_lake_ids": self.fit_lake_ids, "k": self.k_neighbors,
            "components": self._pca.components_.tolist(), "mean": self._pca.mean_.tolist(),
            "reference": self._reference.tolist()})

    def fit(self, embeddings):
        if self._bank is not None:
            raise RuntimeError("density scorer already fitted")
        self.scope.check(embeddings)
        arrays = [finite_matrix(embeddings[key]) for key in sorted(embeddings)]
        if any(a.ndim != 2 or not a.size or not np.isfinite(a).all() for a in arrays):
            raise ValueError("training embeddings must be finite nonempty matrices")
        if len({a.shape[1] for a in arrays}) != 1:
            raise ValueError("embedding dimension mismatch")
        x = np.concatenate(arrays)
        if len(x) < max(2, self.k_neighbors) or self.n_components > min(x.shape):
            raise ValueError("reference bank too small for declared PCA and k")
        if np.all(x == x[0]):
            raise ValueError("reference embeddings are constant: density fit is not estimable")
        pca = PCA(n_components=self.n_components, svd_solver="full").fit(x)
        projected = pca.transform(x)
        if not np.isfinite(projected).all() or not np.isfinite(pca.components_).all():
            raise ValueError("nonfinite fitted embedding transform")
        bank = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm="brute").fit(projected)
        self._pca, self._bank = pca, bank
        self._reference = projected.copy()
        self._reference.setflags(write=False)
        self._fit_lake_ids = tuple(sorted(embeddings))
        return self

    def score(self, embeddings):
        if self._bank is None:
            raise RuntimeError("density scorer must be fitted on declared training data before scoring")
        x = finite_matrix(embeddings)
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError("query embeddings must be finite nonempty matrices")
        result = self._bank.kneighbors(self._pca.transform(x), return_distance=True)[0].mean(axis=1)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite embedding distances")
        return result


class FrozenScoreCombiner:
    """Reference min/max scaling with extrapolation, followed by fixed alpha.

    Output is an anomaly score, not a calibrated event probability. Fit only on
    the declared reference/calibration lakes. Evaluation never changes bounds.
    """
    def __init__(self, scope: FitScope, alpha: float):
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be a finite number in [0,1]")
        self._scope, self._alpha = scope, float(alpha)
        self._bounds = None
        self._fit_lake_ids = ()

    scope = property(lambda self: self._scope)
    alpha = property(lambda self: self._alpha)
    bounds = property(lambda self: self._bounds)
    fit_lake_ids = property(lambda self: self._fit_lake_ids)

    @property
    def state_hash(self):
        if self.bounds is None:
            raise RuntimeError("combiner has not been fitted")
        return digest({"alpha": self.alpha, "bounds": self.bounds, "fit_lake_ids": self.fit_lake_ids})

    def fit(self, score_pairs):
        if self.bounds is not None:
            raise RuntimeError("combiner already fitted")
        self.scope.check(score_pairs)
        aa, bb = [], []
        for a, b in score_pairs.values():
            a, b = finite_vector(a), finite_vector(b)
            if a.shape != b.shape:
                raise ValueError("paired reference scores must have identical shapes")
            aa.extend(a); bb.extend(b)
        bounds = (min(aa), max(aa), min(bb), max(bb))
        if bounds[0] == bounds[1] or bounds[2] == bounds[3]:
            raise ValueError("constant reference score: scaling is not estimable")
        if not np.isfinite(bounds[1] - bounds[0]) or not np.isfinite(bounds[3] - bounds[2]):
            raise ValueError("reference score range cannot be represented")
        self._bounds = tuple(map(float, bounds))
        self._fit_lake_ids = tuple(sorted(score_pairs))
        return self

    def score(self, score_a, score_b):
        if self.bounds is None:
            raise RuntimeError("combiner has not been fitted")
        a, b = finite_vector(score_a), finite_vector(score_b)
        if a.shape != b.shape:
            raise ValueError("Score-A and Score-B must have identical shapes")
        lo_a, hi_a, lo_b, hi_b = self.bounds
        combined = self.alpha*(a-lo_a)/(hi_a-lo_a) + (1-self.alpha)*(b-lo_b)/(hi_b-lo_b)
        if not np.isfinite(combined).all():
            raise ValueError("nonfinite combined score")
        return combined


class MaskedReconstructionScorer:
    """Deterministic cross masking: every observed target is hidden once.

    Observed time positions are distributed deterministically across folds;
    missing positions fill fixed fold capacities. This prevents acquisition
    cadence from placing all observations in one fold.
    Every partition uses only its visible positions; errors over missing targets
    are excluded. Supply normalized float inputs and boolean observation masks.
    No centre-date or streaming availability is inferred by this operator.
    """
    def __init__(self, model, partitions: int = 2):
        if type(partitions) is not int or partitions < 2:
            raise ValueError("at least two mask partitions are required")
        self.model, self._partitions = model, partitions

    partitions = property(lambda self: self._partitions)

    def score(self, x, validity):
        validity = self.model._validate_input(x, validity)
        if self.partitions > x.shape[1]:
            raise ValueError("more partitions than time steps")
        self.model.eval()
        totals = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        counts = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
        with torch.inference_mode():
            for mask in evaluation_masks(validity, self.partitions):
                pred, _ = self.model.reconstruct(x, mask, validity)
                target = mask.unsqueeze(-1) & validity
                totals += ((pred-x).square()*target).sum(dim=(1, 2))
                counts += target.sum(dim=(1, 2))
        if torch.any(counts == 0):
            raise ValueError("no observed targets in one or more scoring windows")
        scores = totals / counts
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite masked reconstruction score")
        return scores.cpu().numpy()


def ema_smooth(scores, span: int):
    scores = finite_vector(scores)
    if type(span) is not int or span < 1:
        raise ValueError("EMA span must be a positive integer")
    alpha = 2/(span+1)
    result = scores.copy()
    for i in range(1, len(scores)):
        result[i] = alpha*scores[i] + (1-alpha)*result[i-1]
    return result
