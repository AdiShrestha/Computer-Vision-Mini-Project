"""Regression tests for defects found in the 2026-11-17 submission-readiness review.

Each test encodes a property the retained SR-1 checkpoints violated:

* the training mask must hide roughly half of each scene's observed patches and
  leave visible context in every scene (review section 4.2, [R1]);
* the validation selection metric must weight blocks by their target count and
  report block counts per fit (review section 4.2 and 4.7);
* Score B must be unavailable for a scene with no observed Zone 0/1 patch
  (review section 4.5).

They were run against the unmodified trainer before the fix and failed; see
docs/protocol/PRE_FIX_REGRESSION_RUN.txt.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinel_gl.sr1_model import train_factorized_sr1_model


class _SpyModel(torch.nn.Module):
    """Records the masks the trainer passes in training and evaluation mode."""

    def __init__(self, *, eval_offset: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.train_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.eval_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.eval_offset = eval_offset

    def forward(self, x, observed, masked, zones, times, modality_id=0):
        record = (observed.detach().clone(), masked.detach().clone())
        (self.train_calls if self.training else self.eval_calls).append(record)
        # Training output depends on the parameter so the optimizer runs; the
        # evaluation output is exactly zero so validation losses are analytic.
        prediction = torch.zeros_like(x) + (self.weight if self.training else 0.0)
        return prediction, prediction

    def count_parameters(self):
        return 1


def _scene(oid, role, day, n_patches, observed_fraction, seed, value=1.0):
    rng = np.random.default_rng(seed)
    observed = np.zeros((n_patches, 5), dtype=bool)
    keep = rng.permutation(n_patches)[: int(round(observed_fraction * n_patches))]
    observed[keep, :] = True
    features = np.where(observed, value, np.nan).astype(np.float32)
    return {"observation_id": oid, "role": role, "site_id": "site-a",
            "acquired_at": f"2020-01-{day:02d}T00:00:00Z", "std_features": features,
            "observed": observed, "zones": np.zeros(n_patches, dtype=np.int64),
            "time_feats": np.zeros(3, dtype=np.float32)}


@pytest.mark.parametrize("fraction", [0.9, 0.5, 0.25])
def test_training_mask_hides_about_half_per_scene_and_keeps_context(fraction):
    torch.manual_seed(0)
    np.random.seed(0)
    fit = [_scene(f"fit-{i}", "FIT", i + 1, 100, fraction, i) for i in range(8)]
    val = [_scene("val-0", "VAL", 20, 100, fraction, 99)]
    model = _SpyModel()
    train_factorized_sr1_model(model, fit + val, fit_observation_ids=[r["observation_id"] for r in fit],
                               validation_observation_ids=["val-0"], max_epochs=2,
                               max_context_scenes=8, plateau_patience=5)
    assert model.train_calls, "trainer made no optimizer call"
    for observed, masked in model.train_calls:
        obs, msk = observed[0], masked[0]
        assert not (msk & ~obs).any(), "a patch without observations was used as a target"
        for t in range(obs.shape[0]):
            n_obs = int(obs[t].sum())
            if n_obs < 2:
                continue
            hidden = int((msk[t] & obs[t]).sum())
            assert 0.4 <= hidden / n_obs <= 0.6, (t, hidden, n_obs)
            assert hidden < n_obs, f"scene {t} has no visible observed patch"


def test_validation_loss_is_target_weighted_and_block_counts_are_per_fit():
    torch.manual_seed(0)
    np.random.seed(0)
    fit = [_scene(f"fit-{i}", "FIT", i + 1, 20, 1.0, i) for i in range(4)]
    # Block 1: one VAL scene with 2 observed patches whose targets are large.
    small = _scene("val-small", "VAL", 10, 20, 0.1, 50, value=10.0)
    # Block 2 (context 1 => one scene per block): 20 observed patches, small targets.
    big = _scene("val-big", "VAL", 11, 20, 1.0, 51, value=1.0)
    model = _SpyModel()
    history = train_factorized_sr1_model(
        model, fit + [small, big], fit_observation_ids=[r["observation_id"] for r in fit],
        validation_observation_ids=["val-small", "val-big"], max_epochs=1,
        max_context_scenes=1, plateau_patience=5)
    # Evaluation prediction is exactly 0, so squared errors equal squared targets.
    small_targets = int(np.ceil(small["observed"][:, 0].sum() / 2)) * 5
    big_targets = int(np.ceil(big["observed"][:, 0].sum() / 2)) * 5
    weighted = (small_targets * 100.0 + big_targets * 1.0) / (small_targets + big_targets)
    unweighted = (100.0 + 1.0) / 2
    assert history["best_val_loss"] == pytest.approx(weighted, rel=1e-5)
    assert history["best_val_loss"] != pytest.approx(unweighted, rel=1e-3)
    assert history["validation_block_count"] == 2
    assert history["validation_blocks_scored"] == 2
    assert history["validation_target_entries"] == small_targets + big_targets


def test_score_b_is_not_estimable_without_observed_zone_patches():
    from sentinel_gl.sr1_scoring import score_b_for_scene

    rng = np.random.default_rng(0)
    n_patches, rank = 6, 2
    zones = np.array([0, 0, 1, 1, 2, 2])
    refs = []
    for i in range(6):
        refs.append({"observation_id": f"fit-{i}", "doy": 100 + i,
                     "projected": rng.normal(size=(n_patches, rank)),
                     "patch_observed": np.ones(n_patches, dtype=bool)})
    empty_query = {"observation_id": "q", "doy": 102,
                   "projected": np.zeros((n_patches, rank)),
                   "patch_observed": np.array([False, False, False, False, True, True])}
    value, status, n_eligible = score_b_for_scene(empty_query, refs, zones, k=5, window_days=45)
    assert value is None
    assert status == "NOT_ESTIMABLE_NO_OBSERVED_ZONE_0_1_PATCH"
    assert n_eligible == 0
    observed_query = dict(empty_query, patch_observed=np.ones(n_patches, dtype=bool),
                          projected=rng.normal(size=(n_patches, rank)))
    value, status, n_eligible = score_b_for_scene(observed_query, refs, zones, k=5, window_days=45)
    assert status == "ESTIMATED" and value is not None and n_eligible == 4
    # Unobserved reference patches never count as neighbours.
    for ref in refs[:2]:
        ref["patch_observed"] = np.zeros(n_patches, dtype=bool)
    value, status, n_eligible = score_b_for_scene(observed_query, refs, zones, k=5, window_days=45)
    assert value is None and status == "NOT_ESTIMABLE_LT_K_OBSERVED_FIT_NEIGHBOURS"
