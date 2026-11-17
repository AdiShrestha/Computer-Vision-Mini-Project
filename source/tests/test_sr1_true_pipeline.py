"""Integrity tests for the corrected SR-1 feature and temporal contracts."""
import numpy as np
import pytest
import torch

from sentinel_gl.sr1_features import (
    apply_robust_scaler,
    compute_guarded_indices,
    fit_robust_scaler,
    tokenize_reflectance_patches,
)
from sentinel_gl.sr1_model import SR1TransformerMAE, train_factorized_sr1_model


def test_guard_preserves_valid_negative_reflectance_and_marks_small_denominator():
    ndwi, mndwi = compute_guarded_indices(
        np.array([-0.02, 0.1]), np.array([0.03, -0.1]), np.array([0.01, 0.2]), delta=0.002,
    )
    assert ndwi[0] == pytest.approx(-5.0)
    assert np.isnan(ndwi[1])
    assert mndwi[1] == pytest.approx(-1.0 / 3.0)


def test_patch_indices_are_computed_from_patch_mean_reflectance():
    values = np.zeros((3, 160, 160), dtype=np.float64)
    values[0, :, :] = 0.1
    values[1, :, :] = 0.2
    values[2, :, :] = 0.1
    values[0, :8, :16] = 0.3
    values[1, :8, :16] = 0.05
    valid = np.ones_like(values, dtype=bool)
    scl = np.full((160, 160), 4, dtype=np.uint8)
    zones = np.full((160, 160), 2, dtype=np.uint8)
    result = tokenize_reflectance_patches(values, valid, scl, zones)
    expected = ((0.1 * 128 + 0.3 * 128) / 256 - (0.2 * 128 + 0.05 * 128) / 256) / (
        (0.1 * 128 + 0.3 * 128) / 256 + (0.2 * 128 + 0.05 * 128) / 256
    )
    assert result["features"][0, 3] == pytest.approx(expected)
    pixelwise = np.mean((values[0, :16, :16] - values[1, :16, :16]) /
                        (values[0, :16, :16] + values[1, :16, :16]))
    assert result["features"][0, 3] != pytest.approx(pixelwise)


def test_undefined_patch_index_is_explicit_missing_with_reason():
    values = np.empty((3, 160, 160), dtype=np.float64)
    values[0] = 0.1
    values[1] = -0.1
    values[2] = 0.2
    valid = np.ones_like(values, dtype=bool)
    scl = np.full((160, 160), 4, dtype=np.uint8)
    zones = np.full((160, 160), 2, dtype=np.uint8)
    result = tokenize_reflectance_patches(values, valid, scl, zones, delta=0.002)
    assert np.isnan(result["features"][0, 3])
    assert result["observed"][0, 3] is np.False_
    assert result["statuses"][0][3] == "UNDEFINED_INDEX_DENOMINATOR"
    assert result["observed"][0, :3].all()


def _record(oid, role, values):
    features = np.asarray(values, dtype=np.float64).reshape(2, 2)
    return {"observation_id": oid, "role": role, "features": features,
            "observed": np.isfinite(features)}


def test_scaler_requires_fit_role_and_excludes_validation_values():
    fit_a = _record("fit-a", "FIT", [1, 2, 3, 4])
    fit_b = _record("fit-b", "FIT", [2, 3, 4, 5])
    val = _record("val", "VAL", [1000, 2000, 3000, 4000])
    first = fit_robust_scaler([fit_a, fit_b, val], ["fit-a", "fit-b"])
    val["features"][:] = -1e9
    second = fit_robust_scaler([fit_a, fit_b, val], ["fit-a", "fit-b"])
    assert first == second
    with pytest.raises(ValueError, match="not independently declared FIT"):
        fit_robust_scaler([fit_a, fit_b, val], ["val"])
    scaled = apply_robust_scaler(val["features"], val["observed"], first)
    assert np.all(scaled < 0)


def test_factorized_temporal_attention_cannot_use_future_scene_values():
    torch.manual_seed(17)
    model = SR1TransformerMAE(dropout=0.0).eval()
    x = torch.randn(1, 3, 4, 5)
    observed = torch.ones(1, 3, 4, dtype=torch.bool)
    masked = torch.zeros_like(observed)
    masked[:, :, 0] = True
    zones = torch.zeros(1, 3, 4, dtype=torch.long)
    times = torch.randn(1, 3, 3)
    with torch.no_grad():
        before, _ = model(x, observed, masked, zones, times)
        changed = x.clone()
        changed[:, 2] += 1000
        after, _ = model(changed, observed, masked, zones, times)
    assert torch.allclose(before[:, 0], after[:, 0], atol=1e-6, rtol=0)
    assert model.count_parameters() == 207685


def test_factorized_temporal_attention_is_finite_with_leading_and_interior_missingness():
    torch.manual_seed(19)
    model = SR1TransformerMAE(num_patches=4, dropout=0.0).eval()
    x = torch.randn(1, 8, 4, 5)
    observed = torch.tensor([[[False, True, True, False],
                              [False, True, False, False],
                              [False, True, True, False],
                              [True, False, True, True],
                              [True, True, False, True],
                              [True, True, True, False],
                              [True, False, True, True],
                              [True, True, True, True]]])
    masked = observed.clone()
    zones = torch.zeros(1, 8, 4, dtype=torch.long)
    times = torch.randn(1, 8, 3)
    with torch.no_grad():
        before, latent_before = model(x, observed, masked, zones, times)
        changed = x.clone()
        changed[:, 7] += 1000
        after, latent_after = model(changed, observed, masked, zones, times)
    assert torch.isfinite(before).all()
    assert torch.isfinite(latent_before).all()
    assert torch.allclose(before[:, :7], after[:, :7], atol=1e-6, rtol=0)
    assert torch.allclose(latent_before[:, :7], latent_after[:, :7], atol=1e-6, rtol=0)


def _train_record(oid, role, year, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(8, 5)).astype(np.float32)
    return {"observation_id": oid, "role": role, "site_id": "site-a",
            "acquired_at": f"{year}-06-01T00:00:00Z", "std_features": values,
            "observed": np.ones_like(values, dtype=bool), "zones": np.zeros(8, dtype=np.int64),
            "time_feats": np.zeros(3, dtype=np.float32)}


def test_training_firewall_rejects_validation_id_in_optimizer_scope():
    records = [_train_record("fit", "FIT", 2020, 1),
               _train_record("val", "VAL", 2022, 2)]
    model = SR1TransformerMAE(num_patches=8, dropout=0.0)
    with pytest.raises(ValueError, match="not declared FIT"):
        train_factorized_sr1_model(model, records, fit_observation_ids=["val"],
                                   validation_observation_ids=["fit"], max_epochs=1)


def test_training_runs_one_real_optimizer_step_with_disjoint_ids():
    records = [_train_record("fit-a", "FIT", 2020, 1),
               _train_record("fit-b", "FIT", 2021, 2),
               _train_record("val", "VAL", 2022, 3)]
    model = SR1TransformerMAE(num_patches=8, dropout=0.0)
    history = train_factorized_sr1_model(
        model, records, fit_observation_ids=["fit-a", "fit-b"],
        validation_observation_ids=["val"], max_epochs=1, max_context_scenes=2,
    )
    assert history["optimizer_steps"] == 1
    assert history["fit_validation_disjoint"] is True
    assert history["best_epoch"] == 1
    assert np.isfinite(history["best_val_loss"])


def test_training_fails_closed_on_nonfinite_validation_block():
    class NonfiniteOnlyDuringValidation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, x, observed, masked, zones, times, modality_id=0):
            prediction = x * self.scale if self.training else torch.full_like(x, float("nan"))
            return prediction, prediction

        def count_parameters(self):
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    records = [_train_record("fit-a", "FIT", 2020, 1),
               _train_record("fit-b", "FIT", 2021, 2),
               _train_record("val", "VAL", 2022, 3)]
    model = NonfiniteOnlyDuringValidation()
    with pytest.raises(FloatingPointError, match="non-finite VAL masked loss; refusing to omit"):
        train_factorized_sr1_model(
            model, records, fit_observation_ids=["fit-a", "fit-b"],
            validation_observation_ids=["val"], max_epochs=1, max_context_scenes=1,
        )
