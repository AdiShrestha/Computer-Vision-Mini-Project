"""Engineering integrity checks for the explicit local diagnostic only."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel_gl.wp06_diagnostic_training import (
    CHANNEL_ORDER,
    CheckpointSelector,
    DiagnosticMaskedMAE,
    FittedTransform,
    _load_checkpoint,
    _load_view,
    _save_checkpoint,
    apply_channel_values,
    apply_transform,
    fit_transform,
    masked_mse,
)


def test_masked_mse_is_independently_recomputable_and_preserves_empty_support():
    prediction = torch.tensor([[[1.0, 20.0], [3.0, 4.0]]], requires_grad=True)
    truth = torch.tensor([[[0.0, -100.0], [1.0, 4.0]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    loss, count = masked_mse(prediction, truth, mask)
    assert count == 2
    assert loss.item() == 2.5  # ((1-0)^2 + (3-1)^2) / 2
    loss.backward()
    assert torch.equal(prediction.grad, torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]))
    empty, empty_count = masked_mse(prediction, truth, torch.zeros_like(mask))
    assert empty_count == 0 and empty.item() == 0.0
    assert empty.requires_grad


def test_masked_model_has_exact_declared_shape_and_finite_output():
    torch.manual_seed(12)
    model = DiagnosticMaskedMAE().eval()
    values = torch.zeros((1, 8, 7))
    observed = torch.ones_like(values, dtype=torch.bool)
    modality = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    time = torch.zeros((1, 8, 4)); time[:, 4:, 3] = 1
    target = torch.zeros((1, 8, 7), dtype=torch.bool); target[:, 0, 5:] = True
    out = model(values, observed, modality, time, target)
    assert out.shape == values.shape and torch.isfinite(out).all()
    assert model.parameter_count == sum(p.numel() for p in model.parameters())
    assert model.parameter_count > 400_000


def test_target_values_do_not_change_encoder_context_projection():
    torch.manual_seed(4)
    model = DiagnosticMaskedMAE().eval()
    values_a = torch.randn(1, 8, 7)
    values_b = values_a.clone(); values_b[:, 0, :] += 1000
    observed = torch.ones_like(values_a, dtype=torch.bool)
    modality = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    time = torch.zeros((1, 8, 4)); target = torch.zeros((1, 8, 7), dtype=torch.bool)
    target[:, 0, :] = True
    # The encoder input contract is that target token values and validity are zeroed
    # by the caller; a perturbation in the source target cannot enter that context.
    for x in (values_a, values_b):
        x[:, 0, :] = 0
    observed[:, 0, :] = False
    with torch.no_grad():
        out_a = model(values_a, observed, modality, time, target)
        out_b = model(values_b, observed, modality, time, target)
    assert torch.equal(out_a, out_b)


def test_fit_transform_ignores_monitor_parity_pixels_and_monitor_applies_frozen_state():
    root = Path.cwd()
    fit_view = _load_view(root, 0)
    transform = fit_transform(fit_view)
    changed_fit_view = deepcopy(fit_view)
    for si, sample in enumerate(changed_fit_view["samples"]):
        for gi, group in enumerate(sample["groups"]):
            monitor_pixels = ~changed_fit_view["partition_by_group"][si][gi]
            group["values"][:, monitor_pixels] += np.float32(1234.5)
    changed_transform = fit_transform(changed_fit_view)
    assert changed_transform.to_dict() == transform.to_dict()

    monitor_view = _load_view(root, 1)
    prepared, receipt = apply_transform(
        monitor_view, transform, application_role="monitor_application"
    )
    assert receipt["fit_transform_sha256"] == transform.state_sha256
    assert receipt["fit_transform_fitted_here"] is False
    for sample in monitor_view["samples"]:
        for group in sample["groups"]:
            offset = 0 if group["metadata"]["sensor"] == "SENTINEL_2_OPTICAL" else 5
            normalized, valid = apply_channel_values(
                group["values"], group["observed_valid"], transform, channel_offset=offset
            )
            for channel in range(group["values"].shape[0]):
                fit_mean = transform.means[offset + channel]
                fit_scale = transform.scales[offset + channel]
                if fit_mean is None or fit_scale is None:
                    assert not valid[channel].any()
                    continue
                expected = (group["values"][channel][valid[channel]] - fit_mean) / fit_scale
                assert np.allclose(
                    normalized[channel][valid[channel]], expected, rtol=1e-5, atol=1e-6
                )
    assert prepared and all(x["values"].shape[-1] == len(CHANNEL_ORDER) for x in prepared)
    with pytest.raises(ValueError, match="only the declared parity-0"):
        fit_transform(monitor_view)
    with pytest.raises(ValueError, match="role does not match"):
        apply_transform(monitor_view, transform, application_role="fit_application")
    with pytest.raises(FrozenInstanceError):
        transform.means = tuple(0.0 for _ in transform.means)


def test_fitted_transform_and_checkpoint_identity_reject_corruption(tmp_path):
    transform = fit_transform(_load_view(Path.cwd(), 0))
    corrupt = transform.to_dict()
    corrupt["means"][0] = 999.0
    with pytest.raises(ValueError, match="SHA-256"):
        FittedTransform.from_dict(corrupt)

    model = DiagnosticMaskedMAE()
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, model, transform, "test", 1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert isinstance(saved["model_state"], OrderedDict)
    _load_checkpoint(checkpoint, DiagnosticMaskedMAE(), transform)

    for corruption, message in (
        ("key", "model keys differ"),
        ("shape", "tensor shape mismatch"),
        ("dtype", "tensor dtype mismatch"),
    ):
        malformed = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = OrderedDict(malformed["model_state"])
        key = next(iter(state))
        if corruption == "key":
            state["unexpected.weight"] = state.pop(key)
        elif corruption == "shape":
            state[key] = state[key].reshape(-1)[:1]
        else:
            state[key] = state[key].to(torch.float64)
        malformed["model_state"] = state
        malformed_path = tmp_path / f"checkpoint_bad_{corruption}.pt"
        torch.save(malformed, malformed_path)
        with pytest.raises(ValueError, match=message):
            _load_checkpoint(malformed_path, DiagnosticMaskedMAE(), transform)

    bad_transform = transform.to_dict()
    bad_transform["fit_source_values_sha256"] = "0" * 64
    bad_transform["state_sha256"] = __import__(
        "sentinel_gl.wp06_diagnostic_training", fromlist=["_canonical_sha256"]
    )._canonical_sha256({k: v for k, v in bad_transform.items() if k != "state_sha256"})
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_checkpoint(checkpoint, DiagnosticMaskedMAE(), FittedTransform.from_dict(bad_transform))


def test_checkpoint_selector_honors_minimum_eligibility_and_null_monitor_semantics():
    selector = CheckpointSelector(
        minimum_optimization_epochs=4,
        earliest_checkpoint_epoch=4,
        patience=2,
        tolerance=0.1,
    )
    early = selector.observe(1, 0.01, 50)
    assert not early["eligible"] and selector.best_epoch is None
    assert selector.nonimprovement_count == 0
    first = selector.observe(4, 2.0, 50)
    assert first["selected"] and selector.best_epoch == 4
    missing = selector.observe(5, None, 0)
    assert not missing["eligible"] and selector.nonimprovement_count == 0
    tie = selector.observe(6, 2.05, 50)
    assert not tie["selected"] and tie["reason"] == "TIE_WITHIN_TOLERANCE_RETAIN_EARLIER"
    assert selector.best_epoch == 4 and selector.nonimprovement_count == 1
    stop = selector.observe(7, 2.2, 50)
    assert stop["stop"] and selector.best_epoch == 4

    unavailable = CheckpointSelector(
        minimum_optimization_epochs=2,
        earliest_checkpoint_epoch=2,
        patience=1,
        tolerance=0.0,
    )
    for epoch in range(1, 4):
        unavailable.observe(epoch, None, 0)
    assert unavailable.best_epoch is None
    assert unavailable.selected_status == "UNAVAILABLE_NO_ELIGIBLE_FINITE_MONITOR_ESTIMATE"
