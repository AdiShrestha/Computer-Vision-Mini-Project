"""Unit tests for the corrected SR-1 scoring, geometry, statistics and positive control.

All inputs here are constructed in the test; none is an observation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch
from shapely.geometry import box

from sentinel_gl.sr1_corpus import role_for_time
from sentinel_gl.sr1_geometry import inventory_zone_map, mask_agreement
from sentinel_gl.sr1_model import SR1TransformerMAE, sample_training_mask
from sentinel_gl.sr1_positive_control import ORIGIN, inject_drainage
from sentinel_gl.sr1_scoring import (
    add_score_c_per_site, entry_errors, model_passes, patch_folds, reference_predictions, score_a,
)
from sentinel_gl.sr1_stats import block_length, paired_ratio_difference, sample_sd


def _scenes(n_scenes=6, n_patches=100, channels=5, seed=0, missing=0.3):
    rng = np.random.default_rng(seed)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_scenes):
        observed = np.repeat(rng.random(n_patches)[:, None] > missing, channels, axis=1)
        feats = np.where(observed, rng.normal(size=(n_patches, channels)), np.nan).astype(np.float32)
        dt = base + timedelta(days=10 * i)
        rows.append({"observation_id": f"s{i}", "site_id": "A", "role": "FIT" if i < 4 else "VAL",
                     "acquired_at": dt.isoformat(), "dt": dt, "std_features": feats, "observed": observed,
                     "zones": np.where(np.arange(n_patches) < 30, 0, 2).astype(np.int64),
                     "time_feats": np.zeros(3, np.float32)})
    return rows


def test_role_boundaries_use_sikkim_event_bracket():
    utc = timezone.utc
    assert role_for_time(datetime(2021, 12, 31, 23, tzinfo=utc)) == "FIT"
    assert role_for_time(datetime(2022, 6, 1, tzinfo=utc)) == "VAL"
    assert role_for_time(datetime(2023, 6, 30, 23, tzinfo=utc)) == "FIT"
    assert role_for_time(datetime(2023, 9, 30, 23, tzinfo=utc)) == "VAL"
    assert role_for_time(datetime(2023, 10, 1, 12, tzinfo=utc)) == "OUT_OF_PROTOCOL_WINDOW"
    assert role_for_time(datetime(2023, 10, 3, 0, tzinfo=utc)) == "EVENT_UNCERTAIN"
    assert role_for_time(datetime(2023, 10, 4, 18, 30, tzinfo=utc)) == "EVAL"


def test_training_mask_never_hides_a_whole_scene():
    torch.manual_seed(1)
    observed = torch.rand(1, 8, 100) < 0.6
    observed[0, 3] = False
    observed[0, 4, :] = False
    observed[0, 4, 7] = True
    masked = sample_training_mask(observed, 0.5)
    assert not (masked & ~observed).any()
    assert not masked[0, 3].any() and not masked[0, 4].any()
    for t in (0, 1, 2, 5, 6, 7):
        n = int(observed[0, t].sum())
        assert int(masked[0, t].sum()) == min(n - 1, max(1, round(0.5 * n)))


def test_checkerboard_cross_mask_hides_every_observed_patch_exactly_once():
    folds = patch_folds()
    assert folds.sum() == 50
    assert folds[0] != folds[1] and folds[0] != folds[10] and folds[0] == folds[11]
    torch.manual_seed(0)
    model = SR1TransformerMAE(dropout=0.0).eval()
    rows = _scenes()
    out = model_passes(model, rows, 5, folds)
    obs = rows[5]["observed"].any(axis=1)
    assert np.array_equal(out["target_observed_patch"], obs)
    for name in ("model", "model_visible_inputs_zeroed", "model_all_context_hidden"):
        assert np.isfinite(out[name][obs]).all(), name
        assert np.isnan(out[name][~obs]).all(), name
    assert out["latent"].shape == (100, 64)


def test_reference_predictors_are_finite_on_targets_and_exclude_query():
    rows = _scenes()
    folds = patch_folds()
    preds, info = reference_predictions(rows, 2, {"s0", "s1", "s2", "s3"}, folds)
    target = rows[2]["std_features"].astype(float)
    tv = rows[2]["observed"] & np.isfinite(target)
    assert info["fit_reference_scenes"] == 3
    for name, pred in preds.items():
        assert np.isfinite(pred[tv]).all(), name
        err = entry_errors(pred, target, tv, rows[2]["zones"])
        assert err["n_all"] == int(tv.sum())
        assert err["n_zone01"] + err["n_zone2"] == err["n_all"]
    # Zero predictor error equals the sum of squared standardized targets.
    assert entry_errors(preds["fit_median"], target, tv, rows[2]["zones"])["sse_all"] == pytest.approx(
        float((target[tv] ** 2).sum()))


def test_score_a_null_without_zone_patch_and_zone_contrast():
    zones = np.array([0, 1, 2, 2])
    target = np.zeros((4, 2))
    tv = np.array([[False, False], [False, False], [True, True], [True, False]])
    pred = np.ones((4, 2))
    a = score_a(pred, target, tv, zones)
    assert a["score_a_zone0_1"] is None and a["score_a_status"] == "NOT_ESTIMABLE_NO_OBSERVED_ZONE_0_1_PATCH"
    assert a["score_a_zone2"] == pytest.approx(1.0)
    tv[0] = [True, True]
    pred[0] = [3.0, 1.0]
    a = score_a(pred, target, tv, zones)
    assert a["score_a_zone0_1"] == pytest.approx(5.0)
    assert a["score_a_zone_contrast"] == pytest.approx(4.0)


def test_score_c_moments_are_fitted_per_site_on_fit_only():
    rows = []
    for site, scale in (("A", 1.0), ("B", 10.0)):
        for i in range(5):
            rows.append({"site_id": site, "role": "FIT", "score_a_zone0_1": scale * (1 + i), "score_b_zone0_1": scale * i})
        rows.append({"site_id": site, "role": "EVAL", "score_a_zone0_1": scale * 1e6, "score_b_zone0_1": scale * 1e6})
    moments = add_score_c_per_site(rows)
    assert moments["A"]["moments"]["b_median"] == pytest.approx(2.0)
    assert moments["B"]["moments"]["b_median"] == pytest.approx(20.0)
    fit_c_a = sorted(r["score_c"] for r in rows if r["site_id"] == "A" and r["role"] == "FIT")
    fit_c_b = sorted(r["score_c"] for r in rows if r["site_id"] == "B" and r["role"] == "FIT")
    assert fit_c_a == pytest.approx(fit_c_b)


def test_inventory_zone_map_and_agreement():
    grid = {"shape": [10, 10], "transform": [20.0, 0.0, 0.0, 0.0, -20.0, 200.0]}
    lake = box(40, 80, 120, 160)  # 4 x 4 pixels
    info = inventory_zone_map(lake, grid, buffer_m=20.0)
    assert info["rasterized_lake_pixels"] == 16
    assert info["outline_fraction_inside_window"] == pytest.approx(1.0)
    assert (info["zones"] == 1).sum() == 16  # 4-neighbour ring at <= 20 m
    agree = mask_agreement(info["lake"], info["lake"])
    assert agree["iou"] == 1.0


def test_positive_control_replaces_only_clear_lake_pixels_and_marks_origin():
    refl = np.full((3, 8, 8), 0.05)
    zones = np.full((8, 8), 2)
    zones[2:6, 2:6] = 0
    refl[:, zones == 2] = 0.3
    valid = np.ones_like(refl, bool)
    scl = np.full((8, 8), 6, np.uint8)
    scl[5, 5] = 9  # cloud inside the lake
    out, info = inject_drainage(refl, valid, scl, zones, 1.0)
    assert out["origin"] == ORIGIN and info["status"] == "APPLIED"
    assert info["replaced_clear_pixels"] == 15
    assert np.allclose(out["reflectance"][:, 2, 2], 0.3)
    assert np.allclose(out["reflectance"][:, 5, 5], 0.05)
    half, info = inject_drainage(refl, valid, scl, zones, 0.5)
    assert info["selected_lake_pixels"] == 8
    assert np.allclose(half["reflectance"][:, 2:6, 4:6][:, scl[2:6, 4:6] != 9], 0.3)
    assert np.allclose(half["reflectance"][:, 2:6, 2:4], 0.05)


def test_block_bootstrap_interval_and_status():
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows = [{"dt": base + timedelta(days=10 * i), "n": 10, "sse_a": 5.0 + (i % 3), "sse_b": 10.0}
            for i in range(20)]
    assert block_length([r["dt"] for r in rows], 30) == 3
    res = paired_ratio_difference({"A": rows}, a="a", b="b", n_boot=500)
    assert res["status"] == "ESTIMATED"
    assert res["lower"] <= res["estimate"] <= res["upper"] < 0
    few = paired_ratio_difference({"A": rows[:2]}, a="a", b="b", n_boot=50)
    assert few["status"] == "NOT_ESTIMABLE_LT_3_SCENES" and few["lower"] is None
    assert sample_sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)
