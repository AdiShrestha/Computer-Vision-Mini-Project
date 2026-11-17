from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import numpy as np
import pytest

from sentinel_gl.admission import AdmissionError
from sentinel_gl.multimodal import (
    RaggedPanelDataset,
    _verify_input_bindings,
    _workspace_member_path,
    collate_ragged_groups,
    validate_multimodal_panel_document,
    verify_multimodal_panel,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL = Path("data/derived/wp05_wp06_multimodal_attempt_004/panel.v1.json")


@pytest.fixture(scope="module")
def panel_document():
    path = ROOT / PANEL
    assert path.is_file(), "source-bound diagnostic panel must be built before these acceptance tests"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_replay():
    return verify_multimodal_panel(ROOT, PANEL)


@pytest.fixture(scope="module")
def diagnostic_dataset():
    return RaggedPanelDataset(ROOT, PANEL, mode="development-diagnostic")


def _verify_mutated_manifest(document, mutate):
    path = ROOT / "data/derived/wp05_wp06_test_corruptions"
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"panel-{uuid.uuid4().hex}.json"
    changed = copy.deepcopy(document)
    mutate(changed)
    target.write_text(json.dumps(changed, allow_nan=False), encoding="utf-8")
    try:
        return verify_multimodal_panel(ROOT, target)
    finally:
        target.unlink(missing_ok=True)
        try:
            path.rmdir()
        except OSError:
            pass


def test_panel_replays_every_emitted_group_from_retained_source_arrays(source_replay):
    assert source_replay["status"] == "PASS_SOURCE_BOUND_DEVELOPMENT_DIAGNOSTIC_ONLY"
    assert source_replay["verified_panels"] == 2
    assert source_replay["verified_groups"] == 4
    assert source_replay["source_array_channel_replays"] == 14
    assert source_replay["research_admissible"] is False
    assert source_replay["training_admitted"] is False


def test_absolute_retained_member_paths_rebase_only_to_verified_workspace_subtree():
    old_root = "/retained/original/project/data/raw/wp03/native-attempt/inputs/member.xml"
    assert _workspace_member_path(ROOT, old_root) == Path("data/raw/wp03/native-attempt/inputs/member.xml")
    with pytest.raises(AdmissionError, match="no unique workspace data anchor"):
        _workspace_member_path(ROOT, "/outside/unrelated/member.xml")


def test_acquisitions_remain_asynchronous_and_per_panel(diagnostic_dataset, panel_document):
    assert len(diagnostic_dataset) == 2
    assert len(panel_document["groups"]) == 4
    for panel in panel_document["panels"]:
        groups = [next(g for g in panel_document["groups"] if g["group_id"] == group_id)
                  for group_id in panel["group_ids"]]
        assert {g["sensor"] for g in groups} == {"SENTINEL_1_IW_GRD", "SENTINEL_2_OPTICAL"}
        sar = next(g for g in groups if g["sensor"] == "SENTINEL_1_IW_GRD")
        optical = next(g for g in groups if g["sensor"] == "SENTINEL_2_OPTICAL")
        assert sar["acquisition"]["start_raw"].startswith("2023-09-01T")
        assert sar["acquisition"]["timezone_status"] == "NOT_EXPLICIT_IN_RETAINED_START_STOP_STRINGS"
        assert optical["acquisition"]["at"].startswith("2023-09-")
        assert sar["group_id"] != optical["group_id"]
    sample = diagnostic_dataset[0]
    collated = collate_ragged_groups([sample])
    assert [g["group_id"] for g in collated["sample_metadata"][0]["groups"]] == [
        g["metadata"]["group_id"] for g in sample["groups"]]


def test_absent_channels_and_unknown_availability_stay_explicit(panel_document):
    for group in panel_document["groups"]:
        assert group["acquisition"]["historical_availability"]["status"] == "UNKNOWN"
        assert group["acquisition"]["historical_availability"]["at"] is None
    optical = [g for g in panel_document["groups"] if g["sensor"] == "SENTINEL_2_OPTICAL"]
    for group in optical:
        assert [c["channel_id"] for c in group["channels"]] == ["B03", "B08", "B11", "NDWI", "MNDWI"]
        assert "NDSI" not in [c["channel_id"] for c in group["channels"]]
        unavailable = {x["channel_id"]: x for x in group["unavailable_channels"]}
        assert set(unavailable) == {"B02", "B04", "B12"}
        assert all(x["status"] == "UNAVAILABLE" and x["value"] is None for x in unavailable.values())
    assert panel_document["missingness_contract"]["no_interpolation_forward_fill_or_repeated_source"] is True
    assert panel_document["unavailable_dependencies"]
    assert all(item["status"] == "UNAVAILABLE" and item["value"] is None
               for item in panel_document["unavailable_dependencies"])


def test_target_masks_only_hide_valid_observations_and_exclude_channel_padding(diagnostic_dataset):
    sample = diagnostic_dataset[0]
    optical_index = next(i for i, group in enumerate(sample["groups"])
                         if group["metadata"]["sensor"] == "SENTINEL_2_OPTICAL")
    group = sample["groups"][optical_index]
    target = np.zeros_like(group["observed_valid"])
    coordinate = tuple(np.argwhere(group["observed_valid"])[0])
    target[coordinate] = True
    masks = [None] * len(sample["groups"])
    masks[optical_index] = target
    batch = collate_ragged_groups([sample], target_masks=[masks])
    assert batch["target_count"] == 1
    assert batch["status"] == "TARGET_SUPPORT_AVAILABLE"
    assert batch["target_mask"][0, optical_index, coordinate[0], coordinate[1] * group["values"].shape[2] + coordinate[2]]
    pixel = coordinate[1] * group["values"].shape[2] + coordinate[2]
    assert batch["context_visible"].sum() == batch["observed_valid"].sum() - 5
    assert batch["artificial_target_mask"].sum() == batch["target_mask"].sum() == 1
    assert batch["dependency_hidden_mask"][0, optical_index, :5, pixel].sum() == 4
    sar_index = next(i for i, value in enumerate(sample["groups"])
                     if value["metadata"]["sensor"] == "SENTINEL_1_IW_GRD")
    assert batch["padding_mask"][0, sar_index, 4].all()
    assert not batch["target_mask"][0, sar_index, 4].any()

    corrupted = {**sample, "groups": [dict(x) for x in sample["groups"]]}
    broken_group = corrupted["groups"][optical_index]
    broken_group["values"] = broken_group["values"].copy()
    broken_group["observed_valid"] = broken_group["observed_valid"].copy()
    broken_group["values"][coordinate] = np.nan
    broken_group["observed_valid"][coordinate] = False
    invalid_target = target.copy(); invalid_target[coordinate] = True
    masks[optical_index] = invalid_target
    with pytest.raises(AdmissionError, match="target includes missing/invalid"):
        collate_ragged_groups([corrupted], target_masks=[masks])


def test_no_target_support_returns_explicit_skip(diagnostic_dataset):
    result = collate_ragged_groups([diagnostic_dataset[0]])
    assert result["status"] == "SKIP_EMPTY_OBSERVED_TARGET_SUPPORT"
    assert result["estimable"] is False
    assert result["target_count"] == 0


def test_masked_b03_hides_same_pixel_indices_and_basebands_only(diagnostic_dataset):
    sample = diagnostic_dataset[0]
    oi = next(i for i,g in enumerate(sample["groups"])
              if g["metadata"]["sensor"] == "SENTINEL_2_OPTICAL")
    si = next(i for i,g in enumerate(sample["groups"])
              if g["metadata"]["sensor"] == "SENTINEL_1_IW_GRD")
    optical = sample["groups"][oi]
    channel = optical["channel_ids"].index("B03")
    y, x = map(int, np.argwhere(optical["observed_valid"][channel])[0])
    pixel = y * optical["values"].shape[2] + x
    mask = np.zeros_like(optical["observed_valid"]); mask[channel,y,x] = True
    targets = [None] * len(sample["groups"]); targets[oi] = mask
    result = collate_ragged_groups([sample], target_masks=[targets])
    assert result["target_count"] == 1
    assert result["target_mask"].sum() == result["artificial_target_mask"].sum() == 1
    assert result["target_values"][0,oi,channel,pixel] == optical["values"][channel,y,x]
    assert np.isnan(result["target_values"][0,oi,1:,pixel]).all()
    assert result["dependency_hidden_mask"][0,oi,1:,pixel].all()
    assert not result["context_visible"][0,oi,:,pixel].any()
    assert np.all(result["encoder_context_values"][0,oi,:,pixel] == 0.0)
    other_pixel = pixel + 1 if x + 1 < optical["values"].shape[2] else pixel - 1
    assert result["context_visible"][0,oi,:,other_pixel].all()
    assert np.array_equal(result["encoder_context_values"][0,oi,:,other_pixel],
                          optical["values"].reshape(5,-1)[:,other_pixel])
    # The SAR scene is a distinct asynchronous observation, not same-acquisition reconstruction.
    assert result["context_visible"][0,si,:len(sample["groups"][si]["channel_ids"]),pixel].all()
    assert result["sample_metadata"][0]["groups"][oi]["context_dependency_policy"] == \
        "OPTICAL_INDEX_DEPENDENCY_CLOSURE_V1"


def test_masked_index_hides_its_optical_basebands_without_adding_targets(diagnostic_dataset):
    sample = diagnostic_dataset[1]
    oi = next(i for i,g in enumerate(sample["groups"])
              if g["metadata"]["sensor"] == "SENTINEL_2_OPTICAL")
    optical = sample["groups"][oi]
    channel = optical["channel_ids"].index("NDWI")
    y, x = map(int, np.argwhere(optical["observed_valid"][channel])[0])
    pixel = y * optical["values"].shape[2] + x
    mask = np.zeros_like(optical["observed_valid"]); mask[channel,y,x] = True
    targets = [None] * len(sample["groups"]); targets[oi] = mask
    result = collate_ragged_groups([sample], target_masks=[targets])
    assert result["target_count"] == int(mask.sum()) == 1
    assert result["target_mask"].sum() == result["artificial_target_mask"].sum()
    assert result["dependency_hidden_mask"][0,oi,[0,1,2,4],pixel].all()
    assert not result["dependency_hidden_mask"][0,oi,channel,pixel]
    assert not result["context_visible"][0,oi,:5,pixel].any()
    assert np.isfinite(result["target_values"][0,oi,channel,pixel])
    assert np.isnan(np.delete(result["target_values"][0,oi,:,pixel], channel)).all()


def test_partial_natural_missingness_stays_distinct_from_dependency_hiding(diagnostic_dataset):
    sample = diagnostic_dataset[0]
    sample = {**sample, "groups": [dict(group) for group in sample["groups"]]}
    oi = next(i for i,g in enumerate(sample["groups"])
              if g["metadata"]["sensor"] == "SENTINEL_2_OPTICAL")
    optical = sample["groups"][oi]
    optical["values"] = optical["values"].copy()
    optical["observed_valid"] = optical["observed_valid"].copy()
    # Explicit test-only mutation of one retained valid coordinate to exercise partial natural absence.
    y, x = 0, 0
    optical["values"][1,y,x] = np.nan
    optical["observed_valid"][1,y,x] = False
    target = np.zeros_like(optical["observed_valid"]); target[0,y,x] = True
    targets = [None] * len(sample["groups"]); targets[oi] = target
    result = collate_ragged_groups([sample], target_masks=[targets])
    pixel = y * optical["values"].shape[2] + x
    assert result["target_count"] == int(target.sum()) == 1
    assert result["natural_missing_mask"][0,oi,1,pixel]
    assert not result["dependency_hidden_mask"][0,oi,1,pixel]
    assert result["dependency_hidden_mask"][0,oi,[2,3,4],pixel].all()
    assert not result["target_mask"][0,oi,1:,pixel].any()
    assert np.isnan(result["target_values"][0,oi,1:,pixel]).all()
    assert result["context_visible"].sum() == result["observed_valid"].sum() - 4


def test_all_missing_groups_skip_targets_and_keep_padding_separate(diagnostic_dataset):
    sample = diagnostic_dataset[0]
    sample = {**sample, "groups": [dict(group) for group in sample["groups"]]}
    for group in sample["groups"]:
        group["values"] = np.full_like(group["values"], np.nan)
        group["observed_valid"] = np.zeros_like(group["observed_valid"], dtype=bool)
    result = collate_ragged_groups([sample])
    assert result["status"] == "SKIP_EMPTY_OBSERVED_TARGET_SUPPORT"
    assert result["target_count"] == 0
    assert not result["observed_valid"].any()
    assert not result["dependency_hidden_mask"].any()
    assert result["natural_missing_mask"].any()
    assert result["padding_mask"].any()
    assert not (result["natural_missing_mask"] & result["padding_mask"]).any()
    assert not result["target_mask"].any()


def test_sar_target_does_not_expose_target_or_raw_db_sidecar_in_context(diagnostic_dataset):
    sample = diagnostic_dataset[0]
    si = next(i for i,g in enumerate(sample["groups"])
              if g["metadata"]["sensor"] == "SENTINEL_1_IW_GRD")
    group = sample["groups"][si]
    y, x = 0, 0; pixel = y * group["values"].shape[2] + x
    mask = np.zeros_like(group["observed_valid"]); mask[0,y,x] = True
    targets = [None] * len(sample["groups"]); targets[si] = mask
    result = collate_ragged_groups([sample], target_masks=[targets])
    assert result["target_count"] == 1
    assert result["values"][0,si,0,pixel] == 0.0
    assert result["target_values"][0,si,0,pixel] == group["values"][0,y,x]
    assert "raw_db" not in result and "representations" not in result
    assert np.isnan(result["target_values"][0,si,1:,pixel]).all()


def test_batch_padding_is_neutral_masked_and_keeps_metadata(diagnostic_dataset):
    sample = diagnostic_dataset[0]
    batch = collate_ragged_groups([sample, sample])
    assert batch["values"].shape[0] == 2
    assert batch["attention_mask"].shape == (2, 2)
    assert batch["attention_mask"].all()
    assert batch["channel_membership_mask"].sum() == 14
    assert batch["padding_mask"].any()
    assert not batch["target_mask"].any()
    assert np.all(batch["values"][batch["padding_mask"]] == 0.0)
    assert batch["sample_metadata"][0]["groups"][0]["acquisition"]
    assert batch["sample_metadata"][0]["groups"][0]["source_id"]
    assert batch["sample_metadata"][0]["groups"][0]["scope_membership"]["role"] is None
    assert batch["sample_metadata"][0]["groups"][0]["sampling_graph"]["status"] == "UNRESOLVED"
    assert batch["sample_metadata"][0]["groups"][0]["provenance_graph"]["status"] == "NO_FITTED_STATE_OR_ROLE_MEMBERSHIP"


def test_quarantined_panel_requires_explicit_diagnostic_mode(panel_document):
    with pytest.raises(AdmissionError, match="research/training loader admission is blocked"):
        RaggedPanelDataset(ROOT, PANEL)
    with pytest.raises(AdmissionError, match="schema constant mismatch"):
        validate_multimodal_panel_document({**panel_document, "research_admissible": True})


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["groups"][0]["scope_membership"].update(role="evaluation"), "scope/admission"),
    (lambda d: d["groups"][0]["acquisition"]["historical_availability"].update(
        at="2023-09-01T00:00:00Z", status="VERIFIED"), "historical availability"),
])
def test_role_and_late_availability_mutations_are_rejected(panel_document, mutation, match):
    changed = copy.deepcopy(panel_document)
    mutation(changed)
    with pytest.raises(AdmissionError, match=match):
        validate_multimodal_panel_document(changed)


def test_wrong_grid_and_hash_bindings_are_rejected(panel_document):
    with pytest.raises(AdmissionError, match="SAR panel grid differs|optical panel grid differs"):
        _verify_mutated_manifest(panel_document,
            lambda d: d["groups"][0]["grid"]["transform"].__setitem__(2,
                d["groups"][0]["grid"]["transform"][2] + 20))
    changed = copy.deepcopy(panel_document)
    retained_index = next(i for i, item in enumerate(changed["inputs"])
                          if (ROOT / item["path"]).is_file())
    changed["inputs"][retained_index]["sha256"] = "0" * 64
    with pytest.raises((AdmissionError, ValueError), match="checksum mismatch|changed"):
        _verify_input_bindings(ROOT, changed)


def test_deleted_excluded_native_input_requires_exact_retirement_identity(panel_document):
    changed = copy.deepcopy(panel_document)
    retired_index = next(i for i, item in enumerate(changed["inputs"])
                         if item.get("role") == "excluded_native_signed_sar_variant" and
                         not (ROOT / item["path"]).is_file())
    retired_path = changed["inputs"][retired_index]["path"]
    changed["inputs"][retired_index]["sha256"] = "0" * 64
    expected = ("missing excluded panel input is not exactly listed in the retirement manifest: "
                f"{retired_path}")
    with pytest.raises(AdmissionError, match=re.escape(expected)):
        _verify_input_bindings(ROOT, changed)


def test_research_cli_stays_fail_closed(panel_document):
    command = [sys.executable, "source/runners/run_wp05_wp06_panel.py", "verify",
               "--root", str(ROOT), "--panel", PANEL.as_posix()]
    env = os.environ.copy(); env["PYTHONPATH"] = "source"
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["literal_error"].startswith("research panel execution is blocked")
