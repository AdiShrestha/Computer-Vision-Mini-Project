"""Versioned, source-bound 2023 observation panel.

This loader is intentionally separate from the retained panel.v1 path. It
validates source bytes and sensor-specific masks, records the actual support,
and never fills missing measurements. Arrays stay in their source NPZ files;
the versioned panel records their hashes and QA counts instead of duplicating
large rasters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo
import hashlib
import json
import math

import numpy as np

from .integrity import file_hash, loads_strict, verified_file


PANEL_V2_VERSION = 2
CHANNELS_BY_SENSOR = {
    "SENTINEL_2_OPTICAL": ("B03", "B08", "B11", "NDWI", "MNDWI"),
    "SENTINEL_1_IW_GRD": (
        "SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR",
    ),
}
CANONICAL_CHANNELS = (
    "B03", "B08", "B11", "NDWI", "MNDWI",
    "SIGMA0_VV_RAW_LINEAR", "SIGMA0_VH_RAW_LINEAR",
)
NEPAL = ZoneInfo("Asia/Kathmandu")
UTC = timezone.utc
FIT_END_EXCLUSIVE_UTC = datetime(2023, 10, 2, 18, 30, tzinfo=UTC)


class PanelV2Error(ValueError):
    """Malformed, unbound, or unsupported version-2 observation input."""


def sha256_file(path: str | Path) -> str:
    return file_hash(path)


def _sha_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise PanelV2Error("source paths must be nonempty root-relative strings")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise PanelV2Error("source path must not be absolute or contain traversal")
    path = root / rel
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise PanelV2Error("source evidence paths may not traverse symlinks")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise PanelV2Error(f"source evidence file is missing or outside root: {value}")
    return path


def _manifest_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError as exc:
            raise PanelV2Error("observations manifest must be inside the workspace") from exc
    return _relative_file(root, candidate.as_posix())


def parse_explicit_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise PanelV2Error("acquired_at must be an explicit UTC ISO timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise PanelV2Error("acquired_at is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise PanelV2Error("acquired_at must carry an explicit zero-offset UTC timezone")
    return parsed.astimezone(UTC)


def scope_for_acquisition(acquired_at: datetime, acquisition_end_at: datetime | None,
                          fit_end_exclusive: datetime = FIT_END_EXCLUSIVE_UTC) -> dict[str, str]:
    """Assign fit membership from the protocol's strict pre-breach end cutoff.

    Parity-1 is a spatial development monitor from the same pre-breach
    acquisitions. Every later observation is a retrospective trajectory, not
    prospective precursor validation. A pre-cutoff observation without a
    verified end strictly before the cutoff is kept out of fitting.
    """
    if acquired_at.tzinfo is None or acquired_at.utcoffset() != timedelta(0):
        raise PanelV2Error("acquisition start must be explicit UTC")
    if fit_end_exclusive.tzinfo is None or fit_end_exclusive.utcoffset() != timedelta(0):
        raise PanelV2Error("fit_end_exclusive must be explicit UTC")
    acquired_at = acquired_at.astimezone(UTC)
    fit_end_exclusive = fit_end_exclusive.astimezone(UTC)
    if acquisition_end_at is not None:
        if acquisition_end_at.tzinfo is None or acquisition_end_at.utcoffset() != timedelta(0):
            raise PanelV2Error("acquisition end must be explicit UTC")
        acquisition_end_at = acquisition_end_at.astimezone(UTC)
        if acquisition_end_at < acquired_at:
            raise PanelV2Error("acquisition end precedes acquisition start")
    local = acquired_at.astimezone(NEPAL)
    if local.year != 2023 or not (1 <= local.month <= 11) or (local.month == 11 and local.day > 15):
        return {"role": "OUTSIDE_2023_EXPANSION_WINDOW", "event_phase": "UNCLASSIFIED",
                "fit_membership_status": "OUTSIDE_DECLARED_EXPANSION_WINDOW"}
    if acquired_at >= fit_end_exclusive:
        role = "EVALUATION_TRAJECTORY"
        fit_status = "AFTER_FIT_END_EXCLUSIVE"
    elif acquisition_end_at is None:
        role = "FIT_MEMBERSHIP_UNRESOLVED"
        fit_status = "ACQUISITION_END_UNAVAILABLE"
    elif acquisition_end_at < fit_end_exclusive:
        role = "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR"
        fit_status = "ACQUISITION_END_STRICTLY_BEFORE_FIT_END_EXCLUSIVE"
    else:
        role = "FIT_MEMBERSHIP_UNRESOLVED"
        fit_status = "ACQUISITION_CROSSES_FIT_END_EXCLUSIVE"
    if local < datetime(2023, 10, 3, tzinfo=NEPAL):
        phase = "PRE_BREACH_CANDIDATE"
    elif local < datetime(2023, 10, 5, tzinfo=NEPAL):
        phase = "EVENT_ONSET_UNCERTAIN_2023_10_03_TO_2023_10_04"
    else:
        phase = "POST_EVENT_CANDIDATE"
    return {"role": role, "event_phase": phase,
            "fit_membership_status": fit_status,
            "fit_end_exclusive_utc": fit_end_exclusive.isoformat().replace("+00:00", "Z")}


def checkerboard_parity_mask(height: int, width: int, parity: int) -> np.ndarray:
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("checkerboard dimensions must be positive integers")
    if type(parity) is not int or parity not in (0, 1):
        raise ValueError("checkerboard parity must be 0 or 1")
    rows, cols = np.indices((height, width))
    return ((rows // 16 + cols // 16) % 2) == parity


def quadrant_masks(height: int, width: int) -> tuple[np.ndarray, ...]:
    if type(height) is not int or type(width) is not int or min(height, width) < 2:
        raise ValueError("quadrant dimensions must be positive and at least two pixels")
    rows, cols = np.indices((height, width))
    row_mid = height // 2
    col_mid = width // 2
    return (
        (rows < row_mid) & (cols < col_mid),
        (rows < row_mid) & (cols >= col_mid),
        (rows >= row_mid) & (cols < col_mid),
        (rows >= row_mid) & (cols >= col_mid),
    )


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _json_object(path: Path, description: str) -> dict:
    try:
        value = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PanelV2Error(f"{description} is not strict JSON: {path}") from exc
    if type(value) is not dict:
        raise PanelV2Error(f"{description} must be a JSON object")
    return value


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_masked_values(values: np.ndarray, valid: np.ndarray,
                            label: str) -> None:
    if values.dtype != np.dtype(np.float32):
        raise PanelV2Error(f"{label} must use float32 source values")
    if valid.dtype != np.dtype(bool) or valid.shape != values.shape:
        raise PanelV2Error(f"{label} validity must be boolean and shape-matched")
    if not np.isfinite(values[valid]).all():
        raise PanelV2Error(f"{label} has non-finite values marked observed")
    if not np.isnan(values[~valid]).all():
        raise PanelV2Error(f"{label} must retain NaN behind every false validity entry")


def _verify_optical_arrays(arrays: Mapping[str, np.ndarray],
                           observed: np.ndarray) -> dict:
    required = {"scl_classes", "qa_clear"}
    if not required <= set(arrays):
        raise PanelV2Error("optical source NPZ must retain scl_classes and qa_clear")
    scl = np.asarray(arrays["scl_classes"])
    qa = np.asarray(arrays["qa_clear"])
    if scl.shape != (499, 499) or scl.dtype.kind not in "ui":
        raise PanelV2Error("scl_classes must be an integer 499x499 source array")
    if qa.shape != (499, 499) or qa.dtype != np.dtype(bool):
        raise PanelV2Error("qa_clear must be boolean and 499x499")
    clear_class = np.isin(scl, (4, 5, 6))
    allowed = qa & clear_class
    if np.any(observed & ~allowed[None, :, :]):
        raise PanelV2Error("optical observed_valid includes pixels outside qa_clear and SCL 4/5/6")

    stored_values = np.asarray(arrays["values"])
    storage_dtype = stored_values.dtype
    # The producer promotes stored float32 reflectances to float64 for index
    # arithmetic, then serializes each index as float32. Replay that sequence.
    b03, b08, b11, ndwi, mndwi = (stored_values[i] for i in range(5))
    masks = observed
    for index, left, right, output in (
        (3, b03, b08, ndwi), (4, b03, b11, mndwi),
    ):
        left64 = left.astype(np.float64)
        right64 = right.astype(np.float64)
        left_index = 0
        right_index = 1 if index == 3 else 2
        denominator = left64 + right64
        expected_valid = (masks[left_index] & masks[right_index] &
                          np.isfinite(denominator) & (denominator != 0))
        if not np.array_equal(masks[index], expected_valid):
            raise PanelV2Error("optical index mask does not equal its source-band dependency closure")
        expected = np.full((499, 499), np.nan, dtype=storage_dtype)
        replayed = ((left64[expected_valid] - right64[expected_valid]) /
                    denominator[expected_valid]).astype(storage_dtype)
        expected[expected_valid] = replayed
        if not np.array_equal(output, expected, equal_nan=True):
            raise PanelV2Error("optical index values do not replay from their source reflectances")
    return {
        "status": "SCL_AND_INDEX_DEPENDENCIES_VERIFIED",
        "scl_class_counts": {str(code): int(np.count_nonzero(scl == code))
                             for code in range(12)},
        "qa_clear_pixels": int(qa.sum()),
        "scl_4_5_6_pixels": int(clear_class.sum()),
        "qa_and_scl_clear_pixels": int(allowed.sum()),
        "observed_valid_counts_by_channel": [int(x.sum()) for x in observed],
    }


def _verify_sar_auxiliaries(arrays: Mapping[str, np.ndarray]) -> dict:
    if not {"signed_values", "signed_valid"} <= set(arrays):
        raise PanelV2Error("SAR source NPZ must retain signed_values and signed_valid")
    signed = np.asarray(arrays["signed_values"])
    signed_valid = np.asarray(arrays["signed_valid"])
    if signed.shape != (2, 499, 499) or signed_valid.shape != signed.shape:
        raise PanelV2Error("signed SAR values and validity must have shape [2,499,499]")
    _validate_masked_values(signed, signed_valid, "signed SAR values")
    primary = np.asarray(arrays["values"])
    primary_valid = np.asarray(arrays["observed_valid"])
    if primary.shape != (2, 499, 499) or primary_valid.shape != primary.shape:
        raise PanelV2Error("primary SAR sigma0 arrays must have shape [2,499,499]")
    _validate_masked_values(primary, primary_valid, "primary raw sigma0 values")
    db_names = {"raw_db", "raw_db_valid", "signed_db", "signed_db_valid"}
    present_db_names = db_names & set(arrays)
    db_inventory = []
    if present_db_names and present_db_names != db_names:
        raise PanelV2Error("SAR dB auxiliaries are partial; all four bound arrays are required")
    if present_db_names:
        for source_name, source_valid_name, db_name, db_valid_name in (
                ("values", "observed_valid", "raw_db", "raw_db_valid"),
                ("signed_values", "signed_valid", "signed_db", "signed_db_valid")):
            source = np.asarray(arrays[source_name], dtype=np.float64)
            source_valid = np.asarray(arrays[source_valid_name], dtype=bool)
            db = np.asarray(arrays[db_name])
            db_valid = np.asarray(arrays[db_valid_name])
            if (db.dtype != np.dtype(np.float32) or db.shape != source.shape or
                    db_valid.dtype != np.dtype(bool) or db_valid.shape != source.shape):
                raise PanelV2Error(f"{db_name} and its validity must be float32/bool [2,499,499]")
            expected_valid = source_valid & np.isfinite(source) & (source > 0)
            if not np.array_equal(db_valid, expected_valid):
                raise PanelV2Error(f"{db_valid_name} does not equal strictly-positive source support")
            expected_db = np.full(source.shape, np.nan, dtype=np.float32)
            expected_db[expected_valid] = (10.0 * np.log10(source[expected_valid])).astype(np.float32)
            if not np.array_equal(db, expected_db, equal_nan=True):
                raise PanelV2Error(f"{db_name} does not replay from strictly-positive linear power")
            db_inventory.append({"key": db_name, "valid_key": db_valid_name,
                "finite_count": int(db_valid.sum()), "array_sha256": _array_digest(db),
                "valid_array_sha256": _array_digest(db_valid),
                "source": source_name, "source_support": source_valid_name,
                "conversion": "10*log10(linear_power) only for finite strictly-positive support"})
    nesz = arrays.get("nesz_values")
    nesz_status = "UNAVAILABLE_IN_SOURCE_NPZ"
    if nesz is not None:
        nesz = np.asarray(nesz)
        if nesz.shape != (2, 499, 499) or nesz.dtype != np.dtype(np.float32):
            raise PanelV2Error("nesz_values must be float32 [2,499,499] when present")
        nesz_status = "PRESENT_SOURCE_BOUND_NOT_USED_AS_PRIMARY_CHANNEL"
    return {
        "status": "SIGNED_SOURCE_ARRAYS_RETAINED",
        "signed_valid_counts_by_polarization": [int(x.sum()) for x in signed_valid],
        "positive_primary_linear_counts_by_polarization": [
            int(np.count_nonzero(arrays["observed_valid"][i] & (arrays["values"][i] > 0)))
            for i in range(2)
        ],
        "d_b_arrays": db_inventory,
        "d_b_status": "VERIFIED_POSITIVE_SUPPORT_DB_REPLAY" if db_inventory
                     else "UNAVAILABLE_IN_SOURCE_NPZ",
        "nesz_status": nesz_status,
        "primary_channel_semantics": "RAW_SIGMA0_LINEAR_NO_DENoising_CLAIM",
    }


@dataclass(frozen=True)
class PanelV2:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    domain_protocol_sha256: str
    document: dict

    @property
    def observations(self) -> tuple[dict, ...]:
        return tuple(self.document["observations"])

    def load_arrays(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Load one source NPZ after rechecking the exact source-file digest."""
        path = verified_file(self.root, observation["values_file"], observation["sha256"])
        if path.stat().st_size != observation["values_bytes"]:
            raise PanelV2Error(f"source values byte count changed: {observation['values_file']}")
        before = file_hash(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        after = file_hash(path)
        if before != after or after != observation["sha256"]:
            raise PanelV2Error(f"source values changed while reading: {observation['values_file']}")
        return arrays


def load_panel_v2(root: str | Path, observations_path: str | Path) -> PanelV2:
    """Validate source manifest, NPZ arrays, receipts, timestamps and QA."""
    root = Path(root).resolve(strict=True)
    protocol_path = _relative_file(root, "configs/domain_2023_protocol.json")
    protocol_sha = file_hash(protocol_path)
    protocol = _json_object(protocol_path, "2023 domain protocol")
    if (type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 2 or
            protocol.get("protocol_id") != "SENTINEL_GL_RETROSPECTIVE_2023_V2"):
        raise PanelV2Error("2023 domain protocol version or scientific identifier is not recognized")
    cutoff_text = protocol.get("fit_end_exclusive")
    if cutoff_text != protocol.get("pre_event_cutoff"):
        raise PanelV2Error("2023 fit and pre-event cutoff bindings disagree")
    fit_end_exclusive = parse_explicit_utc(cutoff_text)
    if (protocol.get("fit_parity") != 0 or protocol.get("monitor_parity") != 1 or
            protocol.get("reconstruction_folds") != 4 or protocol.get("fusion_alpha") != 0.5 or
            protocol.get("scoring_context") != "CAUSAL_ACQUISITION_PREFIX"):
        raise PanelV2Error("2023 domain protocol conflicts with the v2 fitting/scoring contract")
    manifest_path = _manifest_path(root, observations_path)
    manifest_sha = file_hash(manifest_path)
    manifest = _json_object(manifest_path, "observations.v2 manifest")
    if manifest.get("schema_version") != PANEL_V2_VERSION:
        raise PanelV2Error("observations manifest schema_version must be integer 2")
    if type(manifest.get("schema_version")) is not int:
        raise PanelV2Error("observations manifest schema_version must be integer 2")
    rows = manifest.get("observations")
    if type(rows) is not list or not rows:
        raise PanelV2Error("observations.v2 manifest must contain a nonempty observations list")
    ids: set[str] = set()
    source_keys: set[tuple[str, str, str, str]] = set()
    normalized: list[dict] = []
    for row_index, source in enumerate(rows):
        if type(source) is not dict:
            raise PanelV2Error(f"observation {row_index} must be an object")
        required = {"observation_id", "site_id", "sensor", "acquired_at", "product_id",
                    "values_file", "sha256", "channels", "grid", "receipt_file"}
        if not required <= set(source):
            raise PanelV2Error(f"observation {row_index} lacks required v2 fields")
        obs_id = source["observation_id"]
        site_id = source["site_id"]
        product_id = source["product_id"]
        sensor = source["sensor"]
        if any(not isinstance(x, str) or not x.strip() for x in (obs_id, site_id, product_id)):
            raise PanelV2Error("observation_id, site_id, and product_id must be nonempty strings")
        if obs_id in ids:
            raise PanelV2Error(f"duplicate observation_id: {obs_id}")
        ids.add(obs_id)
        if sensor not in CHANNELS_BY_SENSOR:
            raise PanelV2Error(f"unsupported sensor identifier: {sensor}")
        channels = source["channels"]
        if type(channels) is not list or tuple(channels) != CHANNELS_BY_SENSOR[sensor]:
            raise PanelV2Error(f"sensor channel order/membership is invalid for {sensor}")
        acquired = parse_explicit_utc(source["acquired_at"])
        normalized_at = acquired.isoformat(timespec="microseconds").replace("+00:00", "Z")
        source_key = (site_id, sensor, product_id, normalized_at)
        if source_key in source_keys:
            raise PanelV2Error("duplicate site/sensor/product/acquisition support is not allowed")
        source_keys.add(source_key)
        grid = source["grid"]
        if type(grid) is not dict or grid.get("shape") != [499, 499]:
            raise PanelV2Error("each observations.v2 grid must declare shape [499,499]")
        if not isinstance(grid.get("crs"), str) or not grid["crs"]:
            raise PanelV2Error("each observations.v2 grid must retain a CRS identifier")
        transform = grid.get("transform")
        if (type(transform) is not list or len(transform) != 6 or
                any(type(x) not in (int, float) or not math.isfinite(float(x)) for x in transform)):
            raise PanelV2Error("each observations.v2 grid must retain six finite affine coefficients")
        values_path = _relative_file(root, source["values_file"])
        expected_sha = source["sha256"]
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
                any(ch not in "0123456789abcdef" for ch in expected_sha)):
            raise PanelV2Error("source values SHA-256 must be 64 lowercase hexadecimal characters")
        verified_file(root, source["values_file"], expected_sha)
        values_bytes = values_path.stat().st_size
        receipt_path = _relative_file(root, source["receipt_file"])
        receipt_sha = file_hash(receipt_path)
        receipt = _json_object(receipt_path, "observation source receipt")
        for field, expected in (("observation_id", obs_id), ("site_id", site_id),
                                ("product_id", product_id), ("values_file", source["values_file"]),
                                ("sha256", expected_sha)):
            if receipt.get(field) != expected:
                raise PanelV2Error(f"source receipt {field} differs from observation manifest")
        receipt_acquired = parse_explicit_utc(receipt.get("acquired_at", ""))
        if receipt_acquired != acquired:
            raise PanelV2Error("source receipt acquisition time differs from observation manifest")
        if sensor == "SENTINEL_1_IW_GRD":
            end_value = source.get("acquisition_end_at", receipt.get("end_at"))
            if source.get("acquisition_end_at") is not None and receipt.get("end_at") != source["acquisition_end_at"]:
                raise PanelV2Error("manifest and signed SAR receipt acquisition end times differ")
            acquisition_end = parse_explicit_utc(end_value) if end_value is not None else None
            end_status = "EXPLICIT_UTC_SOURCE_RECEIPT" if acquisition_end is not None else "UNAVAILABLE_SOURCE_END_TIME"
        else:
            end_value = source.get("acquisition_end_at", receipt.get("end_at", source["acquired_at"]))
            acquisition_end = parse_explicit_utc(end_value)
            end_status = ("EXPLICIT_UTC_SOURCE_RECEIPT" if receipt.get("end_at") is not None or
                          source.get("acquisition_end_at") is not None else "POINT_SENSING_TIME_NO_INTERVAL")
        if acquisition_end is not None and acquisition_end < acquired:
            raise PanelV2Error("source acquisition end precedes its start")
        receipt_strings = set(_walk_strings(receipt))
        if expected_sha not in receipt_strings:
            raise PanelV2Error("source receipt does not bind the observation NPZ SHA-256")
        if product_id not in receipt_strings:
            raise PanelV2Error("source receipt does not bind the declared provider product_id")
        try:
            with np.load(values_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
        except (OSError, ValueError, KeyError) as exc:
            raise PanelV2Error(f"observation NPZ could not be read without pickle: {source['values_file']}") from exc
        if file_hash(values_path) != expected_sha:
            raise PanelV2Error("observation NPZ changed while it was being inspected")
        if not {"values", "observed_valid"} <= set(arrays):
            raise PanelV2Error("observation NPZ must retain values and observed_valid arrays")
        values = np.asarray(arrays["values"])
        observed = np.asarray(arrays["observed_valid"])
        expected_shape = (len(channels), 499, 499)
        if values.shape != expected_shape or observed.shape != expected_shape:
            raise PanelV2Error("values and observed_valid must match sensor channels and [499,499]")
        _validate_masked_values(values, observed, "primary observation values")
        if sensor == "SENTINEL_2_OPTICAL":
            qa_summary = _verify_optical_arrays(arrays, observed)
        else:
            qa_summary = _verify_sar_auxiliaries(arrays)
        arrays_inventory = []
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise PanelV2Error(f"object arrays are not permitted in source NPZ: {name}")
            arrays_inventory.append({"key": name, "shape": list(array.shape),
                "dtype": str(array.dtype), "array_sha256": _array_digest(array)})
        scope = scope_for_acquisition(acquired, acquisition_end, fit_end_exclusive)
        normalized.append({
            "observation_id": obs_id, "site_id": site_id, "sensor": sensor,
            "acquired_at": normalized_at, "acquired_at_source": source["acquired_at"],
            "acquisition_end_at": (acquisition_end.isoformat(timespec="microseconds").replace("+00:00", "Z")
                                   if acquisition_end is not None else None),
            "acquisition_end_status": end_status,
            "product_id": product_id, "values_file": source["values_file"],
            "values_bytes": values_bytes, "sha256": expected_sha,
            "receipt_file": source["receipt_file"], "receipt_bytes": receipt_path.stat().st_size,
            "receipt_sha256": receipt_sha, "channels": list(channels), "grid": grid,
            "grid_sha256": _sha_json(grid), "scope": scope,
            "observed_valid_counts_by_channel": [int(mask.sum()) for mask in observed],
            "qa_summary": qa_summary, "arrays": arrays_inventory,
            "array_count": len(arrays),
            "origin": "SOURCE_BOUND_INPUT_BYTES_UNDER_RESEARCH_ADMISSION_REVIEW",
        })
    normalized.sort(key=lambda x: (x["site_id"], x["acquired_at"], x["sensor"], x["product_id"]))
    doc = {
        "schema_version": PANEL_V2_VERSION,
        "panel_version": "SOURCE_BOUND_DYNAMIC_2023_PANEL_V2",
        "status": "PASS_SOURCE_BYTES_MASKS_QA_AND_TEMPORAL_BINDINGS",
        "source_manifest": {"path": manifest_path.relative_to(root).as_posix(),
            "bytes": manifest_path.stat().st_size, "sha256": manifest_sha,
            "schema_version": PANEL_V2_VERSION},
        "domain_protocol": {"path": protocol_path.relative_to(root).as_posix(),
            "sha256": protocol_sha, "protocol_id": protocol["protocol_id"],
            "fit_end_exclusive": fit_end_exclusive.isoformat().replace("+00:00", "Z"),
            "publication_authorized": protocol.get("public_publication_authorized") is True},
        "observation_count": len(normalized),
        "site_ids": sorted({x["site_id"] for x in normalized}),
        "sensor_counts": {sensor: sum(x["sensor"] == sensor for x in normalized)
                          for sensor in CHANNELS_BY_SENSOR},
        "observations": normalized,
        "scope_protocol": {
            "fit": "All source acquisitions whose verified end time is strictly earlier than fit_end_exclusive, parity-0 pixels only",
            "fit_end_exclusive_utc": fit_end_exclusive.isoformat().replace("+00:00", "Z"),
            "development_monitor": "parity-1 pixels from the same pre-breach fit acquisitions; explicitly same-scene dependent",
            "calibration": "NO_INDEPENDENT_THRESHOLD_CALIBRATION_SCOPE; closed opportunity schedule and independent support are unavailable",
            "evaluation_trajectory": "Acquisitions at or after the fit cutoff remain retrospective causal-prefix trajectories; no fitted state uses them",
            "event_phase": "time-window descriptors only; 2023-10-03 through 2023-10-04 remains onset-uncertain; no binary labels are assigned",
            "interpretation": "RETROSPECTIVE_DEVELOPMENT_EXPOSED; all pre-breach acquisitions inform fitted states, while each query context remains causal",
            "state_freeze": "normalizer, encoder, embedding reference, and Score C fit standardizers use pre-breach fit support only",
        },
        "graph_status": {
            "physical_sampling_G_S": "UNRESOLVED_UNTIL_VERIFIED_GEOMETRY_AND_ADJACENCY",
            "fitting_provenance_G_P": "SOURCE_PRODUCTS_AND_FITTED_STATES_RECORDED_SEPARATELY",
            "shared_weights_do_not_merge_physical_units": True,
            "seeds_are_not_independent_supports": True,
            "independent_physical_support_count": None,
            "independent_physical_support_status": "UNKNOWN",
        },
        "limits": [
            "The panel binds source bytes and QA masks; it does not establish provider rights, historical availability, geographic independence, or scientific admission.",
            "Shared CRS/shape does not establish SAR-optical pixel registration or merge physical sampling units.",
            "Quarantine, masking and temporal roles are enforced by the v2 loader; no fallback values or binary event/control labels are generated.",
        ],
    }
    return PanelV2(root=root, manifest_path=manifest_path, manifest_sha256=manifest_sha,
                   domain_protocol_sha256=protocol_sha, document=doc)


def write_panel_v2(panel: PanelV2, path: str | Path) -> dict:
    target = Path(path)
    if not target.is_absolute():
        target = panel.root / target
    if not target.resolve(strict=False).is_relative_to(panel.root):
        raise PanelV2Error("panel.v2 output must stay inside the workspace")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing panel.v2 artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(panel.document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(payload, encoding="utf-8")
    return {"path": target.relative_to(panel.root).as_posix(),
            "bytes": target.stat().st_size, "sha256": file_hash(target),
            "source_manifest_sha256": panel.manifest_sha256}


def fit_channel_standardizer(panel: PanelV2) -> dict:
    """Fit per-channel float64 population moments on pre-breach parity 0."""
    counts = np.zeros(len(CANONICAL_CHANNELS), dtype=np.int64)
    running_means = np.zeros(len(CANONICAL_CHANNELS), dtype=np.float64)
    running_m2 = np.zeros(len(CANONICAL_CHANNELS), dtype=np.float64)
    fit_ids = []
    fit_sources = []
    channel_index = {name: i for i, name in enumerate(CANONICAL_CHANNELS)}
    for record in panel.observations:
        if record["scope"]["role"] != "FIT_SUPPORT_WITH_PARITY1_DEVELOPMENT_MONITOR":
            continue
        arrays = panel.load_arrays(record)
        values = arrays["values"]
        valid = arrays["observed_valid"]
        fit_area = checkerboard_parity_mask(499, 499, 0)
        for local_index, channel in enumerate(record["channels"]):
            idx = channel_index[channel]
            mask = valid[local_index] & fit_area
            row = values[local_index][mask].astype(np.float64)
            if row.size:
                batch_count = int(row.size)
                batch_mean = float(row.mean(dtype=np.float64))
                batch_m2 = float(np.square(row - batch_mean).sum(dtype=np.float64))
                prior_count = int(counts[idx])
                combined_count = prior_count + batch_count
                delta = batch_mean - running_means[idx]
                running_means[idx] += delta * batch_count / combined_count
                running_m2[idx] += batch_m2 + delta * delta * prior_count * batch_count / combined_count
                counts[idx] = combined_count
        fit_ids.append(record["observation_id"])
        fit_sources.append({"observation_id": record["observation_id"],
                            "product_id": record["product_id"],
                            "sha256": record["sha256"]})
    if not fit_ids:
        raise PanelV2Error("no source observations end strictly before the declared pre-breach fit cutoff")
    means: list[float | None] = []
    scales: list[float | None] = []
    statuses = []
    for count, mean, m2 in zip(counts, running_means, running_m2):
        if count == 0:
            means.append(None); scales.append(None); statuses.append("UNAVAILABLE_NO_FIT_SUPPORT")
            continue
        variance = m2 / int(count)
        if variance <= 0 or not math.isfinite(variance):
            means.append(float(mean)); scales.append(None); statuses.append("UNAVAILABLE_ZERO_OR_INVALID_FIT_VARIANCE")
        else:
            means.append(float(mean)); scales.append(float(math.sqrt(variance)))
            statuses.append("ESTIMATED_FIT_SUPPORT_ONLY")
    payload = {
        "version": "PANEL_V2_PARITY0_FIT_ONLY_WELFORD_V1",
        "algorithm": "float64 population Welford-equivalent streaming moments; no target fallback",
        "channel_order": list(CANONICAL_CHANNELS),
        "counts": counts.tolist(), "means": means, "scales": scales,
        "channel_statuses": statuses, "fit_role": "FIT",
        "fit_end_exclusive_utc": panel.document["domain_protocol"]["fit_end_exclusive"],
        "fit_scope": "ALL_PRE_BREACH_ACQUISITIONS_ENDING_STRICTLY_BEFORE_CUTOFF",
        "fit_parity": 0, "fit_observation_ids": fit_ids,
        "fit_observation_ids_sha256": _sha_json(fit_ids),
        "fit_source_hashes": fit_sources,
        "source_manifest_sha256": panel.manifest_sha256,
        "monitor_support_used": False, "calibration_or_evaluation_used": False,
    }
    payload["state_sha256"] = _sha_json(payload)
    return payload


def pool_observation_quadrants(panel: PanelV2, observation: Mapping[str, Any],
                               standardizer: Mapping[str, Any], parity: int | None) -> dict:
    """Pool only valid pixels after SCL/index checks, retaining channel counts."""
    if parity is not None and (type(parity) is not int or parity not in (0, 1)):
        raise ValueError("pooling parity must be 0, 1, or None for a time-disjoint query")
    if standardizer.get("channel_order") != list(CANONICAL_CHANNELS):
        raise ValueError("standardizer channel order does not match panel.v2")
    expected_hash = _sha_json({k: v for k, v in standardizer.items() if k != "state_sha256"})
    if standardizer.get("state_sha256") != expected_hash:
        raise ValueError("fit-only standardizer hash mismatch")
    arrays = panel.load_arrays(observation)
    values = arrays["values"]
    valid = arrays["observed_valid"]
    spatial = np.ones((499, 499), dtype=bool) if parity is None else checkerboard_parity_mask(499, 499, parity)
    quadrants = quadrant_masks(499, 499)
    channel_index = {name: i for i, name in enumerate(CANONICAL_CHANNELS)}
    tokens = []
    for q_index, qmask in enumerate(quadrants):
        area = spatial & qmask
        token_values = np.zeros(len(CANONICAL_CHANNELS), dtype=np.float32)
        token_observed = np.zeros(len(CANONICAL_CHANNELS), dtype=bool)
        denominators = [None] * len(CANONICAL_CHANNELS)
        for local_index, channel in enumerate(observation["channels"]):
            idx = channel_index[channel]
            mean = standardizer["means"][idx]
            scale = standardizer["scales"][idx]
            support = valid[local_index] & area
            count = int(support.sum())
            denominators[idx] = count
            if count == 0 or mean is None or scale is None:
                continue
            normalized = ((values[local_index][support].astype(np.float64) - mean) / scale)
            token_values[idx] = np.float32(normalized.mean(dtype=np.float64))
            token_observed[idx] = True
        tokens.append({"quadrant": q_index, "values": token_values,
            "observed": token_observed, "source_pixel_denominators": denominators,
            "status": "OBSERVED" if token_observed.any() else "UNAVAILABLE_NO_TRANSFORMED_SOURCE_SUPPORT"})
    return {"observation_id": observation["observation_id"],
        "site_id": observation["site_id"], "sensor": observation["sensor"],
        "acquired_at": observation["acquired_at"], "product_id": observation["product_id"],
        "source_sha256": observation["sha256"], "parity": parity, "tokens": tokens,
        "grid_sha256": observation["grid_sha256"],
        "spatial_registration_status": "GRID_QUADRANT_CONTEXT_ONLY_PHYSICAL_REGISTRATION_UNVERIFIED"}


def causal_prefix(observations: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
                  query: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return same-site observations no later than the query; never add future rows."""
    query_at = parse_explicit_utc(query["acquired_at"])
    eligible = [row for row in observations
                if row["site_id"] == query["site_id"] and
                parse_explicit_utc(row["acquired_at"]) <= query_at]
    eligible.sort(key=lambda row: (row["acquired_at"], row["sensor"], row["product_id"],
                                   row["observation_id"]))
    if not any(row["observation_id"] == query["observation_id"] for row in eligible):
        raise ValueError("causal prefix does not contain its query observation")
    return tuple(eligible)

