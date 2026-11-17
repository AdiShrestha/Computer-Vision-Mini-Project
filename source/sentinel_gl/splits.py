"""Spatiotemporal split engine, geographic clustering, and availability scheduler.

Enforces zero future-data leakage into retrospective decision windows,
strict geographic cluster isolation (>=50 km buffer) preventing spatial
autocorrelation between basins, context boundary purging across split
horizons, and pre-event quarantining for GLOF target lakes.
"""
from __future__ import annotations
import dataclasses
import datetime
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

SCOPE_ROLES = ("fitting", "development_validation", "threshold_calibration", "evaluation")


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute Great Circle distance between two WGS-84 coordinates in kilometers.

    Uses mean Earth radius R = 6371.0088 km.
    """
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius * c


def parse_utc_timestamp(ts: str) -> datetime.datetime:
    """Parse ISO 8601 timestamp string into timezone-aware UTC datetime."""
    clean_ts = ts.strip()
    if clean_ts.endswith("Z"):
        clean_ts = clean_ts[:-1] + "+00:00"
    dt = datetime.datetime.fromisoformat(clean_ts)
    if dt.tzinfo is None:
        raise ValueError("timestamp timezone is required")
    return dt.astimezone(datetime.timezone.utc)


def cluster_lakes_by_distance(
    lakes: Sequence[Mapping[str, Any]],
    buffer_km: float = 50.0,
) -> Dict[str, List[str]]:
    """Partition lakes into disjoint clusters based on geographic proximity.

    Two lakes belong to the same cluster if they are separated by less than
    buffer_km or if they share an intermediate lake within buffer_km
    (connected component single-linkage clustering).
    """
    if buffer_km < 0:
        raise ValueError(f"buffer_km must be non-negative, got {buffer_km}")

    lake_ids = [str(l["lake_id"]) for l in lakes]
    coords = {
        str(l["lake_id"]): (float(l["centroid_lat"]), float(l["centroid_lon"]))
        for l in lakes
    }

    # Adjacency graph
    adjacency: Dict[str, Set[str]] = {lid: set() for lid in lake_ids}
    n = len(lake_ids)
    for i in range(n):
        id_i = lake_ids[i]
        lat_i, lon_i = coords[id_i]
        for j in range(i + 1, n):
            id_j = lake_ids[j]
            lat_j, lon_j = coords[id_j]
            dist = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            if dist < buffer_km:
                adjacency[id_i].add(id_j)
                adjacency[id_j].add(id_i)

    # Connected components
    visited: Set[str] = set()
    clusters: Dict[str, List[str]] = {}
    cluster_idx = 1

    for lid in sorted(lake_ids):
        if lid not in visited:
            component: List[str] = []
            queue = [lid]
            visited.add(lid)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in sorted(adjacency[curr]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            cluster_name = f"CLS-GEO-{cluster_idx:02d}"
            clusters[cluster_name] = sorted(component)
            cluster_idx += 1

    return clusters


def verify_spatial_disjointness(
    group_a_lakes: Iterable[str],
    group_b_lakes: Iterable[str],
    lake_coords: Mapping[str, Tuple[float, float]],
    min_buffer_km: float = 50.0,
) -> None:
    """Verify that no lake in group A is within min_buffer_km of any lake in group B.

    Raises ValueError if spatial proximity leakage is detected.
    """
    for id_a in group_a_lakes:
        if id_a not in lake_coords:
            raise KeyError(f"Missing coordinates for lake {id_a}")
        lat_a, lon_a = lake_coords[id_a]
        for id_b in group_b_lakes:
            if id_b not in lake_coords:
                raise KeyError(f"Missing coordinates for lake {id_b}")
            lat_b, lon_b = lake_coords[id_b]
            dist = haversine_distance(lat_a, lon_a, lat_b, lon_b)
            if dist < min_buffer_km:
                raise ValueError(
                    f"Spatial leakage detected: lake {id_a} and lake {id_b} are "
                    f"separated by {dist:.2f} km, which is less than the required "
                    f"buffer of {min_buffer_km:.1f} km."
                )


@dataclasses.dataclass(frozen=True)
class DecisionWindow:
    """An operational retrospective decision window at a scheduled evaluation time."""
    window_id: str
    lake_id: str
    decision_timestamp: str  # ISO-8601 UTC
    context_start: str       # ISO-8601 UTC
    context_end: str         # ISO-8601 UTC
    observation_ids: Tuple[str, ...]
    is_eligible: bool
    modalities_present: Tuple[str, ...]
    scope: str
    missingness_reason: Optional[str] = None
    status: str = "ELIGIBLE"
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        decision = parse_utc_timestamp(self.decision_timestamp)
        start = parse_utc_timestamp(self.context_start)
        end = parse_utc_timestamp(self.context_end)
        if start >= decision or end < start or end >= decision:
            raise ValueError("schedule support must be ordered and closed before its decision")
        if not self.window_id or not self.lake_id or not isinstance(self.metadata, Mapping):
            raise ValueError("schedule identity and metadata must be explicit")
        if self.scope not in SCOPE_ROLES:
            raise ValueError("decision schedule row requires one explicit fitting role")
        if self.status not in {"ELIGIBLE", "NOT_ESTIMABLE", "NOT_ASSESSED"}:
            raise ValueError("unknown decision schedule status")
        if type(self.is_eligible) is not bool or self.is_eligible != (self.status == "ELIGIBLE"):
            raise ValueError("eligibility boolean and schedule status conflict")
        if (self.status == "ELIGIBLE") != (self.missingness_reason is None):
            raise ValueError("eligibility and missingness reason conflict")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "lake_id": self.lake_id,
            "decision_timestamp": self.decision_timestamp,
            "context_start": self.context_start,
            "context_end": self.context_end,
            "observation_ids": list(self.observation_ids),
            "is_eligible": self.is_eligible,
            "status": self.status,
            "scope": self.scope,
            "modalities_present": list(self.modalities_present),
            "missingness_reason": self.missingness_reason,
            "metadata": self.metadata,
        }


def verify_retrospective_invariants(
    window: DecisionWindow,
    observations: Sequence[Mapping[str, Any]],
) -> None:
    """Verify every joined source has closed support and verified availability by decision."""
    t_decision = parse_utc_timestamp(window.decision_timestamp)
    t_context_start = parse_utc_timestamp(window.context_start)
    obs_by_id = {str(o.get("record_id")): o for o in observations}
    if len(obs_by_id) != len(observations):
        raise ValueError("duplicate source records in retrospective invariant check")
    if set(window.observation_ids) - set(obs_by_id):
        raise ValueError("window references a missing source record")

    for obs_id in window.observation_ids:
        obs = obs_by_id[obs_id]
        try:
            support_start = parse_utc_timestamp(obs["support_start"])
            support_end = parse_utc_timestamp(obs["support_end"])
        except KeyError as exc:
            raise ValueError("source rows require closed support timestamps") from exc
        availability = _availability_record(obs)
        if support_end < support_start:
            raise ValueError("source support interval is reversed")
        if support_end >= t_decision:
            raise ValueError(f"Future or open source support in window '{window.window_id}'")
        if availability["status"] not in {"PROVIDER_EVIDENCE", "DECLARED_APPROXIMATION"}:
            raise ValueError(f"Source availability is unresolved in window '{window.window_id}'")
        available = parse_utc_timestamp(availability["at"])
        if available > t_decision:
            raise ValueError(f"Late-available source leakage in window '{window.window_id}'")
        if support_start < t_context_start:
            raise ValueError(f"Temporal boundary violation in window '{window.window_id}'")


def _availability_record(observation):
    """Require explicit availability provenance; retrieval and generation are separate axes."""
    record = observation.get("availability")
    if not isinstance(record, Mapping) or set(record) != {
            "status", "at", "evidence_id", "evidence_sha256", "reason"}:
        raise ValueError("source rows require an explicit versioned availability record")
    status = record["status"]
    if status not in {"PROVIDER_EVIDENCE", "DECLARED_APPROXIMATION", "UNVERIFIED"}:
        raise ValueError("unknown source availability status")
    if status == "UNVERIFIED":
        if record["at"] is not None or record["evidence_id"] is not None or record["evidence_sha256"] is not None:
            raise ValueError("unverified availability cannot carry a time or evidence binding")
        if not isinstance(record["reason"], str) or not record["reason"].strip():
            raise ValueError("unverified availability requires an explicit reason")
    else:
        if not isinstance(record["at"], str) or not isinstance(record["evidence_id"], str) or not record["evidence_id"].strip():
            raise ValueError("supported availability requires a time and evidence identity")
        if not isinstance(record["evidence_sha256"], str) or len(record["evidence_sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in record["evidence_sha256"]):
            raise ValueError("supported availability requires an exact SHA-256 evidence binding")
        parse_utc_timestamp(record["at"])
        if record["reason"] is not None:
            raise ValueError("supported availability cannot carry an absence reason")
    axes = observation.get("time_axes")
    required_axes = {"acquisition", "product_generation", "publication", "retrieval",
                     "historical_availability", "execution"}
    if not isinstance(axes, Mapping) or set(axes) != required_axes:
        raise ValueError("source rows require separate acquisition/generation/publication/retrieval/history/execution axes")
    for name, axis in axes.items():
        if not isinstance(axis, Mapping) or set(axis) != {"status", "at", "reason"}:
            raise ValueError("source time axes require explicit status, value and reason fields")
        if axis["status"] == "VERIFIED":
            if not isinstance(axis["at"], str) or axis["reason"] is not None:
                raise ValueError("verified source time requires a timestamp and no absence reason")
            parse_utc_timestamp(axis["at"])
        elif axis["status"] in {"UNKNOWN", "NOT_APPLICABLE"}:
            if axis["at"] is not None or not isinstance(axis["reason"], str) or not axis["reason"].strip():
                raise ValueError("unknown/not-applicable source time requires null and a reason")
        else:
            raise ValueError("unknown source time-axis status")
    if axes["product_generation"]["status"] == "VERIFIED" and status != "UNVERIFIED":
        if parse_utc_timestamp(record["at"]) < parse_utc_timestamp(axes["product_generation"]["at"]):
            raise ValueError("availability cannot precede generation of the exact product version")
    return record


class AvailabilityScheduler:
    """Build complete candidate rows from closed support and declared availability."""

    def __init__(
        self,
        window_days: int,
        stride_days: int,
        minimum_observations_by_modality: Optional[Mapping[str, int]] = None,
    ):
        if type(window_days) is not int or window_days <= 0 or type(stride_days) is not int or stride_days <= 0:
            raise ValueError("window_days and stride_days must be explicit positive integers")
        if minimum_observations_by_modality is not None:
            if (not isinstance(minimum_observations_by_modality, Mapping) or not minimum_observations_by_modality or
                    any(not isinstance(k, str) or not k or type(v) is not int or v < 1
                        for k, v in minimum_observations_by_modality.items())):
                raise ValueError("support rule must map modality names to explicit positive counts")
        self.window_days = window_days
        self.stride_days = stride_days
        self.minimum_observations_by_modality = (dict(minimum_observations_by_modality)
                                                 if minimum_observations_by_modality is not None else None)

    def filter_retrospective_observations(
        self,
        observations: Sequence[Mapping[str, Any]],
        decision_timestamp: str,
        event_cutoff: Optional[str] = None,
    ) -> List[Mapping[str, Any]]:
        """Filter by full closed support and separately recorded availability.

        Acquisition timestamps alone cannot establish whether a product was
        available for a past decision. Every record must supply support_start,
        support_end, and available_at with explicit time zones.
        """
        t_decision = parse_utc_timestamp(decision_timestamp)
        t_window_start = t_decision - datetime.timedelta(days=self.window_days)
        t_cutoff = parse_utc_timestamp(event_cutoff) if event_cutoff else None

        valid_records: List[Mapping[str, Any]] = []
        excluded = []
        seen_ids = set()
        for obs in observations:
            record_id = obs.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise ValueError("source record IDs must be present and unique")
            seen_ids.add(record_id)
            try:
                t_start = parse_utc_timestamp(obs["support_start"])
                t_end = parse_utc_timestamp(obs["support_end"])
                modality = obs["modality"]
            except KeyError as exc:
                raise ValueError("source rows require support_start, support_end, time_axes, availability, and modality") from exc
            availability = _availability_record(obs)
            if not isinstance(modality, str) or not modality or t_end < t_start:
                raise ValueError("source support identity or interval is invalid")
            if t_end >= t_decision:
                excluded.append({"record_id": record_id, "reason": "SUPPORT_NOT_CLOSED_BEFORE_DECISION"})
                continue
            if t_start < t_window_start:
                excluded.append({"record_id": record_id, "reason": "SUPPORT_OUTSIDE_DECLARED_CONTEXT"})
                continue
            if t_cutoff is not None and t_end >= t_cutoff:
                excluded.append({"record_id": record_id, "reason": "SUPPORT_AT_OR_AFTER_EXCLUSIVE_EVENT_CUTOFF"})
                continue
            if availability["status"] == "UNVERIFIED":
                excluded.append({"record_id": record_id, "reason": "HISTORICAL_AVAILABILITY_UNVERIFIED"})
                continue
            available = parse_utc_timestamp(availability["at"])
            if available < t_end:
                raise ValueError("declared availability precedes closed measurement support")
            if available > t_decision:
                excluded.append({"record_id": record_id, "reason": "LATE_AVAILABLE"})
                continue
            valid_records.append(obs)
        self._last_exclusions = sorted(excluded, key=lambda row: row["record_id"])
        return sorted(valid_records, key=lambda obs: str(obs["record_id"]))

    def generate_decision_schedule(
        self,
        lake_id: str,
        start_date: str,
        end_date: str,
        observations: Sequence[Mapping[str, Any]],
        required_modalities: Sequence[str],
        *,
        scope: str,
        event_cutoff: Optional[str] = None,
    ) -> List[DecisionWindow]:
        """Construct sequential decision windows on a fixed cadence for a single lake."""
        if scope not in SCOPE_ROLES:
            raise ValueError("schedule rows require one explicit fitting role")
        dt_start = parse_utc_timestamp(start_date)
        dt_end = parse_utc_timestamp(end_date)
        if dt_start >= dt_end:
            raise ValueError(f"start_date {start_date} must precede end_date {end_date}")
        if not required_modalities or len(set(required_modalities)) != len(required_modalities):
            raise ValueError("required modalities must be an explicit nonempty unique sequence")
        if self.minimum_observations_by_modality is not None and set(self.minimum_observations_by_modality) != set(required_modalities):
            raise ValueError("support rule must exactly match required modalities")

        windows: List[DecisionWindow] = []
        current_decision = dt_start

        while current_decision <= dt_end:
            decision_iso = current_decision.isoformat()
            context_start_dt = current_decision - datetime.timedelta(days=self.window_days)
            context_start_iso = context_start_dt.isoformat()

            # Filter valid observations
            eligible_obs = self.filter_retrospective_observations(
                observations=observations,
                decision_timestamp=decision_iso,
                event_cutoff=event_cutoff,
            )

            # Check eligibility across required modalities
            modality_counts: Dict[str, int] = {}
            for obs in eligible_obs:
                mod = obs.get("modality", "unknown")
                modality_counts[mod] = modality_counts.get(mod, 0) + 1

            modalities_present = tuple(sorted(modality_counts.keys()))
            if self.minimum_observations_by_modality is None:
                is_eligible = False
                missing_reason = "ELIGIBILITY_RULE_NOT_DECLARED"
                win_status = "NOT_ASSESSED"
            else:
                insufficient_modalities = [m for m in required_modalities
                    if modality_counts.get(m, 0) < self.minimum_observations_by_modality[m]]
                is_eligible = not insufficient_modalities
                missing_reason = ("INSUFFICIENT_OBSERVATIONS: missing " + str(insufficient_modalities)
                                  if insufficient_modalities else None)
                win_status = "ELIGIBLE" if is_eligible else "NOT_ESTIMABLE"

            # Compute earliest and latest observation times in window
            obs_support_ends = [parse_utc_timestamp(o["support_end"]) for o in eligible_obs]
            latest_obs_iso = max(obs_support_ends).isoformat() if obs_support_ends else context_start_iso

            win_id = f"WIN-{lake_id}-{current_decision.strftime('%Y%m%d')}"
            obs_hashes = {
                str(o["record_id"]): str(o["sha256"])
                for o in eligible_obs
                if "sha256" in o
            }
            win = DecisionWindow(
                window_id=win_id,
                lake_id=lake_id,
                decision_timestamp=decision_iso,
                context_start=context_start_iso,
                context_end=latest_obs_iso,
                observation_ids=tuple(o["record_id"] for o in eligible_obs),
                is_eligible=is_eligible,
                modalities_present=modalities_present,
                scope=scope,
                missingness_reason=missing_reason,
                status=win_status,
                metadata={
                    "total_observations": len(eligible_obs),
                    "modality_counts": modality_counts,
                    "observation_hashes": obs_hashes,
                    "source_supports": [{k: o[k] for k in ("record_id", "modality", "support_start",
                        "support_end", "availability", "time_axes") if k in o} for o in eligible_obs],
                    "source_exclusions": list(getattr(self, "_last_exclusions", [])),
                    "availability_rule": "EXPLICIT_EVIDENCE_OR_DECLARED_APPROXIMATION_AND_SUPPORT_CLOSED_BY_DECISION",
                },
            )
            verify_retrospective_invariants(win, eligible_obs)
            windows.append(win)
            current_decision += datetime.timedelta(days=self.stride_days)

        return windows


PHYSICAL_SUPPORT_FIELDS = ("glacier_ids", "basin_ids", "era5_cell_ids",
                           "sentinel_frame_ids", "event_time_support_ids")


def build_physical_sampling_graph(lakes, scope_by_lake, *, buffer_km=50.0,
                                  verified_support_bindings=None,
                                  verified_coordinate_bindings=None):
    """Build G_S from independently byte-verified physical support and proximity.

    The binding maps must be produced by an adapter that checked source bytes.
    A registry's self-declared VERIFIED string is never sufficient by itself.
    Fixture bindings can exercise graph arithmetic, but cannot certify research.
    """
    if not math.isclose(float(buffer_km), 50.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("physical sampling adjacency uses the declared 50 km threshold")
    rows = {str(row.get("lake_id")): row for row in lakes}
    if not rows or "" in rows or len(rows) != len(lakes) or set(rows) != set(scope_by_lake):
        raise ValueError("unique lake identities and exact role membership are required")
    if set(scope_by_lake.values()) - set(SCOPE_ROLES):
        raise ValueError("unknown fitting role")
    verified_support_bindings = verified_support_bindings or {}
    verified_coordinate_bindings = verified_coordinate_bindings or {}
    nodes, unresolved_units = {}, []
    for lake_id, row in rows.items():
        support = row.get("physical_support")
        binding = verified_support_bindings.get(lake_id)
        support_digest = __import__("hashlib").sha256(json.dumps(
            support, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() if isinstance(support, Mapping) else None
        bound_support = (isinstance(binding, Mapping) and
                         set(binding) == {"artifact_path", "artifact_sha256", "support_sha256", "origin"} and
                         isinstance(binding["artifact_path"], str) and binding["artifact_path"] and
                         isinstance(binding["artifact_sha256"], str) and len(binding["artifact_sha256"]) == 64 and
                         binding["support_sha256"] == support_digest and
                         binding["origin"] in {"observational", "fixture"})
        known = (bound_support and isinstance(support, Mapping) and
                 set(support) == set(PHYSICAL_SUPPORT_FIELDS) and row.get("physical_support_status") == "VERIFIED")
        normalized = {}
        if known:
            for field in PHYSICAL_SUPPORT_FIELDS:
                values = support[field]
                if (not isinstance(values, (list, tuple)) or
                        any(not isinstance(value, str) or not value.strip() for value in values) or
                        len(values) != len(set(values))):
                    raise ValueError("physical support IDs must be explicit unique strings")
                normalized[field] = tuple(sorted(values))
        else:
            unresolved_units.append(lake_id)
        coordinate_binding = verified_coordinate_bindings.get(lake_id)
        coordinate_digest = __import__("hashlib").sha256(json.dumps(
            [row.get("centroid_lat"), row.get("centroid_lon")], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        bound_coordinates = (isinstance(coordinate_binding, Mapping) and
            set(coordinate_binding) == {"artifact_path", "artifact_sha256", "coordinate_sha256", "origin"} and
            isinstance(coordinate_binding["artifact_path"], str) and coordinate_binding["artifact_path"] and
            isinstance(coordinate_binding["artifact_sha256"], str) and len(coordinate_binding["artifact_sha256"]) == 64 and
            coordinate_binding["coordinate_sha256"] == coordinate_digest and
            coordinate_binding["origin"] in {"observational", "fixture"})
        lat, lon = row.get("centroid_lat"), row.get("centroid_lon")
        coordinate_known = (type(lat) in (int, float) and type(lon) in (int, float) and
            math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            and bound_coordinates)
        if not coordinate_known:
            unresolved_units.append(lake_id)
        nodes[lake_id] = {"lake_id": lake_id, "scope": scope_by_lake[lake_id],
            "physical_support_status": ("VERIFIED" if known and binding["origin"] == "observational" else
                                        "FIXTURE_VERIFIED" if known else "UNKNOWN"),
            "physical_support_evidence": ({"artifact_path": binding["artifact_path"],
                "artifact_sha256": binding["artifact_sha256"]} if known else None),
            "coordinate_status": ("VERIFIED" if coordinate_known and coordinate_binding["origin"] == "observational" else
                                  "FIXTURE_VERIFIED" if coordinate_known else "UNKNOWN"),
            "coordinate_evidence": ({"artifact_path": coordinate_binding["artifact_path"],
                "artifact_sha256": coordinate_binding["artifact_sha256"]} if coordinate_known else None),
            "physical_support": {k: list(v) for k, v in normalized.items()} if known else None}

    edges, unresolved_pairs = [], []
    ids = sorted(rows)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            left, right = rows[left_id], rows[right_id]
            reasons = []
            if nodes[left_id]["physical_support_status"] in {"VERIFIED", "FIXTURE_VERIFIED"} and nodes[right_id]["physical_support_status"] in {"VERIFIED", "FIXTURE_VERIFIED"}:
                for field in PHYSICAL_SUPPORT_FIELDS:
                    shared = sorted(set(nodes[left_id]["physical_support"][field]) &
                                    set(nodes[right_id]["physical_support"][field]))
                    if shared:
                        reasons.append({"kind": "SHARED_" + field.upper(), "ids": shared})
            coords_known = (nodes[left_id]["coordinate_status"] in {"VERIFIED", "FIXTURE_VERIFIED"} and
                            nodes[right_id]["coordinate_status"] in {"VERIFIED", "FIXTURE_VERIFIED"})
            distance = None
            if coords_known:
                distance = haversine_distance(float(left["centroid_lat"]), float(left["centroid_lon"]),
                    float(right["centroid_lat"]), float(right["centroid_lon"]))
                if distance < 50.0:
                    reasons.append({"kind": "DISTANCE_LT_50_KM", "distance_km": distance})
            if reasons:
                edges.append({"left": left_id, "right": right_id, "reasons": reasons})
            if (nodes[left_id]["physical_support_status"] not in {"VERIFIED", "FIXTURE_VERIFIED"} or
                    nodes[right_id]["physical_support_status"] not in {"VERIFIED", "FIXTURE_VERIFIED"} or not coords_known):
                unresolved_pairs.append({"left": left_id, "right": right_id,
                    "reason": "UNKNOWN_PHYSICAL_MEMBERSHIP_OR_COORDINATES"})

    adjacency = {lake_id: set() for lake_id in rows}
    for edge in edges:
        adjacency[edge["left"]].add(edge["right"])
        adjacency[edge["right"]].add(edge["left"])
    components, seen = [], set()
    for lake_id in ids:
        if lake_id in seen:
            continue
        todo, component = [lake_id], []
        seen.add(lake_id)
        while todo:
            current = todo.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current] - seen):
                seen.add(neighbor)
                todo.append(neighbor)
        components.append(sorted(component))
    cross_role_edges = [edge for edge in edges
                        if scope_by_lake[edge["left"]] != scope_by_lake[edge["right"]]]
    fixture_only = any(node["physical_support_status"] == "FIXTURE_VERIFIED" or
                       node["coordinate_status"] == "FIXTURE_VERIFIED" for node in nodes.values())
    isolation = ("UNRESOLVED_UNKNOWN_PHYSICAL_MEMBERSHIP" if unresolved_units or unresolved_pairs else
                 "CROSS_ROLE_PHYSICAL_ADJACENCY" if cross_role_edges else
                 "FIXTURE_ONLY_NOT_RESEARCH_VERIFIED" if fixture_only else
                 "ROLE_ISOLATION_SUPPORTED_BY_EVIDENCE_BOUND_GS")
    return {"graph_id": "G_S", "adjacency_rule": {"shared_support_fields": list(PHYSICAL_SUPPORT_FIELDS),
            "distance_threshold_km_exclusive": 50.0}, "nodes": [nodes[k] for k in ids],
        "edges": edges, "components": components, "unresolved_units": sorted(set(unresolved_units)),
        "unresolved_pairs": unresolved_pairs, "cross_role_edges": cross_role_edges,
        "role_isolation_status": isolation}


def build_fitting_provenance_graph(provenance_by_lake, scope_by_lake):
    """Record G_P dependencies independently; shared weights never alter G_S."""
    nodes, unresolved = [], []
    for lake_id in sorted(scope_by_lake):
        value = provenance_by_lake.get(lake_id) if isinstance(provenance_by_lake, Mapping) else None
        if not isinstance(value, Mapping) or set(value) != {"fit_input_ids", "checkpoint_ids"}:
            unresolved.append(lake_id)
            nodes.append({"lake_id": lake_id, "scope": scope_by_lake[lake_id],
                          "provenance_status": "UNKNOWN", "fit_input_ids": None, "checkpoint_ids": None})
            continue
        normalized = {}
        for field in ("fit_input_ids", "checkpoint_ids"):
            ids = value[field]
            if (not isinstance(ids, (list, tuple)) or any(not isinstance(x, str) or not x for x in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("provenance IDs must be explicit unique strings")
            normalized[field] = sorted(ids)
        nodes.append({"lake_id": lake_id, "scope": scope_by_lake[lake_id],
                      "provenance_status": "DECLARED", **normalized})
    edges = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            if left["provenance_status"] != "DECLARED" or right["provenance_status"] != "DECLARED":
                continue
            shared_fits = sorted(set(left["fit_input_ids"]) & set(right["fit_input_ids"]))
            shared_checkpoints = sorted(set(left["checkpoint_ids"]) & set(right["checkpoint_ids"]))
            if shared_fits or shared_checkpoints:
                edges.append({"left": left["lake_id"], "right": right["lake_id"],
                    "shared_fit_input_ids": shared_fits, "shared_checkpoint_ids": shared_checkpoints})
    return {"graph_id": "G_P", "nodes": nodes, "edges": edges,
            "unresolved_units": unresolved,
            "status": "PROVENANCE_UNRESOLVED" if unresolved else "DECLARED_LINEAGE_RECORDED"}


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    """Versioned four-role schedule; every scheduled candidate retains its status."""
    split_id: str
    protocol: str
    created_at: str
    scope_membership: Mapping[str, Mapping[str, Tuple[str, ...]]]
    schedule_rows: Tuple[DecisionWindow, ...]
    physical_graph: Mapping[str, Any]
    provenance_graph: Mapping[str, Any]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if set(self.scope_membership) != set(SCOPE_ROLES):
            raise ValueError("split manifest requires exactly four explicit fitting roles")
        window_ids = [row.window_id for row in self.schedule_rows]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("schedule window IDs must be unique")
        assigned = set()
        rows_by_id = {row.window_id: row for row in self.schedule_rows}
        for role in SCOPE_ROLES:
            membership = self.scope_membership[role]
            if set(membership) != {"lake_ids", "window_ids"}:
                raise ValueError("scope membership requires exact lake and window IDs")
            lakes, windows = tuple(membership["lake_ids"]), tuple(membership["window_ids"])
            if len(lakes) != len(set(lakes)) or len(windows) != len(set(windows)):
                raise ValueError("duplicate scope membership")
            if assigned & set(windows):
                raise ValueError("window IDs cannot belong to more than one scope")
            if any(not isinstance(lake_id, str) or not lake_id for lake_id in lakes):
                raise ValueError("scope lake IDs must be explicit nonempty strings")
            role_rows = {window_id for window_id, row in rows_by_id.items() if row.scope == role}
            if role_rows != set(windows):
                raise ValueError("scope window membership differs from scheduled role rows")
            if any(rows_by_id[window_id].lake_id not in set(lakes) for window_id in windows):
                raise ValueError("scheduled lake differs from its explicit role membership")
            if set(lakes) != {rows_by_id[window_id].lake_id for window_id in windows}:
                raise ValueError("every scope lake requires complete scheduled membership")
            assigned.update(windows)
        if assigned != set(window_ids):
            raise ValueError("all schedule rows must have exactly one scope assignment")
        for row in self.schedule_rows:
            if row.scope not in SCOPE_ROLES or row.window_id not in self.scope_membership[row.scope]["window_ids"]:
                raise ValueError("schedule row scope differs from explicit membership")

    def to_dict(self) -> Dict[str, Any]:
        result = {"schema_version": 2, "split_id": self.split_id, "protocol": self.protocol,
            "created_at": self.created_at,
            "scope_membership": {role: {"lake_ids": list(self.scope_membership[role]["lake_ids"]),
                "window_ids": list(self.scope_membership[role]["window_ids"])} for role in SCOPE_ROLES},
            "schedule_rows": [window.to_dict() for window in self.schedule_rows],
            "graphs": {"G_S": dict(self.physical_graph), "G_P": dict(self.provenance_graph)},
            "summary": {"schedule_rows": len(self.schedule_rows),
                "eligible_rows": sum(row.is_eligible for row in self.schedule_rows),
                "ineligible_rows": sum(not row.is_eligible for row in self.schedule_rows),
                "scope_row_counts": {role: len(self.scope_membership[role]["window_ids"])
                                      for role in SCOPE_ROLES},
                "physical_role_isolation_status": self.physical_graph["role_isolation_status"]},
            "metadata": dict(self.metadata)}
        from .admission import validate_schema
        validate_schema(result, "split_manifest")
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


class SplitEngine:
    """Construct complete causal rows and preserve G_S/G_P as separate graphs."""
    def __init__(self, *, window_days: int, stride_days: int,
                 minimum_observations_by_modality: Optional[Mapping[str, int]] = None,
                 buffer_km: float = 50.0):
        if type(window_days) is not int or window_days < 1 or type(stride_days) is not int or stride_days < 1:
            raise ValueError("context and stride must be explicitly declared positive candidates")
        if not math.isclose(float(buffer_km), 50.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("physical sampling adjacency uses the conservative 50 km threshold")
        self.window_days = window_days
        self.stride_days = stride_days
        self.buffer_km = 50.0
        self.scheduler = AvailabilityScheduler(window_days, stride_days,
            minimum_observations_by_modality=minimum_observations_by_modality)

    def build_spatiotemporal_split(self, *, split_id: str,
        lakes: Sequence[Mapping[str, Any]],
        observations_by_lake: Mapping[str, Sequence[Mapping[str, Any]]],
        start_date: str, end_date: str, scope_by_lake: Mapping[str, str],
        required_modalities: Sequence[str], event_cutoffs: Optional[Mapping[str, str]] = None,
        provenance_by_lake: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
        verified_support_bindings: Optional[Mapping[str, Mapping[str, str]]] = None,
        verified_coordinate_bindings: Optional[Mapping[str, Mapping[str, str]]] = None) -> SplitManifest:
        lake_ids = [str(row.get("lake_id", "")) for row in lakes]
        if not lake_ids or "" in lake_ids or len(lake_ids) != len(set(lake_ids)):
            raise ValueError("unique lake identities are required")
        if set(scope_by_lake) != set(lake_ids) or set(scope_by_lake.values()) - set(SCOPE_ROLES):
            raise ValueError("every lake must have one explicit assignment to one of the four roles")
        if set(observations_by_lake) - set(lake_ids):
            raise ValueError("observation membership contains an unknown lake")
        if set(event_cutoffs or {}) - set(lake_ids):
            raise ValueError("event cutoff membership contains an unknown lake")
        physical_graph = build_physical_sampling_graph(lakes, scope_by_lake, buffer_km=self.buffer_km,
            verified_support_bindings=verified_support_bindings,
            verified_coordinate_bindings=verified_coordinate_bindings)
        provenance_graph = build_fitting_provenance_graph(provenance_by_lake or {}, scope_by_lake)
        rows = []
        scope_lakes = {role: [] for role in SCOPE_ROLES}
        scope_windows = {role: [] for role in SCOPE_ROLES}
        for lake_id in sorted(lake_ids):
            role = scope_by_lake[lake_id]
            scope_lakes[role].append(lake_id)
            candidate_rows = self.scheduler.generate_decision_schedule(lake_id=lake_id,
                start_date=start_date, end_date=end_date,
                observations=observations_by_lake.get(lake_id, ()),
                scope=role,
                event_cutoff=(event_cutoffs or {}).get(lake_id),
                required_modalities=required_modalities)
            for row in candidate_rows:
                rows.append(row)
                scope_windows[role].append(row.window_id)
        membership = {role: {"lake_ids": tuple(scope_lakes[role]),
                             "window_ids": tuple(scope_windows[role])} for role in SCOPE_ROLES}
        return SplitManifest(split_id=split_id, protocol="four_role_causal_support_graph_v2",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope_membership=membership, schedule_rows=tuple(rows), physical_graph=physical_graph,
            provenance_graph=provenance_graph, metadata={"window_days_candidate": self.window_days,
                "stride_days_candidate": self.stride_days,
                "minimum_observations_by_modality": dict(self.scheduler.minimum_observations_by_modality)
                    if self.scheduler.minimum_observations_by_modality is not None else None,
                "required_modalities": list(required_modalities),
                "event_cutoffs": dict(event_cutoffs or {}),
                "all_candidate_decisions_retained": True,
                "research_admissible": False})
