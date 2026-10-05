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
        dt = dt.replace(tzinfo=datetime.timezone.utc)
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
    missingness_reason: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "lake_id": self.lake_id,
            "decision_timestamp": self.decision_timestamp,
            "context_start": self.context_start,
            "context_end": self.context_end,
            "observation_ids": list(self.observation_ids),
            "is_eligible": self.is_eligible,
            "modalities_present": list(self.modalities_present),
            "missingness_reason": self.missingness_reason,
            "metadata": self.metadata,
        }


class AvailabilityScheduler:
    """Filter raw observations according to strict retrospective availability rules."""

    def __init__(
        self,
        window_days: int = 180,
        stride_days: int = 30,
        min_obs_per_modality: int = 2,
    ):
        if window_days <= 0 or stride_days <= 0:
            raise ValueError("window_days and stride_days must be positive integers")
        self.window_days = window_days
        self.stride_days = stride_days
        self.min_obs_per_modality = min_obs_per_modality

    def filter_retrospective_observations(
        self,
        observations: Sequence[Mapping[str, Any]],
        decision_timestamp: str,
        event_cutoff: Optional[str] = None,
    ) -> List[Mapping[str, Any]]:
        """Filter observations strictly available prior to decision_timestamp and event_cutoff.

        Strict Invariant:
        1. t_acq < t_decision (zero future data leakage).
        2. If event_cutoff is specified, t_acq < t_event_cutoff (zero event/post-event leakage).
        3. t_acq >= t_decision - window_days (support window).
        """
        t_decision = parse_utc_timestamp(decision_timestamp)
        t_window_start = t_decision - datetime.timedelta(days=self.window_days)
        t_cutoff = parse_utc_timestamp(event_cutoff) if event_cutoff else None

        valid_records: List[Mapping[str, Any]] = []
        for obs in observations:
            t_acq = parse_utc_timestamp(obs["acquisition_timestamp"])

            # 1. Temporal support bounds
            if t_acq >= t_decision:
                continue
            if t_acq < t_window_start:
                continue

            # 2. Pre-event quarantine
            if t_cutoff is not None and t_acq >= t_cutoff:
                continue

            valid_records.append(obs)

        return valid_records

    def generate_decision_schedule(
        self,
        lake_id: str,
        start_date: str,
        end_date: str,
        observations: Sequence[Mapping[str, Any]],
        event_cutoff: Optional[str] = None,
        required_modalities: Sequence[str] = ("optical", "sar", "weather"),
    ) -> List[DecisionWindow]:
        """Construct sequential decision windows on a fixed cadence for a single lake."""
        dt_start = parse_utc_timestamp(start_date)
        dt_end = parse_utc_timestamp(end_date)
        if dt_start >= dt_end:
            raise ValueError(f"start_date {start_date} must precede end_date {end_date}")

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
            insufficient_modalities = [
                m for m in required_modalities
                if modality_counts.get(m, 0) < self.min_obs_per_modality
            ]

            is_eligible = len(insufficient_modalities) == 0
            missing_reason = (
                f"INSUFFICIENT_OBSERVATIONS: missing {insufficient_modalities}"
                if not is_eligible else None
            )

            # Compute earliest and latest observation times in window
            obs_timestamps = [parse_utc_timestamp(o["acquisition_timestamp"]) for o in eligible_obs]
            latest_obs_iso = max(obs_timestamps).isoformat() if obs_timestamps else context_start_iso

            win_id = f"WIN-{lake_id}-{current_decision.strftime('%Y%m%d')}"
            win = DecisionWindow(
                window_id=win_id,
                lake_id=lake_id,
                decision_timestamp=decision_iso,
                context_start=context_start_iso,
                context_end=latest_obs_iso,
                observation_ids=tuple(o["record_id"] for o in eligible_obs),
                is_eligible=is_eligible,
                modalities_present=modalities_present,
                missingness_reason=missing_reason,
                metadata={
                    "total_observations": len(eligible_obs),
                    "modality_counts": modality_counts,
                },
            )
            windows.append(win)
            current_decision += datetime.timedelta(days=self.stride_days)

        return windows


class TemporalBoundaryPurger:
    """Purge decision windows across split horizons to eliminate autocorrelation leakage."""

    @staticmethod
    def purge_split_boundary(
        decision_windows: Sequence[DecisionWindow],
        split_timestamp: str,
        window_days: int = 180,
    ) -> Tuple[List[DecisionWindow], List[DecisionWindow], List[DecisionWindow]]:
        """Split a timeline into pre-split, post-split, and purged transition windows.

        Any post-split decision window whose 180-day context extends prior to split_timestamp
        shares historical support with the pre-split period and is purged.

        Returns:
            (pre_split_windows, post_split_windows, purged_windows)
        """
        t_split = parse_utc_timestamp(split_timestamp)
        pre_split: List[DecisionWindow] = []
        post_split: List[DecisionWindow] = []
        purged: List[DecisionWindow] = []

        for win in decision_windows:
            t_decision = parse_utc_timestamp(win.decision_timestamp)
            t_context_start = parse_utc_timestamp(win.context_start)

            if t_decision <= t_split:
                # Historical fitting/training window
                pre_split.append(win)
            else:
                # Prospective evaluation candidate: check if context reaches into training period
                if t_context_start < t_split:
                    # Overlap detected with historical training observation support
                    purged.append(win)
                else:
                    post_split.append(win)

        return pre_split, post_split, purged


@dataclasses.dataclass
class SplitManifest:
    """Comprehensive spatiotemporal split manifest with full provenance."""
    split_id: str
    protocol: str
    created_at: str
    clusters: Dict[str, List[str]]
    train_lakes: Tuple[str, ...]
    val_lakes: Tuple[str, ...]
    test_lakes: Tuple[str, ...]
    train_windows: Tuple[DecisionWindow, ...]
    val_windows: Tuple[DecisionWindow, ...]
    test_windows: Tuple[DecisionWindow, ...]
    purged_windows: Tuple[DecisionWindow, ...]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_id": self.split_id,
            "protocol": self.protocol,
            "created_at": self.created_at,
            "clusters": self.clusters,
            "train_lakes": list(self.train_lakes),
            "val_lakes": list(self.val_lakes),
            "test_lakes": list(self.test_lakes),
            "train_windows": [w.to_dict() for w in self.train_windows],
            "val_windows": [w.to_dict() for w in self.val_windows],
            "test_windows": [w.to_dict() for w in self.test_windows],
            "purged_windows": [w.to_dict() for w in self.purged_windows],
            "summary": {
                "total_train_windows": len(self.train_windows),
                "total_val_windows": len(self.val_windows),
                "total_test_windows": len(self.test_windows),
                "total_purged_windows": len(self.purged_windows),
                "eligible_train_windows": sum(1 for w in self.train_windows if w.is_eligible),
                "eligible_val_windows": sum(1 for w in self.val_windows if w.is_eligible),
                "eligible_test_windows": sum(1 for w in self.test_windows if w.is_eligible),
            },
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SplitEngine:
    """Coordinate spatial clustering, temporal purging, and split manifest creation."""

    def __init__(
        self,
        buffer_km: float = 50.0,
        window_days: int = 180,
        stride_days: int = 30,
        min_obs_per_modality: int = 2,
    ):
        self.buffer_km = buffer_km
        self.window_days = window_days
        self.stride_days = stride_days
        self.min_obs_per_modality = min_obs_per_modality
        self.scheduler = AvailabilityScheduler(
            window_days=window_days,
            stride_days=stride_days,
            min_obs_per_modality=min_obs_per_modality,
        )
        self.purger = TemporalBoundaryPurger()

    def build_spatiotemporal_split(
        self,
        split_id: str,
        lakes: Sequence[Mapping[str, Any]],
        observations_by_lake: Mapping[str, Sequence[Mapping[str, Any]]],
        start_date: str,
        end_date: str,
        split_date: Optional[str] = None,
        test_cluster_ids: Optional[Sequence[str]] = None,
        event_cutoffs: Optional[Mapping[str, str]] = None,
    ) -> SplitManifest:
        """Create a leakage-free spatiotemporal split manifest."""
        clusters = cluster_lakes_by_distance(lakes, buffer_km=self.buffer_km)
        lake_coords = {
            str(l["lake_id"]): (float(l["centroid_lat"]), float(l["centroid_lon"]))
            for l in lakes
        }
        all_lake_ids = {str(l["lake_id"]) for l in lakes}

        # Cluster partition
        assigned_test_clusters = set(test_cluster_ids or [])
        test_lakes: Set[str] = set()
        train_lakes: Set[str] = set()

        for cname, c_lakes in clusters.items():
            if cname in assigned_test_clusters:
                test_lakes.update(c_lakes)
            else:
                train_lakes.update(c_lakes)

        # If no explicit cluster partition, train on non-event or all according to split_date
        if not test_lakes and split_date is None:
            train_lakes = set(all_lake_ids)

        # Validate spatial disjointness if clusters are split
        if train_lakes and test_lakes:
            verify_spatial_disjointness(train_lakes, test_lakes, lake_coords, self.buffer_km)

        train_windows: List[DecisionWindow] = []
        test_windows: List[DecisionWindow] = []
        purged_windows: List[DecisionWindow] = []

        cutoffs = event_cutoffs or {}

        for lake in lakes:
            lid = str(lake["lake_id"])
            obs = observations_by_lake.get(lid, [])
            cutoff = cutoffs.get(lid)

            lake_windows = self.scheduler.generate_decision_schedule(
                lake_id=lid,
                start_date=start_date,
                end_date=end_date,
                observations=obs,
                event_cutoff=cutoff,
            )

            if lid in test_lakes:
                test_windows.extend(lake_windows)
            elif split_date is not None:
                # Temporal split within training cluster
                pre, post, purged = self.purger.purge_split_boundary(
                    lake_windows, split_date, self.window_days
                )
                train_windows.extend(pre)
                test_windows.extend(post)
                purged_windows.extend(purged)
            else:
                train_windows.extend(lake_windows)

        manifest = SplitManifest(
            split_id=split_id,
            protocol="spatiotemporal_purged_cluster",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            clusters=clusters,
            train_lakes=tuple(sorted(train_lakes)),
            val_lakes=(),
            test_lakes=tuple(sorted(test_lakes)),
            train_windows=tuple(train_windows),
            val_windows=(),
            test_windows=tuple(test_windows),
            purged_windows=tuple(purged_windows),
            metadata={
                "buffer_km": self.buffer_km,
                "window_days": self.window_days,
                "stride_days": self.stride_days,
                "split_date": split_date,
            },
        )
        return manifest
