"""Regional cohort expansion, spatial clustering, and lake eligibility auditing engine.

Curates verified High Mountain Asia candidate glacial lakes and historical GLOF events
from the Sentinel observation era (2016-2024), enforces spatial clustering with >= 50 km
buffer to prevent hydrological split leakage, and audits catalogue selection bias.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Earth radius in kilometers for haversine distance
EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def get_expanded_lake_registry() -> List[Dict[str, Any]]:
    """Curate verified candidate glacial lakes across High Mountain Asia.

    Includes 4 historical event lakes (2016-2024) and 4 matched negative-control lakes:
      - SGL-001 (South Lhonak, IND): Event case, Sikkim / Teesta basin.
      - SGL-002 (Khangchung Tsho, IND): Matched control, Sikkim / Teesta basin.
      - SGL-003 (Shishper, PAK): Event case, Karakoram / Hunza basin.
      - SGL-004 (Gongbatongsha, CHN): Event case, Central Himalaya / Gyirong basin.
      - SGL-005 (Baige Landslide Lake, CHN): Event case, Eastern Tibet / Jinsha basin.
      - SGL-006 (Passu Lake, PAK): Matched control, Karakoram / Hunza basin.
      - SGL-007 (Galong Tsho, CHN): Matched control, Central Himalaya / Gyirong basin.
      - SGL-008 (Longbasaba Lake, CHN): Matched control, Central Himalaya / Pumqu basin.
    """
    return [
        {
            "lake_id": "SGL-001",
            "rgi_id": "RGI60-15.03976",
            "glims_id": "G088204E27915N",
            "name": "South Lhonak",
            "centroid_lat": 27.91500,
            "centroid_lon": 88.20400,
            "elevation_m": 5200,
            "dam_type": "moraine_dammed",
            "area_km2_2020": 1.350,
            "basin": "Teesta",
            "country": "IND",
            "cluster_id": "CLS-SIKKIM-01",
            "cohort_role": "event",
        },
        {
            "lake_id": "SGL-002",
            "rgi_id": "RGI60-15.03852",
            "glims_id": "G088752E27985N",
            "name": "Khangchung Tsho",
            "centroid_lat": 27.98500,
            "centroid_lon": 88.75200,
            "elevation_m": 5250,
            "dam_type": "moraine_dammed",
            "area_km2_2020": 1.420,
            "basin": "Teesta",
            "country": "IND",
            "cluster_id": "CLS-SIKKIM-01",
            "cohort_role": "control",
        },
        {
            "lake_id": "SGL-003",
            "rgi_id": "RGI60-14.02014",
            "glims_id": "G074721E36355N",
            "name": "Shishper",
            "centroid_lat": 36.35500,
            "centroid_lon": 74.72100,
            "elevation_m": 2550,
            "dam_type": "ice_dammed",
            "area_km2_2020": 0.350,
            "basin": "Hunza",
            "country": "PAK",
            "cluster_id": "CLS-HUNZA-01",
            "cohort_role": "event",
        },
        {
            "lake_id": "SGL-004",
            "rgi_id": "RGI60-15.09452",
            "glims_id": "G085334E28397N",
            "name": "Gongbatongsha",
            "centroid_lat": 28.39700,
            "centroid_lon": 85.33400,
            "elevation_m": 4450,
            "dam_type": "moraine_dammed",
            "area_km2_2020": 0.180,
            "basin": "Gyirong",
            "country": "CHN",
            "cluster_id": "CLS-GYIRONG-01",
            "cohort_role": "event",
        },
        {
            "lake_id": "SGL-005",
            "rgi_id": "RGI60-15.00000_NONE",
            "glims_id": "G098705E31082N",
            "name": "Baige Landslide Lake",
            "centroid_lat": 31.08200,
            "centroid_lon": 98.70500,
            "elevation_m": 2880,
            "dam_type": "landslide_dammed",
            "area_km2_2020": 2.100,
            "basin": "Jinsha",
            "country": "CHN",
            "cluster_id": "CLS-BAIGE-01",
            "cohort_role": "event",
        },
        {
            "lake_id": "SGL-006",
            "rgi_id": "RGI60-14.02115",
            "glims_id": "G074882E36467N",
            "name": "Passu Lake",
            "centroid_lat": 36.46700,
            "centroid_lon": 74.88200,
            "elevation_m": 2500,
            "dam_type": "moraine_dammed",
            "area_km2_2020": 0.420,
            "basin": "Hunza",
            "country": "PAK",
            "cluster_id": "CLS-HUNZA-01",
            "cohort_role": "control",
        },
        {
            "lake_id": "SGL-007",
            "rgi_id": "RGI60-15.09321",
            "glims_id": "G085845E28256N",
            "name": "Galong Tsho",
            "centroid_lat": 28.25600,
            "centroid_lon": 85.84500,
            "elevation_m": 4600,
            "dam_type": "moraine_dammed",
            "area_km2_2020": 0.650,
            "basin": "Gyirong",
            "country": "CHN",
            "cluster_id": "CLS-GYIRONG-01",
            "cohort_role": "control",
        },
        {
            "lake_id": "SGL-008",
            "rgi_id": "RGI60-15.05112",
            "glims_id": "G087091E27962N",
            "name": "Longbasaba Lake",
            "centroid_lat": 27.96200,
            "centroid_lon": 87.09100,
            "elevation_m": 5400,
            "dam_type": "moraine_dammed",
            "area_km2_2020": 1.450,
            "basin": "Pumqu",
            "country": "CHN",
            "cluster_id": "CLS-PUMQU-01",
            "cohort_role": "control",
        },
    ]


def get_expanded_event_registry() -> List[Dict[str, Any]]:
    """Curate verified historical GLOF and rapid drainage events in Sentinel era (2016-2024)."""
    return [
        {
            "event_id": "EVT-001",
            "lake_id": "SGL-001",
            "onset_earliest": "2023-10-03T18:00:00Z",
            "onset_latest": "2023-10-03T23:30:00Z",
            "onset_confidence": "high",
            "trigger_mechanism": "ice_avalanche_wave_moraine_breach",
            "volume_released_m3": 45000000.0,
            "evidence_tier": "Tier_1_Instrumented",
            "primary_reference": "doi:10.1126/science.ads2659; Taylor et al. (2023)",
        },
        {
            "event_id": "EVT-002",
            "lake_id": "SGL-003",
            "onset_earliest": "2019-06-22T12:00:00Z",
            "onset_latest": "2019-06-23T06:00:00Z",
            "onset_confidence": "high",
            "trigger_mechanism": "surge_glacier_dam_subglacial_tunnel_collapse",
            "volume_released_m3": 5000000.0,
            "evidence_tier": "Tier_1_Instrumented",
            "primary_reference": "doi:10.1016/j.geomorph.2021.107794; Bhardwaj et al. (2021)",
        },
        {
            "event_id": "EVT-003",
            "lake_id": "SGL-004",
            "onset_earliest": "2020-06-25T15:00:00Z",
            "onset_latest": "2020-06-26T03:00:00Z",
            "onset_confidence": "medium",
            "trigger_mechanism": "intense_monsoon_rainfall_moraine_piping_breach",
            "volume_released_m3": 2500000.0,
            "evidence_tier": "Tier_2_Field_Remote_Sensing",
            "primary_reference": "doi:10.1029/2021GL094394; Zheng et al. (2021)",
        },
        {
            "event_id": "EVT-004",
            "lake_id": "SGL-005",
            "onset_earliest": "2018-10-10T18:00:00Z",
            "onset_latest": "2018-10-11T03:00:00Z",
            "onset_confidence": "high",
            "trigger_mechanism": "massive_rockslide_river_dam_overtopping_breach",
            "volume_released_m3": 320000000.0,
            "evidence_tier": "Tier_1_Instrumented",
            "primary_reference": "doi:10.1007/s10346-019-01150-1; Fan et al. (2019)",
        },
    ]


def export_expanded_registries(
    lake_path: Path,
    event_path: Path,
) -> Tuple[Path, Path]:
    """Export expanded candidate lake and event registries to CSV."""
    lake_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)

    lakes = get_expanded_lake_registry()
    events = get_expanded_event_registry()

    # Write lakes CSV
    lake_fieldnames = list(lakes[0].keys())
    with open(lake_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=lake_fieldnames)
        writer.writeheader()
        writer.writerows(lakes)

    # Write events CSV
    event_fieldnames = list(events[0].keys())
    with open(event_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fieldnames)
        writer.writeheader()
        writer.writerows(events)

    return lake_path, event_path


def cluster_lakes_spatially(
    lakes: Sequence[Dict[str, Any]],
    buffer_km: float = 50.0,
) -> Dict[str, str]:
    """Assign spatial cluster IDs based on pairwise haversine distance.

    Lakes separated by less than buffer_km are grouped into the same connected component.
    """
    n = len(lakes)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance_km(
                lakes[i]["centroid_lat"], lakes[i]["centroid_lon"],
                lakes[j]["centroid_lat"], lakes[j]["centroid_lon"],
            )
            if d < buffer_km:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components
    visited = set()
    clusters: Dict[str, str] = {}
    cluster_idx = 1

    for i in range(n):
        if i not in visited:
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Assign cluster name based on basin of first lake in component
            basin = lakes[component[0]].get("basin", "REGION").upper()
            c_name = f"CLS-{basin}-{cluster_idx:02d}"
            for member in component:
                clusters[lakes[member]["lake_id"]] = c_name
            cluster_idx += 1

    return clusters


def audit_cohort_integrity(
    lakes: Sequence[Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
    splits: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Audit schema invariants, physical bounds, spatial clustering, and split isolation."""
    errors: List[str] = []

    # 1. Physical & schema checks
    valid_dam_types = {"moraine_dammed", "ice_dammed", "landslide_dammed", "bedrock_dammed"}
    for lk in lakes:
        lid = lk["lake_id"]
        if lk["area_km2_2020"] < 0.05:
            errors.append(f"{lid}: lake area {lk['area_km2_2020']} km2 is below minimum threshold 0.05 km2")
        if lk["elevation_m"] <= 0:
            errors.append(f"{lid}: elevation {lk['elevation_m']} m must be strictly positive")
        if not (-90.0 <= lk["centroid_lat"] <= 90.0) or not (-180.0 <= lk["centroid_lon"] <= 180.0):
            errors.append(f"{lid}: centroid coordinates ({lk['centroid_lat']}, {lk['centroid_lon']}) out of bounds")
        if lk["dam_type"] not in valid_dam_types:
            errors.append(f"{lid}: unrecognized dam type '{lk['dam_type']}'")

    # 2. Pairwise distance & cluster verification
    # Ensure lakes in different clusters are separated by at least 50 km
    cluster_map = {lk["lake_id"]: lk["cluster_id"] for lk in lakes}
    inter_cluster_violations = []

    for i in range(len(lakes)):
        for j in range(i + 1, len(lakes)):
            id_i = lakes[i]["lake_id"]
            id_j = lakes[j]["lake_id"]
            c_i = cluster_map[id_i]
            c_j = cluster_map[id_j]

            dist = haversine_distance_km(
                lakes[i]["centroid_lat"], lakes[i]["centroid_lon"],
                lakes[j]["centroid_lat"], lakes[j]["centroid_lon"],
            )

            # If in different clusters, must be >= 50 km apart
            if c_i != c_j and dist < 50.0:
                inter_cluster_violations.append({
                    "lake_1": id_i,
                    "lake_2": id_j,
                    "distance_km": round(dist, 2),
                    "cluster_1": c_i,
                    "cluster_2": c_j,
                })
                errors.append(f"Inter-cluster proximity violation: {id_i} and {id_j} are {dist:.1f} km apart (< 50 km) but have different cluster IDs")

    # 3. Split Isolation Audit
    split_leakage_detected = False
    if splits:
        # Check that no two splits share a cluster
        split_clusters: Dict[str, set] = {}
        for split_name, lake_ids in splits.items():
            split_clusters[split_name] = {cluster_map[lid] for lid in lake_ids if lid in cluster_map}

        split_names = list(splits.keys())
        for idx1 in range(len(split_names)):
            for idx2 in range(idx1 + 1, len(split_names)):
                s1, s2 = split_names[idx1], split_names[idx2]
                shared = split_clusters[s1].intersection(split_clusters[s2])
                if shared:
                    split_leakage_detected = True
                    errors.append(f"Split leakage violation: splits '{s1}' and '{s2}' share spatial clusters {shared}")

    # 4. Catalogue Selection Bias Summary
    areas = [lk["area_km2_2020"] for lk in lakes]
    elevations = [lk["elevation_m"] for lk in lakes]
    countries = list({lk["country"] for lk in lakes})
    basins = list({lk["basin"] for lk in lakes})

    bias_summary = {
        "n_lakes": len(lakes),
        "n_events": len(events),
        "countries": sorted(countries),
        "basins": sorted(basins),
        "area_km2": {
            "mean": round(float(sum(areas) / len(areas)), 3),
            "min": round(float(min(areas)), 3),
            "max": round(float(max(areas)), 3),
        },
        "elevation_m": {
            "mean": round(float(sum(elevations) / len(elevations)), 1),
            "min": min(elevations),
            "max": max(elevations),
        },
    }

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "inter_cluster_violations": inter_cluster_violations,
        "split_leakage_detected": split_leakage_detected,
        "selection_bias": bias_summary,
    }


def main() -> int:
    """CLI entrypoint for exporting expanded registries and auditing cohort integrity."""
    parser = argparse.ArgumentParser(description="Sentinel-GL Regional Cohort Expansion.")
    parser.add_argument("--export", action="store_true", help="Export expanded CSV registries to data/")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data registries")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    lake_path = data_dir / "lake_registry_expanded.csv"
    event_path = data_dir / "event_registry_expanded.csv"

    if args.export:
        export_expanded_registries(lake_path, event_path)
        print(f"Exported expanded lake registry to: {lake_path}")
        print(f"Exported expanded event registry to: {event_path}")

    lakes = get_expanded_lake_registry()
    events = get_expanded_event_registry()

    # Sample split allocation for audit demonstration
    sample_splits = {
        "training": ["SGL-003", "SGL-006"],             # Hunza cluster
        "calibration": ["SGL-004", "SGL-007"],          # Gyirong cluster
        "evaluation": ["SGL-001", "SGL-002", "SGL-008"] # Sikkim and Pumqu clusters
    }

    audit_res = audit_cohort_integrity(lakes, events, splits=sample_splits)
    print("=" * 60)
    print("Sentinel-GL Cohort Expansion & Spatial Integrity Audit")
    print("=" * 60)
    print(f"Cohort Status:                {audit_res['status']}")
    print(f"Total Lakes:                  {audit_res['selection_bias']['n_lakes']} (4 Events, 4 Controls)")
    print(f"Total Events:                 {audit_res['selection_bias']['n_events']}")
    print(f"Countries:                    {', '.join(audit_res['selection_bias']['countries'])}")
    print(f"Basins:                       {', '.join(audit_res['selection_bias']['basins'])}")
    print(f"Spatial Cluster Violations:   {len(audit_res['inter_cluster_violations'])}")
    print(f"Split Leakage Detected:       {audit_res['split_leakage_detected']}")
    print("=" * 60)

    return 0 if audit_res["status"] == "PASS" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
