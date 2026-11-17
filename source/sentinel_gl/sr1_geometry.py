"""Independent lake geometry from the Hi-MAG glacial-lake inventory.

Hi-MAG (Chen et al. 2021, Earth Syst. Sci. Data 13:741-766; Zenodo record
4275164, CC BY 4.0) gives annual 30 m Landsat-derived glacial-lake outlines
for High Mountain Asia, 2008-2017. The 2017 outline of each study lake is the
independent, pre-study-period reference used to (a) centre the ROI window on
the lake and (b) define the analysis zones:

* zone 0: pixels whose centre lies inside the 2017 inventory outline;
* zone 1: other pixels within 500 m of zone 0 (shore, dam and moraine buffer);
* zone 2: the remaining window (background).

The outline is not a 2018-2023 shoreline. The spectral FIT-frequency zone rule
used by earlier attempts is retained only for an agreement check.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

HI_MAG_RECORD = "https://doi.org/10.5281/zenodo.4275164"
HI_MAG_ZIP_SHA256 = "b7abd8ed1ae452961e7826418104c896db7e8a8dec329d6d7353d1e469e9085c"
HI_MAG_IDS = {"PILOT-SOUTH-LHONAK": "GL088195E27912N", "PILOT-KHANGCHUNG-TSHO": "GL088815E27990N"}
BUFFER_M = 500.0
PIXEL_M = 20.0


def load_hi_mag_outline(shapefile_stem: Path, gl_id: str, target_epsg: int = 32645):
    """Return (geometry in target CRS, attribute dict) for one Hi-MAG lake ID."""
    import shapefile  # pyshp
    from pyproj import CRS, Transformer

    stem = Path(shapefile_stem)
    source_crs = CRS.from_wkt(stem.with_suffix(".prj").read_text())
    project = Transformer.from_crs(source_crs, f"EPSG:{target_epsg}", always_xy=True).transform
    matches = []
    with shapefile.Reader(str(stem)) as reader:
        for record in reader.iterShapeRecords():
            attrs = record.record.as_dict()
            if attrs.get("GL_ID") == gl_id:
                matches.append((shape(record.shape.__geo_interface__), attrs))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Hi-MAG outline for {gl_id}, found {len(matches)}")
    geometry, attrs = matches[0]
    if not geometry.is_valid or geometry.area <= 0:
        raise ValueError(f"invalid Hi-MAG outline for {gl_id}")
    return shapely_transform(project, geometry), attrs


def outline_centroid_lonlat(geometry_utm, epsg: int = 32645) -> tuple[float, float]:
    from pyproj import Transformer

    back = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    c = geometry_utm.centroid
    lon, lat = back.transform(c.x, c.y)
    return float(lon), float(lat)


def snapped_window(lon: float, lat: float, size: int = 160, epsg: int = 32645) -> dict:
    """160 x 160 window at 20 m whose centre is the 20 m pixel corner nearest (lon, lat)."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    cx = math.floor(x / PIXEL_M + 0.5) * PIXEL_M
    cy = math.floor(y / PIXEL_M + 0.5) * PIXEL_M
    half = size // 2 * PIXEL_M
    transform = [PIXEL_M, 0.0, cx - half, 0.0, -PIXEL_M, cy + half]
    return {"crs": f"EPSG:{epsg}", "shape": [size, size], "transform": transform,
            "center_lon": lon, "center_lat": lat, "center_easting_m": x, "center_northing_m": y}


def rasterize_outline(geometry_utm, grid: dict) -> np.ndarray:
    h, w = grid["shape"]
    affine = rasterio.Affine(*grid["transform"])
    return features.rasterize([(mapping(geometry_utm), 1)], out_shape=(h, w), transform=affine,
                              fill=0, all_touched=False, dtype="uint8").astype(bool)


def inventory_zone_map(geometry_utm, grid: dict, buffer_m: float = BUFFER_M) -> dict:
    lake = rasterize_outline(geometry_utm, grid)
    if not lake.any():
        raise ValueError("inventory outline does not intersect the window")
    buffer = (distance_transform_edt(~lake) * PIXEL_M <= buffer_m) & ~lake
    zones = np.full(lake.shape, 2, dtype=np.uint8)
    zones[buffer] = 1
    zones[lake] = 0
    h, w = grid["shape"]
    affine = rasterio.Affine(*grid["transform"])
    left, top = affine.c, affine.f
    right, bottom = left + w * PIXEL_M, top - h * PIXEL_M
    from shapely.geometry import box

    window_box = box(left, bottom, right, top)
    inside = geometry_utm.intersection(window_box).area / geometry_utm.area
    return {"zones": zones, "lake": lake, "buffer": buffer,
            "outline_area_km2": geometry_utm.area / 1e6,
            "outline_fraction_inside_window": float(inside),
            "rasterized_lake_pixels": int(lake.sum()),
            "rasterized_lake_area_km2": float(lake.sum() * PIXEL_M * PIXEL_M / 1e6)}


def spectral_fit_zone_map(fit_rows: list[dict], compute_indices) -> np.ndarray:
    """Earlier attempts' rule: water in >= 50% of >= 3 supported FIT scenes (for agreement only)."""
    shape_ = fit_rows[0]["scl"].shape
    votes = np.zeros(shape_, dtype=np.uint32)
    support = np.zeros(shape_, dtype=np.uint32)
    for row in fit_rows:
        clear = np.isin(row["scl"], [4, 5, 6]) & row["valid"].all(axis=0)
        ndwi, mndwi = compute_indices(*row["reflectance"], delta=0.002)
        defined = clear & np.isfinite(ndwi) & np.isfinite(mndwi)
        support[defined] += 1
        votes[defined & (row["scl"] == 6) & (ndwi > 0) & (mndwi > 0)] += 1
    frequency = np.divide(votes, support, out=np.zeros(shape_, dtype=np.float64), where=support > 0)
    return (frequency >= 0.5) & (support >= min(3, len(fit_rows)))


def mask_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    inter = int((a & b).sum())
    return {"intersection_pixels": inter, "union_pixels": union,
            "iou": (inter / union) if union else None,
            "a_pixels": int(a.sum()), "b_pixels": int(b.sum()),
            "a_only_pixels": int((a & ~b).sum()), "b_only_pixels": int((b & ~a).sum())}
