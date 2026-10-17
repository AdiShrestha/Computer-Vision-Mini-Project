"""Constructed configuration tests; real accepted grids are metadata only."""
from pathlib import Path
import json

import numpy as np
import pytest
from rasterio.transform import array_bounds
from pyproj import Transformer

from sentinel_gl.ingestion.expansion2023 import grid_for, SITES

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('site', sorted(SITES))
def test_public_roi_grid_contains_its_declared_centroid(site):
    grid = grid_for(ROOT, site)
    west, south, east, north = array_bounds(499, 499, __import__('rasterio').Affine(*grid['transform']))
    x, y = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform(*SITES[site])
    assert west < x < east and south < y < north
    assert east-west == pytest.approx(9980)
    assert north-south == pytest.approx(9980)
    assert abs(np.linalg.det(np.asarray(grid['transform']).reshape(2, 3)[:, :2])) == pytest.approx(400)


def test_public_protocol_retains_declared_fit_firewall():
    protocol = json.loads((ROOT/'configs/domain_2023_protocol.json').read_text())
    assert protocol['protocol_id'] == 'SENTINEL_GL_RETROSPECTIVE_2023_V2'
    assert protocol['fit_end_exclusive'] == protocol['pre_event_cutoff'] == '2023-10-02T18:30:00Z'
    assert protocol['fit_parity'] == 0 and protocol['monitor_parity'] == 1
    assert protocol['reconstruction_folds'] == 4
    assert protocol['seed_ids'] == list(range(5))
