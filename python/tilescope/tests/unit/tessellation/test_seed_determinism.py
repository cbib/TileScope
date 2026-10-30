"""Regression tests for tessellation determinism.

`compute_tessellation` defaults `rng_seed=0` (a concrete int, not
None). Same seed → bit-identical output; different seeds → diverge.
Guards against the historical `rng_seed=None` bug that pulled fresh
OS entropy per call.
"""
import numpy as np
from shapely.geometry import box

from tilescope.tessellation.builder import (
    compute_tessellation,
)


def _fixture():
    rng = np.random.default_rng(123)
    n = 200
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    region_polygon = box(0.0, 0.0, 10.0, 10.0)
    return coords, region_polygon


def _tile_signature(tessellation):
    tiles = tessellation.tiles
    return (
        len(tiles),
        tuple(round(t.polygon.area, 12) for t in tiles),
        tuple(t.n_spots for t in tiles),
    )


def test_compute_tessellation_same_seed_is_bit_identical():
    coords, poly = _fixture()
    a = compute_tessellation("roi", coords, poly, rng_seed=7)
    b = compute_tessellation("roi", coords, poly, rng_seed=7)
    assert _tile_signature(a) == _tile_signature(b)


def test_compute_tessellation_different_seeds_generally_differ():
    coords, poly = _fixture()
    a = compute_tessellation("roi", coords, poly, rng_seed=7)
    b = compute_tessellation("roi", coords, poly, rng_seed=42)
    assert _tile_signature(a) != _tile_signature(b)


def test_compute_tessellation_default_seed_is_deterministic():
    coords, poly = _fixture()
    a = compute_tessellation("roi", coords, poly)
    b = compute_tessellation("roi", coords, poly)
    assert _tile_signature(a) == _tile_signature(b)
