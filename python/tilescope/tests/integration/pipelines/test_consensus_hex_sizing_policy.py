"""Consensus hex sizing — primitive ordering + sizing-authority contract."""
import inspect

import numpy as np
from shapely.geometry import Polygon, box

from tilescope.analyzers.consensus_grid import (
    compute_optimal_hex_radius, winsorized_mean,
)
from tilescope.component_data import Tessellation, Tile
from tilescope.settings import load_run_config


def test_winsor_mean_at_least_median_on_heavy_right_tail():
    """Hex-radius formula is monotonic in `A_base`; on a heavy right tail
    winsor_mean >= median implies hex_radius_winsor >= hex_radius_median."""
    rng = np.random.default_rng(42)
    x = rng.lognormal(mean=0.0, sigma=1.0, size=1000)
    winsor = winsorized_mean(x, 0.0, 0.95)
    median = float(np.median(x))
    assert winsor > 0 and median > 0
    assert winsor >= median


def _fake_tess(tile_areas, tile_polygons):
    """Tessellation whose `Tile.area` values are decoupled from
    `Tile.polygon.area` — lets the test prove which field drives sizing."""
    tiles = tuple(
        Tile(
            tile_id=i, polygon=poly, point_indices=np.zeros(0, dtype=np.int64),
            n_spots=0, area=float(a), tessellation_id="t0", domain_region_id=0,
        )
        for i, (a, poly) in enumerate(zip(tile_areas, tile_polygons))
    )
    return Tessellation(
        tessellation_id="t0", component_id="c0", tiles=tiles,
        method="voronoi", tile_min_points=1, fraction=1.0,
    )


def test_hex_sizing_reads_canonical_tile_and_region_areas():
    """`compute_optimal_hex_radius` reads `Tile.area` + `region_polygon.area`
    directly — canvas precision is not in the signature, and mismatching
    `Tile.polygon.area` vs recorded `Tile.area` proves the recorded area
    is the authority."""
    assert "canvas" not in inspect.signature(compute_optimal_hex_radius).parameters

    config = load_run_config(
        profile="standard",
        set_overrides={
        'output_dir': 'tests/tests_output',
        'tessellation': {
            'n': 1,
        },
        'consensus': {
            'min_tess_coverage_fraction': 1.0,
        },
    },
    )
    # Tile polygons are 1×1 unit squares (polygon.area = 1.0 each), but
    # their canonical Tile.area is set to 100.0 — a mismatch that would
    # not survive a re-snap.
    polys = [box(0, 0, 1, 1), box(2, 0, 3, 1)]
    recorded_area = 100.0
    tess = _fake_tess([recorded_area, recorded_area], polys)
    region = box(-10, -10, 10, 10)          # region.area = 400
    r_via_recorded = compute_optimal_hex_radius(
        tessellation_ensemble={0: tess},
        region_polygon=region, n_roi_spots=1000, config=config,
    )
    # A comparison run where the recorded areas MATCH the polygon areas
    # (both = 1.0). If the primitive had re-snapped polygon.area it would
    # match the second run, not the first.
    tess_matched = _fake_tess([1.0, 1.0], polys)
    r_via_polygon = compute_optimal_hex_radius(
        tessellation_ensemble={0: tess_matched},
        region_polygon=region, n_roi_spots=1000, config=config,
    )
    assert r_via_recorded != r_via_polygon, (
        "hex radius should differ when recorded Tile.area differs from "
        "polygon.area — but they matched, suggesting a hidden re-snap."
    )

    # region_polygon.area drives A_roi: doubling the region area shifts
    # A_required / cap and therefore r_hex (unless A_base dominates end-to-end).
    r_small_region = compute_optimal_hex_radius(
        tessellation_ensemble={0: tess},
        region_polygon=box(0, 0, 5, 5), n_roi_spots=1000, config=config,
    )
    assert r_small_region != r_via_recorded
