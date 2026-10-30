"""Batched geometry preserves scalar assignment and polygon serialization."""
import sklearn.neighbors
from tilescope.tessellation import engine_geos
from shapely.geometry import GeometryCollection

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box
from sklearn.neighbors import KDTree

from tilescope.tessellation import geometry
from tilescope.tessellation.engine_geos import GeosVoronoiConfig, GeosVoronoiEngine


@pytest.mark.parametrize("batch_size", [1, 7, 4096])
def test_batched_covering_matches_scalar_with_holes_and_ties(monkeypatch, batch_size):
    monkeypatch.setattr(geometry, "_POINT_QUERY_BATCH_SIZE", batch_size)
    holed = box(0, 0, 4, 4).difference(box(1, 1, 3, 3))
    polygons = [Polygon(), holed, box(2, 0, 6, 4),
                MultiPolygon([box(7, 0, 8, 1), box(7, 2, 8, 3)])]
    random_xy = np.random.default_rng(12).uniform([-1, -1], [9, 5], (4200, 2))
    xy = np.vstack([random_xy, [[0, 0], [1, 2], [2, 2], [4, 4], [7, 1], [7.5, 1.5]]])
    for ordered in [polygons, polygons[::-1]]:
        expected = [next((j for j, poly in enumerate(ordered) if poly.covers(Point(p))), -1)
                    for p in xy]
        np.testing.assert_array_equal(
            geometry.assign_points_to_first_covering_polygon(ordered, xy), expected,
        )


def test_batched_covering_empty_inputs():
    assert geometry.assign_points_to_first_covering_polygon([box(0, 0, 1, 1)], np.empty((0, 2))).size == 0
    np.testing.assert_array_equal(geometry.assign_points_to_first_covering_polygon([], np.zeros((3, 2))), [-1] * 3)


def _assert_same_build(a, b):
    assert a.seed_index == b.seed_index
    assert a.precision_grid == b.precision_grid
    assert [p.wkb for p in a.tiles] == [p.wkb for p in b.tiles]


def test_batched_seed_queries_match_scalar_including_ties(monkeypatch):

    seeds = np.asarray([[0., 1.], [2., 1.], [0., 3.], [2., 3.]])
    region = box(-1, -1, 3, 5)
    # Representative points lie exactly on seed bisectors.
    cells = GeometryCollection([box(.5, .5, 1.5, 1.5), box(.5, 2.5, 1.5, 3.5)])
    monkeypatch.setattr(engine_geos, "voronoi_diagram", lambda *a, **k: cells)
    engine = GeosVoronoiEngine(GeosVoronoiConfig(enforce_seed_in_tile=False))
    batched = engine.build(region, seeds, d1=1.)
    query_sizes = []

    class ScalarQueries:
        def __init__(self, xy):
            self.tree = KDTree(xy)

        def query(self, xy, k):
            query_sizes.append(len(xy))
            results = [self.tree.query([p], k=k) for p in xy]
            return np.vstack([r[0] for r in results]), np.vstack([r[1] for r in results])

    monkeypatch.setattr(sklearn.neighbors, "KDTree", ScalarQueries)
    scalar = engine.build(region, seeds, d1=1.)
    assert query_sizes == [2]
    _assert_same_build(batched, scalar)


@pytest.mark.parametrize("multipart", [False, True])
def test_prepared_region_preserves_exact_tiles_across_partitions(multipart):
    region = box(0, 0, 10, 10).difference(box(3, 3, 7, 7))
    if multipart:
        region = MultiPolygon([region, box(12, 0, 14, 10)])
    engine = GeosVoronoiEngine(GeosVoronoiConfig())
    prepared = engine.prepare_region(region, d1=.75)
    for seed in [0, 1, 2]:
        xy = np.random.default_rng(seed).uniform([0, 0], [14 if multipart else 10, 10], (30, 2))
        xy = np.asarray([p for p in xy if region.covers(Point(p))])
        _assert_same_build(
            engine.build(region, xy, d1=.75),
            engine.build(region, xy, d1=.75, prepared_region=prepared),
        )


def test_prepared_region_rejects_different_precision_settings():
    engine = GeosVoronoiEngine(GeosVoronoiConfig())
    region = box(0, 0, 4, 4)
    prepared = engine.prepare_region(region, d1=1.)
    with pytest.raises(ValueError, match="different precision"):
        engine.build(region, np.asarray([[1., 1.], [3., 3.]]), d1=2., prepared_region=prepared)


def test_seed_fallback_when_kdtree_construction_fails_preserves_tiles(monkeypatch):
    engine = GeosVoronoiEngine(GeosVoronoiConfig())
    region = box(0, 0, 10, 10)
    xy = np.asarray([[1., 1.], [2., 8.], [8., 3.]])
    expected = engine.build(region, xy, d1=1.)

    def construction_error(*a, **k):
        raise ValueError("test construction failure")

    monkeypatch.setattr(sklearn.neighbors, "KDTree", construction_error)
    _assert_same_build(expected, engine.build(region, xy, d1=1.))
