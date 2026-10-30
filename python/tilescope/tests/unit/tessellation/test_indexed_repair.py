"""Indexed search must retain the previous exhaustive merge decisions exactly."""
from copy import deepcopy
import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

from tilescope.tessellation import repair


class ExhaustiveCandidates:
    """Test-only oracle: visit all tile indices as the retired search did."""
    def __init__(self, boundaries):
        self.n = len(boundaries)

    def query(self, boundary):
        return np.arange(self.n)


def tiles_from_polygons(polygons):
    return [dict(polygon=p, area=p.area, point_indices=np.array([i]), n_spots=1)
            for i, p in enumerate(polygons)]


def signature(result):
    tiles, count = result
    return count, [(t['polygon'].wkb, t['point_indices'].tobytes(), t['n_spots'], t['area'])
                   for t in tiles]


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('update_area', [False, True])
def test_indexed_search_matches_exhaustive_chain_merges(monkeypatch, seed, update_area):
    rng = np.random.default_rng(seed)
    polygons = [box(x, y, x+1, y+1) for y in range(8) for x in range(8)]
    # Holes, disconnected tiles, corner contacts, empty tiles and seed fallback.
    polygons += [box(50, 50, 51, 51), Polygon(),
                 Polygon([(60,60),(65,60),(65,65),(60,65)],
                         holes=[[(61,61),(62,61),(62,62),(61,62)]])]
    tiles = tiles_from_polygons(polygons)
    seeds = np.array([(i % 8 + .5, i // 8 + .5) for i in range(len(tiles))])
    order = rng.permutation(len(tiles))[:45].tolist()
    kwargs = dict(indices_to_merge=order, seeds=seeds, update_area=update_area)
    indexed = repair._merge_tiles_generic(deepcopy(tiles), **kwargs)
    monkeypatch.setattr(repair, 'STRtree', ExhaustiveCandidates)
    exhaustive = repair._merge_tiles_generic(deepcopy(tiles), **kwargs)
    assert signature(indexed) == signature(exhaustive)
    for tile in indexed[0]:
        assert tile['n_spots'] == len(tile['point_indices'])
        assert np.all(np.diff(tile['point_indices']) > 0)


@pytest.mark.parametrize('tolerance', [0, 1e-9, 2.0])
def test_ties_and_nonstandard_tolerance(monkeypatch, tolerance):
    tiles = tiles_from_polygons([box(x, 0, x+1, 1) for x in range(5)])
    kwargs = dict(indices_to_merge=[2, 1, 3], seeds=None, edge_tol=tolerance)
    indexed = repair._merge_tiles_generic(deepcopy(tiles), **kwargs)
    monkeypatch.setattr(repair, 'STRtree', ExhaustiveCandidates)
    exhaustive = repair._merge_tiles_generic(deepcopy(tiles), **kwargs)
    assert signature(indexed) == signature(exhaustive)


def test_index_prunes_distant_tiles(monkeypatch):
    sizes = []
    class CountingTree(STRtree):
        def query(self, boundary):
            found = super().query(boundary)
            sizes.append(len(found))
            return found
    monkeypatch.setattr(repair, 'STRtree', CountingTree)
    tiles = tiles_from_polygons([box(x, y, x+1, y+1) for x in range(30) for y in range(30)])
    repair._merge_tiles_generic(tiles, list(range(0, 900, 13)))
    assert sizes and max(sizes) <= 9
    assert sum(sizes) < 900 * len(sizes) / 50


@pytest.mark.parametrize('tolerance', [-1e-9, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('entry', ['generic', 'sliver', 'small'])
def test_invalid_tolerance_rejected_even_for_empty_inputs(tolerance, entry):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        if entry == 'generic':
            repair._merge_tiles_generic([], [], edge_tol=tolerance)
        elif entry == 'sliver':
            repair.fix_degenerate_tiles([], np.empty((0, 2)), d1=1, edge_tol=tolerance)
        else:
            repair.merge_small_tiles_geos([], 3, edge_tol=tolerance)
