"""Indexed checks retain exhaustive coverage, ordering and graph membership."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial import component_domains as domains


def frame(rows):
    result = pd.DataFrame(rows, columns=[COMPONENT_ID, COORD_X, COORD_Y])
    result[ROI_ID] = result[COMPONENT_ID]
    result[SOURCE_ROW_ID] = np.arange(len(result), dtype=np.int64) + 100
    return result


def diagnostics(polygons, data):
    result = []
    for cid, geometry in polygons.items():
        own = data.loc[data[COMPONENT_ID].eq(cid)]
        covered = sum(geometry.covers(Point(x, y)) for x, y in own[[COORD_X, COORD_Y]].to_numpy())
        result.append(domains.ComponentDomainDiagnostic(
            cid, cid, len(own), covered, 0, 0, geometry.area, 0., 0., geometry.area, 'ok',
        ))
    return result


def validate(polygons, data, diags=None):
    return domains._validate_component_domains(
        polygons, data, diagnostics(polygons, data) if diags is None else diags,
        [], area_tol=1e-9,
    )


def test_holes_multipart_boundaries_and_empty_domains_match_scalar_coverage():
    shell = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)],
                    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
    polygons = {'A': MultiPolygon([shell, box(10, 0, 11, 1)]), 'B': Polygon()}
    data = frame([('A', 0, 0), ('A', 1, 1.5), ('A', 1.5, 1.5),
                  ('A', 10.5, .5), ('A', 6, 6), ('B', 20, 20)])
    validate(polygons, data)


@pytest.mark.parametrize('x,y', [(0, 0), (1, .5), (.5, .5)])
def test_foreign_interior_and_boundary_cells_are_rejected(x, y):
    polygons = {'A': box(0, 0, 1, 1), 'B': box(10, 0, 11, 1)}
    data = frame([('A', .25, .25), ('B', x, y)])
    with pytest.raises(AssertionError, match="invariant 4.*src_id=101.*foreign component 'A'"):
        validate(polygons, data)


def test_foreign_failure_order_is_component_then_input_row_across_batches():
    polygons = {'Z': box(0, 0, 1, 1), 'A': box(10, 0, 11, 1), 'B': box(20, 0, 21, 1)}
    data = frame([('B', 10.5, .5)] + [('B', 20.5, .5)] * 4100 + [('B', .5, .5)])
    with pytest.raises(AssertionError, match="invariant 4.*src_id=4201.*foreign component 'Z'"):
        validate(polygons, data)


def test_own_coverage_failure_precedes_foreign_coverage_failure():
    polygons = {'A': box(0, 0, 1, 1), 'B': box(10, 0, 11, 1)}
    data = frame([('A', .5, .5), ('B', .5, .5)])
    diags = diagnostics(polygons, data)
    diags[0] = replace(diags[0], n_spots_in_domain=0)
    with pytest.raises(AssertionError, match='invariant 3.*assignment count'):
        validate(polygons, data, diags)


def test_point_work_is_batched_once_not_repeated_for_each_domain(monkeypatch):
    data = frame([(str(i % 20), 10 * (i % 20) + .5, .5) for i in range(10000)])
    polygons = {str(i): box(i * 10, 0, i * 10 + 1, 1) for i in range(20)}
    diags = diagnostics(polygons, data)
    sizes = []
    original = domains.points
    def bounded_points(values):
        sizes.append(len(values))
        return original(values)
    monkeypatch.setattr(domains, 'points', bounded_points)
    validate(polygons, data, diags)
    assert sizes == [4096, 4096, 1808]


def test_nonoverlapping_polygon_pairs_do_not_call_intersection(monkeypatch):
    polygons = {str(i): box(i * 10, 0, i * 10 + 1, 1) for i in range(60)}
    def forbidden(*args, **kwargs):
        pytest.fail('Disjoint bounding boxes must not require an exact intersection')
    monkeypatch.setattr(BaseGeometry, 'intersection', forbidden)
    validate(polygons, frame([]))


def test_overlap_first_failure_keeps_sorted_component_order():
    polygons = {'Z': box(.5, 0, 2, 1), 'B': box(.25, 0, 1.5, 1), 'A': box(0, 0, 1, 1)}
    with pytest.raises(AssertionError, match="invariant 5.*'A' and 'B'"):
        validate(polygons, frame([]))


def test_touching_polygons_are_not_material_overlap():
    validate({'A': box(0, 0, 1, 1), 'B': box(1, 0, 2, 1)}, frame([]))


def test_component_edge_lookup_matches_exhaustive_edges_for_reordered_subsets():
    rng = np.random.default_rng(35)
    n = 100
    edges = sorted({tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(240)})
    neighbours = [[] for _ in range(n)]
    for left, right in edges:
        neighbours[left].append(right)
        neighbours[right].append(left)
    ids = rng.permutation(n) + 1000
    adjacency = SimpleNamespace(source_spot_table=SimpleNamespace(spot_id=tuple(map(str, ids))),
                                neighbors=tuple(tuple(sorted(v)) for v in neighbours))
    lookup = domains._ComponentEdgeLookup(adjacency)
    for size in [0, 1, 5, 25, n]:
        selected = rng.choice(n, size, replace=False)
        local = {int(global_id): i for i, global_id in enumerate(selected)}
        expected = {tuple(sorted((local[a], local[b]))) for a, b in edges if a in local and b in local}
        assert lookup.induced_edges(ids[selected]) == expected
    with pytest.raises(ValueError, match='absent.*9999'):
        lookup.induced_edges(np.array([9999]))


def test_component_edge_lookup_visits_only_requested_vertices():
    class Neighbours:
        def __init__(self): self.visits = []
        def __getitem__(self, i):
            self.visits.append(i)
            return {0: (1,), 1: (0, 2), 2: (1,)}.get(i, ())
    neighbours = Neighbours()
    adjacency = SimpleNamespace(source_spot_table=SimpleNamespace(spot_id=tuple(map(str, range(100)))),
                                neighbors=neighbours)
    lookup = domains._ComponentEdgeLookup(adjacency)
    assert lookup.induced_edges(np.array([1, 0])) == {(0, 1)}
    assert neighbours.visits == [1, 0]
