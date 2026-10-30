"""
Geometry utilities for tessellation - pure Shapely operations.
Extracted from .old/roi_utils.py with no dependencies on ROI class.
"""
from shapely.strtree import STRtree


from typing import Sequence, Union

import numpy as np
from shapely import points
from shapely.geometry import MultiPolygon, Polygon


UNASSIGNED = -1
_POINT_QUERY_BATCH_SIZE = 4096


def assign_points_to_first_covering_polygon(
    polygons: Sequence[Union[Polygon, MultiPolygon]],
    coordinates: np.ndarray,
) -> np.ndarray:
    """Canonical one-to-one point -> polygon assignment.

    Returns an integer array of length `len(coordinates)`. For each
    point, entry i is the index of the FIRST polygon (in the given
    order) that covers it, or ``UNASSIGNED`` (-1) if no polygon does.

    Shapely `covers` is boundary-inclusive, so a point lying on a
    shared edge between two polygons is a real event. Deterministic
    first-wins tie resolution guarantees no point is assigned to
    more than one polygon; the caller controls the canonical order.
    Uncovered points remain explicitly unassigned: no reassignment.

    Query points in bounded batches. ``covered_by`` includes polygon
    boundaries; taking the minimum matching polygon index preserves
    first-wins ordering independently of STRtree traversal order.
    """

    n_points = int(len(coordinates))
    assignment = np.full(n_points, UNASSIGNED, dtype=np.int64)
    if n_points == 0 or not polygons:
        return assignment
    tree = STRtree(list(polygons))
    points_xy = np.asarray(coordinates, dtype=float)
    for start in range(0, n_points, _POINT_QUERY_BATCH_SIZE):
        stop = min(start + _POINT_QUERY_BATCH_SIZE, n_points)
        point_ids, polygon_ids = tree.query(
            points(points_xy[start:stop]), predicate="covered_by",
        )
        first = np.full(stop - start, len(polygons), dtype=np.int64)
        np.minimum.at(first, point_ids, polygon_ids)
        first[first == len(polygons)] = UNASSIGNED
        assignment[start:stop] = first
    return assignment
