"""
Spatial coordinate handling and transformations.

Canvas-based pipeline for continuous float coordinates (ℝ²).
"""

from typing import Dict, Union

from shapely.geometry import MultiPolygon, Polygon

from tilescope.spatial.canvas import Canvas, make_canvas
from tilescope.spatial.hex import hex_radius_from_area
from tilescope.spatial.precision import precision_grid_size

# `ComponentPolygon` is the final clipped polygonal extent of one
# component: a Shapely `Polygon` or `MultiPolygon` carrying no spot
# / tile / diagnostic payload. `ComponentPolygonDict` is the
# component_id-keyed dict of such geometries. Distinct from
# `ComponentDomainPartition` in `spatial.component_domains`, which
# is the typed partitioning record derived from the polygon.
ComponentPolygon = Union[Polygon, MultiPolygon]
ComponentPolygonDict = Dict[str, ComponentPolygon]

__all__ = [
    'Canvas',
    'ComponentPolygon',
    'ComponentPolygonDict',
    'hex_radius_from_area',
    'make_canvas',
    'precision_grid_size',
]
