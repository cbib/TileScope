"""
Geometry validation and cleaning helpers for visualization debugging.
Diagnoses gruyère holes by classifying drop reasons and cleaning geometries.
"""

import logging
from typing import Dict, Mapping, Optional

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely import set_precision, make_valid
from shapely.ops import unary_union

from tilescope.viz.painters import _ring_to_verts_codes

log = logging.getLogger(__name__)


def summarize_geom(g) -> str:
    """Log-friendly geometry summary."""
    if g is None or not hasattr(g, "geom_type"):
        return f"{type(g)}"
    a = getattr(g, "area", float("nan"))
    b = getattr(g, "bounds", (np.nan,) * 4)
    return (
        f"{g.geom_type}, area={a:.4g}, "
        f"bounds=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}), "
        f"valid={g.is_valid}"
    )


def classify_drop(g, min_area: float) -> str:
    """Classify why a geometry can't be drawn (reason codes)."""
    if g is None:
        return "none"
    if g.is_empty:
        return "empty"
    if not g.is_valid:
        return "invalid"
    if g.area <= min_area:
        return "too_small"
    return "ok"


def clean_if_needed(g, precision=None, simplify_tol: float = 0.0):
    """Clean geometry only if invalid. Returns (geometry, was_fixed, reason)."""
    if g is None or g.is_empty:
        return g, False, "empty"

    # Fast path: already valid
    if g.is_valid and g.area > 0:
        return g, False, "ok"

    # Slow path: fix invalid geometry
    gg = make_valid(g)
    if precision is not None:
        gg = set_precision(gg, grid_size=precision)
    if simplify_tol > 0.0:
        gg = gg.simplify(simplify_tol, preserve_topology=True)
    gg = gg.buffer(0)  # finalize

    reason = "fixed" if (not g.is_valid and gg.is_valid) else ("ok" if gg.is_valid else "invalid")
    return gg, True, reason


def iter_polygons(g):
    """Yield individual polygons from Polygon or MultiPolygon.

    GeometryCollection should never reach here - it must be filtered at repair boundaries.
    """
    if g is None or g.is_empty:
        return
    if g.geom_type == "Polygon":
        yield g
    elif g.geom_type == "MultiPolygon":
        yield from g.geoms
    else:
        raise TypeError(f"Unexpected geometry type {g.geom_type} - GeometryCollection should be filtered at repair boundaries")


def patch_from_polygon(poly, **kwargs) -> PathPatch:
    """PathPatch from a Shapely polygon with proper hole rendering."""
    verts_all, codes_all = [], []
    v, c = _ring_to_verts_codes(poly.exterior)
    verts_all.extend(v); codes_all.extend(c)
    for interior in poly.interiors:
        v, c = _ring_to_verts_codes(interior)
        verts_all.extend(v); codes_all.extend(c)
    return PathPatch(Path(verts_all, codes_all), **kwargs)


def dissolved_roi_polygons(
    component_polygons: Optional[Mapping[str, object]],
    component_to_roi: Optional[Mapping[str, str]],
) -> Dict[str, object]:
    """{roi_id: unary_union(component_polygons for that ROI)}.

    Single authority for the ROI-dissolution step used by both the
    consensus ROI-grain panels and the expression-summary constructed
    correlation panel. Missing/empty polygons are skipped."""
    grouped: Dict[str, list] = {}
    for cid, roi_id in (component_to_roi or {}).items():
        poly = (component_polygons or {}).get(str(cid))
        if poly is None or poly.is_empty:
            continue
        grouped.setdefault(str(roi_id), []).append(poly)
    return {
        roi: unary_union(polys) if len(polys) > 1 else polys[0]
        for roi, polys in grouped.items()
    }


def paint_roi_value(
    ax, polygon, value: Optional[float], *,
    scalar_mappable, undefined_fill, undefined_edge, undefined_edge_linewidth,
    finite_edge: str = "black", finite_linewidth: float = 0.3,
) -> bool:
    """Paint a dissolved ROI polygon with `value` via the mappable, or
    with the undefined-support style when `value` is None/NaN. Returns
    True when the undefined style was used so the caller can gate the
    figure-level availability legend entry."""
    if value is not None and np.isfinite(value):
        face = scalar_mappable.to_rgba(float(value))
        edge, lw = finite_edge, finite_linewidth
        is_undefined = False
    else:
        face = undefined_fill
        edge, lw = undefined_edge, undefined_edge_linewidth
        is_undefined = True
    for part in iter_polygons(polygon):
        ax.add_patch(patch_from_polygon(
            part, facecolor=face, edgecolor=edge, linewidth=lw,
        ))
    return is_undefined
