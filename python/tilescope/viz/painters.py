"""Vector-based painters for spot / polygon visualization (world coordinates).

- SpotSplatPainter: filled circles tied to local sampling scale.
- draw_polygon_boundaries: stroked ring outlines from any mapping of ids
  to Shapely Polygon / MultiPolygon (component-grain, ROI-grain, etc.).
- add_identity_header: two-line figure heading with A/B identity mapping;
  callers pass ``feature_a_id``/``feature_b_id`` from the pair-authority
  dict (see VisualizationRunner). Empty or generic identities are rejected.
- _shapely_to_mpl_path: single filled-with-holes matplotlib Path constructor
  (also consumed by viz/geometry_helpers.patch_from_polygon).
"""

import logging
from typing import Mapping, Optional, Union

import matplotlib.colors as _mcolors
import matplotlib.pyplot as _plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from shapely.geometry import MultiPolygon, Polygon

from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.spatial.spacing import median_nn_distance


PEARSON_CMAP = "RdBu_r"
PEARSON_NORM = _mcolors.Normalize(vmin=-1.0, vmax=1.0)


def pearson_scalar_mappable():
    """Shared ScalarMappable for Pearson r in [-1, +1] on the RdBu_r cmap.
    Used by consensus ROI-correlation panel and expression correlation
    panels so a single palette + norm authority governs both PDFs."""
    sm = _plt.cm.ScalarMappable(cmap=PEARSON_CMAP, norm=PEARSON_NORM)
    sm.set_array([])
    return sm


def pearson_face_color(r: float):
    """RdBu_r color for a scalar Pearson r; normalized to [-1, +1]."""
    return _plt.get_cmap(PEARSON_CMAP)(PEARSON_NORM(float(r)))


logger = logging.getLogger(__name__)


def _ring_to_verts_codes(ring):
    """Return (verts_list, codes_list) for a Shapely LinearRing as one MOVETO -> LINETO*
    -> CLOSEPOLY sub-path. Single source of truth for ring↔matplotlib conversion."""
    xy = np.asarray(ring.coords, float)
    verts = [(float(x), float(y)) for x, y in xy]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return verts, codes


def _shapely_to_mpl_path(poly) -> MplPath:
    """Convert Shapely Polygon/MultiPolygon into one filled matplotlib Path with holes."""
    if poly.geom_type == "Polygon":
        polys = [poly]
    elif poly.geom_type == "MultiPolygon":
        polys = [g for g in poly.geoms if not g.is_empty]
    else:
        polys = []

    verts_all, codes_all = [], []
    for g in polys:
        v, c = _ring_to_verts_codes(g.exterior)
        verts_all.extend(v); codes_all.extend(c)
        for hole in g.interiors:
            v, c = _ring_to_verts_codes(hole)
            verts_all.extend(v); codes_all.extend(c)

    return MplPath(verts_all, codes_all)


def draw_polygon_boundaries(
    ax: Axes,
    polygons: Optional[Mapping[str, Union[Polygon, MultiPolygon]]],
    *,
    color: str = "#FF6700",
    linewidth: float = 1.2,
    alpha: float = 1.0,
    include_interiors: bool = True,
) -> None:
    """Stroke polygon exteriors and, when requested, interior rings."""
    if polygons is None:
        return
    for _pid, polygon in polygons.items():
        polys = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
        for poly in polys:
            _stroke_ring(ax, poly.exterior, color, linewidth, alpha)
            if include_interiors:
                for interior in poly.interiors:
                    _stroke_ring(ax, interior, color, linewidth, alpha)


def _stroke_ring(ax: Axes, ring, color: str, linewidth: float, alpha: float) -> None:
    """Stroke a single ring as a non-filled PathPatch."""
    try:
        verts, codes = _ring_to_verts_codes(ring)
    except (TypeError, ValueError) as e:
        logger.error("[COMPONENT_OVERLAY] Invalid ring coordinates: %s", e)
        raise
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor='none', edgecolor=color,
        linewidth=linewidth, alpha=alpha, antialiased=True,
    )
    ax.add_patch(patch)


def add_identity_header(fig, feature_a_id, feature_b_id, heading: str) -> None:
    """Two-line suptitle: representation heading + A/B identity mapping.
    Wraps long identities rather than silently truncating them."""
    a_id, b_id = require_semantic_feature_ids(
        feature_a_id, feature_b_id, owner="add_identity_header",
    )
    fig.suptitle(
        f"{heading}\nA = {a_id}    B = {b_id}",
        fontsize=14, wrap=True,
    )


class SpotSplatPainter:
    """Filled circles tied to local sampling scale (default, universal).

    Renders each spot as a filled circle with radius proportional to
    median NN distance. True to data (no interpolation), vector-based.
    """

    def __init__(self, radius_scale: float = 0.45):
        # 0.45 leaves a ~10% inter-spot gap so overlay rings stay visible.
        self.radius_scale = radius_scale

    def draw(self, ax, points_xy, values, component_polygon=None, *, vmin=None, vmax=None, cmap, canvas, median_nn_override=None, norm=None):
        """Draw spots as filled circles, optionally clipped to `component_polygon`.

        When `component_polygon` is None splats render unclipped: used
        by the spot/cell expression view where boundaries are drawn
        separately as line overlays.
        """
        if len(points_xy) < 2:
            return

        x = points_xy[:, 0]
        y = points_xy[:, 1]

        median_nn = median_nn_override if median_nn_override is not None \
                    else median_nn_distance(points_xy)
        radius = self.radius_scale * median_nn

        fig = ax.figure
        x0, y0 = x[0], y[0]
        p1 = ax.transData.transform([[x0, y0], [x0 + radius, y0]])
        dx_px = np.abs(p1[1, 0] - p1[0, 0])
        dx_pt = dx_px * 72.0 / fig.dpi
        s = (dx_pt * 2) ** 2

        scatter_kw = dict(
            c=values, s=s, cmap=cmap,
            marker='o', linewidths=0, edgecolors='none', rasterized=False,
        )
        if norm is not None:
            scatter_kw['norm'] = norm
        else:
            scatter_kw['vmin'] = vmin
            scatter_kw['vmax'] = vmax
        collection = ax.scatter(x, y, **scatter_kw)

        if component_polygon is not None:
            roi_path = _shapely_to_mpl_path(component_polygon)
            clip_patch = PathPatch(roi_path, transform=ax.transData, linewidth=0)
            collection.set_clip_path(clip_patch)

        return collection
