"""GEOS Voronoi engine: Shapely voronoi_diagram + precision model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union
import logging
import numpy as np
import sklearn.neighbors

from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point
from shapely.ops import voronoi_diagram, unary_union
from shapely.validation import make_valid
from shapely import set_precision
from tilescope.tessellation.clipping import ordered_clips
from tilescope.tessellation.geometry import UNASSIGNED, assign_points_to_first_covering_polygon

logger = logging.getLogger(__name__)


@dataclass
class VoronoiBuildResult:
    """GEOS Voronoi build output. `precision_grid` is the engine's
    `set_precision` value; downstream containment / area tolerances
    must derive from it so they mirror the engine's snap."""
    tiles: List[Union[Polygon, MultiPolygon]]
    seed_index: List[int]
    precision_grid: float

    def assign_points_geos(self, points_xy: np.ndarray, seeds_xy: np.ndarray) -> np.ndarray:
        """Assign points to tiles via the shared first-covering primitive,
        then KDTree nearest-seed fallback for any uncovered points."""
        assignments = assign_points_to_first_covering_polygon(
            self.tiles, points_xy,
        ).astype(int, copy=True)

        uncovered_mask = assignments == UNASSIGNED
        if not uncovered_mask.any():
            return assignments

        try:
            kd = sklearn.neighbors.KDTree(seeds_xy[np.asarray(self.seed_index)])
        except Exception as e:
            logger.warning(
                "[SKLEARN] KDTree construction failed (%s); uncovered points "
                "remain unassigned: %s", type(e).__name__, e,
            )
            return assignments

        uncovered_idx = np.where(uncovered_mask)[0]
        _, nearest = kd.query(points_xy[uncovered_idx], k=1)
        assignments[uncovered_idx] = nearest[:, 0].astype(int)
        return assignments


@dataclass
class GeosVoronoiConfig:
    """GEOS Voronoi engine configuration."""
    precision_factor_d1: float = 1e-4
    envelope_pad_d1: float = 2.0
    enforce_seed_in_tile: bool = True


@dataclass(frozen=True)
class PreparedVoronoiRegion:
    """Snapped region and padded envelope shared by all its partitions."""
    polygon: Union[Polygon, MultiPolygon]
    envelope: Polygon
    precision_grid: float
    envelope_pad: float


class GeosVoronoiEngine:
    """Voronoi tessellation using GEOS (Shapely) with precision model.

    Applies set_precision to ROI and Voronoi cells before intersection to
    prevent hairline seams from floating-point overlay; produces a watertight
    partition with no epsilon buffering needed for assignment.
    """

    def __init__(self, cfg: GeosVoronoiConfig):
        self.cfg = cfg

    def prepare_region(self, region_polygon, *, d1: float) -> PreparedVoronoiRegion:
        precision_grid = max(self.cfg.precision_factor_d1 * float(d1), 1e-9)
        roi_prec = set_precision(make_valid(region_polygon), precision_grid)
        pad = self.cfg.envelope_pad_d1 * d1
        envelope = set_precision(roi_prec.envelope.buffer(pad), precision_grid)
        return PreparedVoronoiRegion(roi_prec, envelope, precision_grid, pad)

    def build(self,
              region_polygon: Union[Polygon, MultiPolygon],
              seeds_xy: np.ndarray,
              *, d1: float,
              prepared_region: PreparedVoronoiRegion | None = None) -> VoronoiBuildResult:
        """Build GEOS Voronoi tessellation with precision model. `d1` is
        supplied by `compute_tessellation` (single d1 authority)."""
        if prepared_region is None:
            prepared_region = self.prepare_region(region_polygon, d1=d1)
        elif (prepared_region.precision_grid != max(self.cfg.precision_factor_d1 * float(d1), 1e-9)
              or prepared_region.envelope_pad != self.cfg.envelope_pad_d1 * d1):
            raise ValueError("Prepared region uses different precision or envelope settings")
        precision_grid = prepared_region.precision_grid
        roi_prec = prepared_region.polygon
        envelope_prec = prepared_region.envelope

        sites = MultiPoint([tuple(s) for s in seeds_xy])
        vd = voronoi_diagram(sites, envelope=envelope_prec, edges=False)
        vd_prec = set_precision(make_valid(vd), precision_grid)

        try:
            kd = sklearn.neighbors.KDTree(seeds_xy)
        except Exception as e:
            logger.warning(
                "[SKLEARN] KDTree construction failed (%s); using centroid labeling only: %s",
                type(e).__name__, e
            )
            kd = None

        tiles_raw: List[Union[Polygon, MultiPolygon]] = []
        representative_xy = []

        def clip_cell(cell):
            cell = set_precision(make_valid(cell), precision_grid)
            if cell.is_empty:
                return None

            clipped = set_precision(make_valid(cell.intersection(roi_prec)), precision_grid)
            if clipped.is_empty or clipped.area < 1e-12:
                return None

            rp = clipped.representative_point()
            return clipped, (rp.x, rp.y)

        with ordered_clips(clip_cell, vd_prec.geoms) as clipped_cells:
            for result in clipped_cells:
                if result is not None:
                    clipped, xy = result
                    tiles_raw.append(clipped)
                    representative_xy.append(xy)

        tiles_merged: List[Union[Polygon, MultiPolygon]] = []
        seed_index_merged: List[int] = []

        if tiles_raw:
            if kd is not None:
                _, idx = kd.query(np.asarray(representative_xy), k=1)
                seed_index_raw = idx[:, 0].tolist()
            else:
                seed_index_raw = [
                    int(np.argmin(np.sum((seeds_xy - np.asarray(xy))**2, axis=1)))
                    for xy in representative_xy
                ]
            by_seed = {}
            for poly, j in zip(tiles_raw, seed_index_raw):
                by_seed.setdefault(j, []).append(poly)

            for j, polys in by_seed.items():
                merged_poly = set_precision(make_valid(unary_union(polys)), precision_grid)
                if merged_poly.is_empty or merged_poly.area < 1e-12:
                    continue
                tiles_merged.append(merged_poly)
                seed_index_merged.append(j)

        if self.cfg.enforce_seed_in_tile:
            tiles_fixed = []
            for poly, j in zip(tiles_merged, seed_index_merged):
                if poly.is_empty:
                    tiles_fixed.append(poly)
                    continue

                seed_pt = Point(*seeds_xy[j])
                if not poly.covers(seed_pt):
                    poly = set_precision(make_valid(poly.buffer(precision_grid)), precision_grid)

                tiles_fixed.append(poly)

            tiles_merged = tiles_fixed

        return VoronoiBuildResult(
            tiles=tiles_merged,
            seed_index=seed_index_merged,
            precision_grid=float(precision_grid),
        )
