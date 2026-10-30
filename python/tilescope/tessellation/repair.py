"""Degenerate tile repair for Voronoi tessellations.

Handles zero-area tiles created by clipping Voronoi cells to concave ROI
boundaries (e.g. dumbbell necks); merges them into neighbors to keep a
complete spatial partition without gaps or invalid geometries.
"""
from shapely.errors import ShapelyError
from shapely import minimum_clearance


import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from shapely import make_valid, set_precision
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from tilescope.defaults import SLIVER_TAU_D1, SLIVER_TAU_MED

logger = logging.getLogger(__name__)


class NoPolygonalContentError(RuntimeError):
    """Raised when the clipped tile has no positive-area polygonal support.

    The upstream tessellator assigns spots to every emitted tile, so a reclip
    result with only points, line residue, or empty geometry is an integrity
    failure. `residue_summary` describes the non-areal content encountered.
    """

    def __init__(self, tile_id, residue_summary: str) -> None:
        super().__init__(
            f"Tile {tile_id!r} has no positive-area polygonal support after "
            f"reclip; residue={residue_summary}."
        )
        self.tile_id = tile_id
        self.residue_summary = residue_summary


def extract_polygonal_support(
    geometry,
) -> Tuple[Optional[object], str]:
    """Extract the positive-area polygonal support of a Shapely geometry.

    Returns `(support, residue_summary)`: `support` is a Polygon / MultiPolygon
    with strictly positive area or None; `residue_summary` describes the
    discarded non-areal content ("" when none).
    """
    if geometry is None or geometry.is_empty:
        return None, "empty"
    if isinstance(geometry, Polygon):
        if geometry.area > 0:
            return geometry, ""
        return None, f"polygon_zero_area (perimeter={geometry.length:.3g})"
    if isinstance(geometry, MultiPolygon):
        parts = [p for p in geometry.geoms if (not p.is_empty) and p.area > 0]
        if not parts:
            return None, "multipolygon_all_zero_area"
        if len(parts) == 1:
            return parts[0], ""
        return MultiPolygon(parts), ""
    if isinstance(geometry, GeometryCollection):
        polygonal = []
        residue_types: List[str] = []
        for sub in geometry.geoms:
            # Recurse so nested GeometryCollections still surface polygonal
            # content and accumulate non-areal residue.
            inner_support, inner_residue = extract_polygonal_support(sub)
            if inner_support is not None:
                polygonal.append(inner_support)
            if inner_residue.startswith("non_areal="):
                # Merge inner "non_areal={...}" set into the outer set.
                inner_types = inner_residue[len("non_areal={"):-1].split(",")
                residue_types.extend(t for t in inner_types if t)
        residue_summary = ""
        if residue_types:
            residue_summary = f"non_areal={{{','.join(sorted(set(residue_types)))}}}"
        if not polygonal:
            return None, residue_summary or "collection_no_polygonal_members"
        union = unary_union(polygonal)
        if isinstance(union, (Polygon, MultiPolygon)):
            return union, residue_summary
        # Rare: union collapsed further; recurse once.
        return extract_polygonal_support(union)
    # Non-areal geometry (Point, LineString, MultiPoint, ...). Emit the same
    # `non_areal={NAME}` format the collection branch uses so recursive callers
    # can merge nested residue types uniformly.
    return None, f"non_areal={{{type(geometry).__name__}}}"


def clip_to_region_and_refresh(
    tile: Dict,
    snapped_region: Polygon,
    precision_grid: float,
) -> Dict:
    """Reclip a tile against the snapped region and normalize polygonal support.

    Normalization: Polygon / MultiPolygon are kept as-is; GeometryCollection
    is reduced to the union of positive-area members with lower-dimensional
    residue recorded on `tile["non_areal_residue"]`; no positive-area content
    raises `NoPolygonalContentError`. Interior rings are preserved.
    `tile["area"]` is refreshed from the polygonal support's total area.
    """
    poly = tile["polygon"]
    poly = make_valid(poly)
    poly = set_precision(poly, precision_grid)
    clipped = poly.intersection(snapped_region)
    clipped = make_valid(clipped)
    clipped = set_precision(clipped, precision_grid)
    support, residue_summary = extract_polygonal_support(clipped)
    if support is None:
        raise NoPolygonalContentError(
            tile_id=tile.get("tile_id"),
            residue_summary=residue_summary,
        )
    tile["polygon"] = support
    tile["area"] = float(support.area)
    tile["non_areal_residue"] = residue_summary
    return tile


def _validate_edge_tolerance(edge_tol: float) -> None:
    if not np.isfinite(edge_tol) or edge_tol < 0:
        raise ValueError("edge_tol must be finite and nonnegative.")


def _merge_tiles_generic(
    tiles: List[Dict],
    indices_to_merge: List[int],
    seeds: Optional[np.ndarray] = None,
    *,
    edge_tol: float = 1e-9,
    get_seed_idx: Optional[callable] = None,
    update_area: bool = False,
) -> tuple:
    """Merge specified tiles into their best neighbors.

    Selection of which tiles to merge is the caller's responsibility.
    Returns `(merged_tiles, merged_count)`.
    """
    _validate_edge_tolerance(edge_tol)
    if len(tiles) == 0 or not indices_to_merge:
        return tiles, 0

    if get_seed_idx is None:
        get_seed_idx = lambda i: i

    boundaries = [t["polygon"].boundary if not t["polygon"].is_empty else None
                  for t in tiles]
    # Index the same frozen boundaries used by the merge selection rule.
    # Updating boundaries after a merge would change subsequent decisions.
    boundary_tree = STRtree(boundaries)

    def shared_edge_len(i: int, j: int) -> float:
        if i == j or tiles[i]["polygon"].is_empty or tiles[j]["polygon"].is_empty:
            return 0.0
        if boundaries[i] is None or boundaries[j] is None:
            return 0.0
        try:
            inter = boundaries[i].intersection(boundaries[j])
            return float(inter.length) if hasattr(inter, "length") else 0.0
        except Exception as e:

            if isinstance(e, ShapelyError):
                logger.debug(
                    "[GEOM] Boundary intersection failed for tiles %d,%d (%s): %s",
                    i, j, type(e).__name__, e
                )
            else:
                logger.warning(
                    "[GEOM] Unexpected error in boundary intersection tiles %d,%d (%s): %s",
                    i, j, type(e).__name__, e
                )
            return 0.0

    alive = [True] * len(tiles)
    merged_count = 0

    for i in indices_to_merge:
        if not alive[i] or tiles[i]["polygon"].is_empty:
            continue

        # Prefer the neighbor with the longest shared boundary.
        best_j, best_len = None, 0.0
        # STRtree query order is unspecified; restore the original ascending
        # tile order so equal shared edges retain the same tie-breaking.
        candidates = sorted(boundary_tree.query(boundaries[i]))
        for j in candidates:
            if i == j or not alive[j] or tiles[j]["polygon"].is_empty:
                continue
            L = shared_edge_len(i, j)
            if L > best_len + edge_tol:
                best_len, best_j = L, j

        # Fallback to nearest seed when no shared edge is found.
        if best_j is None and seeds is not None:
            seed_idx_i = get_seed_idx(i)
            if seed_idx_i < len(seeds):
                px, py = seeds[seed_idx_i]
                d_min, best_j = float("inf"), None
                for j in range(len(tiles)):
                    if i == j or not alive[j]:
                        continue
                    seed_idx_j = get_seed_idx(j)
                    if seed_idx_j < len(seeds):
                        qx, qy = seeds[seed_idx_j]
                        d = np.hypot(px - qx, py - qy)
                        if d < d_min:
                            d_min, best_j = d, j

        if best_j is not None:
            merged_poly = unary_union([tiles[best_j]["polygon"], tiles[i]["polygon"]])
            merged_poly = merged_poly.buffer(0)  # Clean topology.

            merged_indices = np.union1d(
                tiles[best_j]["point_indices"], tiles[i]["point_indices"]
            )
            merged_n_points = int(merged_indices.size)

            tiles[best_j]["polygon"] = merged_poly
            tiles[best_j]["point_indices"] = merged_indices
            tiles[best_j]["n_spots"] = merged_n_points

            if update_area:
                tiles[best_j]["area"] = merged_poly.area

            alive[i] = False
            merged_count += 1

    merged_tiles = [t for a, t in zip(alive, tiles) if a]

    return merged_tiles, merged_count


def fix_degenerate_tiles(
    tiles: List[Dict],
    seeds: np.ndarray,
    d1: float,
    *,
    tau_d1: float = SLIVER_TAU_D1,
    tau_med: float = SLIVER_TAU_MED,
    edge_tol: float = 1e-9,
    min_width_factor: Optional[float] = None,
) -> List[Dict]:
    """Merge sliver tiles into neighbors: area < max(tau_d1*d1**2, tau_med*median_area)."""
    _validate_edge_tolerance(edge_tol)
    if len(tiles) == 0:
        return tiles

    areas = np.array([t["polygon"].area if not t["polygon"].is_empty else 0.0
                      for t in tiles])
    A_med = float(np.median(areas[areas > 0])) if np.any(areas > 0) else 0.0

    A_d1 = tau_d1 * (d1 ** 2)
    A_median = tau_med * A_med
    eps_area = max(1e-12, A_d1, A_median)

    logger.debug(f"  [REPAIR] d1={d1:.4f}, A_med={A_med:.4e}, "
                 f"A_cut=max(tau_d1*d1**2={A_d1:.4e}, tau_med*A_med={A_median:.4e}) = {eps_area:.4e}")

    def is_sliver(poly: Polygon) -> bool:
        if min_width_factor is None:
            return False
        try:
            mc = minimum_clearance(poly)
            return mc < (min_width_factor * d1)
        except Exception as e:

            if isinstance(e, ShapelyError):
                logger.debug(
                    "[GEOM] minimum_clearance failed; skipping sliver check: %s", e
                )
            else:
                logger.debug(
                    "[GEOM] Unexpected error in minimum_clearance; skipping sliver check: %s", e
                )
            return False

    bad = [i for i, t in enumerate(tiles)
           if t["polygon"].is_empty or
              (t["polygon"].area < eps_area) or
              is_sliver(t["polygon"])]

    if not bad:
        return tiles

    repaired_tiles, merged_count = _merge_tiles_generic(
        tiles=tiles,
        indices_to_merge=bad,
        seeds=seeds,
        edge_tol=edge_tol,
        get_seed_idx=lambda i: i,
        update_area=False,
    )

    removal_pct = 100.0 * merged_count / len(tiles) if len(tiles) > 0 else 0.0
    logger.debug(f"  [REPAIR] removed {merged_count}/{len(tiles)} tiles ({removal_pct:.2f}%)")

    return repaired_tiles


def merge_small_tiles_geos(
    tiles: List[Dict],
    tile_min_points: int,
    seeds: Optional[np.ndarray] = None,
    edge_tol: float = 1e-9
) -> List[Dict]:
    """Merge tiles with n_spots < tile_min_points into the neighbor with the longest shared boundary."""
    _validate_edge_tolerance(edge_tol)
    if len(tiles) == 0:
        return tiles

    small = [i for i, t in enumerate(tiles) if t["n_spots"] < tile_min_points]

    if not small:
        return tiles

    logger.debug(f"  [MERGE] Found {len(small)}/{len(tiles)} tiles with n_spots < {tile_min_points}")

    merged_tiles, merged_count = _merge_tiles_generic(
        tiles=tiles,
        indices_to_merge=small,
        seeds=seeds,
        edge_tol=edge_tol,
        get_seed_idx=lambda i: tiles[i].get("seed_idx", i),
        update_area=True,
    )

    merge_pct = 100.0 * merged_count / len(tiles) if len(tiles) > 0 else 0.0
    logger.debug(f"  [MERGE] Merged {merged_count}/{len(tiles)} small tiles ({merge_pct:.2f}%)")

    return merged_tiles
