"""High-level tessellation builder: GEOS Voronoi engine + repair."""
import math


import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors

from tilescope.component_data import Tessellation, Tile
from tilescope.defaults import (
    DEFAULT_MIN_TILE_SPOTS, DEFAULT_VORONOI_FRACTION,
    VALIDATION_BOUNDARY_THRESH_MULT, VALIDATION_MAX_BOUNDARY_FRAC,
)
from tilescope.enums import GridMethod
from tilescope.spatial.spacing import median_nn_distance
from tilescope.tessellation.engine_geos import (
    GeosVoronoiEngine, GeosVoronoiConfig, PreparedVoronoiRegion,
)
from tilescope.tessellation.repair import fix_degenerate_tiles, merge_small_tiles_geos
from tilescope.tessellation.voronoi import select_seeds

logger = logging.getLogger(__name__)


def _group_point_indices(assignments: np.ndarray, n_tiles: int) -> List[np.ndarray]:
    """Group rows by tile in O(N log N) time and O(N + M) storage.

    Stable sorting preserves the ascending row order of Boolean selection,
    including floating-point summation order in downstream expression summaries.
    Unassigned rows are omitted, as in the previous per-tile equality masks.
    """
    valid = np.flatnonzero((assignments >= 0) & (assignments < n_tiles))
    tile_ids = assignments[valid]
    order = valid[np.argsort(tile_ids, kind="stable")]
    counts = np.bincount(tile_ids, minlength=n_tiles)
    cuts = np.concatenate(([0], np.cumsum(counts)))
    return [order[cuts[i]:cuts[i + 1]] for i in range(n_tiles)]


def _validate_tessellation(tiles: List[Dict],
                           roi_coords: np.ndarray,
                           seeds: np.ndarray,
                           d1: float) -> Tuple[bool, Dict]:
    """Validate tessellation quality: no unassigned + bounded boundary-suspect fraction."""
    n_spots = len(roi_coords)
    n_tiles = len(tiles)
    n_assigned = sum(t["n_spots"] for t in tiles)

    boundary_threshold = VALIDATION_BOUNDARY_THRESH_MULT * d1

    nn_seeds = NearestNeighbors(n_neighbors=min(2, len(seeds))).fit(seeds)
    seed_distances, _ = nn_seeds.kneighbors(roi_coords)

    if seed_distances.shape[1] >= 2:
        gap = seed_distances[:, 1] - seed_distances[:, 0]
        boundary_suspect = gap < boundary_threshold
        frac_boundary_suspect = boundary_suspect.sum() / n_spots if n_spots > 0 else 0.0
    else:
        frac_boundary_suspect = 0.0

    n_unassigned = n_spots - n_assigned
    max_boundary_frac = VALIDATION_MAX_BOUNDARY_FRAC

    passed = (n_unassigned == 0) and (frac_boundary_suspect <= max_boundary_frac)

    metrics = {
        "n_tiles": n_tiles,
        "n_spots": n_spots,
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "boundary_suspect_frac": frac_boundary_suspect,
        "boundary_threshold": boundary_threshold,
        "max_boundary_frac": max_boundary_frac,
        "passed": passed,
    }
    return passed, metrics


def compute_tessellation(component_id: str,
                        roi_coords: np.ndarray,
                        region_polygon: Union[Polygon, MultiPolygon],
                        tile_min_points: int = DEFAULT_MIN_TILE_SPOTS,
                        fraction: float = DEFAULT_VORONOI_FRACTION,
                        rng_seed: int = 0,
                        tessellation_id: Optional[str] = None,
                        precomputed_d1: Optional[float] = None,
                        precomputed_unique_coords: Optional[np.ndarray] = None,
                        prepared_geometry: Optional[PreparedVoronoiRegion] = None):
    """Build Voronoi tessellation within ROI bounds (GEOS engine).

    Single d1 authority: resolves `precomputed_d1` (validate finite) or
    computes once, then passes the value into `GeosVoronoiEngine.build`.
    """

    if precomputed_d1 is not None:
        d1_val = float(precomputed_d1)
        if not math.isfinite(d1_val):
            raise ValueError(
                f"compute_tessellation: precomputed_d1 must be finite, "
                f"got {precomputed_d1!r}."
            )
    else:
        d1_val = median_nn_distance(roi_coords)

    rng = np.random.default_rng(rng_seed)

    seeds = select_seeds(
        roi_coords, fraction=fraction, rng=rng,
        precomputed_unique_coords=precomputed_unique_coords,
    )
    if seeds.shape[0] == 0:
        return Tessellation(
            tessellation_id=tessellation_id or "empty",
            component_id=component_id,
            tiles=[],
            method=GridMethod.VORONOI.value,
            tile_min_points=tile_min_points,
            fraction=fraction,
            rng_seed=rng_seed
        )

    engine = GeosVoronoiEngine(GeosVoronoiConfig())

    result = engine.build(
        region_polygon=region_polygon,
        seeds_xy=seeds,
        d1=d1_val,
        prepared_region=prepared_geometry,
    )

    assignments = result.assign_points_geos(roi_coords, seeds)
    n_fallback = np.sum(assignments < 0)
    fallback_rate = n_fallback / len(assignments) if len(assignments) > 0 else 0
    logger.debug(f"  [ASSIGNMENT] GEOS, {len(result.tiles)} tiles, "
                 f"fallbacks={n_fallback}/{len(assignments)} ({fallback_rate:.1%})")

    members = _group_point_indices(assignments, len(result.seed_index))
    tiles_with_assignments = []
    for tile_idx, point_indices in enumerate(members):
        n_pts = int(point_indices.size)

        if n_pts == 0:
            continue

        if result.tiles and tile_idx < len(result.tiles):
            polygon = result.tiles[tile_idx]
            area = polygon.area
        else:
            raise RuntimeError(f"Missing polygon for tile {tile_idx}")

        tiles_with_assignments.append({
            "polygon": polygon,
            "point_indices": point_indices,
            "n_spots": n_pts,
            "area": area,
        })

    logger.debug(f"  [TILE_DEBUG] {len(tiles_with_assignments)} tiles with points")

    passed, metrics = _validate_tessellation(
        tiles_with_assignments, roi_coords, seeds, d1_val,
    )

    logger.debug(f"  [VALIDATION] tiles={metrics['n_tiles']}, points={metrics['n_spots']}, "
                 f"assigned={metrics['n_assigned']}, "
                 f"boundary_suspect={metrics['boundary_suspect_frac']:.1%} "
                 f"(threshold={metrics['boundary_threshold']:.3f})")
    logger.debug(f"  [VALIDATION] PASS={passed} "
                 f"(max_boundary={metrics['max_boundary_frac']:.1%}, unassigned={metrics['n_unassigned']})")

    logger.debug(f"  [DEBUG] Before repair: {len(tiles_with_assignments)} tiles")
    tiles_repaired = fix_degenerate_tiles(
        tiles_with_assignments, seeds, d1_val,
    )
    logger.debug(f"  [DEBUG] After repair: {len(tiles_repaired)} tiles")

    tiles_final = merge_small_tiles_geos(
        tiles_repaired, tile_min_points, seeds=seeds,
    )

    tiles = []
    for tile_data in tiles_final:
        if tile_data["n_spots"] > 0:
            tile = Tile(
                tile_id=len(tiles),
                polygon=tile_data["polygon"],
                point_indices=tile_data["point_indices"],
                n_spots=tile_data["n_spots"],
                area=tile_data["area"],
            )
            tiles.append(tile)

    return Tessellation(
        tessellation_id=tessellation_id or "0",
        component_id=component_id,
        tiles=tiles,
        method=GridMethod.VORONOI.value,
        tile_min_points=tile_min_points,
        fraction=fraction,
        rng_seed=rng_seed,
        precision_grid=result.precision_grid,
    )
