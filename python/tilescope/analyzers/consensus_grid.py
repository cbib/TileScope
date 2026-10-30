"""
Consensus grid builder - hexagonal overlay aggregating tessellation ensemble statistics.
Exact reproduction of .old/roi_consensus.py algorithm with clean I/O contracts.
"""
from tilescope.spatial.hex import make_flat_top_hexagon
from tilescope.spatial import hex_radius_from_area


import logging
import numpy as np
from typing import Dict, List, Tuple
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from tilescope.component_data import Tessellation
from tilescope.tessellation.repair import extract_polygonal_support
from tilescope.cols import (
    N_SPOTS, SUM_G1, SUM_G2, N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
    N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
)
from tilescope.schema.sufficient_stats import sufficient_stat_columns
from tilescope.defaults import CONSENSUS_HEX_SCALING_FACTOR

logger = logging.getLogger(__name__)


class OverlapIndex:
    """STRtree spatial index over tile polygons for tile↔hex overlap queries."""

    def __init__(self, tiles: list):
        self._tiles = tiles
        self._tree = STRtree([t.polygon for t in tiles])

    def query(self, hex_poly: Polygon) -> Dict[int, float]:
        """Return {tile_id: intersection_area} for tiles overlapping hex_poly."""
        out: Dict[int, float] = {}
        for idx in self._tree.query(hex_poly, predicate='intersects'):
            tile = self._tiles[idx]
            inter = hex_poly.intersection(tile.polygon)
            if not inter.is_empty and hasattr(inter, 'area'):
                out[tile.tile_id] = inter.area
        return out


def winsorized_mean(x: np.ndarray, q_low: float, q_high: float) -> float:
    """Compute winsorized mean by clipping extreme values.

    Winsorization replaces values below q_low quantile with q_low value,
    and values above q_high quantile with q_high value, then computes mean.
    Robust to outliers while retaining all data spots.

    Args:
        x: Array of tile areas
        q_low: Lower quantile (clip values below this, typically 0.0)
        q_high: Upper quantile (clip values above this, typically 0.95-0.98)

    Returns:
        Mean of clipped distribution

    Raises:
        ValueError: If quantiles are invalid
    """
    if x.size == 0:
        return 0.0
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"Invalid winsor quantiles: ({q_low}, {q_high})")

    # Edge case: no clipping (should equal plain mean)
    if q_low == 0.0 and q_high == 1.0:
        return float(x.mean())

    # Compute quantile bounds
    lo, hi = np.quantile(x, [q_low, q_high])

    # Clip and compute mean
    x_clipped = np.clip(x, lo, hi)
    mean_winsor = float(x_clipped.mean())

    # Debug logging: compare estimators
    median = float(np.median(x))
    mean_plain = float(x.mean())
    n_clipped = np.sum((x < lo) | (x > hi))
    logger.debug(
        f"  [WINSOR] n={len(x)}, clipped={n_clipped} "
        f"(median={median:.1f}, winsor={mean_winsor:.1f}, mean={mean_plain:.1f})"
    )

    return mean_winsor


def _init_per_tess_stats() -> Dict[str, float]:
    """Initialize per-tessellation statistics accumulator (uses standard constant names as keys)."""
    return {field: 0.0 for field in sufficient_stat_columns(N_SPOTS)}


def _accumulate_stats(stats_dict: Dict[str, float], overlap_fraction: float, voronoi_tile: Dict) -> None:
    """Accumulate count-weighted statistics from Voronoi tile (in-place, uses standard constant names)."""
    stats_dict[N_SPOTS] += overlap_fraction * voronoi_tile[N_SPOTS]
    stats_dict[SUM_G1] += overlap_fraction * voronoi_tile[SUM_G1]
    stats_dict[SUM_G2] += overlap_fraction * voronoi_tile[SUM_G2]
    stats_dict[N_G1_HIGH] += overlap_fraction * voronoi_tile[N_G1_HIGH]
    stats_dict[N_G2_HIGH] += overlap_fraction * voronoi_tile[N_G2_HIGH]
    stats_dict[N_BOTH_HIGH] += overlap_fraction * voronoi_tile[N_BOTH_HIGH]
    stats_dict[N_G1_LOW] += overlap_fraction * voronoi_tile[N_G1_LOW]
    stats_dict[N_G2_LOW] += overlap_fraction * voronoi_tile[N_G2_LOW]
    stats_dict[N_BOTH_LOW] += overlap_fraction * voronoi_tile[N_BOTH_LOW]


def _aggregate_per_tess_stats(per_tess_stats: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-tessellation statistics into single aggregated dict (uses standard constant names as keys)."""
    return {field: sum(s[field] for s in per_tess_stats.values()) for field in sufficient_stat_columns(N_SPOTS)}


def _generate_hex_grid(
    region_polygon: Polygon,
    hex_radius: float,
    *,
    anchor_x: float,
    anchor_y: float,
) -> List[Tuple[int, int, float, float, Polygon]]:
    """Generate a flat-top hexagonal lattice covering the region's
    padded bounding box.

    Lattice origin is `(anchor_x, anchor_y)`: cell `(col, row) = (0, 0)`
    has centre exactly at that anchor. Downstream, two regions that
    share an anchor and hex_radius produce cell-for-cell aligned hex
    lattices (a sample-wide anchor makes hexes align across every
    component/region of the sample). Callers that want a per-region
    origin (the pre-anchor behaviour) pass
    `(region_bounds[0] - hex_radius, region_bounds[1] - hex_radius)`
    and get output identical to the previous positional-arg form.

    Returns:
        List of `(col, row, center_x, center_y, hex_polygon)`. The
        `(center_x, center_y)` are the lattice-cell centres, the
        canonical anchor for neighbour construction and any per-hex
        coordinate readout. Callers must NOT recompute centres from
        `polygon.centroid`: after clipping / holes / multipart
        support the centroid drifts off the lattice cell.
        Cell indices can be negative when the anchor lies to the
        right of / above the region.
    """
    minx, miny, maxx, maxy = region_polygon.bounds

    # Regular hexagonal spacing (flat-top): dx = 1.5*r, dy = sqrt(3)*r.
    dx = 1.5 * hex_radius
    dy = np.sqrt(3) * hex_radius

    # Pad the covered extent so hexes on the border have full cells
    # produced even if their centre is just outside the region bbox.
    padding = hex_radius
    extent_minx = minx - padding
    extent_maxx = maxx + padding
    extent_miny = miny - padding
    extent_maxy = maxy + padding

    # First / last cell whose centre lies in the padded extent.
    first_col = int(np.floor((extent_minx - anchor_x) / dx))
    last_col = int(np.ceil((extent_maxx - anchor_x) / dx))
    first_row = int(np.floor((extent_miny - anchor_y) / dy))
    last_row = int(np.ceil((extent_maxy - anchor_y) / dy))


    cells: List[Tuple[int, int, float, float, Polygon]] = []
    # Iterate (row, col) in a stable order; matches the pre-anchor
    # traversal so downstream tile-id enumeration is unchanged when
    # the caller passes the equivalent per-region anchor.
    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            x = anchor_x + dx * col
            y = anchor_y + dy * row + (dy / 2) * (col % 2)
            cells.append((
                col, row, float(x), float(y),
                make_flat_top_hexagon(x, y, hex_radius),
            ))
    return cells


def _clip_hex_to_roi(hex_polygon: Polygon, region_polygon_prep,
                     region_polygon: Polygon, min_area_fraction: float = 0.1):
    """Clip a hexagon to the region boundary and return its positive-
    area polygonal support (`Polygon` or `MultiPolygon`).

    A multipart intersection (concave region, retained holes, multi-
    connected support) is preserved as a `MultiPolygon`: all faces
    belong to the same hex identity and their total area matters for
    downstream aggregation. Discarding the smaller faces would silently
    drop hex-covered analysis area.

    Discards the hex only if:
      - the region and hex do not intersect at all; or
      - the intersection has no positive-area polygonal content
        (points/lines/empty only); or
      - total polygonal support area < `min_area_fraction * hex area`.
    """
    if not region_polygon_prep.intersects(hex_polygon):
        return None
    raw_intersection = hex_polygon.intersection(region_polygon)
    if raw_intersection.is_empty:
        return None
    support, _residue = extract_polygonal_support(raw_intersection)
    if support is None:
        return None
    if support.area < min_area_fraction * hex_polygon.area:
        return None
    return support


def compute_optimal_hex_radius(
    tessellation_ensemble: Dict[int, Tessellation],
    region_polygon: Polygon,
    n_roi_spots: int,
    config: "RunConfigModel",
    scaling_factor: float = CONSENSUS_HEX_SCALING_FACTOR
) -> float:
    """Density-aware hex radius ensures rho * A_target >= min_spots_per_hex.

    Sizing authority: `Tile.area` (engine-snap consistent) for the
    tile-area distribution + `region_polygon.area` for the ROI area.
    Canvas precision is deliberately not consulted here: mixing it
    with the engine's own snap grid is incorrect.
    """
    min_spots_per_hex = config.run_config.consensus.min_spots_per_hex
    occupancy_safety = config.resolved_policy.consensus.occupancy_safety
    max_hex_area_fraction_of_roi = config.resolved_policy.consensus.max_hex_area_fraction_of_roi

    if not tessellation_ensemble:
        return 2.0  # Fallback

    # Read canonical Tile.area: the engine-authoritative area committed
    # at tessellation construction.
    areas = []
    for tess in tessellation_ensemble.values():
        if tess.tiles:
            areas.append(np.fromiter(
                (float(t.area) for t in tess.tiles),
                dtype=float, count=len(tess.tiles),
            ))

    A_tiles = np.concatenate(areas) if areas else np.array([], float)
    A_tiles = A_tiles[A_tiles > 0]

    if len(A_tiles) == 0:
        return 2.0  # Fallback

    method = config.resolved_policy.consensus.area_sizing_method
    if method == "winsor_mean":
        q_low, q_high = config.resolved_policy.consensus.winsor_limits
        A_base = winsorized_mean(A_tiles, q_low, q_high) * scaling_factor
    elif method == "median":
        q = config.resolved_policy.consensus.base_area_quantile
        A_base = float(np.quantile(A_tiles, q)) * scaling_factor
    else:
        raise ValueError(f"Unknown area_sizing_method: {method}")

    # ROI area from the canonical domain-region polygon directly.
    A_roi = float(region_polygon.area)
    rho = n_roi_spots / A_roi if A_roi > 0 else 0.0  # spots / u²
    A_required = ((min_spots_per_hex * occupancy_safety) / rho) if rho > 0 else 0.0

    cap = max_hex_area_fraction_of_roi * A_roi
    A_target = min(max(A_base, A_required), cap if cap > 0 else max(A_base, A_required))
    if A_target <= 0:
        A_target = A_base if A_base > 0 else 1.0

    r_hex = hex_radius_from_area(A_target)

    logger.info(
        f"[CONSENSUS] rho={rho:.3f} A_base={A_base:.3f} A_req={A_required:.3f} "
        f"A_roi={A_roi:.3f} -> A_target={A_target:.3f} r_hex={r_hex:.3f}"
    )
    return r_hex


def _aggregate_hex_from_ensemble(hex_poly: Polygon,
                                 overlap_indexes: Dict[int, 'OverlapIndex'],
                                 tessellation_ensemble: Dict[int, Tessellation],
                                 tiles_by_tess: Dict[str, Dict[int, Dict]]) -> Tuple[
                                     Dict[int, Dict[str, float]],
                                     List[Tuple[int, int, float, float]],
                                 ]:
    """Accumulate count-weighted stats per tessellation for one hex.

    Returns `(per_tess_stats, edges)`: `per_tess_stats[tess_idx]` is
    the aggregated stat block for tessellations that overlap the
    hex; `edges` is `[(tess_idx, tile_id, overlap_area, tile_area)]`
    captured from the same intersection pass so the bank can consume
    them without re-querying. `overlap_area` sums across MultiPolygon
    faces of a tile.
    """
    per_tess_stats: Dict[int, Dict[str, float]] = {}
    edges: List[Tuple[int, int, float, float]] = []
    for tess_idx, overlap_index in overlap_indexes.items():
        overlaps = overlap_index.query(hex_poly)
        if not overlaps:
            continue

        per_tess_stats[tess_idx] = _init_per_tess_stats()
        tess_id = tessellation_ensemble[tess_idx].tessellation_id
        tiles_for_tess = tiles_by_tess.get(tess_id, {})

        for tile_id, overlap_area in overlaps.items():
            voronoi_tile = tiles_for_tess.get(int(tile_id))
            if voronoi_tile is None:
                continue
            tile_area = voronoi_tile['area']
            if tile_area > 0:
                overlap_fraction = overlap_area / tile_area
                _accumulate_stats(per_tess_stats[tess_idx], overlap_fraction, voronoi_tile)
                edges.append((
                    int(tess_idx), int(tile_id),
                    float(overlap_area), float(tile_area),
                ))

    return per_tess_stats, edges
