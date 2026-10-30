"""One-region-at-a-time tessellation execution.

Given a single connected region of a component's final domain, this
module runs the GEOS Voronoi engine on the region's spot subset,
reclips each produced tile against the snapped region (so holes
retained in the region geometry are excluded from tile interiors),
and returns a rebuilt `Tessellation` whose `.tiles` are the
canonical, hole-aware, component-local-indexed tile set: the ONLY
tile authority for downstream consumers (correlation, consensus,
reporters).

Contract:

- The runner is the ONLY authority that pairs regions with their
  spot subsets. Callers do not construct region row frames
  themselves; the runner slices the canonical `ComponentData` using
  `source_row_ids_in_region` (which must be a subset of
  `component_data.expression_data[SOURCE_ROW_ID]`).
- The returned `Tessellation.tiles` carries hole-reclipped polygons
  and component-local point indices. The pre-reclip engine output
  is NOT exposed; it would silently defeat the hole and remap
  contracts if a consumer read it.
- Restricted-cell semantics: a Voronoi cell clipped
  to a non-convex domain may become MultiPolygon. Such a tile is
  KEPT as one statistical tile: one seed, one identity, one point
  assignment, area = total face area. The only reclip failure is
  `NoPolygonalContentError` (points/lines only, or empty support),
  which the runner maps to
  `RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT` with
  `n_tiles=0`. Missing columns, invalid point indices, engine
  crashes, and other contract violations propagate unchanged.
- 1- or 2-tile regions are `CONTRIBUTED`; the >=3-tile requirement
  lives on the pooled component-replicate correlation, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus,
    RegionReplicateTessellationDiagnostic,
)
from tilescope.cols import COORD_X, COORD_Y, SOURCE_ROW_ID
from tilescope.component_data import (
    ComponentData, Tessellation, Tile, require_integer_point_indices,
)
from tilescope.spatial.spacing import median_nn_distance
from tilescope.tessellation.builder import compute_tessellation
from tilescope.tessellation.clipping import ordered_clips
from tilescope.tessellation.engine_geos import (
    GeosVoronoiConfig, GeosVoronoiEngine, PreparedVoronoiRegion,
)
from tilescope.tessellation.repair import (
    NoPolygonalContentError,
    clip_to_region_and_refresh,
)


@dataclass(frozen=True)
class RegionTessellationOutcome:
    """Per (component, region, replicate) runner output.

    `tessellation` is `None` only for `TILE_NO_POLYGONAL_CONTENT`
    (the replicate is not downstream-consumable). For `CONTRIBUTED`
    and `NO_TILES` it is the canonical rebuilt tessellation whose
    tiles may be `Polygon` or `MultiPolygon` under restricted-cell
    semantics;
    `NO_TILES` yields `tessellation.tiles == []`.
    """
    tessellation: Optional[Tessellation]
    diagnostic: RegionReplicateTessellationDiagnostic


@dataclass(frozen=True)
class PreparedComponent:
    """Component-scoped, region-independent state.

    Built once per component OUTSIDE the region loop and reused for
    every region on that component. Owns the ``SOURCE_ROW_ID -> row
    position`` mapping (a component-wide relation that does not change
    across regions or replicates).
    """
    component_id: str
    component_position_by_source_id: Mapping[int, int]
    component_row_count: int


def prepare_component(component_data: ComponentData) -> PreparedComponent:
    """Extract the component-wide identity relation once."""
    expr = component_data.expression_data
    if SOURCE_ROW_ID not in expr.columns:
        raise ValueError(
            "prepare_component: component_data.expression_data is missing "
            "SOURCE_ROW_ID; the orchestrator must retain it when "
            "constructing ComponentData."
        )
    component_source_ids = expr[SOURCE_ROW_ID].to_numpy()
    position_by_id: Dict[int, int] = {
        int(sid): int(pos) for pos, sid in enumerate(component_source_ids)
    }
    return PreparedComponent(
        component_id=str(component_data.component_id),
        component_position_by_source_id=MappingProxyType(position_by_id),
        component_row_count=int(len(component_source_ids)),
    )


@dataclass(frozen=True)
class PreparedRegion:
    """Replicate-independent per-region construction state.

    Built once per (component, region) OUTSIDE the K-replicate loop
    and reused by every replicate. Caches:

    - ``region_local_to_component_local``: the argument-order-
      preserving `int64` array mapping region-local row positions to
      component-local row positions (consumed by `_remap_point_indices`).
    - ``coords``: the region's spot coordinate array in the same
      argument order.
    - ``unique_coords``: the sorted-unique-tuple reduction of
      ``coords`` (consumed by ``select_seeds``).
    - ``d1_median_nn``: the region's median nearest-neighbour
      distance. ``compute_tessellation`` is the single downstream d1
      authority; ``prepare_region`` performs finiteness validation on
      the computed value.
    - ``geometry``: the snapped region and padded Voronoi envelope,
      reused by the engine and final tile clipping.
    """
    component_id: str
    domain_region_id: int
    region_polygon: Union[Polygon, MultiPolygon]
    region_local_to_component_local: np.ndarray
    component_row_count: int
    coords: np.ndarray
    unique_coords: np.ndarray
    d1_median_nn: float
    geometry: PreparedVoronoiRegion


def prepare_region(
    *,
    component_id: str,
    domain_region_id: int,
    region_polygon: Union[Polygon, MultiPolygon],
    component_data: ComponentData,
    source_row_ids_in_region: Sequence[int],
    prepared_component: Optional[PreparedComponent] = None,
) -> PreparedRegion:
    """Build one region's replicate-independent construction state.

    Slices ``component_data.expression_data`` to the region's rows,
    validates identity contracts, computes the region coord array +
    unique-coordinate reduction + median-NN distance ONCE. Consumers
    pass the returned ``PreparedRegion`` to ``run_region_tessellation``
    for every replicate in the K-ensemble.

    ``prepared_component``: pre-built by ``prepare_component`` and
    reused across every region on the same component. When absent, a
    fresh ``PreparedComponent`` is built here (backward-compat path
    for standalone callers such as focused tests).
    """
    if prepared_component is None:
        prepared_component = prepare_component(component_data)
    component_position_by_source_id = (
        prepared_component.component_position_by_source_id
    )
    region_source_ids_arr = np.asarray(source_row_ids_in_region, dtype=np.int64)
    if region_source_ids_arr.size == 0:
        raise ValueError(
            f"prepare_region: component {component_id!r} region "
            f"{domain_region_id} has no source_row_ids; eligibility gate "
            f"should have prevented this call."
        )
    try:
        region_local_to_component_local = np.asarray(
            [component_position_by_source_id[int(sid)] for sid in region_source_ids_arr],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"prepare_region: component {component_id!r} region "
            f"{domain_region_id}: source_row_id {exc.args[0]!r} not "
            f"present in component_data.expression_data — contract "
            f"violation upstream."
        ) from exc
    expr = component_data.expression_data
    region_rows = expr.iloc[region_local_to_component_local].reset_index(drop=True)
    coords = region_rows[[COORD_X, COORD_Y]].to_numpy(dtype=float)
    if coords.size:
        unique_coords = np.array(
            sorted(set(map(tuple, coords))), dtype=float,
        )
        d1 = float(median_nn_distance(coords))
        if not (d1 != d1 or d1 in (float("inf"), float("-inf"))):
            pass  # finite; accepted
        else:
            raise ValueError(
                f"prepare_region: component {component_id!r} region "
                f"{domain_region_id}: median-NN distance is {d1!r}; "
                f"cannot proceed."
            )
    else:
        unique_coords = coords.reshape(0, 2)
        d1 = float("nan")
    return PreparedRegion(
        component_id=str(component_id),
        domain_region_id=int(domain_region_id),
        region_polygon=region_polygon,
        region_local_to_component_local=region_local_to_component_local,
        component_row_count=prepared_component.component_row_count,
        coords=coords,
        unique_coords=unique_coords,
        d1_median_nn=d1,
        geometry=GeosVoronoiEngine(GeosVoronoiConfig()).prepare_region(
            region_polygon, d1=d1,
        ),
    )


def run_region_tessellation(
    *,
    prepared: PreparedRegion,
    replicate_index: int,
    tile_min_points: int,
    voronoi_fraction: float,
    rng_seed: int,
) -> RegionTessellationOutcome:
    """Tessellate one region for one replicate.

    Consumes a ``PreparedRegion`` built once outside the K-replicate
    loop. Per replicate: seed draw (reusing the cached
    ``unique_coords`` and ``d1_median_nn``) -> Voronoi build ->
    per-tile assignment / repair / merge -> reclip each tile against
    the region's snapped geometry -> remap point indices to
    component-local.

    The region's snapped geometry and envelope are prepared once
    at its fixed precision grid and reused across replicates.

    ``clip_to_region_and_refresh`` raises ``NoPolygonalContentError``
    when the reclip leaves no positive-area polygonal content; the
    runner maps that to ``TILE_NO_POLYGONAL_CONTENT`` with
    ``n_tiles=0`` and returns ``tessellation=None``. Restricted-cell
    tile geometry (Polygon or MultiPolygon) is preserved.
    """
    component_id = prepared.component_id
    domain_region_id = prepared.domain_region_id
    region_local_to_component_local = prepared.region_local_to_component_local

    tessellation = compute_tessellation(
        component_id=component_id,
        roi_coords=prepared.coords,
        region_polygon=prepared.region_polygon,
        tile_min_points=tile_min_points,
        fraction=voronoi_fraction,
        rng_seed=rng_seed,
        tessellation_id=f"{component_id}::r{domain_region_id}::t{replicate_index}",
        precomputed_d1=prepared.d1_median_nn,
        precomputed_unique_coords=prepared.unique_coords,
        prepared_geometry=prepared.geometry,
    )
    precision_grid = prepared.geometry.precision_grid
    snapped_region = prepared.geometry.polygon

    if not tessellation.tiles:
        return RegionTessellationOutcome(
            tessellation=dataclass_replace(tessellation, tiles=[]),
            diagnostic=RegionReplicateTessellationDiagnostic(
                component_id=component_id,
                domain_region_id=domain_region_id,
                replicate_index=replicate_index,
                n_tiles=0,
                region_status=RegionExecutionStatus.NO_TILES,
            ),
        )

    def reclip(tile):
        return clip_to_region_and_refresh(
            tile={"tile_id": tile.tile_id, "polygon": tile.polygon, "area": tile.area},
            snapped_region=snapped_region,
            precision_grid=precision_grid,
        )

    kept: list[Tile] = []
    with ordered_clips(reclip, tessellation.tiles) as reclipped_tiles:
        for tile in tessellation.tiles:
            try:
                reclipped = next(reclipped_tiles)
            except NoPolygonalContentError:
                return RegionTessellationOutcome(
                    tessellation=None,
                    diagnostic=RegionReplicateTessellationDiagnostic(
                        component_id=component_id,
                        domain_region_id=domain_region_id,
                        replicate_index=replicate_index,
                        n_tiles=0,
                        region_status=RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT,
                    ),
                )
            remapped_indices = _remap_point_indices(
                tile.point_indices, region_local_to_component_local,
                component_row_count=prepared.component_row_count,
            )
            n_spots_derived = int(remapped_indices.size)
            kept.append(Tile(
                tile_id=tile.tile_id,
                polygon=reclipped["polygon"],
                point_indices=remapped_indices,
                n_spots=n_spots_derived,
                area=float(reclipped["area"]),
                tessellation_id=tessellation.tessellation_id,
                domain_region_id=domain_region_id,
            ))

    if not kept:
        return RegionTessellationOutcome(
            tessellation=dataclass_replace(tessellation, tiles=[]),
            diagnostic=RegionReplicateTessellationDiagnostic(
                component_id=component_id,
                domain_region_id=domain_region_id,
                replicate_index=replicate_index,
                n_tiles=0,
                region_status=RegionExecutionStatus.NO_TILES,
            ),
        )

    return RegionTessellationOutcome(
        tessellation=dataclass_replace(tessellation, tiles=kept),
        diagnostic=RegionReplicateTessellationDiagnostic(
            component_id=component_id,
            domain_region_id=domain_region_id,
            replicate_index=replicate_index,
            n_tiles=len(kept),
            region_status=RegionExecutionStatus.CONTRIBUTED,
        ),
    )


def _remap_point_indices(
    region_indices: np.ndarray,
    region_local_to_component_local: np.ndarray,
    *,
    component_row_count: int,
) -> np.ndarray:
    """Map region-local tile indices to component-local positions.

    The engine supplies integer region-local positions. These become sorted,
    unique integer component positions.
    Storage is proportional to tile membership, not component size; ascending
    order preserves the numerical reduction order of Boolean selection.

    Fail-loud on out-of-range: an index whose target position
    does not exist in the region-local frame is a contract violation.
    """
    idx = require_integer_point_indices(
        region_indices, upper_bound=region_local_to_component_local.shape[0],
    )
    if idx.size == 0:
        return np.empty(0, dtype=np.int64)
    out = np.unique(region_local_to_component_local[idx]).astype(np.int64, copy=False)
    return require_integer_point_indices(out, upper_bound=component_row_count)
