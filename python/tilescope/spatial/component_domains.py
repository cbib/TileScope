"""Component-domain geometry builder.

Composes G_c = C_c intersect T_parent(c) intersect A per component,
where C_c is component support, T_p is Voronoi parent ownership minus
ineligible-spot exclusion disks, and A is acquisition support. The
builder retains the full mathematical geometry; estimator-specific
eligibility thresholds live with their estimators.
"""
from __future__ import annotations

import hashlib


import logging
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from shapely import make_valid, points
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import polygonize, unary_union, voronoi_diagram
from shapely.strtree import STRtree

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial import ComponentPolygon, ComponentPolygonDict
from tilescope.spatial.source_spot_adjacency import (
    SourceSpotAdjacency,
    build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentDomainDiagnostic:
    """One row per component. Scalar-only (no Shapely geometry).

    Estimator-eligibility fields are per-estimator and live with those
    estimators, not here.
    """
    component_id: str
    parent_roi_id: str
    n_component_spots: int
    n_spots_in_domain: int
    n_domain_regions: int
    n_regions_with_spots: int
    component_support_area: float
    ownership_excluded_area: float
    acquisition_excluded_area: float
    final_domain_area: float
    geometry_status: str


@dataclass(frozen=True)
class DomainRegionDiagnostic:
    """One row per connected region within a component's MultiPolygon.

    `source_row_fingerprint` is a stable digest of the sorted
    SOURCE_ROW_IDs; downstream coordinators cross-check by recomputing
    and asserting equality, since counts + area alone can coincide
    across different partitions.
    """
    component_id: str
    domain_region_id: int
    n_component_spots: int
    area: float
    n_holes: int
    geometry_status: str
    source_row_fingerprint: str


def _validate_source_row_ids(df: pd.DataFrame, frame_name: str) -> None:
    """Input contract: `SOURCE_ROW_ID` present, integer-typed, non-null, unique.

    Enforces dtype directly to reject booleans, which would otherwise
    slip through int-comparison via True == 1 and let a downstream join
    silently pair spots by boolean truthiness.
    """
    if SOURCE_ROW_ID not in df.columns:
        raise ValueError(
            f"[component_domains] {frame_name} missing required column "
            f"{SOURCE_ROW_ID!r}."
        )
    col = df[SOURCE_ROW_ID]
    if col.isna().any():
        n_bad = int(col.isna().sum())
        raise ValueError(
            f"[component_domains] {frame_name} has {n_bad} row(s) with "
            f"null {SOURCE_ROW_ID!r}."
        )
    if pd.api.types.is_bool_dtype(col):
        raise ValueError(
            f"[component_domains] {frame_name} {SOURCE_ROW_ID!r} has bool "
            f"dtype; expected an integer dtype."
        )
    if not pd.api.types.is_integer_dtype(col):
        raise ValueError(
            f"[component_domains] {frame_name} {SOURCE_ROW_ID!r} is not "
            f"integer-typed (got dtype {col.dtype!r})."
        )
    values = col.to_numpy()
    if len(np.unique(values)) != len(values):
        raise ValueError(
            f"[component_domains] {frame_name} {SOURCE_ROW_ID!r} is not unique."
        )


def _validate_supported_frame_agreement(
    source_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
) -> None:
    """Input contract: `spatial_df` rows are a subset of `source_df` by
    `SOURCE_ROW_ID`, and their `(COORD_X, COORD_Y, ROI_ID)` match.

    Align rows by source ID once, then compare coordinate and ROI columns
    separately. This preserves identifier dtypes and reports the first
    offending supported row without constructing a Series for every cell.
    """
    src_ids = source_df[SOURCE_ROW_ID].astype(np.int64).to_numpy()
    sup_ids = spatial_df[SOURCE_ROW_ID].astype(np.int64).to_numpy()
    positions = pd.Index(src_ids).get_indexer(sup_ids)
    unknown = sup_ids[positions < 0]
    if len(unknown):
        raise ValueError(
            f"[component_domains] {len(unknown)} supported SOURCE_ROW_ID(s) "
            f"absent from source frame (e.g. {unknown[:5].tolist()!r})."
        )
    source_xy = source_df[[COORD_X, COORD_Y]].to_numpy(dtype=float)[positions]
    supported_xy = spatial_df[[COORD_X, COORD_Y]].to_numpy(dtype=float)
    coordinate_match = (source_xy == supported_xy).all(axis=1)
    source_rois = source_df[ROI_ID].map(str).to_numpy()[positions]
    parent_match = source_rois == spatial_df[ROI_ID].map(str).to_numpy()
    bad = np.flatnonzero(~(coordinate_match & parent_match))
    if len(bad):
        i = int(bad[0])
        row = spatial_df.iloc[i]
        src_row_id = int(sup_ids[i])
        if not coordinate_match[i]:
            raise ValueError(
                f"[component_domains] SOURCE_ROW_ID={src_row_id}: coordinates "
                f"in spatial frame ({row[COORD_X]!r}, {row[COORD_Y]!r}) "
                f"disagree with source frame "
                f"({source_xy[i, 0]!r}, {source_xy[i, 1]!r})."
            )
        if not parent_match[i]:
            raise ValueError(
                f"[component_domains] SOURCE_ROW_ID={src_row_id}: "
                f"{ROI_ID!r} in spatial frame ({row[ROI_ID]!r}) disagrees "
                f"with source frame ({source_rois[i]!r})."
            )


def _validate_radius_factors(**factors: float) -> None:
    """Input contract: every named radius factor is finite and strictly positive."""
    for name, value in factors.items():
        v = float(value)
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(
                f"[component_domains] {name} must be finite and > 0; got {value!r}."
            )


def _validate_component_parent_consistency(
    spatial_df: pd.DataFrame,
    component_to_roi: Dict[str, str],
) -> None:
    """Each COMPONENT_ID maps to exactly one non-null ROI_ID, and the
    row-level parent agrees with `component_to_roi`.
    """
    grouped = spatial_df.groupby(COMPONENT_ID)[ROI_ID]
    for cid, roi_series in grouped:
        cid_s = str(cid)
        parents = roi_series.dropna().astype(str).unique()
        if len(parents) == 0:
            raise ValueError(
                f"[component_domains] component {cid_s!r} has no non-null "
                f"{ROI_ID!r} rows."
            )
        if len(parents) > 1:
            raise ValueError(
                f"[component_domains] component {cid_s!r} has multiple "
                f"parent ROI_IDs {sorted(parents)!r}; expected exactly one."
            )
        row_parent = parents[0]
        mapped = component_to_roi.get(cid_s)
        if mapped is None:
            raise ValueError(
                f"[component_domains] component_to_roi missing key {cid_s!r}."
            )
        if str(mapped) != row_parent:
            raise ValueError(
                f"[component_domains] component {cid_s!r}: row-level "
                f"{ROI_ID!r}={row_parent!r} disagrees with component_to_roi "
                f"mapping {str(mapped)!r}."
            )


def _validate_source_coordinate_contract(source_expression_df: pd.DataFrame) -> None:
    """Fail loud on structural problems in the source frame.

    Checks column presence, coordinate finiteness, no duplicate (x, y),
    and non-null ROI_ID. Duplicate coordinates trigger a
    conflicting-label subcase message when ROI_IDs disagree.
    """
    for col in (COORD_X, COORD_Y, ROI_ID):
        if col not in source_expression_df.columns:
            raise ValueError(
                f"[component_domains] source frame missing required column "
                f"{col!r}."
            )

    xy = source_expression_df[[COORD_X, COORD_Y]].to_numpy()
    if not np.isfinite(xy).all():
        n_bad = int((~np.isfinite(xy)).any(axis=1).sum())
        raise ValueError(
            f"[component_domains] source frame has {n_bad} row(s) with "
            f"non-finite coordinates."
        )

    if source_expression_df[ROI_ID].isna().any():
        n_bad = int(source_expression_df[ROI_ID].isna().sum())
        raise ValueError(
            f"[component_domains] source frame has {n_bad} row(s) with "
            f"missing {ROI_ID!r}."
        )

    dup_mask = source_expression_df.duplicated(subset=[COORD_X, COORD_Y], keep=False)
    if dup_mask.any():
        dup_df = source_expression_df.loc[dup_mask, [COORD_X, COORD_Y, ROI_ID]]
        # Detect the label-conflict subcase for a more informative message.
        labels_per_xy = (
            dup_df.astype({ROI_ID: str})
            .groupby([COORD_X, COORD_Y])[ROI_ID]
            .nunique()
        )
        n_dup_coords = int((labels_per_xy >= 1).sum())
        n_conflict = int((labels_per_xy > 1).sum())
        if n_conflict > 0:
            raise ValueError(
                f"[component_domains] {n_conflict} coordinate(s) carry "
                f"conflicting {ROI_ID!r} labels: same (x, y), different "
                f"parent; refusing to guess ownership."
            )
        raise ValueError(
            f"[component_domains] source frame has {n_dup_coords} duplicated "
            f"(x, y) coordinate(s); duplicates are forbidden."
        )


def _buffered_point_support(
    coordinates: np.ndarray, radius: float,
) -> ComponentPolygon:
    """Union equal-radius point buffers without filling holes or islands."""
    if len(coordinates) == 0:
        return Polygon()
    return unary_union([
        Point(float(x), float(y)).buffer(radius) for x, y in coordinates
    ])


def _topology_aware_point_support(
    coordinates: np.ndarray,
    radius: float,
    allowed_edges: set[tuple[int, int]],
    *,
    optimize_planar_face_union: bool = False,
) -> ComponentPolygon:
    """Point footprints plus local Delaunay faces admitted by one graph."""
    base = _buffered_point_support(coordinates, radius)
    if len(coordinates) < 3:
        return base
    try:
        triangulation = Delaunay(coordinates)
        simplices = triangulation.simplices
    except QhullError:
        return base
    retained = np.zeros(len(simplices), dtype=bool)
    boundary_edge_counts: Dict[tuple[int, int], int] = {}
    for simplex_index, simplex in enumerate(simplices):
        edges = (
            tuple(sorted((int(simplex[0]), int(simplex[1])))),
            tuple(sorted((int(simplex[1]), int(simplex[2])))),
            tuple(sorted((int(simplex[0]), int(simplex[2])))),
        )
        triangle_edges = {
            edges[0], edges[1], edges[2],
        }
        if triangle_edges.issubset(allowed_edges):
            retained[simplex_index] = True
            if optimize_planar_face_union:
                for edge in edges:
                    boundary_edge_counts[edge] = (
                        boundary_edge_counts.get(edge, 0) + 1
                    )
    if not retained.any():
        return base

    if not optimize_planar_face_union:
        # Preserve Visium construction order exactly; changing polygon
        # construction order can perturb boundary-touch decisions at
        # floating-point precision.
        return unary_union([
            base,
            *(Polygon(coordinates[simplex])
              for simplex in simplices[retained]),
        ])

    # Retained Delaunay triangles form a planar mesh; its union is
    # determined by edges incident to exactly one retained face.
    # Polygonize also emits polygons for enclosed holes, so retain only
    # polygons whose interior representative point falls in a retained
    # Delaunay simplex.
    boundary_lines = [
        LineString(coordinates[list(edge)])
        for edge, count in boundary_edge_counts.items()
        if count == 1
    ]
    face_parts = []
    for polygon in polygonize(boundary_lines):
        representative = polygon.representative_point()
        simplex_index = int(triangulation.find_simplex(np.array([[
            representative.x, representative.y,
        ]]))[0])
        if simplex_index >= 0 and retained[simplex_index]:
            face_parts.append(polygon)
    if not face_parts:
        raise RuntimeError(
            "Retained Delaunay faces produced no polygonized support."
        )
    face_support = unary_union(face_parts)
    return unary_union([base, face_support])


class _ComponentEdgeLookup:
    """Reuse source-row lookup and visit only a component's neighbour lists."""

    def __init__(self, adjacency: SourceSpotAdjacency):
        self.adjacency = adjacency
        self.source_ids = pd.Index(np.fromiter(
            (int(v) for v in adjacency.source_spot_table.spot_id), dtype=np.int64,
        ))

    def induced_edges(self, source_row_ids: np.ndarray) -> set[tuple[int, int]]:
        global_positions = self.source_ids.get_indexer(source_row_ids)
        missing = np.flatnonzero(global_positions < 0)
        if len(missing):
            raise ValueError(
                "Component-domain source_row_id is absent from the canonical "
                f"SourceSpotAdjacency: {str(int(source_row_ids[missing[0]]))!r}."
            )
        local_by_global = {
            int(global_position): local_position
            for local_position, global_position in enumerate(global_positions)
        }
        out: set[tuple[int, int]] = set()
        for left, global_left in enumerate(global_positions):
            for global_right in self.adjacency.neighbors[int(global_left)]:
                right = local_by_global.get(global_right)
                if right is not None and left < right:
                    out.add((left, right))
        return out


def _build_parent_ownership(
    source_expression_df: pd.DataFrame,
    spatial_component_expression_df: pd.DataFrame,
    ineligible_src_row_ids: Iterable[int],
    envelope: ComponentPolygon,
    exclusion_radius: float,
) -> Dict[str, ComponentPolygon]:
    """Voronoi ownership per parent ROI, minus ineligible-spot exclusions.

    Every parent's owned domain has exclusion disks around EVERY
    ineligible spot or cell subtracted, including disks that fall in a
    sibling supported spot's Voronoi cell. Returns
    str(parent_roi_id) -> owned_domain_polygon; parents with zero
    supported spots are absent.
    """
    if len(spatial_component_expression_df) == 0:
        return {}

    supported_coords = spatial_component_expression_df[
        [COORD_X, COORD_Y]
    ].to_numpy()
    supported_parents = (
        spatial_component_expression_df[ROI_ID].astype(str).to_numpy()
    )

    if len(supported_coords) == 1:
        # Single-point Voronoi degenerates; the owned domain is
        # the entire envelope for that parent.
        parent = str(supported_parents[0])
        owned = envelope
    else:
        seeds = MultiPoint([(float(x), float(y)) for x, y in supported_coords])
        vd = voronoi_diagram(seeds, envelope=envelope)
        cells: List[ComponentPolygon] = [g for g in vd.geoms]
        # Shapely 2 does not preserve seed order in the cell list; match
        # each seed to its containing cell.
        cell_tree = STRtree(cells)
        cell_index_by_seed: List[int] = []
        for x, y in supported_coords:
            seed_pt = Point(float(x), float(y))
            candidates = cell_tree.query(seed_pt)
            matched = None
            for idx in candidates:
                if cells[int(idx)].intersects(seed_pt):
                    matched = int(idx)
                    break
            if matched is None:
                # Extreme boundary condition: fall back to linear scan.
                for idx in range(len(cells)):
                    if cells[idx].intersects(seed_pt):
                        matched = idx
                        break
            if matched is None:
                raise RuntimeError(
                    "[component_domains] Voronoi cell lookup failed for "
                    f"seed ({x!r}, {y!r})."
                )
            cell_index_by_seed.append(matched)

        cells_by_parent: Dict[str, List[ComponentPolygon]] = {}
        for seed_i, cell_i in enumerate(cell_index_by_seed):
            parent = str(supported_parents[seed_i])
            cells_by_parent.setdefault(parent, []).append(cells[cell_i])

        owned_by_parent: Dict[str, ComponentPolygon] = {
            parent: unary_union(cell_list).intersection(envelope)
            for parent, cell_list in cells_by_parent.items()
        }

    if len(supported_coords) == 1:
        owned_by_parent = {parent: owned}

    ineligible_ids = frozenset(int(i) for i in ineligible_src_row_ids)
    if ineligible_ids:
        ineligible_mask = source_expression_df[SOURCE_ROW_ID].isin(ineligible_ids)
        ineligible_coords = source_expression_df.loc[
            ineligible_mask, [COORD_X, COORD_Y]
        ].to_numpy()
        exclusion = _buffered_point_support(
            ineligible_coords, exclusion_radius,
        )
    else:
        exclusion = Polygon()

    if not exclusion.is_empty:
        return {
            parent: owned.difference(exclusion)
            for parent, owned in owned_by_parent.items()
        }
    return owned_by_parent


def _extract_polygonal_geometry(geom) -> ComponentPolygon:
    """Retain only the Polygon / MultiPolygon parts of a mixed geometry.

    Shapely intersections can produce GeometryCollections mixing
    polygons, lines, and points; downstream analyses only consume 2-D
    area.
    """
    if geom.is_empty:
        return geom if isinstance(geom, (Polygon, MultiPolygon)) else Polygon()
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return Polygon()
        return unary_union(polys)
    # Points, LineStrings, etc. carry zero area.
    return Polygon()


def _drop_sliver_holes(geom: ComponentPolygon, area_tol: float) -> ComponentPolygon:
    """Remove interior rings whose area is at most `area_tol`.

    Unioning disk buffers with Delaunay triangles can leave a zero-area
    ring along a shared triangle edge, depending on floating-point
    rounding at the given coordinate scale. Such rings are construction
    residue, not holes; keeping them makes `n_holes` and the ring
    structure of `G_c` scale-dependent.
    """
    def _clean(poly: Polygon) -> Polygon:
        kept = [r for r in poly.interiors if Polygon(r).area > area_tol]
        if len(kept) == len(poly.interiors):
            return poly
        return Polygon(poly.exterior, kept)

    if isinstance(geom, Polygon) and not geom.is_empty:
        return _clean(geom)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([_clean(p) for p in geom.geoms])
    return geom


_REGION_ORDER_INF = float("inf")


def _regions_of(geom: ComponentPolygon) -> List[Polygon]:
    """Split a component's `G_c` into its constituent polygons."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return list(geom.geoms)


def _assign_component_spots_to_regions(
    regions: List[Polygon],
    component_coords: np.ndarray,
    component_src_ids: np.ndarray,
) -> Dict[int, int]:
    """Point-in-polygon: map each component spot to exactly one region.

    Uses `covers` so boundary spots are assigned. Boundary tie-break
    picks the lowest region index in construction order; STRtree query
    order is undefined, so candidates are sorted before the covers scan
    to remove that source of nondeterminism.
    """
    if not regions:
        return {}
    tree = STRtree(regions)
    out: Dict[int, int] = {}
    for i in range(len(component_coords)):
        pt = Point(float(component_coords[i, 0]), float(component_coords[i, 1]))
        src_id = int(component_src_ids[i])
        candidates = sorted(int(r) for r in tree.query(pt))
        for r_idx_i in candidates:
            if regions[r_idx_i].covers(pt):
                out[src_id] = r_idx_i
                break
    return out


def _canonical_region_order(
    regions: List[Polygon],
    spots_per_region: List[List[int]],
) -> List[int]:
    """Spot-informed canonical sort key. Returns permutation of region indices.

    Key: (-spot_count, min_source_row_id, -area, centroid.x, centroid.y).
    NOT invariant under rotation/reflection; metamorphic tests must
    match empty regions geometrically, not by domain_region_id.
    """
    def key(i: int):
        n = len(spots_per_region[i])
        min_src = (
            min(spots_per_region[i]) if n > 0 else _REGION_ORDER_INF
        )
        centroid = regions[i].centroid
        return (-n, min_src, -regions[i].area, centroid.x, centroid.y)

    return sorted(range(len(regions)), key=key)


def _geometric_sort_key(region: Polygon) -> tuple:
    """Total, coordinate-and-order-agnostic key for a region.

    Applied before any spot-informed step so shared-boundary tie-break
    is stable under any incoming `MultiPolygon.geoms` permutation.
    Final tiebreaker is normalized WKB so distinct polygons never
    collide.
    """
    minx, miny, maxx, maxy = region.bounds
    centroid = region.centroid
    return (
        -float(region.area),
        float(minx), float(miny), float(maxx), float(maxy),
        float(centroid.x), float(centroid.y),
        region.wkb,
    )


def _source_row_fingerprint(source_row_ids: Iterable[int]) -> str:
    """Stable digest of the sorted SOURCE_ROW_IDs assigned to a region.

    Coordinators cross-check by recomputing and asserting equality;
    counts + area alone can coincide across different partitions.
    """
    sorted_ids = sorted(int(i) for i in source_row_ids)
    payload = ",".join(str(i) for i in sorted_ids).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class ComponentDomainPartition:
    """Deterministic partition of a component domain into canonical regions.

    Structurally immutable. `canonical_regions[i]` carries
    `domain_region_id == i`; `canonical_ids_by_source_row` maps spot ->
    region (uncovered spots absent); the other two mappings hold the
    per-region SOURCE_ROW_IDs (sorted) and their stable digest.
    """
    canonical_regions: Tuple[Polygon, ...]
    canonical_ids_by_source_row: Mapping[int, int]
    source_row_ids_by_canonical_id: Mapping[int, Tuple[int, ...]]
    source_row_fingerprint_by_canonical_id: Mapping[int, str]


@dataclass(frozen=True)
class ComponentPolygonBuild:
    """Canonical output of component-polygon construction.

    Downstream tessellation consumes ``partitions`` directly; it must
    never re-partition the same polygon.
    """

    component_polygons: Mapping[str, ComponentPolygon]
    partitions: Mapping[str, ComponentDomainPartition]
    component_diagnostics: Tuple[ComponentDomainDiagnostic, ...]
    region_diagnostics: Tuple[DomainRegionDiagnostic, ...]


@dataclass(frozen=True)
class _ComponentMaskComposition:
    """One component's masked geometry and exclusion-area diagnostics."""
    raw_geometry: ComponentPolygon
    ownership_excluded_area: float
    acquisition_excluded_area: float


def _area_difference(container_area: float, subset_area: float) -> float:
    """Return a numerically non-negative set-area difference.

    Overlay area is mathematically monotone, but GEOS float
    accumulation can leave a tiny negative residue; clamping preserves
    the set-measure result without another overlay.
    """
    return max(0.0, float(container_area) - float(subset_area))


def _compose_component_masks(
    component_support: ComponentPolygon,
    parent_ownership: ComponentPolygon,
    acquisition_support: ComponentPolygon,
    *,
    component_is_within_acquisition: bool,
    reuse_constructive_overlays: bool = True,
) -> _ComponentMaskComposition:
    """Compose ``C intersect T intersect A`` while reusing constructive intersections.

    Materialises each intersection at most once and derives diagnostic
    set measures algebraically. When component and acquisition support
    share the source graph and r_C <= r_A, C is a subset of A by
    construction, so only ``C intersect T`` is needed.
    """
    if not reuse_constructive_overlays:
        # Preserve Visium operation order for byte-for-byte
        # reproducibility of existing lattice results.
        raw = component_support.intersection(
            parent_ownership,
        ).intersection(acquisition_support)
        return _ComponentMaskComposition(
            raw_geometry=raw,
            ownership_excluded_area=float(
                component_support.difference(parent_ownership)
                .intersection(acquisition_support).area
            ),
            acquisition_excluded_area=float(
                component_support.difference(acquisition_support)
                .intersection(parent_ownership).area
            ),
        )

    component_in_parent = component_support.intersection(parent_ownership)
    if component_is_within_acquisition:
        raw = component_in_parent
        raw_area = float(raw.area)
        return _ComponentMaskComposition(
            raw_geometry=raw,
            ownership_excluded_area=_area_difference(
                component_support.area, raw_area,
            ),
            acquisition_excluded_area=0.0,
        )

    component_in_acquisition = component_support.intersection(
        acquisition_support,
    )
    raw = component_in_parent.intersection(acquisition_support)
    raw_area = float(raw.area)
    return _ComponentMaskComposition(
        raw_geometry=raw,
        ownership_excluded_area=_area_difference(
            component_in_acquisition.area, raw_area,
        ),
        acquisition_excluded_area=_area_difference(
            component_in_parent.area, raw_area,
        ),
    )


def partition_component_domain(
    geometry: ComponentPolygon,
    component_coords: np.ndarray,
    component_src_ids: np.ndarray,
) -> ComponentDomainPartition:
    """Split a component's final domain into canonically-ordered regions.

    Canonical order depends on which spots each region holds; the
    shared-boundary tie-break is coordinate-and-order-agnostic so
    ordering does not depend on incoming MultiPolygon.geoms order.
    """
    if len(component_coords) != len(component_src_ids):
        raise ValueError(
            f"partition_component_domain: coords/src_ids length mismatch "
            f"({len(component_coords)} vs {len(component_src_ids)})."
        )
    regions = _regions_of(geometry)
    if not regions:
        return ComponentDomainPartition(
            canonical_regions=(),
            canonical_ids_by_source_row=MappingProxyType({}),
            source_row_ids_by_canonical_id=MappingProxyType({}),
            source_row_fingerprint_by_canonical_id=MappingProxyType({}),
        )
    geom_order = sorted(range(len(regions)), key=lambda i: _geometric_sort_key(regions[i]))
    geom_sorted_regions = [regions[i] for i in geom_order]
    spot_to_geom_idx = _assign_component_spots_to_regions(
        geom_sorted_regions, component_coords, component_src_ids,
    )
    spots_per_geom_region: List[List[int]] = [[] for _ in geom_sorted_regions]
    for src_id, geom_idx in spot_to_geom_idx.items():
        spots_per_geom_region[geom_idx].append(int(src_id))
    canonical_order = _canonical_region_order(geom_sorted_regions, spots_per_geom_region)
    canonical_regions = [geom_sorted_regions[i] for i in canonical_order]
    canonical_spots = [spots_per_geom_region[i] for i in canonical_order]

    canonical_ids_by_source_row: Dict[int, int] = {}
    ids_by_canonical: Dict[int, tuple] = {}
    fp_by_canonical: Dict[int, str] = {}
    for canonical_id, spots in enumerate(canonical_spots):
        sorted_spots = tuple(sorted(spots))
        ids_by_canonical[canonical_id] = sorted_spots
        fp_by_canonical[canonical_id] = _source_row_fingerprint(sorted_spots)
        for src_id in sorted_spots:
            canonical_ids_by_source_row[src_id] = canonical_id

    return ComponentDomainPartition(
        canonical_regions=tuple(canonical_regions),
        canonical_ids_by_source_row=MappingProxyType(canonical_ids_by_source_row),
        source_row_ids_by_canonical_id=MappingProxyType(ids_by_canonical),
        source_row_fingerprint_by_canonical_id=MappingProxyType(fp_by_canonical),
    )


def _validate_component_domains(
    domains: ComponentPolygonDict,
    spatial_component_expression_df: pd.DataFrame,
    comp_diags: List[ComponentDomainDiagnostic],
    region_diags: List[DomainRegionDiagnostic],
    *,
    area_tol: float,
) -> None:
    """Fail loud on construction, cross-component, and accounting violations.

    `area_tol` is a scale-aware snap-boundary allowance for polygon-union
    residue, not a substantive-fit tolerance. Per-component parent
    consistency is checked pre-construction by the input-contract guard.
    """
    # Invariant 1: final geometry validity.
    for cid, geom in domains.items():
        if not geom.is_valid:
            raise AssertionError(
                f"[component_domains invariant 1] component {cid!r} has "
                f"invalid final geometry: {geom.wkt!r}."
            )

    # Invariant 2: retained-region area finite and > 0 for regions
    # holding spots. Consult both diagnostics and geometry to catch a
    # bookkeeping-only regression.
    diag_by_comp = {d.component_id: d for d in comp_diags}
    for rd in region_diags:
        if rd.n_component_spots == 0:
            continue
        if not (np.isfinite(rd.area) and rd.area > 0):
            raise AssertionError(
                f"[component_domains invariant 2] component "
                f"{rd.component_id!r} region id={rd.domain_region_id} "
                f"holds {rd.n_component_spots} spot(s) but has area "
                f"{rd.area!r}; expected finite and > 0."
            )

    # Query a polygon index in bounded point batches. This independently
    # re-derives coverage without scanning every cell for every component.
    # Keep the first foreign hit in original component/input-row order.
    comp_geoms = list(domains.items())
    coverage = np.zeros(len(comp_geoms), dtype=np.int64)
    foreign_hit = None
    coords = spatial_component_expression_df[[COORD_X, COORD_Y]].to_numpy(dtype=float)
    owners = spatial_component_expression_df[COMPONENT_ID].astype(str).to_numpy()
    component_ids = np.asarray([cid for cid, _ in comp_geoms], dtype=object)
    if comp_geoms:
        tree = STRtree([geom for _, geom in comp_geoms])
        for start in range(0, len(coords), 4096):
            row_ids, domain_ids = tree.query(
                points(coords[start:start + 4096]), predicate="covered_by",
            )
            row_ids = row_ids + start
            own = owners[row_ids] == component_ids[domain_ids]
            coverage += np.bincount(domain_ids[own], minlength=len(comp_geoms))
            if (~own).any():
                foreign_domains = domain_ids[~own]
                first_domain = int(foreign_domains.min())
                first_row = int(row_ids[~own][foreign_domains == first_domain].min())
                candidate = (first_domain, first_row)
                if foreign_hit is None or candidate < foreign_hit:
                    foreign_hit = candidate

    # Invariant 3: every own spot covered by G_c is assigned to exactly
    # one region. Re-derive coverage from geometry so the check is
    # independent of the assignment loop.
    for i, (cid, geom) in enumerate(comp_geoms):
        if geom.is_empty:
            continue
        n_covered = int(coverage[i])
        diag = diag_by_comp.get(cid)
        if diag is None:
            raise AssertionError(
                f"[component_domains invariant 3] component {cid!r}: no "
                f"diagnostic row emitted; assignment count cannot be "
                f"verified against geometric coverage."
            )
        if diag.n_spots_in_domain != n_covered:
            raise AssertionError(
                f"[component_domains invariant 3] component {cid!r}: "
                f"assignment count {diag.n_spots_in_domain} disagrees with "
                f"geometric coverage count {n_covered}."
            )
    # Invariant 4: foreign spot exclusion. Uses `covers` (strict) so a
    # boundary-hit fires; boundary-hit rate is near zero in practice.
    if foreign_hit is not None:
        i, row_position = foreign_hit
        row = spatial_component_expression_df.iloc[row_position]
        raise AssertionError(
            f"[component_domains invariant 4] foreign supported "
            f"spot src_id={int(row[SOURCE_ROW_ID])} "
            f"(component {row[COMPONENT_ID]!r}) lies inside "
            f"foreign component {comp_geoms[i][0]!r}'s domain."
        )

    # Invariants 5 and 6: pairwise material non-overlap. Voronoi
    # ownership prevents cross-parent overlap by construction; sibling
    # overlap is not, and must be checked. `area_tol` absorbs
    # polygon-union snap residue only.
    cids_all = sorted(diag_by_comp.keys())
    overlap_tree = STRtree([domains[cid] for cid in cids_all])
    for i in range(len(cids_all)):
        for j in sorted(int(v) for v in overlap_tree.query(domains[cids_all[i]]) if v > i):
            ci, cj = cids_all[i], cids_all[j]
            gi = domains[ci]
            gj = domains[cj]
            if gi.is_empty or gj.is_empty:
                continue
            overlap = gi.intersection(gj).area
            if overlap <= area_tol:
                continue
            same_parent = diag_by_comp[ci].parent_roi_id == diag_by_comp[cj].parent_roi_id
            invariant = "6" if same_parent else "5"
            scope = "sibling" if same_parent else "cross-parent"
            raise AssertionError(
                f"[component_domains invariant {invariant}] {scope} components "
                f"{ci!r} and {cj!r} overlap by {overlap!r} > area_tol="
                f"{area_tol!r}."
            )

    # Invariant 8: source-row accounting.
    total_supported = len(spatial_component_expression_df)
    n_in = sum(d.n_spots_in_domain for d in comp_diags)
    n_out = sum(
        d.n_component_spots - d.n_spots_in_domain for d in comp_diags
    )
    if n_in + n_out != total_supported:
        raise AssertionError(
            f"[component_domains invariant 8] source-row accounting: "
            f"n_in={n_in} + n_out={n_out} != total_supported={total_supported}."
        )


def build_component_polygons(
    source_expression_df: pd.DataFrame,
    spatial_component_expression_df: pd.DataFrame,
    component_to_roi: Dict[str, str],
    sample_median_nn: float,
    *,
    source_spot_adjacency: SourceSpotAdjacency | None = None,
    component_support_radius_factor: float,
    acquisition_support_radius_factor: float,
    ineligible_spot_exclusion_radius_factor: float,
) -> ComponentPolygonBuild:
    """Build the canonical component-domain map plus diagnostics.

    Retains the complete `G_c = C_c intersect T_p intersect A` per
    component; no estimator-eligibility pruning happens here.
    """
    if not (np.isfinite(sample_median_nn) and sample_median_nn > 0):
        raise ValueError(
            f"[component_domains] sample_median_nn must be finite and > 0; "
            f"got {sample_median_nn!r}."
        )

    _validate_radius_factors(
        component_support_radius_factor=component_support_radius_factor,
        acquisition_support_radius_factor=acquisition_support_radius_factor,
        ineligible_spot_exclusion_radius_factor=ineligible_spot_exclusion_radius_factor,
    )

    _validate_source_coordinate_contract(source_expression_df)
    _validate_source_row_ids(source_expression_df, "source frame")
    _validate_source_row_ids(spatial_component_expression_df, "spatial component frame")
    _validate_supported_frame_agreement(
        source_expression_df, spatial_component_expression_df,
    )
    _validate_component_parent_consistency(
        spatial_component_expression_df, component_to_roi,
    )

    if source_spot_adjacency is None:
        # Focused-geometry caller path. Production supplies the sample
        # scope graph materialised before splitting.
        source_spot_adjacency = build_source_spot_adjacency(
            build_source_spot_table_from_frame(source_expression_df),
            technology="visium",
        )
    source_ids_in_order = (
        source_expression_df[SOURCE_ROW_ID].astype(int).to_numpy()
    )
    if tuple(str(v) for v in source_ids_in_order) != (
        source_spot_adjacency.source_spot_table.spot_id
    ):
        raise ValueError(
            "build_component_polygons requires SourceSpotAdjacency built "
            "from source_expression_df in canonical row order."
        )
    optimize_planar_delaunay_geometry = (
        source_spot_adjacency.method == "locally_pruned_delaunay"
    )
    edge_lookup = _ComponentEdgeLookup(source_spot_adjacency)
    geometry_start = time.monotonic()
    logger.info("[GEOMETRY] acquisition support start: %d locations", len(source_expression_df))

    r_C = float(component_support_radius_factor) * float(sample_median_nn)
    r_A = float(acquisition_support_radius_factor) * float(sample_median_nn)
    r_E = (
        float(ineligible_spot_exclusion_radius_factor) * float(sample_median_nn)
    )
    # Absolute area tolerance for construction-residue overlaps; a
    # precision-band floor keeps snap-boundary residues from firing
    # invariant assertions.
    precision_grid = max(1e-4 * float(sample_median_nn), 1e-9)
    area_tol = max(1e-9 * float(sample_median_nn) ** 2, precision_grid ** 2)

    A_geom = _topology_aware_point_support(
        source_expression_df[[COORD_X, COORD_Y]].to_numpy(),
        r_A,
        edge_lookup.induced_edges(source_ids_in_order),
        optimize_planar_face_union=optimize_planar_delaunay_geometry,
    )
    if A_geom.is_empty:
        raise ValueError(
            "[component_domains] acquisition support is empty; the source "
            "frame appears to have no spots."
        )

    all_ids = frozenset(source_expression_df[SOURCE_ROW_ID].astype(int).tolist())
    supported_ids = frozenset(
        spatial_component_expression_df[SOURCE_ROW_ID].astype(int).tolist()
    )
    ineligible_ids = all_ids - supported_ids

    envelope = A_geom.envelope
    logger.info("[GEOMETRY] parent ownership start; elapsed=%.1fs", time.monotonic() - geometry_start)
    ownership_by_parent = _build_parent_ownership(
        source_expression_df=source_expression_df,
        spatial_component_expression_df=spatial_component_expression_df,
        ineligible_src_row_ids=ineligible_ids,
        envelope=envelope,
        exclusion_radius=r_E,
    )
    del all_ids, supported_ids, ineligible_ids

    # Both supports come from the same canonical source graph; when
    # r_C <= r_A every component footprint is already contained in A,
    # letting the loop skip repeated overlays against the large sample
    # acquisition geometry. Only applied on the optimized Delaunay path
    # to preserve Visium float-order compatibility.
    component_is_within_acquisition = (
        optimize_planar_delaunay_geometry and r_C <= r_A
    )

    domains: ComponentPolygonDict = {}
    partitions: Dict[str, ComponentDomainPartition] = {}
    comp_diags: List[ComponentDomainDiagnostic] = []
    region_diags: List[DomainRegionDiagnostic] = []
    logger.info("[GEOMETRY] component domains start; elapsed=%.1fs", time.monotonic() - geometry_start)

    for component_id, group in spatial_component_expression_df.groupby(
        COMPONENT_ID, sort=True,
    ):
        cid_s = str(component_id)
        if cid_s not in component_to_roi:
            raise ValueError(
                f"[component_domains] component_to_roi missing key {cid_s!r}."
            )
        parent = str(component_to_roi[cid_s])
        if parent not in ownership_by_parent:
            raise ValueError(
                f"[component_domains] parent {parent!r} of component "
                f"{cid_s!r} has no ownership (no supported spots)."
            )

        component_coords = group[[COORD_X, COORD_Y]].to_numpy()
        component_src_ids = group[SOURCE_ROW_ID].astype(int).to_numpy()

        C_c = _topology_aware_point_support(
            component_coords, r_C,
            edge_lookup.induced_edges(component_src_ids),
            optimize_planar_face_union=optimize_planar_delaunay_geometry,
        )
        T_p = ownership_by_parent[parent]

        mask_composition = _compose_component_masks(
            C_c,
            T_p,
            A_geom,
            component_is_within_acquisition=(
                component_is_within_acquisition
            ),
            reuse_constructive_overlays=optimize_planar_delaunay_geometry,
        )
        G_c = _drop_sliver_holes(
            _extract_polygonal_geometry(
                make_valid(mask_composition.raw_geometry)
            ),
            area_tol,
        )

        partition = partition_component_domain(
            G_c, component_coords, component_src_ids,
        )
        partitions[cid_s] = partition
        sorted_regions = partition.canonical_regions
        sorted_spots_per_region = [
            list(partition.source_row_ids_by_canonical_id.get(i, ()))
            for i in range(len(sorted_regions))
        ]

        for canonical_id, region in enumerate(sorted_regions):
            n_holes = len(region.interiors) if isinstance(region, Polygon) else 0
            region_diags.append(DomainRegionDiagnostic(
                component_id=cid_s,
                domain_region_id=canonical_id,
                n_component_spots=len(sorted_spots_per_region[canonical_id]),
                area=float(region.area),
                n_holes=int(n_holes),
                geometry_status="ok",
                source_row_fingerprint=partition.source_row_fingerprint_by_canonical_id[canonical_id],
            ))

        if not sorted_regions:
            final: ComponentPolygon = Polygon()
        elif len(sorted_regions) == 1:
            final = sorted_regions[0]
        else:
            final = MultiPolygon(sorted_regions)
        domains[cid_s] = final

        n_in = sum(len(spots) for spots in sorted_spots_per_region)
        n_regions_with_spots = sum(
            1 for spots in sorted_spots_per_region if spots
        )
        geometry_status = (
            "ok" if not final.is_empty else "empty_after_masking"
        )
        comp_diags.append(ComponentDomainDiagnostic(
            component_id=cid_s,
            parent_roi_id=parent,
            n_component_spots=len(component_coords),
            n_spots_in_domain=n_in,
            n_domain_regions=len(sorted_regions),
            n_regions_with_spots=n_regions_with_spots,
            component_support_area=float(C_c.area),
            ownership_excluded_area=(
                mask_composition.ownership_excluded_area
            ),
            acquisition_excluded_area=(
                mask_composition.acquisition_excluded_area
            ),
            final_domain_area=float(final.area),
            geometry_status=geometry_status,
        ))

    # Construction-only indices and sample/ROI polygons are not needed by
    # the independent validation of the final component domains.
    del edge_lookup, ownership_by_parent, A_geom
    logger.info("[GEOMETRY] validation start: %d components; elapsed=%.1fs", len(domains), time.monotonic() - geometry_start)
    _validate_component_domains(
        domains=domains,
        spatial_component_expression_df=spatial_component_expression_df,
        comp_diags=comp_diags,
        region_diags=region_diags,
        area_tol=area_tol,
    )

    logger.info("[GEOMETRY] complete; elapsed=%.1fs", time.monotonic() - geometry_start)
    return ComponentPolygonBuild(
        component_polygons=MappingProxyType(dict(domains)),
        partitions=MappingProxyType(dict(partitions)),
        component_diagnostics=tuple(comp_diags),
        region_diagnostics=tuple(region_diags),
    )
