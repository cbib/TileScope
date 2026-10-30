"""Canonical materialiser for the per-tile tessellation surface.

Builds the single canonical tabular materialisation of the typed
:class:`~tilescope.component_data.SampleTessellationResult`. The
typed record remains the stage authority; this DataFrame is its
one derived surface. Consumers (ROI x replicate correlation, ROI x
replicate enrichment, visualization, CSV exporter) read the SAME
DataFrame; the exporter serialises it verbatim, never rebuilds it.

Row grain: one row per Voronoi tile per replicate per contributing
region per component per sample. Composite key
``(sample_id, component_id, domain_region_id, replicate_index,
tile_id)`` where ``tile_id`` is a plain integer local to the
enclosing ``(component_id, domain_region_id, replicate_index)``
triple (the composite key disambiguates it across regions +
replicates without embedding ``tessellation_id`` into the
identifier). ``tessellation_id`` is exposed as its own provenance
column.

``roi_id`` is retained on every row as biological-parent provenance
under the functional dependency ``(sample_id, component_id) →
exactly one roi_id``. The builder validates this against the typed
``sample_tess_result.component_to_roi`` authority; the redundant
convenience column cannot drift from the typed source.

Column set (LOCKED):

- Identity + provenance (str / Int64): ``sample_id``, ``roi_id``,
  ``component_id``, ``domain_region_id``, ``replicate_index``,
  ``tessellation_id``, ``tile_id``.
- Sufficient stats (Int64 counts / Float64 sums): ``n_spots``,
  ``sum_g1``, ``sum_g2``, ``n_g1_high``, ``n_g2_high``,
  ``n_both_high``, ``n_g1_low``, ``n_g2_low``, ``n_both_low``.
- Derived densities (Float64, nullable when ``n_spots == 0``):
  ``g1_density``, ``g2_density``.
- Geometry diagnostics (str / Float64): ``tile_geometry_wkt``,
  ``tile_area``. Diagnostic only; area NEVER weights the
  estimator; ROI x replicate estimators consume equal-weighted
  tile densities.

Zero-tile outcomes are NOT synthesised here. A replicate that
produced no tiles for a region simply contributes no rows; the
authoritative "zero-tile" surface is
``component_region_replicate_status.csv``. Absence from this table
never becomes an unexplained drop; consumers cross-reference the
status frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Mapping

import numpy as np
import pandas as pd

from tilescope import cols
from tilescope.schema.table_validation import TessellationTilesSchema
from tilescope.spatial.consensus_bank import TileKey

if TYPE_CHECKING:
    from tilescope.analyzers.tessellation_analysis import (
        SampleTessellationResult,
    )
    from tilescope.metrics.enrichment_thresholds import PercentileThresholds
    from tilescope.spatial.feature_projection import FeatureSpatialProjection


TESSELLATION_TILES_COLUMNS = tuple(TessellationTilesSchema.columns)

TESSELLATION_TILE_SUPPORT_COLUMNS = (
    cols.SAMPLE_ID, cols.ROI_ID, cols.COMPONENT_ID,
    cols.DOMAIN_REGION_ID, cols.REPLICATE_INDEX,
    cols.TESSELLATION_ID, cols.TILE_ID, cols.N_SPOTS,
    cols.TILE_GEOMETRY_WKT, cols.TILE_AREA,
)


def _iter_records(frame: pd.DataFrame):
    """Yield one row mapping at a time without copying the complete table."""
    columns = tuple(frame.columns)
    for values in frame.itertuples(index=False, name=None):
        yield dict(zip(columns, values))


@dataclass(frozen=True)
class TessellationTileSupport:
    """Feature-independent tile rows and source membership for one sample."""

    sample_id: str
    support_frame: pd.DataFrame
    sample_positions_by_tile: Mapping[TileKey, np.ndarray]
    component_to_roi: Mapping[str, str]

    def __post_init__(self) -> None:
        frame = self.support_frame.copy(deep=True).reset_index(drop=True)
        if tuple(frame.columns) != TESSELLATION_TILE_SUPPORT_COLUMNS:
            raise ValueError(
                "TessellationTileSupport.support_frame columns must equal "
                f"{TESSELLATION_TILE_SUPPORT_COLUMNS!r}."
            )
        object.__setattr__(self, "support_frame", frame)
        frozen_positions: dict[TileKey, np.ndarray] = {}
        for key, positions in self.sample_positions_by_tile.items():
            arr = np.array(positions, dtype=np.int64, copy=True)
            if arr.ndim != 1:
                raise ValueError(
                    f"TessellationTileSupport tile {key!r} membership must be 1-D."
                )
            arr.flags.writeable = False
            frozen_positions[key] = arr
        object.__setattr__(
            self, "sample_positions_by_tile",
            MappingProxyType(frozen_positions),
        )
        object.__setattr__(
            self, "component_to_roi",
            MappingProxyType({
                str(cid): str(roi) for cid, roi in self.component_to_roi.items()
            }),
        )
        frame_keys = tuple(
            _tile_key_from_row(row)
            for row in _iter_records(frame)
        )
        if len(set(frame_keys)) != len(frame_keys):
            raise ValueError(
                "TessellationTileSupport.support_frame contains duplicate "
                "canonical tile keys."
            )
        if set(frame_keys) != set(frozen_positions):
            raise ValueError(
                "TessellationTileSupport support rows and membership keys "
                "must match exactly."
            )
        for row, key in zip(_iter_records(frame), frame_keys):
            n_spots = int(row[cols.N_SPOTS])
            if n_spots != len(frozen_positions[key]):
                raise ValueError(
                    f"TessellationTileSupport tile {key!r} n_spots={n_spots} "
                    f"but membership has {len(frozen_positions[key])} rows."
                )
            if float(row[cols.TILE_AREA]) <= 0.0:
                raise ValueError(
                    f"TessellationTileSupport tile {key!r} area must be "
                    "strictly positive."
                )


def _tile_key_from_row(row: Mapping) -> TileKey:
    return (
        str(row[cols.COMPONENT_ID]),
        int(row[cols.DOMAIN_REGION_ID]),
        int(row[cols.REPLICATE_INDEX]),
        int(row[cols.TILE_ID]),
    )


def build_tessellation_tile_support(
    *, sample_tess_result: "SampleTessellationResult", sample_id: str = "",
) -> TessellationTileSupport:
    """Traverse typed tiles once and freeze pair-independent support fields."""
    typed_sample_id = str(sample_tess_result.sample_id)
    if sample_id and str(sample_id) != typed_sample_id:
        raise ValueError(
            f"sample_id identity conflict: caller supplied {sample_id!r} "
            f"but sample_tess_result.sample_id is {typed_sample_id!r}. "
            "Two identity authorities are not permitted."
        )
    component_to_roi = dict(sample_tess_result.component_to_roi)
    rows: list[dict] = []
    positions_by_tile: dict[TileKey, np.ndarray] = {}
    for cid, comp in sample_tess_result.components.items():
        parent_roi_id = str(component_to_roi.get(str(cid), comp.parent_roi_id))
        source_row_ids = np.asarray(
            comp.component_data.expression_data[cols.SOURCE_ROW_ID],
            dtype=np.int64,
        )
        for region_id in sorted(comp.tessellations):
            for replicate_index, tess in comp.tessellations[region_id].items():
                for tile in tess.tiles:
                    key = (
                        str(cid), int(region_id), int(replicate_index),
                        int(tile.tile_id),
                    )
                    local_positions = tile.point_indices
                    sample_positions = source_row_ids[local_positions]
                    if key in positions_by_tile:
                        raise ValueError(
                            f"Duplicate canonical tessellation tile key {key!r}."
                        )
                    positions_by_tile[key] = sample_positions
                    rows.append({
                        cols.SAMPLE_ID: typed_sample_id,
                        cols.ROI_ID: parent_roi_id,
                        cols.COMPONENT_ID: str(cid),
                        cols.DOMAIN_REGION_ID: int(region_id),
                        cols.REPLICATE_INDEX: int(replicate_index),
                        cols.TESSELLATION_ID: str(tess.tessellation_id),
                        cols.TILE_ID: int(tile.tile_id),
                        cols.N_SPOTS: int(local_positions.size),
                        cols.TILE_GEOMETRY_WKT: tile.polygon.wkt,
                        cols.TILE_AREA: float(tile.area),
                    })
    frame = pd.DataFrame(rows, columns=list(TESSELLATION_TILE_SUPPORT_COLUMNS))
    _assert_roi_id_functional_dependency(frame, component_to_roi)
    return TessellationTileSupport(
        sample_id=typed_sample_id,
        support_frame=frame,
        sample_positions_by_tile=positions_by_tile,
        component_to_roi=component_to_roi,
    )


def _tile_joint_counts(
    *,
    tile_positions: np.ndarray,
    g1_vals: np.ndarray,
    g2_vals: np.ndarray,
    q_g1_high: float, q_g2_high: float,
    q_g1_low: float, q_g2_low: float,
) -> tuple[int, int]:
    """Joint (`n_both_high`, `n_both_low`): only counts the projection
    layer cannot supply because they cross the two features."""
    if tile_positions.size == 0:
        return 0, 0
    g1 = g1_vals[tile_positions]
    g2 = g2_vals[tile_positions]
    g1_hi = g1 > q_g1_high
    g2_hi = g2 > q_g2_high
    g1_lo = g1 < q_g1_low
    g2_lo = g2 < q_g2_low
    return int((g1_hi & g2_hi).sum()), int((g1_lo & g2_lo).sum())


def build_tessellation_tiles_df(
    *,
    tile_support: TessellationTileSupport,
    thresholds: "PercentileThresholds",
    projection_g1: "FeatureSpatialProjection",
    projection_g2: "FeatureSpatialProjection",
) -> pd.DataFrame:
    """Materialise the canonical tile-grain DataFrame.

    One row per (component, contributing region, replicate, tile).
    Per-tile marginals (`sum_g*`, `n_g*_high`, `n_g*_low`) are read
    from `projection_g1` / `projection_g2`; the feature-projection
    layer is the sole authority for marginal accumulators. Only the
    joint counts (`n_both_high`, `n_both_low`) are computed here,
    because they cross the two features.

    Args:
        tile_support: sample-level feature-independent tile authority.
        thresholds: pair-scope percentile thresholds; used for joint
            counts only (marginal thresholds come from the projections'
            own `FeatureThresholds`).
        projection_g1 / projection_g2: per-feature marginal authorities.
    Returns:
        A Pandera-validated DataFrame conforming to
        :data:`~tilescope.schema.table_validation.TessellationTilesSchema`.
        Empty (zero-row, contract-shaped) when no component produced
        any tiles under any replicate.
    """
    if projection_g1.bank_fingerprint != projection_g2.bank_fingerprint:
        raise ValueError(
            "build_tessellation_tiles_df: projection_g1 and projection_g2 "
            "carry different bank fingerprints — refusing to mix."
        )
    # Sample-scope feature vectors for joint counts (marginals come
    # from projections; joints require both features on the same tile).
    g1_vals = np.asarray(projection_g1.source_values, dtype=float)
    g2_vals = np.asarray(projection_g2.source_values, dtype=float)
    q_g1_high = thresholds.q_g1_threshold
    q_g2_high = thresholds.q_g2_threshold
    q_g1_low = thresholds.q_g1_low_threshold
    q_g2_low = thresholds.q_g2_low_threshold
    tile_marginals_g1 = projection_g1.tile_marginal_sums
    tile_marginals_g2 = projection_g2.tile_marginal_sums

    rows: list[dict] = []
    for support_row in _iter_records(tile_support.support_frame):
        tk = _tile_key_from_row(support_row)
        tms_g1 = tile_marginals_g1[tk]
        tms_g2 = tile_marginals_g2[tk]
        sample_positions = tile_support.sample_positions_by_tile[tk]
        n_both_high, n_both_low = _tile_joint_counts(
            tile_positions=sample_positions,
            g1_vals=g1_vals, g2_vals=g2_vals,
            q_g1_high=q_g1_high, q_g2_high=q_g2_high,
            q_g1_low=q_g1_low, q_g2_low=q_g2_low,
        )
        n_spots = int(support_row[cols.N_SPOTS])
        sum_g1 = float(tms_g1.sum_g)
        sum_g2 = float(tms_g2.sum_g)
        n_g1_high = int(tms_g1.n_high)
        n_g2_high = int(tms_g2.n_high)
        n_g1_low = int(tms_g1.n_low)
        n_g2_low = int(tms_g2.n_low)
        _assert_tile_invariants(
            tessellation_id=str(support_row[cols.TESSELLATION_ID]),
            tile_id=int(support_row[cols.TILE_ID]),
            n_spots=n_spots, sum_g1=sum_g1, sum_g2=sum_g2,
            n_g1_high=n_g1_high, n_g2_high=n_g2_high,
            n_both_high=n_both_high,
            n_g1_low=n_g1_low, n_g2_low=n_g2_low,
            n_both_low=n_both_low,
        )
        rows.append({
            **support_row,
            cols.SUM_G1: sum_g1,
            cols.SUM_G2: sum_g2,
            cols.N_G1_HIGH: n_g1_high,
            cols.N_G2_HIGH: n_g2_high,
            cols.N_BOTH_HIGH: n_both_high,
            cols.N_G1_LOW: n_g1_low,
            cols.N_G2_LOW: n_g2_low,
            cols.N_BOTH_LOW: n_both_low,
            cols.GENE1_DENSITY: (
                sum_g1 / n_spots if n_spots > 0 else pd.NA
            ),
            cols.GENE2_DENSITY: (
                sum_g2 / n_spots if n_spots > 0 else pd.NA
            ),
        })

    df = pd.DataFrame(rows, columns=list(TESSELLATION_TILES_COLUMNS))
    df = TessellationTilesSchema.validate(df, lazy=True)
    _assert_roi_id_functional_dependency(df, dict(tile_support.component_to_roi))
    return df


def _assert_tile_invariants(
    *,
    tessellation_id: str,
    tile_id: int,
    n_spots: int,
    sum_g1: float,
    sum_g2: float,
    n_g1_high: int,
    n_g2_high: int,
    n_both_high: int,
    n_g1_low: int,
    n_g2_low: int,
    n_both_low: int,
) -> None:
    """Guard pair-value counts; support geometry is validated once upstream."""
    ctx = (
        f"tile_id={tile_id!r} in tessellation_id={tessellation_id!r}"
    )
    if n_spots < 0:
        raise ValueError(f"{ctx}: n_spots={n_spots} is negative")
    for name, value in (
        ("n_g1_high", n_g1_high), ("n_g2_high", n_g2_high),
        ("n_both_high", n_both_high),
        ("n_g1_low", n_g1_low), ("n_g2_low", n_g2_low),
        ("n_both_low", n_both_low),
    ):
        if not (0 <= value <= n_spots):
            raise ValueError(
                f"{ctx}: {name}={value} outside [0, n_spots={n_spots}]"
            )
    if n_both_high > min(n_g1_high, n_g2_high):
        raise ValueError(
            f"{ctx}: n_both_high={n_both_high} exceeds "
            f"min(n_g1_high={n_g1_high}, n_g2_high={n_g2_high})"
        )
    if n_both_low > min(n_g1_low, n_g2_low):
        raise ValueError(
            f"{ctx}: n_both_low={n_both_low} exceeds "
            f"min(n_g1_low={n_g1_low}, n_g2_low={n_g2_low})"
        )
    # High + low on the same gene cannot together exceed the tile's
    # spot count: a spot is at most one of {high, low, neither}.
    if n_g1_high + n_g1_low > n_spots:
        raise ValueError(
            f"{ctx}: n_g1_high({n_g1_high}) + n_g1_low({n_g1_low}) "
            f"exceeds n_spots={n_spots}"
        )
    if n_g2_high + n_g2_low > n_spots:
        raise ValueError(
            f"{ctx}: n_g2_high({n_g2_high}) + n_g2_low({n_g2_low}) "
            f"exceeds n_spots={n_spots}"
        )


def _assert_roi_id_functional_dependency(
    df: pd.DataFrame,
    component_to_roi: Dict[str, str],
) -> None:
    """Enforce ``(sample_id, component_id) → exactly one roi_id`` and
    require that the emitted ``roi_id`` matches the typed
    ``sample_tess_result.component_to_roi`` authority.

    The redundant convenience ``roi_id`` column cannot silently drift
    from the typed component-to-parent mapping; if a caller ever
    tries to publish a mixed or invented parent, construction fails.
    """
    if df.empty:
        return
    per_key_unique = (
        df.groupby([cols.SAMPLE_ID, cols.COMPONENT_ID])[cols.ROI_ID]
        .nunique()
    )
    offenders = per_key_unique[per_key_unique > 1]
    if len(offenders) > 0:
        raise ValueError(
            "tessellation_tiles_df: (sample_id, component_id) with "
            f"multiple roi_id values: {offenders.to_dict()!r}. Every "
            "component maps to exactly one parent ROI."
        )
    # Cross-check every (sample_id, component_id) → roi_id pair against
    # the typed authority. `component_to_roi` values are the source of
    # truth; the emitted column must match.
    for (sid, cid), roi in (
        df.groupby([cols.SAMPLE_ID, cols.COMPONENT_ID])[cols.ROI_ID]
        .first()
        .items()
    ):
        expected = component_to_roi.get(str(cid))
        if expected is None:
            raise ValueError(
                f"tessellation_tiles_df: component {cid!r} not in "
                "sample_tess_result.component_to_roi authority."
            )
        if str(roi) != str(expected):
            raise ValueError(
                f"tessellation_tiles_df: (sample={sid!r}, component="
                f"{cid!r}) emitted roi_id={roi!r} but typed authority "
                f"has {expected!r}."
            )


__all__ = [
    "build_tessellation_tile_support",
    "build_tessellation_tiles_df",
    "TessellationTileSupport",
    "TESSELLATION_TILE_SUPPORT_COLUMNS",
    "TESSELLATION_TILES_COLUMNS",
]
