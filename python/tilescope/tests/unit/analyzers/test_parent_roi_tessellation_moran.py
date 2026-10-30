"""Contract tests for :func:`build_roi_tessellation_moran_df`.

Per-replicate bivariate Moran on tile densities + region-blocked kNN
adjacency; reduced to median + Q25 + Q75 across replicates that cover
every eligible sibling component. Replicates missing a component are
skipped; if none survive the ROI reports NO_VALID_REPLICATES.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Sequence

import numpy as np
import pandas as pd
from shapely.geometry import box

from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.analyzers.parent_roi_tessellation_common import (
    RoiReplicatePool, RoiTessellationSupport, TessellationPoolingIndex,
    TessellationSupportIndex,
)
from tilescope.analyzers.parent_roi_tessellation_moran import (
    build_roi_tessellation_moran_df as _build_roi_tessellation_moran_df,
)
from tilescope.cols import (
    ADJACENCY_SUPPORTED_FRACTION, COMPONENT_ID, DOMAIN_REGION_ID,
    GENE1_DENSITY, GENE2_DENSITY,
    MORAN_VALUE, MORAN_VALUE_P25_ACROSS_REPLICATES,
    MORAN_VALUE_P75_ACROSS_REPLICATES, N_ADJACENCY_SUPPORTED_UNITS,
    N_FINITE_REPLICATES, N_REPLICATES, N_SPOTS, PARENT_MORAN_STATUS,
    REPLICATE_INDEX, ROI_ID, SAMPLE_ID, TILE_AREA, TILE_GEOMETRY_WKT, TILE_ID,
)


def build_roi_tessellation_moran_df(**kwargs):
    return _build_roi_tessellation_moran_df(
        g1_id="Neuron", g2_id="AC", **kwargs,
    )


def _tile_row(cid, replicate, tile_id, cx, cy, g1, g2, *, region_id=0,
              roi_id="R"):
    poly = box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    return {
        SAMPLE_ID: "S",
        ROI_ID: roi_id,
        COMPONENT_ID: cid,
        DOMAIN_REGION_ID: region_id,
        REPLICATE_INDEX: replicate,
        TILE_ID: tile_id,
        N_SPOTS: 4,
        GENE1_DENSITY: g1,
        GENE2_DENSITY: g2,
        TILE_GEOMETRY_WKT: poly.wkt,
        TILE_AREA: 1.0,
    }


def _tiles_df(rows):
    return pd.DataFrame(rows).astype({
        REPLICATE_INDEX: int, DOMAIN_REGION_ID: int, TILE_ID: int, N_SPOTS: int,
    })


def _support(roi_ids: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        ROI_ID: list(roi_ids),
        "source_spots": [100] * len(roi_ids),
        "spatial_support_class": ["supported"] * len(roi_ids),
    })


def _pooling_index_from_tiles(
    tiles: pd.DataFrame,
    *,
    n_replicates: int,
    roi_components: dict[str, tuple[str, ...]],
) -> TessellationPoolingIndex:
    """Build a valid pooling index whose positions index into `tiles`."""
    support_by_roi = {
        roi: RoiTessellationSupport(
            component_ids=components,
            eligible_component_ids=components,
            n_source_spots=100,
            n_eligible_regions=len(components),
            n_eligible_spots=100,
        )
        for roi, components in roi_components.items()
    }
    component_to_roi = {
        cid: roi for roi, cids in roi_components.items() for cid in cids
    }
    pool_by_roi_replicate: dict[tuple[str, int], RoiReplicatePool] = {}
    for roi, components in roi_components.items():
        for t in range(n_replicates):
            positions: list[int] = []
            contributing_components = set()
            for idx, row in tiles.iterrows():
                if int(row[REPLICATE_INDEX]) != t:
                    continue
                cid = str(row[COMPONENT_ID])
                if component_to_roi.get(cid) != roi:
                    continue
                positions.append(int(idx))
                contributing_components.add(cid)
            pool_by_roi_replicate[(roi, t)] = RoiReplicatePool(
                tile_row_positions=tuple(positions),
                n_contributing_components=len(contributing_components),
                n_contributing_regions=len(positions),
                n_tiles=len(positions),
                n_spots=int(tiles.iloc[positions][N_SPOTS].sum())
                if positions else 0,
            )
    identity_cols = (COMPONENT_ID, DOMAIN_REGION_ID, REPLICATE_INDEX, TILE_ID)
    identity_rows = tuple(
        tuple(row) for row in tiles[list(identity_cols)].itertuples(
            index=False, name=None,
        )
    )
    empty_pool = RoiReplicatePool((), 0, 0, 0, 0)
    ctx_pool = {
        key: empty_pool for key in pool_by_roi_replicate
    }
    ctx_support = {
        roi: RoiTessellationSupport((), (), 0, 0, 0) for roi in roi_components
    }
    support_index = TessellationSupportIndex(
        sample_id="S",
        n_replicates_planned=n_replicates,
        roi_ids=tuple(roi_components),
        support_by_roi=MappingProxyType(support_by_roi),
        pool_by_roi_replicate=MappingProxyType(pool_by_roi_replicate),
        context_support_by_roi=MappingProxyType(ctx_support),
        context_pool_by_roi_replicate=MappingProxyType(ctx_pool),
        tile_identity_columns=identity_cols,
        tile_identity_rows=identity_rows,
    )
    return TessellationPoolingIndex(
        support_index=support_index,
        gene1_density=tiles[GENE1_DENSITY].to_numpy(dtype=float),
        gene2_density=tiles[GENE2_DENSITY].to_numpy(dtype=float),
    )


def _linear_replicate_rows(replicate: int, cid: str = "CA"):
    return [
        _tile_row(cid, replicate, i, float(i), 0.0,
                  float(i + 1), float(i + 1))
        for i in range(12)
    ]


def test_positive_spatial_autocorrelation_yields_positive_median():
    n_replicates = 3
    rows = []
    for t in range(n_replicates):
        rows.extend(_linear_replicate_rows(t))
    tiles = _tiles_df(rows)
    pi = _pooling_index_from_tiles(
        tiles, n_replicates=n_replicates,
        roi_components={"R": ("CA",)},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pi, tessellation_tiles_df=tiles,
        support_status_df=_support(("R",)),
        k_neighbors=2, moran_enabled=True, sample_id="S",
    )
    finite = df[df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert (finite[MORAN_VALUE] > 0).all()
    assert finite[MORAN_VALUE_P25_ACROSS_REPLICATES].notna().all()
    assert finite[MORAN_VALUE_P75_ACROSS_REPLICATES].notna().all()
    assert (finite[N_FINITE_REPLICATES] == n_replicates).all()
    assert (finite[N_REPLICATES] == n_replicates).all()
    assert (finite[N_ADJACENCY_SUPPORTED_UNITS] == 12).all()
    assert (finite[ADJACENCY_SUPPORTED_FRACTION] == 1.0).all()


def test_complete_small_graph_reports_no_spatial_discrimination():
    rows = [
        _tile_row(
            "CA", 0, i, float(i), 0.0, float(i + 1), float(i + 1),
        )
        for i in range(4)
    ]
    tiles = _tiles_df(rows)
    pooling = _pooling_index_from_tiles(
        tiles, n_replicates=1, roi_components={"R": ("CA",)},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pooling,
        tessellation_tiles_df=tiles,
        support_status_df=_support(("R",)),
        k_neighbors=6,
        moran_enabled=True,
        sample_id="S",
    )
    assert (
        df[PARENT_MORAN_STATUS]
        == ParentMoranStatus.NO_SPATIAL_DISCRIMINATION.value
    ).all()
    assert df[MORAN_VALUE].isna().all()


def test_anti_correlated_tile_densities_yield_negative_median():
    n_replicates = 2
    rows = []
    for t in range(n_replicates):
        for i in range(12):
            rows.append(_tile_row(
                "CA", t, i, float(i), 0.0, float(i + 1), float(12 - i),
            ))
    tiles = _tiles_df(rows)
    pi = _pooling_index_from_tiles(
        tiles, n_replicates=n_replicates,
        roi_components={"R": ("CA",)},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pi, tessellation_tiles_df=tiles,
        support_status_df=_support(("R",)),
        k_neighbors=2, moran_enabled=True, sample_id="S",
    )
    finite = df[df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert (finite[MORAN_VALUE] < 0).all()


def test_incomplete_support_replicate_is_skipped():
    """A replicate missing an eligible sibling component is skipped;
    replicates with full sibling coverage still contribute."""
    n_replicates = 3
    # ROI R has eligible components (CA, CB). Complete replicate 0
    # contributes both; replicate 1 contributes only CA (skipped);
    # replicate 2 contributes both.
    rows = _linear_replicate_rows(0, cid="CA") + [
        _tile_row("CB", 0, 100 + i, float(i), 2.0,
                  float(i + 1), float(i + 1)) for i in range(12)
    ]
    rows.extend(_linear_replicate_rows(1, cid="CA"))  # missing CB
    rows.extend(_linear_replicate_rows(2, cid="CA") + [
        _tile_row("CB", 2, 200 + i, float(i), 2.0,
                  float(i + 1), float(i + 1)) for i in range(12)
    ])
    tiles = _tiles_df(rows)
    pi = _pooling_index_from_tiles(
        tiles, n_replicates=n_replicates,
        roi_components={"R": ("CA", "CB")},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pi, tessellation_tiles_df=tiles,
        support_status_df=_support(("R",)),
        k_neighbors=2, moran_enabled=True, sample_id="S",
    )
    finite = df[df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert (finite[N_FINITE_REPLICATES] == 2).all()
    assert (finite[N_REPLICATES] == n_replicates).all()


def test_no_valid_replicates_when_every_replicate_incomplete():
    """Every replicate misses at least one eligible component → NO_VALID_REPLICATES."""
    n_replicates = 2
    # Two eligible components (CA, CB), but each replicate only carries CA.
    rows = []
    for t in range(n_replicates):
        rows.extend(_linear_replicate_rows(t, cid="CA"))
    tiles = _tiles_df(rows)
    pi = _pooling_index_from_tiles(
        tiles, n_replicates=n_replicates,
        roi_components={"R": ("CA", "CB")},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pi, tessellation_tiles_df=tiles,
        support_status_df=_support(("R",)),
        k_neighbors=2, moran_enabled=True, sample_id="S",
    )
    assert (
        df[PARENT_MORAN_STATUS]
        == ParentMoranStatus.NO_VALID_REPLICATES.value
    ).all()
    assert df[MORAN_VALUE].isna().all()


def test_no_tessellation_eligible_component_reports_status():
    n_replicates = 2
    empty_tiles = pd.DataFrame(columns=[
        SAMPLE_ID, ROI_ID, COMPONENT_ID, DOMAIN_REGION_ID,
        REPLICATE_INDEX, TILE_ID, N_SPOTS,
        GENE1_DENSITY, GENE2_DENSITY, TILE_GEOMETRY_WKT, TILE_AREA,
    ]).astype({
        REPLICATE_INDEX: int, DOMAIN_REGION_ID: int, TILE_ID: int, N_SPOTS: int,
    })
    pi = _pooling_index_from_tiles(
        empty_tiles, n_replicates=n_replicates,
        roi_components={"R": ()},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pi, tessellation_tiles_df=empty_tiles,
        support_status_df=_support(("R",)),
        k_neighbors=2, moran_enabled=True, sample_id="S",
    )
    assert (
        df[PARENT_MORAN_STATUS]
        == ParentMoranStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    ).all()


def test_moran_disabled_emits_total_table():
    n_replicates = 2
    rows = []
    for t in range(n_replicates):
        for i in range(4):
            rows.append(_tile_row(
                "CA", t, i, float(i), 0.0, float(i), float(i),
            ))
    tiles = _tiles_df(rows)
    pi = _pooling_index_from_tiles(
        tiles, n_replicates=n_replicates,
        roi_components={"R": ("CA",)},
    )
    df = build_roi_tessellation_moran_df(
        pooling_index=pi, tessellation_tiles_df=tiles,
        support_status_df=_support(("R",)),
        k_neighbors=2, moran_enabled=False, sample_id="S",
    )
    assert len(df) == 2
    assert (
        df[PARENT_MORAN_STATUS] == ParentMoranStatus.MORAN_DISABLED.value
    ).all()
    assert df[MORAN_VALUE].isna().all()
