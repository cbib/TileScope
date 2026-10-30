"""Sample-level tessellation support index and pair density binding."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tilescope.analyzers.parent_roi_common import group_components_by_parent
from tilescope.analyzers.tessellation_outcomes import RegionExecutionStatus
from tilescope.cols import (
    COMPONENT_ID, DOMAIN_REGION_ID, GENE1_DENSITY, GENE2_DENSITY,
    N_SPOTS, REPLICATE_INDEX, ROI_ID, TILE_ID,
)
from tilescope.schema.helpers import natural_sort_key


@dataclass(frozen=True)
class RoiTessellationSupport:
    """Fixed support for one biological ROI."""

    component_ids: tuple[str, ...]
    eligible_component_ids: tuple[str, ...]
    n_source_spots: int
    n_eligible_regions: int
    n_eligible_spots: int

    @property
    def n_eligible_components(self) -> int:
        return len(self.eligible_component_ids)


@dataclass(frozen=True)
class RoiReplicatePool:
    """Pair-independent row selection and support for one ROI realization."""

    tile_row_positions: tuple[int, ...]
    n_contributing_components: int
    n_contributing_regions: int
    n_tiles: int
    n_spots: int


@dataclass(frozen=True)
class TessellationSupportIndex:
    """Sample-level ROI/replicate support and target/context row selections."""

    sample_id: str
    n_replicates_planned: int
    roi_ids: tuple[str, ...]
    support_by_roi: Mapping[str, RoiTessellationSupport]
    pool_by_roi_replicate: Mapping[tuple[str, int], RoiReplicatePool]
    context_support_by_roi: Mapping[str, RoiTessellationSupport]
    context_pool_by_roi_replicate: Mapping[tuple[str, int], RoiReplicatePool]
    tile_identity_columns: tuple[str, ...]
    tile_identity_rows: tuple[tuple, ...]


@dataclass(frozen=True)
class TessellationPoolingIndex:
    """Pair density vectors bound to one sample-level support index."""

    support_index: TessellationSupportIndex
    gene1_density: np.ndarray
    gene2_density: np.ndarray

    def __post_init__(self) -> None:
        g1 = np.array(self.gene1_density, dtype=float, copy=True)
        g2 = np.array(self.gene2_density, dtype=float, copy=True)
        expected = len(self.support_index.tile_identity_rows)
        if g1.ndim != 1 or g2.ndim != 1 or len(g1) != expected or len(g2) != expected:
            raise ValueError(
                "TessellationPoolingIndex density vectors must be 1-D and "
                f"match the {expected} support rows."
            )
        g1.flags.writeable = False
        g2.flags.writeable = False
        object.__setattr__(self, "gene1_density", g1)
        object.__setattr__(self, "gene2_density", g2)

    @property
    def sample_id(self) -> str:
        return self.support_index.sample_id

    @property
    def n_replicates_planned(self) -> int:
        return self.support_index.n_replicates_planned

    @property
    def roi_ids(self) -> tuple[str, ...]:
        return self.support_index.roi_ids

    @property
    def support_by_roi(self):
        return self.support_index.support_by_roi

    @property
    def pool_by_roi_replicate(self):
        return self.support_index.pool_by_roi_replicate

    @property
    def context_support_by_roi(self):
        return self.support_index.context_support_by_roi

    @property
    def context_pool_by_roi_replicate(self):
        return self.support_index.context_pool_by_roi_replicate

    def density_values(
        self, pool: RoiReplicatePool,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(pool.tile_row_positions, dtype=np.int64)
        return self.gene1_density[positions], self.gene2_density[positions]


def build_tessellation_support_index(
    *, sample_tess_result, tessellation_tile_support_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
) -> TessellationSupportIndex:
    """Build feature-independent ROI support and index selections once/sample."""
    tile_support = tessellation_tile_support_df.reset_index(drop=True)
    required = {COMPONENT_ID, REPLICATE_INDEX, N_SPOTS}
    missing = required - set(tile_support.columns)
    if missing:
        raise ValueError(
            "tessellation tile support is missing required columns "
            f"{sorted(missing)!r}."
        )
    component_to_roi = dict(sample_tess_result.component_to_roi)
    parents_to_components = (
        group_components_by_parent(component_to_roi)
        if component_to_roi else {}
    )
    roi_ids = tuple(sorted(
        support_status_df[ROI_ID].astype(str).unique(), key=natural_sort_key,
    ))
    unknown = set(parents_to_components) - set(roi_ids)
    if unknown:
        raise ValueError(
            "roi_support_status has no row for parent ROI(s) "
            f"{sorted(unknown)!r} despite retained component(s)."
        )

    source_spots = {
        str(row[ROI_ID]): int(row["source_spots"])
        for _, row in support_status_df.iterrows()
    }
    if len(source_spots) != len(support_status_df):
        raise ValueError("roi_support_status must contain exactly one row per ROI")

    eligible_regions: dict[str, int] = {}
    eligible_spots: dict[str, int] = {}
    contributing_regions: dict[tuple[str, int], int] = {}
    for component_id, component in sample_tess_result.components.items():
        cid = str(component_id)
        eligible = component.tessellation_eligible_region_diagnostics
        eligible_regions[cid] = len(eligible)
        eligible_spots[cid] = sum(d.n_component_spots for d in eligible)
        for diagnostic in component.region_replicate_diagnostics:
            if diagnostic.region_status is RegionExecutionStatus.CONTRIBUTED:
                key = (cid, int(diagnostic.replicate_index))
                contributing_regions[key] = contributing_regions.get(key, 0) + 1

    support_by_roi: dict[str, RoiTessellationSupport] = {}
    for roi_id in roi_ids:
        components = tuple(parents_to_components.get(roi_id, ()))
        eligible = tuple(
            cid for cid in components if eligible_regions.get(cid, 0) > 0
        )
        support = RoiTessellationSupport(
            component_ids=components,
            eligible_component_ids=eligible,
            n_source_spots=source_spots[roi_id],
            n_eligible_regions=sum(eligible_regions[cid] for cid in eligible),
            n_eligible_spots=sum(eligible_spots[cid] for cid in eligible),
        )
        if support.n_eligible_spots > support.n_source_spots:
            raise ValueError(
                f"Invariant violation for parent ROI {roi_id!r}: "
                f"n_eligible_spots ({support.n_eligible_spots}) > "
                f"n_source_spots ({support.n_source_spots})."
            )
        support_by_roi[roi_id] = support

    component_tiles = _index_tile_positions(tile_support)
    pool_by_roi_replicate = {
        (roi_id, replicate): _pool_components(
            support_by_roi[roi_id].eligible_component_ids,
            replicate, component_tiles, contributing_regions,
        )
        for roi_id in roi_ids
        for replicate in range(sample_tess_result.n_replicates_planned)
    }

    global_support = _combine_supports(
        tuple(support_by_roi[roi_id] for roi_id in roi_ids)
    )
    global_pools = {
        replicate: _pool_components(
            global_support.eligible_component_ids,
            replicate, component_tiles, contributing_regions,
        )
        for replicate in range(sample_tess_result.n_replicates_planned)
    }
    context_support_by_roi = {
        roi_id: _subtract_support(global_support, support_by_roi[roi_id])
        for roi_id in roi_ids
    }
    context_pool_by_roi_replicate = {
        (roi_id, replicate): _subtract_pool(
            global_pools[replicate], pool_by_roi_replicate[(roi_id, replicate)]
        )
        for roi_id in roi_ids
        for replicate in range(sample_tess_result.n_replicates_planned)
    }

    identity_columns = tuple(
        column for column in (
            COMPONENT_ID, DOMAIN_REGION_ID, REPLICATE_INDEX, TILE_ID,
        )
        if column in tile_support.columns
    )
    identity_rows = tuple(
        tuple(row)
        for row in tile_support[list(identity_columns)].itertuples(
            index=False, name=None,
        )
    )
    return TessellationSupportIndex(
        sample_id=str(sample_tess_result.sample_id),
        n_replicates_planned=int(sample_tess_result.n_replicates_planned),
        roi_ids=roi_ids,
        support_by_roi=MappingProxyType(support_by_roi),
        pool_by_roi_replicate=MappingProxyType(pool_by_roi_replicate),
        context_support_by_roi=MappingProxyType(context_support_by_roi),
        context_pool_by_roi_replicate=MappingProxyType(
            context_pool_by_roi_replicate,
        ),
        tile_identity_columns=identity_columns,
        tile_identity_rows=identity_rows,
    )


def bind_tessellation_pooling_index(
    *, support_index: TessellationSupportIndex,
    tessellation_tiles_df: pd.DataFrame,
) -> TessellationPoolingIndex:
    """Bind one pair's density vectors without rebuilding ROI/context support."""
    frame = tessellation_tiles_df.reset_index(drop=True)
    required = {GENE1_DENSITY, GENE2_DENSITY, *support_index.tile_identity_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "pair tessellation tiles are missing required columns "
            f"{sorted(missing)!r}."
        )
    identity_rows = tuple(
        tuple(row)
        for row in frame[list(support_index.tile_identity_columns)].itertuples(
            index=False, name=None,
        )
    )
    if identity_rows != support_index.tile_identity_rows:
        raise ValueError(
            "pair tessellation tile row identity/order disagrees with the "
            "sample-level TessellationSupportIndex."
        )
    return TessellationPoolingIndex(
        support_index=support_index,
        gene1_density=frame[GENE1_DENSITY].to_numpy(dtype=float),
        gene2_density=frame[GENE2_DENSITY].to_numpy(dtype=float),
    )


def build_tessellation_pooling_index(
    *, sample_tess_result, tessellation_tiles_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
) -> TessellationPoolingIndex:
    """Compatibility composition; production builds support once before pairs."""
    support_index = build_tessellation_support_index(
        sample_tess_result=sample_tess_result,
        tessellation_tile_support_df=tessellation_tiles_df,
        support_status_df=support_status_df,
    )
    return bind_tessellation_pooling_index(
        support_index=support_index,
        tessellation_tiles_df=tessellation_tiles_df,
    )


def _empty_pool() -> RoiReplicatePool:
    return RoiReplicatePool(
        tile_row_positions=(), n_contributing_components=0,
        n_contributing_regions=0, n_tiles=0, n_spots=0,
    )


def _pool_components(
    component_ids: Sequence[str], replicate: int,
    component_tiles: Mapping[tuple[str, int], tuple[tuple[int, ...], int]],
    contributing_regions: Mapping[tuple[str, int], int],
) -> RoiReplicatePool:
    positions: list[int] = []
    n_regions = 0
    n_components = 0
    n_spots = 0
    for component_id in component_ids:
        key = (str(component_id), int(replicate))
        n_regions += contributing_regions.get(key, 0)
        indexed = component_tiles.get(key)
        if indexed is not None and indexed[0]:
            positions.extend(indexed[0])
            n_spots += indexed[1]
            n_components += 1
    return RoiReplicatePool(
        tile_row_positions=tuple(positions),
        n_contributing_components=n_components,
        n_contributing_regions=n_regions,
        n_tiles=len(positions),
        n_spots=n_spots,
    )


def _combine_supports(
    supports: Sequence[RoiTessellationSupport],
) -> RoiTessellationSupport:
    return RoiTessellationSupport(
        component_ids=tuple(
            cid for support in supports for cid in support.component_ids
        ),
        eligible_component_ids=tuple(
            cid for support in supports for cid in support.eligible_component_ids
        ),
        n_source_spots=sum(s.n_source_spots for s in supports),
        n_eligible_regions=sum(s.n_eligible_regions for s in supports),
        n_eligible_spots=sum(s.n_eligible_spots for s in supports),
    )


def _subtract_support(
    whole: RoiTessellationSupport, target: RoiTessellationSupport,
) -> RoiTessellationSupport:
    target_components = set(target.component_ids)
    target_eligible = set(target.eligible_component_ids)
    if not target_components.issubset(whole.component_ids):
        raise ValueError("Target ROI components are not a subset of sample support.")
    if not target_eligible.issubset(whole.eligible_component_ids):
        raise ValueError(
            "Target ROI eligible components are not a subset of sample support."
        )
    if (
        target.n_source_spots > whole.n_source_spots
        or target.n_eligible_regions > whole.n_eligible_regions
        or target.n_eligible_spots > whole.n_eligible_spots
    ):
        raise ValueError("Target ROI support counts exceed sample-wide support.")
    return RoiTessellationSupport(
        component_ids=tuple(
            cid for cid in whole.component_ids if cid not in target_components
        ),
        eligible_component_ids=tuple(
            cid for cid in whole.eligible_component_ids if cid not in target_eligible
        ),
        n_source_spots=whole.n_source_spots - target.n_source_spots,
        n_eligible_regions=whole.n_eligible_regions - target.n_eligible_regions,
        n_eligible_spots=whole.n_eligible_spots - target.n_eligible_spots,
    )


def _subtract_pool(
    whole: RoiReplicatePool, target: RoiReplicatePool,
) -> RoiReplicatePool:
    target_positions = set(target.tile_row_positions)
    if not target_positions.issubset(whole.tile_row_positions):
        raise ValueError("Target ROI tile positions are not a subset of sample pool.")
    if (
        target.n_contributing_components > whole.n_contributing_components
        or target.n_contributing_regions > whole.n_contributing_regions
        or target.n_tiles > whole.n_tiles
        or target.n_spots > whole.n_spots
    ):
        raise ValueError("Target ROI pool counts exceed sample-wide pool.")
    positions = tuple(
        pos for pos in whole.tile_row_positions if pos not in target_positions
    )
    return RoiReplicatePool(
        tile_row_positions=positions,
        n_contributing_components=(
            whole.n_contributing_components - target.n_contributing_components
        ),
        n_contributing_regions=(
            whole.n_contributing_regions - target.n_contributing_regions
        ),
        n_tiles=whole.n_tiles - target.n_tiles,
        n_spots=whole.n_spots - target.n_spots,
    )


def _index_tile_positions(
    tile_support_df: pd.DataFrame,
) -> dict[tuple[str, int], tuple[tuple[int, ...], int]]:
    if tile_support_df.empty:
        return {}
    indexed: dict[tuple[str, int], tuple[tuple[int, ...], int]] = {}
    for (component_id, replicate), group in tile_support_df.groupby(
        [COMPONENT_ID, REPLICATE_INDEX], sort=False,
    ):
        positions = tuple(int(i) for i in group.index)
        indexed[(str(component_id), int(replicate))] = (
            positions, int(group[N_SPOTS].sum()),
        )
    return indexed


__all__ = [
    "RoiReplicatePool",
    "RoiTessellationSupport",
    "TessellationPoolingIndex",
    "TessellationSupportIndex",
    "bind_tessellation_pooling_index",
    "build_tessellation_pooling_index",
    "build_tessellation_support_index",
]
