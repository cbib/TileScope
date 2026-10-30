"""Shared fixtures for ROI-grain tessellation analyzer tests.

Helper module (not a test file — no ``test_`` functions). Used by
``test_parent_roi_tessellation_correlation.py`` and
``test_parent_roi_tessellation_enrichment.py`` to avoid duplicating
the ~130 LOC of ComponentTessellationResult stubbing.
"""
from __future__ import annotations

from tilescope.component_data import Tessellation
from tilescope.component_data import Tile
from tilescope.analyzers.tessellation_outcomes import RegionExecutionStatus, RegionReplicateTessellationDiagnostic
from tilescope.component_data import ComponentData
from tilescope.cols import COORD_X as _COORD_X, COORD_Y as _COORD_Y, ROI_ID as _ROI_ID, GENE1 as _GENE1, GENE2 as _GENE2, SOURCE_ROW_ID as _SRC_ROW_ID
from tilescope.cols import GENE1, GENE2, SOURCE_ROW_ID


from types import MappingProxyType
from unittest.mock import Mock

import numpy as np
import pandas as pd
from shapely.geometry import box as _box

from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionIneligibilityReason,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.cols import (
    COMPONENT_ID, GENE1_DENSITY, GENE2_DENSITY, N_SPOTS, REPLICATE_INDEX,
)
from tilescope.spatial.component_domains import ComponentDomainPartition


def contrib(replicate_index: int, *, n_eligible: int,
            n_contributing: int, n_tiles: int):
    """Per-replicate CONTRIBUTED-region declaration consumed by
    :func:`component_result` to synthesise matching diagnostics."""
    return {
        "replicate_index": int(replicate_index),
        "n_eligible": int(n_eligible),
        "n_contributing": int(n_contributing),
        "n_tiles": int(n_tiles),
    }


def support_frame(rows):
    return pd.DataFrame(
        rows, columns=["roi_id", "source_spots", "spatial_support_class"],
    )


def region_diag(rid: int, n_spots: int, eligible: bool = True, cid: str = "C"):
    return RegionTessellationEligibilityDiagnostic(
        component_id=cid, domain_region_id=rid, n_component_spots=n_spots,
        region_area=1.0,
        ineligibility_reason=(
            None if eligible else RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN
        ),
    )


def component_result(
    *,
    cid: str = "C",
    parent_roi_id: str = "R",
    region_diagnostics,
    contributed_regions=(),
):
    """Stub ComponentTessellationResult with synthesised region-replicate
    diagnostics + matching per-tile Tessellations. The first
    ``n_contributing`` eligible regions get CONTRIBUTED with a single
    stub Tile; the remainder get NO_TILES."""
    # Real minimal ComponentData so the bank builder can lift a
    # ComponentSupport out of it (points / polygon / source_row_ids).
    _n = sum(d.n_component_spots for d in region_diagnostics)
    _pts = np.zeros((max(_n, 1), 2), dtype=float)
    _expr = pd.DataFrame({
        _SRC_ROW_ID: np.arange(max(_n, 1), dtype=np.int64),
        _COORD_X: _pts[:, 0], _COORD_Y: _pts[:, 1],
        _GENE1: np.zeros(max(_n, 1)), _GENE2: np.zeros(max(_n, 1)),
        _ROI_ID: [parent_roi_id] * max(_n, 1),
    })
    stub = ComponentData(
        component_id=cid, points=_pts, canvas=None,
        polygon=_box(0.0, 0.0, 1.0, 1.0), expression_data=_expr,
    )
    eligible_rids = sorted(
        d.domain_region_id for d in region_diagnostics
        if d.is_tessellation_eligible
    )
    rids = sorted({d.domain_region_id for d in region_diagnostics})
    ids_by_rid: dict = {}
    fp_by_rid: dict = {}
    canonical_ids_by_src: dict = {}
    next_src = 0
    for rid in rids:
        d = next(rd for rd in region_diagnostics if rd.domain_region_id == rid)
        assigned = tuple(range(next_src, next_src + d.n_component_spots))
        ids_by_rid[rid] = assigned
        fp_by_rid[rid] = f"stub_fp_r{rid}"
        for sid in assigned:
            canonical_ids_by_src[sid] = rid
        next_src += d.n_component_spots
    partition = ComponentDomainPartition(
        canonical_regions=tuple(
            _box(float(rid), 0.0, float(rid) + 1.0, 1.0) for rid in rids
        ),
        canonical_ids_by_source_row=MappingProxyType(canonical_ids_by_src),
        source_row_ids_by_canonical_id=MappingProxyType(ids_by_rid),
        source_row_fingerprint_by_canonical_id=MappingProxyType(fp_by_rid),
    )
    rr_diags = []
    tessellations: dict = {}
    stub_polygon = _box(0.0, 0.0, 1.0, 1.0)
    for c in contributed_regions:
        t = int(c["replicate_index"])
        n_contrib = int(c["n_contributing"])
        for i, rid in enumerate(eligible_rids):
            if i < n_contrib:
                rr_diags.append(RegionReplicateTessellationDiagnostic(
                    component_id=cid, domain_region_id=rid,
                    replicate_index=t, n_tiles=1,
                    region_status=RegionExecutionStatus.CONTRIBUTED,
                ))
                tile_stub = Tile(
                    tile_id=0, polygon=stub_polygon,
                    point_indices=np.array([], dtype=int),
                    n_spots=0, area=1.0, tessellation_id=f"{cid}_r{rid}_t{t}",
                    domain_region_id=rid,
                )
                tessellations.setdefault(rid, {})[t] = Tessellation(
                    tessellation_id=f"{cid}_r{rid}_t{t}",
                    component_id=cid, tiles=[tile_stub],
                    method="voronoi", tile_min_points=1, fraction=1.0,
                )
            else:
                rr_diags.append(RegionReplicateTessellationDiagnostic(
                    component_id=cid, domain_region_id=rid,
                    replicate_index=t, n_tiles=0,
                    region_status=RegionExecutionStatus.NO_TILES,
                ))
    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=stub,
        tessellations=MappingProxyType({
            rid: MappingProxyType(dict(rmap))
            for rid, rmap in tessellations.items()
        }),
        region_diagnostics=tuple(region_diagnostics),
        region_replicate_diagnostics=tuple(rr_diags),
        domain_partition=partition,
    )


def tile_df(g1, g2, n_spots):
    return pd.DataFrame({
        GENE1_DENSITY: g1, GENE2_DENSITY: g2, N_SPOTS: n_spots,
    })


def tiles_from_dict(cid_to_rep):
    """Flatten a nested ``{component_id: {replicate_index: per_tile_df}}``
    fixture into the canonical ``tessellation_tiles_df`` row shape the
    ROI × replicate builders consume."""
    frames = []
    for cid, rep_dfs in cid_to_rep.items():
        for rep, df in rep_dfs.items():
            f = df.copy()
            f[COMPONENT_ID] = str(cid)
            f[REPLICATE_INDEX] = int(rep)
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=[
            COMPONENT_ID, REPLICATE_INDEX,
            GENE1_DENSITY, GENE2_DENSITY, N_SPOTS,
        ])
    return pd.concat(frames, ignore_index=True)


def real_component_result(
    *,
    cid: str,
    parent_roi_id: str,
    g1_values,
    g2_values,
    tile_membership_per_replicate,
    source_row_offset: int = 0,
):
    """Build a ComponentTessellationResult whose ``component_data``
    carries real GENE1 / GENE2 columns and whose per-replicate tiles
    carry matching component-local ``point_indices``. Used only where
    the reduced-null primitive needs real expression data."""
    g1_arr = np.asarray(g1_values, dtype=float)
    g2_arr = np.asarray(g2_values, dtype=float)
    n = len(g1_arr)
    source_row_ids = np.arange(
        source_row_offset, source_row_offset + n, dtype=np.int64,
    )
    expression = pd.DataFrame({
        GENE1: g1_arr,
        GENE2: g2_arr,
        SOURCE_ROW_ID: source_row_ids,
    })
    canvas_stub = Mock(spec=[])
    component = ComponentData(
        component_id=cid,
        points=np.zeros((n, 2), dtype=float),
        canvas=canvas_stub,
        polygon=_box(0.0, 0.0, 1.0, 1.0),
        expression_data=expression,
    )
    partition = ComponentDomainPartition(
        canonical_regions=(_box(0.0, 0.0, 1.0, 1.0),),
        canonical_ids_by_source_row=MappingProxyType(
            {int(i): 0 for i in source_row_ids},
        ),
        source_row_ids_by_canonical_id=MappingProxyType(
            {0: tuple(int(i) for i in source_row_ids)},
        ),
        source_row_fingerprint_by_canonical_id=MappingProxyType(
            {0: "stub_fp"},
        ),
    )
    rd = RegionTessellationEligibilityDiagnostic(
        component_id=cid, domain_region_id=0, n_component_spots=n,
        region_area=1.0, ineligibility_reason=None,
    )
    rr_diags = []
    tessellations: dict = {0: {}}
    stub_polygon = _box(0.0, 0.0, 1.0, 1.0)
    for t, tile_index_lists in enumerate(tile_membership_per_replicate):
        tiles = []
        for tile_id, idx in enumerate(tile_index_lists):
            idx_arr = np.unique(np.asarray(idx, dtype=np.int64))
            tiles.append(Tile(
                tile_id=tile_id, polygon=stub_polygon,
                point_indices=idx_arr,
                n_spots=int(idx_arr.size),
                area=1.0, tessellation_id=f"{cid}_r0_t{t}",
                domain_region_id=0,
            ))
        tessellations[0][t] = Tessellation(
            tessellation_id=f"{cid}_r0_t{t}",
            component_id=cid, tiles=tiles,
            method="voronoi", tile_min_points=1, fraction=1.0,
        )
        rr_diags.append(RegionReplicateTessellationDiagnostic(
            component_id=cid, domain_region_id=0,
            replicate_index=t, n_tiles=len(tiles),
            region_status=RegionExecutionStatus.CONTRIBUTED,
        ))
    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=component,
        tessellations=MappingProxyType({
            0: MappingProxyType(dict(tessellations[0])),
        }),
        region_diagnostics=(rd,),
        region_replicate_diagnostics=tuple(rr_diags),
        domain_partition=partition,
    )
