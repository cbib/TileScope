"""Corrective contract tests for the consensus spatial bank.

Seven semantic scenarios:

1. Direct membership uses canonical SOURCE_ROW_ID (not component-local
   position), and every spot or cell has exactly one membership row.
2. Empty / all-scattered path composes a real SampleSpatialBank.
3. Result-view identity delegation: hex.geometry `is` bank hex; component
   support `is` bank component; sample spatial_bank required.
4. TileHexOverlap edges are canonically ordered, unique, reference known
   hexes, and carry positive finite area weights.
5. Ensemble occupancy is reconstructible from tile occupancy × overlap
   fraction; the bank does not need feature-side values.
6. Failure-direction parametrisation for ComponentConsensusSupport invariants.
7. Overlap geometry is computed once — `_aggregate_hex_from_ensemble` runs
   exactly once per (region, hex) even when the bank materialises edges.
"""
from __future__ import annotations

from tilescope.analyzers.consensus_stage import SampleConsensusResult
from tilescope.analyzers.sample_support_stage import build_empty_sample_support_bank
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
from tilescope.analyzers import consensus_stage


from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.analyzers.consensus_stage import run_consensus_stage
from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult, SampleTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus, RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.cols import GENE1, GENE2, SOURCE_ROW_ID
from tilescope.component_data import ComponentData, Tessellation, Tile
from tilescope.spatial.canvas import make_canvas
from tilescope.spatial.component_domains import partition_component_domain
from tilescope.spatial.consensus_bank import (
    ComponentConsensusSupport,
    ConsensusHexGeometry,
    DirectHexMembership,
    TileHexOverlap,
)
from tilescope.settings import load_run_config


def _build_simple_sample():
    """One component with one region, one replicate, four spots on a 2x2 grid."""
    region_polygon = box(0.0, 0.0, 4.0, 4.0)
    coords = np.array([
        [1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0],
    ], dtype=float)
    n = len(coords)
    # Canonical SOURCE_ROW_IDs offset from component-local index to prove
    # the direct-membership rewrite consumes SOURCE_ROW_ID, not enumerate().
    source_row_ids = np.array([100, 101, 102, 103], dtype=np.int64)
    canvas = make_canvas(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}))
    expr = pd.DataFrame({
        GENE1: np.ones(n),
        GENE2: np.arange(n, dtype=float),
        SOURCE_ROW_ID: source_row_ids,
        "x": coords[:, 0], "y": coords[:, 1],
    })
    comp_data = ComponentData(
        component_id="C", points=coords.astype(float),
        canvas=canvas, polygon=region_polygon, expression_data=expr,
    )
    tid = "C_r0_rep0"
    mask = np.ones(n, dtype=bool)
    tess = Tessellation(
        tessellation_id=tid, component_id="C",
        tiles=[Tile(
            tile_id=0, polygon=region_polygon, point_indices=np.flatnonzero(mask),
            n_spots=int(mask.sum()), area=float(region_polygon.area),
            tessellation_id=tid, domain_region_id=0,
        )],
        method="voronoi", tile_min_points=1, fraction=1.0, rng_seed=42,
    )
    partition = partition_component_domain(
        comp_data.polygon, coords, source_row_ids,
    )
    comp_result = ComponentTessellationResult(
        parent_roi_id="R", component_data=comp_data,
        tessellations=MappingProxyType({0: {0: tess}}),
        region_diagnostics=(RegionTessellationEligibilityDiagnostic(
            component_id="C", domain_region_id=0,
            n_component_spots=n, region_area=float(region_polygon.area),
            ineligibility_reason=None,
        ),),
        region_replicate_diagnostics=(RegionReplicateTessellationDiagnostic(
            component_id="C", domain_region_id=0, replicate_index=0,
            n_tiles=1, region_status=RegionExecutionStatus.CONTRIBUTED,
        ),),
        domain_partition=partition,
    )
    sample_tess = SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({"C": comp_result}),
        n_replicates_planned=1,
    )
    return sample_tess, canvas, source_row_ids


def _run_stage(sample_tess, canvas):
    return run_consensus_stage(
        sample_tessellation_result=sample_tess, canvas=canvas,
        sample_anchor=(0.0, 0.0),
        consensus_config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
            'consensus': {
                'min_tess_coverage_fraction': 0.1,
            },
        },
        ),
    )


# ---------------------------------------------------------------------------
# 1. DirectHexMembership uses canonical SOURCE_ROW_ID
# ---------------------------------------------------------------------------


def test_direct_hex_membership_uses_canonical_source_row_ids():
    sample_tess, canvas, source_row_ids = _build_simple_sample()
    result = _run_stage(sample_tess, canvas)
    support = result.components["C"].support
    membership = support.direct_hex_membership
    stored_ids = sorted(m.source_row_id for m in membership)
    assert stored_ids == sorted(source_row_ids.tolist()), (
        f"membership stores {stored_ids}; expected canonical {source_row_ids.tolist()}"
    )
    # Exactly one row per source spot.
    assert len(membership) == len(source_row_ids)


# ---------------------------------------------------------------------------
# 2. Empty / all-scattered path composes a real SampleSpatialBank
# ---------------------------------------------------------------------------


def test_all_scattered_path_composes_spatial_bank():
    """All-scattered composition via the sample-authority helpers."""
    source_expr = pd.DataFrame({
        "x": [0.0, 1.0], "y": [0.0, 1.0],
        GENE1: [0.0, 1.0], GENE2: [0.0, 1.0],
        SOURCE_ROW_ID: [0, 1],
        "roi_id": ["R1", "R1"],
    })
    empty_bank = build_empty_sample_support_bank(
        sample_id="S",
        source_spot_adjacency=build_source_spot_adjacency(
            build_source_spot_table_from_frame(source_expr),
        ),
        config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
        },
        ), n_replicates=1, base_seed=0,
    )
    cons = SampleConsensusResult.from_components(
        sample_id="S", components=MappingProxyType({}),
        n_replicates_planned=1,
        sample_anchor_x=0.0, sample_anchor_y=0.0,
        support_bank=empty_bank,
    )
    assert cons.spatial_bank is not None
    assert cons.spatial_bank.support_bank.sample_id == "S"
    assert cons.spatial_bank.consensus_bank.sample_id == "S"
    assert dict(cons.spatial_bank.consensus_bank.components) == {}


# ---------------------------------------------------------------------------
# 3. Identity delegation: view records reference bank objects
# ---------------------------------------------------------------------------


def test_view_records_are_is_identical_to_bank_records():
    sample_tess, canvas, _ = _build_simple_sample()
    result = _run_stage(sample_tess, canvas)
    comp = result.components["C"]
    bank_support = result.spatial_bank.consensus_bank.components["C"]
    assert comp.support is bank_support
    for i, hx in enumerate(comp.hexes):
        assert hx.geometry is bank_support.hexes[i]


# ---------------------------------------------------------------------------
# 4. Overlap-edge canonical ordering / uniqueness / referential integrity
# ---------------------------------------------------------------------------


def test_overlap_edges_canonical_and_reference_known_hexes():
    sample_tess, canvas, _ = _build_simple_sample()
    result = _run_stage(sample_tess, canvas)
    support = result.components["C"].support
    keys = [
        (e.domain_region_id, e.replicate_index, e.tile_id, e.hex_id)
        for e in support.tile_hex_overlaps
    ]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    canonical_hex_ids = {h.hex_id for h in support.hexes}
    for edge in support.tile_hex_overlaps:
        assert edge.hex_id in canonical_hex_ids
        assert edge.overlap_area > 0 and np.isfinite(edge.overlap_area)
        assert edge.tile_area > 0 and np.isfinite(edge.tile_area)


# ---------------------------------------------------------------------------
# 5. Ensemble occupancy reconstructible from bank-owned edges
# ---------------------------------------------------------------------------


def test_ensemble_occupancy_reconstructs_from_bank_edges():
    sample_tess, canvas, _ = _build_simple_sample()
    result = _run_stage(sample_tess, canvas)
    comp = result.components["C"]
    bank_support = result.spatial_bank.consensus_bank.components["C"]
    tile_n_spots = {}
    for cid, cr in sample_tess.components.items():
        for rid, reps in cr.tessellations.items():
            for rep_idx, tess in reps.items():
                for tile in tess.tiles:
                    tile_n_spots[
                        (rid, int(rep_idx), int(tile.tile_id))
                    ] = int(tile.n_spots)
    reconstructed = {int(h.hex_id): 0.0 for h in comp.hexes}
    for e in bank_support.tile_hex_overlaps:
        n_spots = tile_n_spots[(e.domain_region_id, e.replicate_index, e.tile_id)]
        reconstructed[int(e.hex_id)] += (e.overlap_area / e.tile_area) * n_spots
    for h in comp.hexes:
        assert reconstructed[int(h.hex_id)] == pytest.approx(
            h.n_spots_ensemble_weighted, rel=0, abs=1e-9,
        )


# ---------------------------------------------------------------------------
# 6. ComponentConsensusSupport invariant failure directions
# ---------------------------------------------------------------------------


def _base_geometry(hex_id: int = 0) -> ConsensusHexGeometry:
    return ConsensusHexGeometry(
        component_id="C", hex_id=hex_id, domain_region_id=0,
        polygon=box(float(hex_id), 0.0, float(hex_id) + 1.0, 1.0),
        lattice_col=hex_id, lattice_row=0,
        center_x=hex_id + 0.5, center_y=0.5,
        n_tessellations_covered=1, n_spots_direct=0,
        n_spots_ensemble_weighted=0.0,
    )


@pytest.mark.parametrize("mutator,pattern", [
    (
        lambda: dict(
            hexes=(_base_geometry(0),),
            tile_hex_overlaps=(TileHexOverlap(
                component_id="OTHER", domain_region_id=0, replicate_index=0,
                tile_id=0, hex_id=0, overlap_area=1.0, tile_area=1.0,
            ),),
        ),
        "overlap.component_id",
    ),
    (
        lambda: dict(
            hexes=(_base_geometry(0),),
            tile_hex_overlaps=(TileHexOverlap(
                component_id="C", domain_region_id=0, replicate_index=0,
                tile_id=0, hex_id=99, overlap_area=1.0, tile_area=1.0,
            ),),
        ),
        "references unknown hex",
    ),
    (
        lambda: dict(
            hexes=(_base_geometry(0),),
            tile_hex_overlaps=(
                TileHexOverlap(component_id="C", domain_region_id=0, replicate_index=0, tile_id=0, hex_id=0, overlap_area=1.0, tile_area=1.0),
                TileHexOverlap(component_id="C", domain_region_id=0, replicate_index=0, tile_id=0, hex_id=0, overlap_area=1.0, tile_area=1.0),
            ),
        ),
        "duplicate overlap edge",
    ),
    (
        lambda: dict(
            hexes=(_base_geometry(0),),
            tile_hex_overlaps=(TileHexOverlap(
                component_id="C", domain_region_id=0, replicate_index=0,
                tile_id=0, hex_id=0, overlap_area=-1.0, tile_area=1.0,
            ),),
        ),
        "overlap_area",
    ),
    (
        lambda: dict(
            hexes=(_base_geometry(0),),
            direct_hex_membership=(
                DirectHexMembership(source_row_id=1, hex_id=0),
                DirectHexMembership(source_row_id=1, hex_id=-1),
            ),
        ),
        "duplicate direct membership",
    ),
], ids=["overlap_cid_mismatch", "overlap_unknown_hex",
        "overlap_duplicate", "overlap_negative_area",
        "membership_duplicate_source_row_id"])
def test_component_consensus_support_invariants(mutator, pattern):
    kwargs = dict(
        component_id="C", parent_roi_id="R",
        hex_radius=1.0, n_tessellation_eligible_regions=1,
        hexes=(_base_geometry(0),),
        tile_hex_overlaps=(),
        direct_hex_membership=(),
    )
    kwargs.update(mutator())
    with pytest.raises(ValueError, match=pattern):
        ComponentConsensusSupport(**kwargs)


# ---------------------------------------------------------------------------
# 7. Overlap geometry is computed once — no double aggregation
# ---------------------------------------------------------------------------


def test_aggregate_hex_from_ensemble_called_once_per_retained_hex():
    sample_tess, canvas, _ = _build_simple_sample()

    original = consensus_stage._aggregate_hex_from_ensemble
    call_count = {"n": 0}

    def _counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    with patch.object(
        consensus_stage, "_aggregate_hex_from_ensemble", _counting,
    ):
        result = _run_stage(sample_tess, canvas)

    n_candidate_hexes = call_count["n"]
    n_retained = len(result.components["C"].hexes)
    assert n_retained > 0
    assert n_candidate_hexes >= n_retained
    n_edges = len(result.components["C"].support.tile_hex_overlaps)
    assert n_candidate_hexes < n_retained + n_edges, (
        "Overlap edges appear to be re-aggregated per edge; expected "
        "single aggregation per candidate hex."
    )
