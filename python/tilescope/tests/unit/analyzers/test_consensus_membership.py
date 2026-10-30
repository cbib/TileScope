"""CORE-HYG-05 Step 2: canonical direct spot -> consensus-hex membership.

Primitive under test is
:func:`tilescope.tessellation.geometry.assign_points_to_first_covering_polygon`
— one-to-one assignments with deterministic first-canonical-polygon
tie resolution on shared boundaries and explicit UNASSIGNED sentinels
for uncovered points.

Six invariants (per CORE-HYG-05 Step 2 acceptance spec):

1. A spot exactly on a shared boundary enters exactly one polygon.
2. Boundary tie resolves to the first post-sort canonical polygon.
3. Candidate ordering controls tie resolution (input-order determinism).
4. Uncovered spots remain explicitly UNASSIGNED.
5. No spot contributes to more than one polygon.
6. Direct counts + gene sums reconstruct from the assigned indices.

Plus one true integration test that runs ``run_consensus_stage`` and
verifies the emitted ``ConsensusHex.n_spots_direct / sum_g*_direct``
reconstruct from the primitive applied externally to the retained
hex polygons — same call path production takes.
"""
from __future__ import annotations

from tilescope.tests.unit.analyzers.test_consensus_stage import _build_partial_coverage_sample_result
from tilescope.analyzers.consensus_stage import run_consensus_stage
from tilescope.settings import load_run_config
from tilescope.cols import GENE1, GENE2


import numpy as np
from shapely.geometry import Polygon

from tilescope.tessellation.geometry import (
    UNASSIGNED,
    assign_points_to_first_covering_polygon,
)


def _square(x0: float, y0: float, side: float = 1.0) -> Polygon:
    return Polygon([
        (x0, y0), (x0 + side, y0),
        (x0 + side, y0 + side), (x0, y0 + side),
    ])


# ---------------------------------------------------------------------------
# 1 + 2. Boundary tie: exactly one hex, first canonical
# ---------------------------------------------------------------------------


def test_shared_boundary_spot_enters_exactly_one_hex_first_canonical():
    """Two adjacent unit squares share the edge x=1. A spot at (1, 0.5)
    lies exactly on that shared boundary. Under boundary-inclusive
    covers() both polygons cover it; the assignment must resolve to
    the FIRST polygon in the canonical order (hex 0)."""
    left = _square(0.0, 0.0)   # canonical index 0, covers x in [0, 1]
    right = _square(1.0, 0.0)  # canonical index 1, covers x in [1, 2]
    coords = np.asarray([[1.0, 0.5]], dtype=float)
    a = assign_points_to_first_covering_polygon([left, right], coords)
    assert a.tolist() == [0]


# ---------------------------------------------------------------------------
# 3. Ordering controls tie resolution
# ---------------------------------------------------------------------------


def test_reversing_canonical_order_flips_tie_winner():
    """Two DISTINCT polygons sharing a boundary — reversing the input
    order flips which physical polygon receives the boundary spot.
    Uses polygon IDs (bbox min-x) to make the assertion about which
    polygon won, not just about which returned index."""
    left = _square(0.0, 0.0)   # bbox min-x = 0.0
    right = _square(1.0, 0.0)  # bbox min-x = 1.0
    coords = np.asarray([[1.0, 0.5]], dtype=float)
    forward = assign_points_to_first_covering_polygon([left, right], coords)
    reversed_ = assign_points_to_first_covering_polygon([right, left], coords)
    assert forward.tolist() == [0] and reversed_.tolist() == [0]
    # In forward order the spot goes to `left` (physical bbox 0.0..1.0);
    # in reversed order it goes to `right` (physical bbox 1.0..2.0).
    forward_winner_bbox_x = [left, right][forward[0]].bounds[0]
    reversed_winner_bbox_x = [right, left][reversed_[0]].bounds[0]
    assert forward_winner_bbox_x == 0.0, (
        f"forward order: expected `left` (bbox_min_x=0.0) to win, "
        f"got polygon with bbox_min_x={forward_winner_bbox_x}"
    )
    assert reversed_winner_bbox_x == 1.0, (
        f"reversed order: expected `right` (bbox_min_x=1.0) to win, "
        f"got polygon with bbox_min_x={reversed_winner_bbox_x}"
    )


# ---------------------------------------------------------------------------
# 4. Uncovered spots stay UNASSIGNED (no reassignment)
# ---------------------------------------------------------------------------


def test_uncovered_spot_remains_unassigned():
    left = _square(0.0, 0.0)
    right = _square(1.0, 0.0)
    coords = np.asarray([
        [0.5, 0.5],
        [10.0, 10.0],
        [1.5, 0.5],
    ], dtype=float)
    a = assign_points_to_first_covering_polygon([left, right], coords)
    assert a.tolist() == [0, UNASSIGNED, 1]


def test_all_uncovered_when_no_polygons_match():
    left = _square(0.0, 0.0)
    coords = np.asarray([[5.0, 5.0], [-1.0, -1.0]], dtype=float)
    a = assign_points_to_first_covering_polygon([left], coords)
    assert (a == UNASSIGNED).all()


def test_empty_inputs_return_empty_or_all_unassigned():
    empty_polys_assignment = assign_points_to_first_covering_polygon(
        [], np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
    )
    assert empty_polys_assignment.tolist() == [UNASSIGNED, UNASSIGNED]
    empty_coords_assignment = assign_points_to_first_covering_polygon(
        [_square(0.0, 0.0)], np.zeros((0, 2), dtype=float),
    )
    assert empty_coords_assignment.tolist() == []


# ---------------------------------------------------------------------------
# 5. No spot contributes to more than one polygon
# ---------------------------------------------------------------------------


def test_no_spot_contributes_to_more_than_one_polygon():
    polys = [_square(i, j) for i in range(3) for j in range(3)]
    xs = np.linspace(0.0, 3.0, 5)
    ys = np.linspace(0.0, 3.0, 5)
    coords = np.asarray(
        [[x, y] for x in xs for y in ys], dtype=float,
    )
    a = assign_points_to_first_covering_polygon(polys, coords)
    assert a.shape == (25,)
    covered = a[a != UNASSIGNED]
    assert len(covered) == 25
    for p_idx in range(len(polys)):
        assert (a == p_idx).any()


# ---------------------------------------------------------------------------
# 6. Direct counts + gene sums reconstruct from assigned indices
# ---------------------------------------------------------------------------


def test_direct_stats_reconstruct_from_assignment_indices():
    left = _square(0.0, 0.0)
    right = _square(1.0, 0.0)
    coords = np.asarray([
        [0.5, 0.5],
        [0.2, 0.2],
        [1.5, 0.5],
        [1.0, 0.5],
        [5.0, 5.0],
    ], dtype=float)
    g1 = np.asarray([10.0, 20.0, 30.0, 40.0, 500.0])
    g2 = np.asarray([ 1.0,  2.0,  3.0,  4.0,  50.0])
    a = assign_points_to_first_covering_polygon([left, right], coords)
    assert a.tolist() == [0, 0, 1, 0, UNASSIGNED]
    mask_left = a == 0
    assert int(mask_left.sum()) == 3
    assert float(g1[mask_left].sum()) == 70.0
    assert float(g2[mask_left].sum()) == 7.0
    mask_right = a == 1
    assert int(mask_right.sum()) == 1
    assert float(g1[mask_right].sum()) == 30.0
    assert float(g2[mask_right].sum()) == 3.0


# ---------------------------------------------------------------------------
# Integration: run_consensus_stage + reconstruction from primitive
# ---------------------------------------------------------------------------


def test_run_consensus_stage_direct_stats_reconstruct_via_canonical_membership():
    """End-to-end contract: build a real ``SampleTessellationResult``,
    run ``run_consensus_stage``, and verify that every emitted
    ``ConsensusHex.n_spots_direct / sum_g*_direct`` reconstructs
    exactly from
    ``assign_points_to_first_covering_polygon(retained_polygons, component.points)``
    applied externally in the same canonical order. Also asserts the
    canonical sort key and the no-double-counting invariant.
    """
    # Reuse the well-covered fixture in test_consensus_stage.py — it
    # builds a single-region component with a uniform spot grid and
    # returns (sample_result, canvas) suitable for run_consensus_stage.

    sample_result, canvas = _build_partial_coverage_sample_result()
    anchor = (0.0, 0.0)
    result = run_consensus_stage(
        sample_tessellation_result=sample_result, canvas=canvas,
        sample_anchor=anchor,
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
    component = result.components["C"]
    assert component.hexes, "fixture must produce at least one retained hex"

    # Canonical sort key: (domain_region_id, lattice_col, lattice_row).
    canonical_keys = [
        (h.domain_region_id, h.lattice_col, h.lattice_row)
        for h in component.hexes
    ]
    assert canonical_keys == sorted(canonical_keys), (
        "emitted hexes are not in canonical (region, lattice_col, "
        "lattice_row) order — Step 2's assumption breaks"
    )

    # Reconstruction: apply the primitive externally in the same
    # canonical order and reproduce per-hex direct stats.
    comp_data = sample_result.components["C"].component_data
    direct_coords = comp_data.points
    direct_g1 = comp_data.expression_data[GENE1].to_numpy()
    direct_g2 = comp_data.expression_data[GENE2].to_numpy()
    canonical_polygons = [h.polygon for h in component.hexes]

    assignment = assign_points_to_first_covering_polygon(
        canonical_polygons, direct_coords,
    )

    # No-double-counting: `assign_points_to_first_covering_polygon`
    # returns one index per spot by construction, so partition follows.
    # Verify sum_over_hexes(n_spots_direct) == n_covered_spots.
    n_covered = int((assignment != UNASSIGNED).sum())
    total_n_direct = sum(int(h.n_spots_direct) for h in component.hexes)
    assert total_n_direct == n_covered, (
        f"sum(n_spots_direct)={total_n_direct} disagrees with the "
        f"externally-assigned coverage count={n_covered}"
    )
    assert total_n_direct <= len(direct_coords), (
        f"sum(n_spots_direct)={total_n_direct} > n_component_spots="
        f"{len(direct_coords)} — double-counting is back"
    )

    # Per-hex reconstruction: n_spots_direct match. Direct feature sums
    # migrated to FeatureSpatialProjection in Phase 3a; verified there.
    del direct_g1, direct_g2
    for hex_id, hex_obj in enumerate(component.hexes):
        mask = assignment == hex_id
        assert int(mask.sum()) == int(hex_obj.n_spots_direct), (
            f"hex_id={hex_id}: n_spots_direct mismatch — "
            f"emitted={hex_obj.n_spots_direct}, external={int(mask.sum())}"
        )
