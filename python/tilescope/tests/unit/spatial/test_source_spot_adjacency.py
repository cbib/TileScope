"""Canonical spot/cell adjacency: symmetry, classifier parity, interface ops.

Minimum coverage for the shared authority — five semantic tests, no
per-method characterisation.
"""
from __future__ import annotations

from tilescope.analyzers.sample_support_stage import build_empty_sample_support_bank
from tilescope.cols import GENE1, GENE2
from tilescope.settings import load_run_config


import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.support_eligibility import classify_sample_support
from tilescope.cols import COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.source_spot_adjacency import (
    UNREACHABLE_HOP, build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import (
    build_source_spot_table_from_frame,
)


def _table(coords, rois):
    df = pd.DataFrame({
        SOURCE_ROW_ID: list(range(len(coords))),
        COORD_X: [c[0] for c in coords],
        COORD_Y: [c[1] for c in coords],
        ROI_ID: list(rois),
    })
    return build_source_spot_table_from_frame(df)


def test_adjacency_symmetry_no_self_edges_deterministic_ordering():
    # 4-spot square + one outlier; edges depend only on positions.
    coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (10.0, 10.0)]
    sst = _table(coords, ["R"] * 5)
    adj = build_source_spot_adjacency(sst, radius_factor=1.5)
    assert adj.method == "fixed_radius"

    # No self-edges.
    assert (adj.edges[:, 0] < adj.edges[:, 1]).all()
    # Canonical (lexicographic) row order.
    order = np.lexsort((adj.edges[:, 1], adj.edges[:, 0]))
    assert np.array_equal(adj.edges, adj.edges[order])
    # Neighbour lists mirror the edge set: (i, j) in edges  ⇔  j in nb[i] AND i in nb[j].
    for i, j in adj.edges:
        assert int(j) in adj.neighbors[int(i)]
        assert int(i) in adj.neighbors[int(j)]
    for u, nbs in enumerate(adj.neighbors):
        for v in nbs:
            assert u in adj.neighbors[int(v)]
            assert u != v
    # Outlier is isolated.
    assert adj.neighbors[4] == ()


def test_local_delaunay_connects_sparse_chain_without_long_bridge():
    # The old 1.5 × global-median radius disconnects the final two cells:
    # median NN = 1, while both sparse-chain gaps are 2.  Local Delaunay
    # retains those edges, but the separate 4 × median cap still rejects a
    # macroscopic outlier bridge.
    coords = [(0, 0), (1, 0), (2, 0), (4, 0), (6, 0), (100, 0)]
    adj = build_source_spot_adjacency(
        _table(coords, ["R"] * len(coords)), technology="xenium",
    )
    assert adj.method == "locally_pruned_delaunay"
    assert adj.neighbors[0] == (1,)
    assert 3 in adj.neighbors[2]
    assert 4 in adj.neighbors[3]
    assert adj.neighbors[5] == ()


def test_classify_sample_support_result_stable_after_adjacency_extraction():
    """Regression: the classifier still returns the pre-refactor per-ROI component sizes."""
    coords = (
        [(x, 0.0) for x in range(5)]            # ROI A: chain of 5
        + [(x, 10.0) for x in range(3)]         # ROI B: chain of 3
        + [(100.0, 100.0), (100.5, 100.0)]      # ROI B: pair far away
    )
    labels = ["A"] * 5 + ["B"] * 3 + ["B"] * 2
    df = pd.DataFrame({
        SOURCE_ROW_ID: list(range(len(coords))),
        COORD_X: [c[0] for c in coords],
        COORD_Y: [c[1] for c in coords],
        ROI_ID: labels,
    })
    sizes, median_nn = classify_sample_support(df, factor=1.5)
    assert median_nn == pytest.approx(1.0)
    assert sizes == {"A": [5], "B": [3, 2]}


def test_interface_endpoints_and_within_responder_hop_distances():
    # Two touching 3-spot chains; A at y=0, B shifted +1 in y.
    #   A: (0,0) (1,0) (2,0)      indices 0..2
    #   B: (0,1) (1,1) (2,1)      indices 3..5
    coords = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    sst = _table(coords, ["A", "A", "A", "B", "B", "B"])
    adj = build_source_spot_adjacency(sst, radius_factor=1.5)

    anchor_ep, responder_ep = adj.interface_endpoints("A", "B")
    # Every A spot has a corresponding B spot within 1.0 → all 3 sit on the interface.
    assert list(anchor_ep) == [0, 1, 2]
    assert list(responder_ep) == [3, 4, 5]

    # From every responder-interface seed, hop distance on the B-induced subgraph is 0.
    hop = adj.hop_distance_within_roi("B", responder_ep)
    assert list(hop[3:]) == [0, 0, 0]
    assert list(hop[:3]) == [-1, -1, -1]  # anchor spots are outside the ROI subgraph


def test_unreachable_responder_spots_marked_when_component_disconnected():
    # Responder B has two disconnected pieces (near-interface trio + far pair);
    # only the near trio is reachable from A↔B interface seeds.
    coords = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),           # A: chain at y=0
        (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),           # B: chain at y=1  (touches A)
        (100.0, 100.0), (101.0, 100.0),               # B: disconnected pair
    ]
    labels = ["A"] * 3 + ["B"] * 5
    sst = _table(coords, labels)
    adj = build_source_spot_adjacency(sst, radius_factor=1.5)

    _, responder_ep = adj.interface_endpoints("A", "B")
    hop = adj.hop_distance_within_roi("B", responder_ep)
    # Near trio: reachable (all seeds at 0).
    assert (hop[3:6] == 0).all()
    # Far pair: in ROI B but unreachable from the interface seed set.
    assert hop[6] == UNREACHABLE_HOP
    assert hop[7] == UNREACHABLE_HOP
    # A spots (outside ROI B) also marked unreachable.
    assert (hop[:3] == UNREACHABLE_HOP).all()


def test_all_scattered_bank_still_carries_adjacency():
    """The all-scattered path routes the caller-supplied adjacency into the empty bank."""

    df = pd.DataFrame({
        SOURCE_ROW_ID: [0, 1], COORD_X: [0.0, 1.0], COORD_Y: [0.0, 1.0],
        ROI_ID: ["R", "R"], GENE1: [1.0, 2.0], GENE2: [1.0, 2.0],
    })
    sst = build_source_spot_table_from_frame(df)
    adj = build_source_spot_adjacency(sst)
    config = load_run_config(
        profile="standard", set_overrides={"output_dir": "tests/tests_output"},
    )
    bank = build_empty_sample_support_bank(
        sample_id="S", source_spot_adjacency=adj, config=config,
        n_replicates=1, base_seed=0,
    )
    assert bank.components == {}
    assert bank.source_spot_adjacency is adj  # exact `is`-identity, not just carried
    assert bank.source_spots is adj.source_spot_table
    # Two spots at (0,0) / (1,1): median_nn = √2, threshold = 1.5·√2 ≈ 2.12 → one edge.
    assert adj.edges.shape == (1, 2)
    assert adj.sample_median_nn == pytest.approx(np.sqrt(2))
