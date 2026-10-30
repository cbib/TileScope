"""Semantic coverage for production and compatibility ROI splitters."""
from __future__ import annotations

import pandas as pd

from tilescope.analyzers.roi_splitter import (
    filter_components,
    split_disconnected_rois,
    split_rois_from_source_adjacency,
)
from tilescope.cols import COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame


def _two_fragments() -> pd.DataFrame:
    xs = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    return pd.DataFrame({
        SOURCE_ROW_ID: [f"spot-{i}" for i in range(len(xs))],
        ROI_ID: ["roi1"] * len(xs),
        COORD_X: xs,
        COORD_Y: [0.0] * len(xs),
    })


def test_production_splitter_projects_canonical_source_components():
    df = _two_fragments()
    adjacency = build_source_spot_adjacency(
        build_source_spot_table_from_frame(df), technology="visium",
    )

    assert [component.n_spots for component in adjacency.within_roi_components] == [
        3, 3,
    ]
    split = split_rois_from_source_adjacency(
        df, adjacency, split_min_component_observations=1,
    )
    assert split.groupby("cc_idx", sort=True).size().tolist() == [3, 3]
    assert split.loc[split[COORD_X] < 5, "cc_idx"].unique().tolist() == [0]
    assert split.loc[split[COORD_X] > 5, "cc_idx"].unique().tolist() == [1]


def test_compatibility_splitter_separates_a_large_gap():
    df = _two_fragments().drop(columns=[SOURCE_ROW_ID])
    split = split_disconnected_rois(
        df, k=2, gamma=1.5, symmetrize="union", min_component_spots=1,
    )

    assert split.groupby("cc_idx", sort=True).size().tolist() == [3, 3]
    assert split["comp_id"].nunique() == 2


def test_component_filter_uses_component_size():
    frame = pd.DataFrame({
        "comp_id": [0] * 4 + [1] * 2,
        COORD_X: range(6),
        COORD_Y: [0.0] * 6,
    })

    filtered = filter_components(frame, min_component_spots=3)

    assert filtered["comp_id"].unique().tolist() == [0]
    assert len(filtered) == 4
