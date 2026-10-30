"""Semantic guards for the CORE-VIZ-02 conditional 1x3 / 2x3 expression
summary. Covers: canonical membership + None-only legacy fallback, the
1x3 vs 2x3 conditional layout, native spot/cell values under shared
full-slice normalization, grouped identity contours across all three bottom
panels, exact Pearson lookup, and the legend contents.

No assertions on ``axes.collections`` by index — the two-pass render
makes collection order an implementation detail."""
from __future__ import annotations

from matplotlib.colors import to_rgba
from tilescope.viz.expression import _SCATTERED_IDENTITY_PALETTE


import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tilescope.cols import GENE1, GENE2, ROI_ID
from tilescope.viz import expression as expr_mod
from tilescope.viz.expression import (
    _paint_scattered_value_panel, _per_spot_pearson,
    _resolve_scattered_roi_ids, _roi_identity_color_map,
    _scattered_categorical_legend_handles, _scattered_roi_ids_from_status,
    plot_expression_summary,
)
from tilescope.viz.painters import PEARSON_CMAP, PEARSON_NORM


def _status_frame(rows):
    return pd.DataFrame(rows, columns=[ROI_ID, "spatial_support_class"])


def _source_frame(rows):
    return pd.DataFrame(
        rows, columns=["x", "y", GENE1, GENE2, ROI_ID],
    )


# --------------------------------------------------------------------------- #
# 1. Canonical scattered membership + None-only legacy fallback               #
# --------------------------------------------------------------------------- #


def test_canonical_status_authority_reads_only_scattered_class():
    status = _status_frame([
        ("1", "scattered"),
        ("2", "mixed"),
        ("3", "fully_connected"),
    ])
    assert _scattered_roi_ids_from_status(status) == ["1"]


def test_legacy_fallback_fires_only_when_status_is_none():
    resolved = _resolve_scattered_roi_ids(
        roi_support_status_df=None,
        correlation_map={"1": -0.1, "2": 0.5},
        constructible_roi_ids={"2"},
    )
    assert resolved == ["1"]


def test_empty_status_frame_is_authoritative_not_fallback():
    """Zero-row status frame => no scattered ROIs. A correlation_map ROI
    that is neither constructible nor scattered must NOT be coerced."""
    empty_status = _status_frame([])
    with pytest.raises(ValueError, match="silently reclassify"):
        _resolve_scattered_roi_ids(
            roi_support_status_df=empty_status,
            correlation_map={"7": 0.1},
            constructible_roi_ids=set(),
        )


# --------------------------------------------------------------------------- #
# 2. Conditional 1x3 / 2x3 layout                                             #
# --------------------------------------------------------------------------- #


def _minimal_source():
    return _source_frame([
        (0.0, 0.0, 1.1, 2.1, "1"),
        (1.0, 0.0, 1.2, 2.2, "1"),
        (2.0, 0.0, 1.3, 2.3, "1"),
    ])


def _capture_subplots(monkeypatch):
    """Return {'nrows': int, 'ncols': int} the first time plt.subplots
    is called from expression.py."""
    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        if "nrows" not in captured:
            nrows = args[0] if len(args) >= 1 else kwargs.get("nrows", 1)
            ncols = args[1] if len(args) >= 2 else kwargs.get("ncols", 1)
            captured["nrows"] = nrows
            captured["ncols"] = ncols
        return real_subplots(*args, **kwargs)

    monkeypatch.setattr(expr_mod.plt, "subplots", spy)
    return captured


def test_layout_is_1x3_when_no_scattered_rois(tmp_path, monkeypatch):
    captured = _capture_subplots(monkeypatch)
    plot_expression_summary(
        source_expression_df=_minimal_source(),
        correlation_map={},
        component_to_roi={},
        roi_support_status_df=_status_frame([("1", "fully_connected")]),
        output_dir=str(tmp_path),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    assert captured == {"nrows": 1, "ncols": 3}


def test_layout_is_2x3_when_scattered_rois_present(tmp_path, monkeypatch):
    captured = _capture_subplots(monkeypatch)
    plot_expression_summary(
        source_expression_df=_minimal_source(),
        correlation_map={"1": 0.5},
        component_to_roi={},
        roi_support_status_df=_status_frame([("1", "scattered")]),
        output_dir=str(tmp_path),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    assert captured == {"nrows": 2, "ncols": 3}


# --------------------------------------------------------------------------- #
# 3. Native spot/cell values under shared full-slice normalization               #
# --------------------------------------------------------------------------- #


def test_bottom_fill_is_source_values_under_top_row_norm():
    """The scattered-panel value fill collection carries the exact
    spot/cell values and the SAME `norm` as the top-row full-slice
    normalization."""
    src = _source_frame([
        (0.0, 0.0, -1.5, 0.0, "7"),
        (1.0, 0.0,  0.4, 0.0, "7"),
        (2.0, 0.0,  2.9, 0.0, "7"),
    ])
    vmin, vmax = float(src[GENE1].min()), float(src[GENE1].max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ids = ["7"]
    id_colors = _roi_identity_color_map(ids)
    coords = src[["x", "y"]].to_numpy(dtype=float)
    fill = src[GENE1].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    _paint_scattered_value_panel(
        ax, coords, values=fill, cmap="viridis", norm=norm,
        roi_ids=src[ROI_ID].astype(str).to_numpy(),
        roi_identity_colors=id_colors, roi_polygons={}, median_nn=1.0,
    )
    # Locate the value-fill collection: the one whose colour array
    # matches the input `fill`. Do NOT rely on index.
    fill_coll = None
    for coll in ax.collections:
        arr = coll.get_array()
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.shape == fill.shape and np.allclose(arr, fill):
            fill_coll = coll
            break
    assert fill_coll is not None, "value-fill collection missing"
    assert float(fill_coll.norm.vmin) == vmin
    assert float(fill_coll.norm.vmax) == vmax
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Grouped identity contours shared across all three bottom panels          #
# --------------------------------------------------------------------------- #


def test_grouped_identity_contours_are_shared_across_bottom_panels():
    """Adjacent spots in one ROI receive one outer identity contour;
    a second ROI receives a distinct contour on every bottom panel."""
    src = _source_frame([
        (0.0, 0.0, 1.0, 2.0, "1"),
        (1.0, 0.0, 1.2, 2.2, "1"),
        (4.0, 0.0, 1.4, 2.4, "2"),
    ])
    id_colors = _roi_identity_color_map(["1", "2"])
    coords = src[["x", "y"]].to_numpy(dtype=float)
    roi_ids = src[ROI_ID].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 3)
    _paint_scattered_value_panel(
        axes[0], coords, values=src[GENE1].to_numpy(dtype=float),
        cmap="viridis", norm=mcolors.Normalize(0, 3),
        roi_ids=roi_ids, roi_identity_colors=id_colors,
        roi_polygons={}, median_nn=1.0,
    )
    _paint_scattered_value_panel(
        axes[1], coords, values=src[GENE2].to_numpy(dtype=float),
        cmap="viridis", norm=mcolors.Normalize(0, 3),
        roi_ids=roi_ids, roi_identity_colors=id_colors,
        roi_polygons={}, median_nn=1.0,
    )
    _paint_scattered_value_panel(
        axes[2], coords, values=np.array([0.1, -0.2, 0.1]),
        cmap=PEARSON_CMAP, norm=PEARSON_NORM,
        roi_ids=roi_ids, roi_identity_colors=id_colors,
        roi_polygons={}, median_nn=1.0,
    )
    expected = {
        tuple(mcolors.to_rgba(id_colors["1"])),
        tuple(mcolors.to_rgba(id_colors["2"])),
    }
    for ax in axes:
        identity_edges = {
            tuple(patch.get_edgecolor())
            for patch in ax.patches
            if tuple(patch.get_edgecolor()) in expected
        }
        assert identity_edges == expected
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Exact Pearson lookup                                                      #
# --------------------------------------------------------------------------- #


def test_per_spot_pearson_matches_correlation_map_exactly():
    src = _source_frame([
        (0.0, 0.0, 0.0, 0.0, "7"),
        (1.0, 0.0, 0.0, 0.0, "7"),
        (2.0, 0.0, 0.0, 0.0, "9"),
    ])
    corr = {"7": -0.141, "9": 0.62}
    assert _per_spot_pearson(src, corr).tolist() == [-0.141, -0.141, 0.62]


# --------------------------------------------------------------------------- #
# 6. Legend contains every scattered ROI                                       #
# --------------------------------------------------------------------------- #


def test_legend_contains_every_scattered_roi():
    ids = ["1", "2", "5"]
    id_colors = _roi_identity_color_map(ids)
    handles = _scattered_categorical_legend_handles(ids, id_colors)
    labels = [h.get_label() for h in handles]
    assert labels == ["ROI 1", "ROI 2", "ROI 5"]
    # Edge colours on the legend markers reflect the identity map.
    for roi_id, handle in zip(ids, handles):
        assert tuple(handle.get_markeredgecolor()) == tuple(id_colors[str(roi_id)])


# --------------------------------------------------------------------------- #
# 7. Fluorescent identity palette assigned deterministically by sorted ROI id #
# --------------------------------------------------------------------------- #


def test_roi_identity_color_map_uses_fluorescent_palette_by_sorted_id():
    """Palette starts green→purple→yellow and assigns by sorted ROI
    string id, so a caller passing ids in any order gets the same
    mapping and repeated ids across renders remain stable."""
    ids_shuffled = ["7", "3", "1"]
    id_colors = _roi_identity_color_map(ids_shuffled)
    # Sorted order is ["1", "3", "7"] → palette indices 0, 1, 2.
    assert id_colors["1"] == to_rgba(_SCATTERED_IDENTITY_PALETTE[0])
    assert id_colors["3"] == to_rgba(_SCATTERED_IDENTITY_PALETTE[1])
    assert id_colors["7"] == to_rgba(_SCATTERED_IDENTITY_PALETTE[2])


def test_roi_identity_color_map_cycles_when_scattered_count_exceeds_palette():
    """More scattered ROIs than palette entries → cycle from index 0."""
    n = len(_SCATTERED_IDENTITY_PALETTE)
    ids = [str(i) for i in range(n + 2)]
    id_colors = _roi_identity_color_map(ids)
    sorted_ids = sorted(ids)
    assert id_colors[sorted_ids[0]] == to_rgba(_SCATTERED_IDENTITY_PALETTE[0])
    assert id_colors[sorted_ids[n]] == to_rgba(_SCATTERED_IDENTITY_PALETTE[0])
    assert id_colors[sorted_ids[n + 1]] == to_rgba(_SCATTERED_IDENTITY_PALETTE[1])
