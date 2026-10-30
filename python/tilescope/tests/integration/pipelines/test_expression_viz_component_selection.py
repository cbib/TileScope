"""Source-native expression viz contract.

Under the spot/cell contract, `plot_expression_summary` calls the
`SpotSplatPainter` ONCE per gene panel on the FULL `source_expression_df`
with `component_polygon=None` (no clipping). Component polygons are boundary
overlays only, drawn separately.

This replaces the earlier per-component painter-call contract (retired
along with the polygon-clipped multi-call pattern).

The fixture below includes THREE spot conditions the current viz
contract must render without dropping any:

  * spots inside a component polygon (retained-and-covered);
  * a source-only fragment with NO polygon (mixed-ROI fragment stand-in);
  * a point placed OUTSIDE all polygons (retained-but-not-covered stand-in).

The assertion "every spot or cell reaches a painter call with
`component_polygon=None`" directly covers visibility. Whether the actual
pipeline places any given real spot into any given category depends on
component splitting and the bounded-support component-domain contract —
those are exercised by their own dedicated tests and are deliberately
out of scope here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, GENE1, GENE2, ROI_ID
from tilescope.spatial import make_canvas
from tilescope.viz import expression as expression_module


def _square(cx: float, cy: float, side: float) -> Polygon:
    h = side / 2
    return Polygon([
        (cx - h, cy - h), (cx + h, cy - h),
        (cx + h, cy + h), (cx - h, cy + h),
    ])


def test_source_native_painter_call_contract(tmp_path: Path, monkeypatch):
    """One `SpotSplatPainter.draw` call per gene panel, on the FULL
    source frame, with `component_polygon=None` (no clip).

    Fixture layout:
      * A_left  (8 spots inside a polygon)     — retained-and-covered
      * A_right (8 spots inside a polygon)     — retained-and-covered
      * B_main  (8 spots inside a polygon)     — retained-and-covered
      * A_frag  (5 spots, no polygon)          — source-only fragment
      * outlier (1 spot outside every polygon) — retained-but-not-covered

    Total 30 spots. The viz must show all 30, so both painter calls
    must receive n_points == 30.
    """
    a_left_xy = np.array([(i % 4, i // 4) for i in range(8)], dtype=float)
    a_right_xy = np.array([(20 + i % 4, i // 4) for i in range(8)], dtype=float)
    b_main_xy = np.array([(40 + i % 4, i // 4) for i in range(8)], dtype=float)
    # Source-only fragment far from any polygon.
    a_frag_xy = np.array([(60 + i, 0) for i in range(5)], dtype=float)
    # Retained-but-not-covered: nominally belongs to A_left component
    # but sits outside its 6×6 polygon (which spans (-1.5, -2)–(4.5, 4)).
    outlier_xy = np.array([[10.0, 8.0]], dtype=float)

    coords = np.vstack([a_left_xy, a_right_xy, b_main_xy, a_frag_xy, outlier_xy])
    source_df = pd.DataFrame({
        ROI_ID: (
            ["A"] * 8 + ["A"] * 8 + ["B"] * 8 + ["A"] * 5 + ["A"] * 1
        ),
        COMPONENT_ID: (
            ["A_left"] * 8 + ["A_right"] * 8 + ["B_main"] * 8
            + ["A_frag"] * 5      # component that never gets a polygon
            + ["A_left"] * 1      # outlier assigned to A_left, outside its poly
        ),
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: np.arange(30, dtype=float),
        GENE2: np.arange(30, dtype=float) * 2.0,
    })

    # Polygons for the three retained components ONLY. A_frag has no
    # polygon (source-only fragment). Outlier's component (A_left)
    # DOES have a polygon, but the outlier sits outside it.
    component_polygons = {
        "A_left":  _square(1.5, 1.0, side=6.0),
        "A_right": _square(21.5, 1.0, side=6.0),
        "B_main":  _square(41.5, 1.0, side=6.0),
    }

    calls: list = []

    def fake_draw(self, ax, points_xy, values, component_polygon=None, **kwargs):
        calls.append({
            "n_points": len(points_xy),
            "component_polygon": component_polygon,
            "first_value": float(values[0]) if len(values) else None,
        })
        return None

    monkeypatch.setattr(
        expression_module.SpotSplatPainter, "draw", fake_draw,
    )

    canvas = make_canvas(source_df, target_px_per_step=1, target_px_per_nn=4, margin_px=10)
    correlation_map = {"A": 0.5, "B": 0.5}
    component_to_roi = {
        "A_left": "A", "A_right": "A", "B_main": "B", "A_frag": "A",
    }
    expression_module.plot_expression_summary(
        source_expression_df=source_df,
        correlation_map=correlation_map,
        component_to_roi=component_to_roi,
        component_polygons=component_polygons,
        canvas=canvas,
        output_dir=str(tmp_path),
        feature_a_id="Neuron", feature_b_id="AC",
    )

    assert len(calls) == 2, (
        f"expected 2 painter calls (Gene1 + Gene2); got {len(calls)}. "
        "Source-native contract: one call per panel, not one per component."
    )
    for call in calls:
        assert call["n_points"] == 30, (
            f"painter received {call['n_points']} points; expected 30 "
            "(retained-and-covered + source-only fragment + outlier). "
            "Any lower count means at least one condition is not being "
            "rendered — breaks the spot/cell contract."
        )
        assert call["component_polygon"] is None, (
            "painter must be called with component_polygon=None under the "
            "spot/cell contract; polygon clipping is off."
        )
    # First call carries Gene1 (values 0..29), second carries Gene2 (2×).
    assert calls[0]["first_value"] == pytest.approx(0.0)
    assert calls[1]["first_value"] == pytest.approx(0.0)
