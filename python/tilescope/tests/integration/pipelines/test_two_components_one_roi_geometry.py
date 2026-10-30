"""Load-bearing identity + geometry contract fixture.

One source ROI containing two spatially disconnected patches must:
1. Emerge as two per-component datasets from `_prepare_roi_context`
   (keyed by COMPONENT_ID, iterating on the geometric analysis unit).
2. Retain the source biological identity in ROI_ID on every row.
3. Produce two distinct component polygons from `build_component_polygons`,
   each built solely from its component's own spatial support (neither
   A polygon covers any sibling A-component spot).

Discriminates against pre-refactor grouping-on-ROI_ID: under that
implementation `_prepare_roi_context` would return two entries (one
per source parent), not three; the assertion `len(roi_datasets) == 3`
fails.

Splitter is intentionally not invoked — COMPONENT_ID is assigned
directly so the test isolates the identity/geometry contract from
splitter calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

from tilescope.analyzers.sample_authority import (
    _build_component_to_roi,
    _prepare_component_datasets,
)
from tilescope.cols import (
    COMPONENT_ID,
    COORD_X,
    COORD_Y,
    GENE1,
    GENE2,
    ROI_ID,
    SOURCE_ROW_ID,
)
from tilescope.settings.loader import load_run_config
from tilescope.spatial.component_domains import build_component_polygons


_HEX_DY = np.sqrt(3.0) / 2.0


def _hex_lattice(rows: int, cols: int, x0: float = 0.0, y0: float = 0.0,
                 spacing: float = 1.0) -> np.ndarray:
    pts = []
    for r in range(rows):
        y = y0 + r * spacing * _HEX_DY
        x_offset = (spacing / 2.0) if (r % 2 == 1) else 0.0
        for c in range(cols):
            x = x0 + x_offset + c * spacing
            pts.append((x, y))
    return np.array(pts, dtype=float)


def _make_three_component_df() -> pd.DataFrame:
    """Source ROI A has two disconnected patches (A_left, A_right) at
    x=0 and x=40; source ROI B has one patch (B_main) at x=80. Each
    patch is 5x9 = 45 spots. COMPONENT_ID is assigned directly so
    the splitter's calibration does not enter the test."""
    a_left = _hex_lattice(5, 9, x0=0.0)
    a_right = _hex_lattice(5, 9, x0=40.0)
    b_main = _hex_lattice(5, 9, x0=80.0)

    coords = np.vstack([a_left, a_right, b_main])
    roi_labels = ["A"] * 45 + ["A"] * 45 + ["B"] * 45
    component_labels = ["A_left"] * 45 + ["A_right"] * 45 + ["B_main"] * 45

    df = pd.DataFrame({
        ROI_ID: roi_labels,
        COMPONENT_ID: component_labels,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: np.zeros(len(coords)),
        GENE2: np.zeros(len(coords)),
        SOURCE_ROW_ID: np.arange(len(coords), dtype=np.int64),
    })
    return df


def test_component_datasets_and_polygons_from_one_roi(tmp_path: Path):
    df = _make_three_component_df()
    assert len(df) == 135

    # Sanity: source ROI_ID is the biological label, COMPONENT_ID is
    # the geometric key. Two components under A, one under B.
    assert set(df[ROI_ID].unique()) == {"A", "B"}
    assert set(df[COMPONENT_ID].unique()) == {"A_left", "A_right", "B_main"}

    # --- Identity contract on dataset preparation --------------------
    component_to_roi = _build_component_to_roi(df)
    assert component_to_roi == {"A_left": "A", "A_right": "A", "B_main": "B"}

    roi_datasets = _prepare_component_datasets(df)

    # Exactly three per-component datasets (one per COMPONENT_ID).
    # Pre-refactor grouping-on-ROI_ID would return two.
    assert set(roi_datasets.keys()) == {"A_left", "A_right", "B_main"}

    # Both A-component datasets retain source biological identity in
    # ROI_ID; each has exactly the 45 spots of its own component.
    for comp_id in ("A_left", "A_right"):
        ds = roi_datasets[comp_id]
        assert len(ds) == 45
        assert set(ds[ROI_ID].unique()) == {"A"}
        assert set(ds[COMPONENT_ID].unique()) == {comp_id}

    # --- Geometry contract on polygon construction -------------------
    config = load_run_config(path=None, profile=None,
                              flag_overrides={"output_dir": str(tmp_path)},
                              set_overrides=None)
    component_build = build_component_polygons(
        source_expression_df=df,
        spatial_component_expression_df=df,
        component_to_roi=component_to_roi,
        sample_median_nn=1.0,
        component_support_radius_factor=(
            config.resolved_policy.component_domain_radii.component_support_radius_factor
        ),
        acquisition_support_radius_factor=(
            config.resolved_policy.component_domain_radii.acquisition_support_radius_factor
        ),
        ineligible_spot_exclusion_radius_factor=(
            config.resolved_policy.component_domain_radii.ineligible_spot_exclusion_radius_factor
        ),
    )
    component_polygons = component_build.component_polygons

    # Three polygons, keys matching the three COMPONENT_IDs.
    assert set(component_polygons.keys()) == {"A_left", "A_right", "B_main"}

    # Every polygon is well-formed.
    for comp_id, poly in component_polygons.items():
        assert poly is not None, f"missing polygon for {comp_id}"
        assert not poly.is_empty, f"empty polygon for {comp_id}"
        assert poly.is_valid, f"invalid polygon for {comp_id}"

    # --- Isolation: neither A polygon covers any sibling-component spot
    # This is the load-bearing spatial-support-boundary assertion:
    # each component polygon is built solely from its own spots and
    # does not reach across the empty space to a sibling. Neighbouring
    # or buffered geometries are permitted to touch or overlap; here
    # the two A patches are separated by ~30 spot-spacings, so any
    # coverage of a sibling's spot indicates a support-spans-gap bug.
    a_left_spots = df.loc[df[COMPONENT_ID] == "A_left", [COORD_X, COORD_Y]].to_numpy()
    a_right_spots = df.loc[df[COMPONENT_ID] == "A_right", [COORD_X, COORD_Y]].to_numpy()

    poly_a_left = component_polygons["A_left"]
    poly_a_right = component_polygons["A_right"]

    for xy in a_right_spots:
        assert not poly_a_left.covers(Point(xy)), (
            f"A_left polygon covers A_right spot {tuple(xy)}; polygon spans "
            "sibling-component support"
        )
    for xy in a_left_spots:
        assert not poly_a_right.covers(Point(xy)), (
            f"A_right polygon covers A_left spot {tuple(xy)}; polygon spans "
            "sibling-component support"
        )
