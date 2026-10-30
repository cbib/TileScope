"""End-to-end grain and identity acceptance contract.

One deterministic fixture: parent ROI A with two spatially
disconnected components (A_c1, A_c2), each 45 spots; parent ROI B
with one component (B_main), 45 spots. All three components
satisfy every default gate (size ≥ 40, sibling-excluded context
≥ 40 spots), so every component-grain path emits a row for each.

Two assertion blocks:

Block 1 — component-grain geometry export:
    `component_geometry.csv` has one row per component (3 rows;
    A × 2, B × 1); both identity columns present.

Block 2 — parent-grain in-memory outputs:
    `roi_enrichment` and `roi_correlations` each have exactly 2
    rows (one per biological ROI); no COMPONENT_ID column;
    distinct values {A, B}. Retirement contract: the
    `spot_metrics` / `spot_correlations` keys and
    `summary_cross_scale.csv` must NOT appear.

Kept to ONE fixture + ONE test to avoid scenario-suite bloat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.orchestrator import run_analysis
from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, GENE1, GENE2, ROI_ID
from tilescope.settings.loader import load_run_config


_HEX_DY = np.sqrt(3.0) / 2.0


def _hex_lattice(rows: int, cols: int, x0: float, y0: float = 0.0,
                 spacing: float = 1.0) -> np.ndarray:
    pts = []
    for r in range(rows):
        y = y0 + r * spacing * _HEX_DY
        x_offset = (spacing / 2.0) if (r % 2 == 1) else 0.0
        for c in range(cols):
            x = x0 + x_offset + c * spacing
            pts.append((x, y))
    return np.array(pts, dtype=float)


def _two_parent_three_component_fixture() -> pd.DataFrame:
    """A_left + A_right + B_main, each 45 spots, well-separated patches.

    Frame carries only `ROI_ID` (biological label); `COMPONENT_ID` is
    NOT pre-assigned — the splitter creates it downstream from the
    coordinate geometry. Parent A's two patches are ~100 units apart
    (well beyond the splitter's kNN threshold), so the splitter
    produces two components for A and one for B, yielding the
    three-component structure the test asserts.

    Component sizes chosen so all three pass the component-retention gate
    and produce multi-tile tessellations under the construction floor.
    - sibling-excluded context for each ≥ 40 (A_*'s context = B's 45;
      B's context = A's 90).
    """
    a_left = _hex_lattice(5, 9, x0=0.0)
    a_right = _hex_lattice(5, 9, x0=100.0)  # wide gap ensures splitter finds it
    b_main = _hex_lattice(5, 9, x0=200.0)

    coords = np.vstack([a_left, a_right, b_main])
    n = 45
    labels = ["A"] * n + ["A"] * n + ["B"] * n
    rng = np.random.default_rng(2026)
    return pd.DataFrame({
        ROI_ID: labels,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.1, 5.0, len(labels)),
        GENE2: rng.uniform(0.1, 5.0, len(labels)),
    })


def test_two_grain_semantic_contract_end_to_end(tmp_path: Path):
    df = _two_parent_three_component_fixture()
    assert len(df) == 135

    config = load_run_config(
        path=None, profile=None,
        flag_overrides={"output_dir": str(tmp_path)},
        set_overrides=None,
    )

    results = run_analysis(
        df, config, feature_a_id="Neuron", feature_b_id="AC",
    )

    # Sanity: the splitter recovered the two-component structure of
    # parent A. If this fails the whole fixture is misconfigured;
    # subsequent blocks would be nonsense.
    analysis_df = results.get("analysis_df")
    assert analysis_df is not None, "results['analysis_df'] absent"
    assert COMPONENT_ID in analysis_df.columns
    assert ROI_ID in analysis_df.columns
    n_components = analysis_df[COMPONENT_ID].nunique()
    n_biological_rois = analysis_df[ROI_ID].nunique()
    assert n_components == 3, (
        f"Fixture expects 3 components (A_c1, A_c2, B_main); splitter "
        f"produced {n_components}. Check kNN calibration or patch spacing."
    )
    assert n_biological_rois == 2, (
        f"Fixture expects 2 biological ROIs (A, B); got {n_biological_rois}."
    )
    a_component_ids = set(
        analysis_df.loc[analysis_df[ROI_ID] == "A", COMPONENT_ID].unique()
    )
    assert len(a_component_ids) == 2, "parent A should map to 2 components"

    # -----------------------------------------------------------------
    # BLOCK 1 — component-grain geometry surface
    # -----------------------------------------------------------------
    # `component_geometry_df` is placed on the return dict by
    # `run_analysis`; result_exporter is the sole CSV writer. The
    # semantic contract lives on the DataFrame, not the file.
    geom = results["component_geometry_df"]
    assert COMPONENT_ID in geom.columns, "component_geometry_df missing COMPONENT_ID"
    assert ROI_ID in geom.columns, "component_geometry_df missing ROI_ID"

    # Exact three-row / component-mapping contract. Any row-count
    # deviation, sibling-merging under a single row, or misrouting of
    # a component to the wrong parent breaks this assertion.
    assert len(geom) == 3, f"expected 3 rows (one per component), got {len(geom)}"
    parent_by_component = dict(zip(
        geom[COMPONENT_ID].astype(str), geom[ROI_ID].astype(str),
    ))
    parent_to_components = {"A": set(), "B": set()}
    for comp_id, parent in parent_by_component.items():
        parent_to_components.setdefault(parent, set()).add(comp_id)
    assert set(parent_to_components) == {"A", "B"}, (
        f"component_geometry.csv must map to exactly parents {{A, B}}; "
        f"got parents {sorted(parent_to_components)}"
    )
    assert len(parent_to_components["A"]) == 2, (
        f"parent A must own exactly 2 components; got "
        f"{sorted(parent_to_components['A'])}"
    )
    assert len(parent_to_components["B"]) == 1, (
        f"parent B must own exactly 1 component; got "
        f"{sorted(parent_to_components['B'])}"
    )

    # -----------------------------------------------------------------
    # BLOCK 2 — parent-grain outputs
    # -----------------------------------------------------------------
    for key in ("roi_enrichment", "roi_correlations"):
        assert key in results, f"required parent frame results[{key!r}] is missing"
        frame = results[key]
        assert not frame.empty, f"results[{key!r}] must not be empty"
        assert ROI_ID in frame.columns, f"{key} missing ROI_ID column"
        assert COMPONENT_ID not in frame.columns, (
            f"{key} is parent-grain and MUST NOT carry COMPONENT_ID; "
            "carrying it would signal a grain leak from the component "
            "analyzer into the parent-analyzer output."
        )
        distinct_rois = set(frame[ROI_ID].astype(str).unique())
        assert distinct_rois == {"A", "B"}, (
            f"{key} must have exactly 2 biological ROIs (A, B); got "
            f"{sorted(distinct_rois)}"
        )
        assert frame[ROI_ID].nunique() == 2

    # Retirement contract: the retired component-spot keys and the
    # cross-scale summary must be gone.
    for retired_key in ("spot_metrics", "spot_correlations"):
        assert retired_key not in results, (
            f"retired results key {retired_key!r} is still present"
        )
    assert not (tmp_path / "summary_cross_scale.csv").exists(), (
        "retired summary_cross_scale.csv is still emitted"
    )
