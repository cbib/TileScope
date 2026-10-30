"""Regression test for the scattered-ROI eligibility gate.

One deterministic fixture with three biological ROIs:

  - A: coherent 30-spot lattice (`fully_connected`).
  - B: coherent 25-spot lattice (`fully_connected`).
  - C: 15 spots scattered across the sample so no same-label
       sample-support component reaches min_component_spots (`scattered`).

Assertions:

  1. `roi_support_status` classifies C as `scattered`, A/B not.
  2. A's and B's tessellation-eligible context DataFrames contain
     ONLY each other's rows, not C's. (Regression against the bug
     where `_prepare_component_datasets` ran before the gate and
     baked scattered spots into sibling contexts.)
  3. C appears in `roi_correlations` and `roi_enrichment` with
     `sample_size == 15` and `n_roi == 15` respectively — pooled
     parent analysis is unaffected by the gate.
  4. C is absent from `component_geometry`, tessellation and
     consensus outputs.
  5. `source_expression_df` in results carries EVERY source spot
     (all 78 = 48 A + 15 C + 15 B in `_fixture`). Panels 1/2/3 of
     the expression viz consume this source frame directly under
     the spot/cell contract, so scattered ROIs, mixed-ROI
     fragments, and retained spots-outside-polygon are all visible
     via this single frame.
"""
from __future__ import annotations

from tilescope.cols import PARENT_REPLICATE_CORRELATION_STATUS
from tilescope.api import run_tilescope_from_df
from tilescope.cols import PARENT_CORRELATION_STATUS, PARENT_ENRICHMENT_STATUS
from tilescope.analyzers import pre_analysis as pa_mod
from tilescope.analyzers import support_eligibility as se_mod
import re


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers import sample_authority as orch_mod
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


def _fixture() -> pd.DataFrame:
    """Three parents: A coherent 48-spot, B coherent 42-spot, C 15 scattered.

    A and B are large enough to produce multi-tile tessellations under
    the default construction floor. C's 15 spots are
    spread widely enough that no same-label sample-support component
    reaches `min_component_spots` — the sample-median NN is dominated
    by A + B's spacing=1.0 lattice, so radius = 1.5 leaves C as
    singletons under sample-support connectivity.
    """
    a_pts = _hex_lattice(8, 6, x0=0.0)                             # 48 spots
    b_pts = _hex_lattice(7, 6, x0=100.0)                           # 42 spots
    c_pts = np.array(
        [(x, y) for x, y in zip(
            np.linspace(-10, 110, 15),
            np.linspace(-10, 12, 15),
        )],
        dtype=float,
    )

    coords = np.vstack([a_pts, b_pts, c_pts])
    labels = ["A"] * len(a_pts) + ["B"] * len(b_pts) + ["C"] * len(c_pts)
    rng = np.random.default_rng(2026)
    return pd.DataFrame({
        ROI_ID: labels,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.1, 5.0, len(labels)),
        GENE2: rng.uniform(0.1, 5.0, len(labels)),
    })


def test_scattered_roi_gate_end_to_end(tmp_path: Path, monkeypatch):
    df = _fixture()
    assert len(df) == 48 + 42 + 15

    # Spy on `_prepare_component_datasets` to prove the gate runs
    # BEFORE context construction: the frame handed to it must never
    # contain scattered-ROI rows. If someone moves the gate below
    # `_prepare_component_datasets` again (Issue 1's original bug),
    # this assertion fires because C's rows would leak into the
    # input frame.
    prepare_calls: list[pd.DataFrame] = []
    real_prepare = orch_mod._prepare_component_datasets

    def _spy(component_expression_df, *args, **kwargs):
        prepare_calls.append(component_expression_df.copy())
        return real_prepare(component_expression_df, *args, **kwargs)

    monkeypatch.setattr(orch_mod, "_prepare_component_datasets", _spy)

    # Lower parent-ROI gates so C's 15 spots enter roi_correlations
    # (the default `min_roi_spots=40` would exclude C from parent
    # analysis, hiding whether the gate preserves pooled statistics
    # for a scattered ROI).
    config = load_run_config(
        path=None, profile=None,
        flag_overrides={
            "output_dir": str(tmp_path),
            "roi_filters": {"min_roi_spots": 15, "min_context_spots": 15},
        },
        set_overrides=None,
    )

    results = run_analysis(
        df, config, feature_a_id="Neuron", feature_b_id="AC",
    )

    # -----------------------------------------------------------------
    # 1. roi_support_status classification
    # -----------------------------------------------------------------
    support = results["roi_support_status"]
    assert set(support["roi_id"]) == {"A", "B", "C"}
    row_c = support[support["roi_id"] == "C"].iloc[0]
    assert row_c["spatial_support_class"] == "scattered", (
        f"C must be scattered; got {row_c['spatial_support_class']} "
        f"(largest support = {row_c['largest_support_component_spots']})"
    )
    # A and B individually not scattered.
    for parent in ("A", "B"):
        assert (
            support.loc[support["roi_id"] == parent, "spatial_support_class"].iloc[0]
            != "scattered"
        ), f"{parent} must not be scattered"

    # -----------------------------------------------------------------
    # 2. Scattered ROI absent from spatial CSVs on disk
    # -----------------------------------------------------------------
    geom = results["component_geometry_df"]
    assert "C" not in set(geom[ROI_ID].astype(str)), (
        "C's spots must NOT produce a component_geometry row"
    )
    # Tessellation + consensus outputs are inside results (not always
    # written to disk at this scale), inspect the frames directly.
    # The ROI × replicate table is a total table over source ROIs: a
    # scattered ROI IS present (one row per planned replicate) with a
    # typed NO_TESSELLATION_ELIGIBLE_COMPONENT status. No numeric
    # correlation is manufactured; the row records that the ROI had
    # no tessellation-eligible component to contribute.
    replicate_corr = results.get("roi_replicate_correlations_df")
    if replicate_corr is not None and not replicate_corr.empty:
        c_rows = replicate_corr[replicate_corr[ROI_ID].astype(str) == "C"]
        assert not c_rows.empty, (
            "scattered ROI 'C' must have rows in the ROI×replicate "
            "total table (typed status, not row absence)"
        )
        assert set(c_rows[PARENT_REPLICATE_CORRELATION_STATUS]) == {
            "no_tessellation_eligible_component"
        }
    # Canonical consensus per-hex surface (CORE-OUT-01). Every
    # component_id present must have a corresponding component_geometry
    # row; a scattered ROI's component would otherwise leak into the
    # consensus surface.
    cons_hexes = results.get("consensus_hexes_df")
    if cons_hexes is not None and not cons_hexes.empty:
        geom_component_ids = set(geom[COMPONENT_ID].astype(str))
        for view_key in cons_hexes[COMPONENT_ID].astype(str).unique():
            assert view_key in geom_component_ids, (
                f"consensus hex for {view_key!r} has no geometry row — "
                "scattered ROI's component leaked into consensus"
            )

    # -----------------------------------------------------------------
    # 3. Parent analyzer sees ALL of C's spots
    # -----------------------------------------------------------------
    roi_corr = results["roi_correlations"]
    row_c_corr = roi_corr[roi_corr[ROI_ID].astype(str) == "C"].iloc[0]
    assert int(row_c_corr["sample_size"]) == 15
    roi_enr = results["roi_enrichment"]
    if not roi_enr.empty:
        matching = roi_enr[roi_enr[ROI_ID].astype(str) == "C"]
        if not matching.empty:
            assert int(matching.iloc[0]["n_roi"]) == 15

    # -----------------------------------------------------------------
    # 4. Source frame carries EVERY spot or cell (all 105 = 48 A + 42 B
    # + 15 C). Panels 1/2/3 consume this directly under the source-
    # native contract. No derived scattered-only sub-frame:
    # (a) mixed-ROI fragments would be missed by any ROI-class filter
    # (same ROI has geometry-supported and unsupported spots), and
    # (b) the viz doesn't need it — it renders the full source frame.
    # Scattered-vs-supported labelling in the viz uses
    # `roi_support_status` per-ROI, not a manufactured subset.
    # -----------------------------------------------------------------
    source_df = results["source_expression_df"]
    assert len(source_df) == 48 + 42 + 15
    assert set(source_df[ROI_ID].astype(str).unique()) == {"A", "B", "C"}

    # -----------------------------------------------------------------
    # 5. Issue-1 regression: _prepare_component_datasets receives a
    # frame WITHOUT any scattered-ROI rows. This is the assertion
    # that fires if the gate is moved back below context construction.
    # -----------------------------------------------------------------
    assert len(prepare_calls) == 1, (
        f"expected exactly one _prepare_component_datasets call; "
        f"got {len(prepare_calls)}"
    )
    prepared_frame = prepare_calls[0]
    leaked_rows = prepared_frame[
        prepared_frame[ROI_ID].astype(str) == "C"
    ]
    assert leaked_rows.empty, (
        f"Issue-1 regression: {len(leaked_rows)} row(s) belonging to "
        "scattered ROI 'C' leaked into the context-builder input. "
        "The scattered-ROI gate must run BEFORE "
        "_prepare_component_datasets, otherwise C's spots enter A's "
        "and B's spatial contexts."
    )


def _all_scattered_empty_post_filter_fixture() -> pd.DataFrame:
    """Two small ROIs (3 spots each) placed far apart. Each ROI on its
    own is below `min_component_spots` so `filter_components` empties
    the adaptive-splitter frame entirely, and every biological ROI is
    classified `scattered` by sample-support.

    This exercises the empty-post-filter parent-only path: without the
    guard the pipeline crashed while rebuilding the canvas
    (`zero-size array to reduction operation minimum`). Three-spot
    ROIs (not singletons) so `min_roi_spots=3` still yields meaningful
    parent-level statistics."""
    a_pts = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], dtype=float)
    b_pts = np.array([(200.0, 0.0), (201.0, 0.0), (200.0, 1.0)], dtype=float)
    coords = np.vstack([a_pts, b_pts])
    labels = ["A", "A", "A", "B", "B", "B"]
    rng = np.random.default_rng(2027)
    return pd.DataFrame({
        ROI_ID: labels,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.1, 5.0, len(labels)),
        GENE2: rng.uniform(0.1, 5.0, len(labels)),
    })


def test_all_scattered_empty_post_filter_returns_parent_only(tmp_path: Path):
    """Public-API exercise of the all-scattered + empty-post-filter path.

    Two 3-spot ROIs → `filter_components` empties every adaptive
    component AND every ROI is classified scattered. Runs through
    `run_tilescope_from_df` with `export=True, html_report=True` so
    the crash surface previously hit only via the public path (the
    expression-summary NN estimator, HTML file listing) is exercised.

    Asserts:
      1. No crash.
      2. Parent CSVs + roi_support_status.csv + expression_summary.pdf +
         analysis_report.html are produced.
      3. Spatial component outputs absent or empty (component_geometry.csv
         absent; consensus/tessellation PDFs absent).
      4. In-memory result frames confirm schema-stable empty spatial
         outputs.
    """

    df = _all_scattered_empty_post_filter_fixture()
    results = run_tilescope_from_df(
        df=df,
        config={
            "output_dir": str(tmp_path),
            "roi_filters": {"min_roi_spots": 3, "min_context_spots": 3},
        },
        output_dir=str(tmp_path),
        export=True,
        html_report=True,
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    # --- roi_support_status ------------------------------------------------
    support = results["roi_support_status"]
    assert set(support["spatial_support_class"]) == {"scattered"}
    assert set(support["roi_id"].astype(str)) == {"A", "B"}

    # --- Parent CSVs (total tables) ---------------------------------------
    # Both parents present on both tables with typed status codes.
    # roi_correlations: FINITE for supported parents.
        # roi_enrichment: CONTEXT_PROPORTION_ZERO — the strict primitive
    # raises when the required context proportions collapse under the
    # all-scattered slice; the row records the status, no numeric ratio
    # is manufactured.
    assert set(results["roi_correlations"][ROI_ID].astype(str)) == {"A", "B"}
    assert set(results["roi_enrichment"][ROI_ID].astype(str)) == {"A", "B"}
    assert set(
        results["roi_correlations"][PARENT_CORRELATION_STATUS]
    ) == {"finite"}
    assert set(
        results["roi_enrichment"][PARENT_ENRICHMENT_STATUS]
        ) == {"context_proportion_zero"}

    # --- In-memory spatial artifacts empty (schema-stable) -----------------
    assert results["component_polygons"] == {}
    # ROI × replicate correlation total table: rows PRESENT for A + B
    # under the all-scattered path, every row carries the typed
    # NO_TESSELLATION_ELIGIBLE_COMPONENT status. No numeric correlation
    # is manufactured (correlation_value is NA).
    rrc = results["roi_replicate_correlations_df"]
    assert not rrc.empty
    assert set(rrc[ROI_ID].astype(str)) == {"A", "B"}
    assert set(rrc[PARENT_REPLICATE_CORRELATION_STATUS]) == {
        "no_tessellation_eligible_component"
    }
    # Canonical per-hex frame is empty on the all-scattered path.
    assert results["consensus_hexes_df"].empty

    # --- source_expression_df carries every spot or cell --------------------
    # No scattered-subset assertion: the viz consumes the source frame
    # directly under the spot/cell contract. Scattered-labelling in
    # the viz reads roi_support_status per-ROI, not a manufactured
    # subset — which is also the only correct approach because a mixed
    # ROI contains both supported and unsupported spots that no
    # ROI-class filter could separate.
    source_df = results["source_expression_df"]
    assert len(source_df) == 6
    assert set(source_df[ROI_ID].astype(str)) == {"A", "B"}

    # --- On-disk expected artifacts present --------------------------------
    for produced in (
        "roi_correlations.csv",
        "roi_support_status.csv",
        "expression_summary.pdf",
        "analysis_report.html",
    ):
        assert (tmp_path / produced).exists(), (
            f"{produced!r} must be produced under the all-scattered public "
            "API path with export=True, html_report=True"
        )

    # --- Spatial component outputs absent ---------------------------------
    for absent in (
        "component_geometry.csv",
        "consensus_tessellation_summary.pdf",
        "tessellation_densities_1.pdf",
    ):
        assert not (tmp_path / absent).exists(), (
            f"{absent!r} must NOT be produced when every ROI is scattered"
        )


def test_empty_post_filter_but_some_roi_supported_raises(tmp_path: Path, monkeypatch):
    """Contradiction: `component_expression_df` is empty but the
    sample-support classifier flags some ROI as supported. The pipeline
    must NOT silently treat this as all-scattered; it must raise a
    clear invariant error so the disagreement is surfaced.

    Monkeypatches `derive_roi_support_status` to force a mismatch on
    the same empty-post-filter fixture."""

    df = _all_scattered_empty_post_filter_fixture()
    config = load_run_config(
        path=None, profile=None,
        flag_overrides={
            "output_dir": str(tmp_path),
            "roi_filters": {"min_roi_spots": 3, "min_context_spots": 3},
        },
        set_overrides=None,
    )

    real_derive = se_mod.derive_roi_support_status

    def _forge_one_supported(sizes, min_component_spots):
        out = real_derive(sizes, min_component_spots=min_component_spots)
        # Overwrite ROI A's class to `fully_connected`, keeping B scattered.
        if not out.empty:
            out.loc[out["roi_id"] == "A", "spatial_support_class"] = "fully_connected"
        return out

    # `derive_roi_support_status` is bound into `pre_analysis` at module
    # load time; patch there (patching `se_mod` doesn't rebind the
    # already-imported name).
    monkeypatch.setattr(pa_mod, "derive_roi_support_status", _forge_one_supported)

    with pytest.raises(ValueError, match="Invariant violation.*empty.*supported"):
        run_analysis(
            df, config, feature_a_id="Neuron", feature_b_id="AC",
        )


def _multi_component_fixture() -> pd.DataFrame:
    """Parent A decomposed into TWO components + parent B one component.
    Distinct biological ROI count (2) vs component count (3) — catches
    the 'component count presented as ROI count' rendering defect."""
    a_left = _hex_lattice(6, 5, x0=0.0)
    a_right = _hex_lattice(6, 5, x0=100.0)
    b_main = _hex_lattice(7, 6, x0=200.0)
    coords = np.vstack([a_left, a_right, b_main])
    labels = ["A"] * len(a_left) + ["A"] * len(a_right) + ["B"] * len(b_main)
    rng = np.random.default_rng(2028)
    return pd.DataFrame({
        ROI_ID: labels,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.1, 5.0, len(labels)),
        GENE2: rng.uniform(0.1, 5.0, len(labels)),
    })


def test_html_report_true_no_retired_terminology_and_grain_explicit_counts(tmp_path: Path):
    """HTML must (a) not contain retired vocabulary and
    (b) render biological-ROI and component counts with their
    grain-explicit values — asserts the rendered numbers, not just
    the label text. Also verifies the semantic producer contract on
    the same run: `run_metadata` reports the same distinct counts
    and `component_stats` carries both identity columns.

    Multi-component fixture (2 biological ROIs, 3 geometry-eligible
    components) makes the two counts distinct so the renderer cannot
    pass by accidentally showing the same value in both fields."""

    df = _multi_component_fixture()
    results = run_tilescope_from_df(
        df=df,
        config={
            "output_dir": str(tmp_path),
            "roi_filters": {
                "min_roi_spots": 25,
                "min_context_spots": 25,
            },
        },
        output_dir=str(tmp_path),
        export=True,
        html_report=True,
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    # --- Semantic producer contract (run_metadata) --------------------
    pop = results["run_metadata"]["population"]
    assert pop["biological_roi_count"] == 2, (
        f"biological_roi_count must be 2 (A + B); got {pop['biological_roi_count']}"
    )
    assert pop["spatially_supported_component_count"] == 3, (
        f"spatially_supported_component_count must be 3 (A_left + A_right + B_main); "
        f"got {pop['spatially_supported_component_count']}"
    )
    cstats = results["run_metadata"]["component_stats"]
    assert len(cstats) == 3
    parent_counts = {"A": 0, "B": 0}
    for row in cstats:
        parent_counts[row["roi_id"]] += 1
    assert parent_counts == {"A": 2, "B": 1}

    # --- Rendered HTML contract --------------------------------------
    html = (tmp_path / "analysis_report.html").read_text()

    # Retired vocabulary absent.
    for retired in (
        "Original Cluster",
        "Connected Component ID",
        "Final ROIs",
        # Superseded by the reader-facing "Populations" vocabulary.
        "Retained spots",
        "Geometry-eligible components",
        "Tessellation-eligible components",
        "ROI Page",
        "across all ROIs",
        "Component Spatial Processing",
        "Run configuration",
        # CORE-REPORT-01 retirements:
        "Sample-support connectivity classification",
        "receive no polygon",
        "A <strong>qualifying</strong> support",
        "Analysis settings and provenance",
        "not scientific results",
    ):
        assert retired not in html, (
            f"HTML still contains retired term {retired!r}"
        )

    # Reader-facing section titles + labels present.
    for expected in (
        "<h2>Spatial support by ROI</h2>",
        "<h2>Analysis coverage</h2>",
        "<h2>Downloads</h2>",
        # Analysis coverage sample-level rows — actual pipeline counts.
        "Input biological ROIs",
        "Input spots",
        "Spatially supported components",
        "Spots included in spatial geometry",
        "Tessellated components",
        "Spots included in tessellation",
        "min_component_spots",
        "min_roi_spots",
        # Spatial support by ROI — reader-facing intro + diagnostic columns.
        "TileScope checks whether each ROI contains connected regions",
        "Qualifying support components",
        "Spots in qualifying components",
        # Vignette link is rendered in every report header.
        "How to read this report",
        "docs/HOW_TO_READ_THE_REPORT.md",
        # Collapsed <details> sections.
        "Construction provenance",
        "Analysis settings",
        "All generated files",
        # Component-details reader-facing column headers.
        "Biological ROI",
        "Spatial component",
        "Processing status",
        "Mean tiles per tessellation",
        "Consensus regions",
    ):
        assert expected in html, f"HTML missing expected term {expected!r}"

    assert "min_tessellation_component_spots" not in html

    # Rendered numeric contract: biological_roi_count = 2, spatially
    # supported components = 3. Matches label → value across HTML
    # whitespace / attribute noise.
    bio_match = re.search(
        r"Input biological ROIs</td>\s*<td[^>]*>\s*<strong>\s*2\s*</strong>",
        html,
    )
    assert bio_match, (
        "HTML population table must render 'Input biological ROIs' = 2. "
        "Rendering defect suspected: label present but value wrong."
    )
    supported_match = re.search(
        r"Spatially supported components</td>\s*<td[^>]*>\s*3\s*</td>",
        html,
    )
    assert supported_match, (
        "HTML population table must render 'Spatially supported components' = 3. "
        "Rendering defect suspected: label present but value wrong."
    )
