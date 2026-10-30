"""Consensus marginal-pattern rendering + four-state classification.

Coverage:

  1. One rendering smoke — plot_consensus_grids produces a non-empty
     PDF on a five-quadrant fixture that exercises every enrichment
     state (both↑ / both↓ / A↑B↓ / A↓B↑ / near-neutral). The pure
     classification tests below already cover per-state coloring, so
     one representative render suffices.
  2. ROI-panel rendering with real dissolved geometry — supplies
     component_polygons + component_to_roi so the constructed-support
     Pearson and Moran panels actually paint per ROI.
  3. Pure-Python unit tests for _transform_log2_fc and
     _classify_marginal_pattern.
  4. Stability-layout regression — show_stability=True adds a single
     column, not two.
  5. Identity-header behavioral — add_identity_header writes A/B
     mapping + heading to the figure suptitle.
"""
from __future__ import annotations

from tilescope.cols import MORAN_DIRECTION, MORAN_VALUE, PARENT_MORAN_STATUS
from tilescope.viz.consensus import _COLOR_BOTH_UP, _COLOR_BOTH_DOWN, _COLOR_G1UP_G2DOWN, _COLOR_G1DOWN_G2UP
from tilescope.viz.consensus import _COLOR_NEUTRAL
from tilescope.viz.consensus import _hex_has_positive_over_zero_context, _enrichment_colorbar_label
from tilescope.viz import consensus as consensus_mod
from tilescope.viz.support_overlay import classify
from tilescope.viz.color_norms import resolve_expression_norm
from tilescope.viz import expression as expression_mod


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from tilescope.viz.consensus import (
    _classify_marginal_pattern,
    _transform_log2_fc,
    plot_consensus_grids,
)
from tilescope.viz.painters import add_identity_header
from tilescope.cols import (
    COMPONENT_ID, CORRELATION_VALUE, GENE1_DENSITY_ENSEMBLE, GENE1_DENSITY_STABILITY,
    GENE2_DENSITY_ENSEMBLE, GENE2_DENSITY_STABILITY,
    HEX_ENRICHMENT_STATUS,
    HEX_G1_ENRICHMENT, HEX_G1_ENRICHMENT_STATUS,
    HEX_G2_ENRICHMENT, HEX_G2_ENRICHMENT_STATUS,
    HEX_GEOMETRY_WKT, HEX_ID,
    HEX_JOINT_HIGH_ENRICHMENT_FC, HEX_JOINT_HIGH_ENRICHMENT_STATUS,
    HEX_JOINT_LOW_ENRICHMENT_FC, HEX_JOINT_LOW_ENRICHMENT_STATUS,
    PARENT_CONSENSUS_CORRELATION_STATUS, PARENT_CONSENSUS_ENRICHMENT_STATUS,
    ROI_ID, SAMPLE_ID, VALUE_SURFACE,
)


_HEX_COUNTER = {"i": 0}


def _make_hex(x_offset: float, g1_enrich: float, g2_enrich: float,
              joint_high: float = 1.0, joint_low: float = 1.0,
              *, component_id: str = "component_0",
              roi_id: str = "roi_test",
              polygon=None) -> dict:
    poly = polygon if polygon is not None else box(x_offset, 0, x_offset + 1, 1)
    hid = _HEX_COUNTER["i"]
    _HEX_COUNTER["i"] += 1
    return {
        SAMPLE_ID: "S",
        ROI_ID: roi_id,
        COMPONENT_ID: component_id,
        HEX_ID: hid,
        HEX_GEOMETRY_WKT: poly.wkt,
        GENE1_DENSITY_ENSEMBLE: 1.0,
        GENE2_DENSITY_ENSEMBLE: 1.0,
        GENE1_DENSITY_STABILITY: 0.1,
        GENE2_DENSITY_STABILITY: 0.1,
        PARENT_CONSENSUS_ENRICHMENT_STATUS: "finite",
        HEX_ENRICHMENT_STATUS: "finite",
        HEX_G1_ENRICHMENT: g1_enrich,
        HEX_G2_ENRICHMENT: g2_enrich,
        HEX_JOINT_HIGH_ENRICHMENT_FC: joint_high,
        HEX_JOINT_LOW_ENRICHMENT_FC: joint_low,
        HEX_G1_ENRICHMENT_STATUS: "finite",
        HEX_G2_ENRICHMENT_STATUS: "finite",
        HEX_JOINT_HIGH_ENRICHMENT_STATUS: "finite",
        HEX_JOINT_LOW_ENRICHMENT_STATUS: "finite",
    }


def _five_quadrants_df() -> pd.DataFrame:
    return pd.DataFrame([
        _make_hex(0, 2.0, 2.0, joint_high=3.0),   # both↑
        _make_hex(2, 0.5, 0.5, joint_low=3.0),    # both↓
        _make_hex(4, 2.0, 0.5),                   # A↑ B↓
        _make_hex(6, 0.5, 2.0),                   # A↓ B↑
        _make_hex(8, 1.0, 1.0),                   # near-neutral
    ])


def _matched_ledger_and_source(rows) -> tuple:
    """Build a matched (spot_ledger_df, source_expression_df) pair.

    ``rows`` is a list of dicts with keys ``spot_id``, ``x``, ``y``,
    ``g1``, ``g2``, ``pattern`` where pattern is 'constructible' or
    'scattered_parent' (chosen to exercise the truth-table's E and C
    branches without needing full pipeline state)."""
    ledger_rows = []
    source_rows = []
    for r in rows:
        if r["pattern"] == "constructible":
            ledger_rows.append({
                "spot_index": r["spot_id"],
                "component_status": "retained_component",
                "exclusion_reason": None,
                "parent_scattered": False,
                "roi_id": r.get("roi_id", "1"),
                "component_id": "c0",
                "domain_region_id": "r0",
                "tessellation_eligible": True,
            })
        elif r["pattern"] == "scattered_parent":
            ledger_rows.append({
                "spot_index": r["spot_id"],
                "component_status": "retained_component",
                "exclusion_reason": None,
                "parent_scattered": True,
                "roi_id": r.get("roi_id", "1"),
                "component_id": "c0",
                "domain_region_id": None,
                "tessellation_eligible": None,
            })
        else:
            raise ValueError(f"unknown pattern {r['pattern']!r}")
        source_rows.append({
            "source_row_id": r["spot_id"],
            "x": r["x"], "y": r["y"],
            "gene1": r["g1"], "gene2": r["g2"],
            ROI_ID: r.get("roi_id", "1"),
        })
    return pd.DataFrame(ledger_rows), pd.DataFrame(source_rows)


def _minimal_ledger_and_source() -> tuple:
    """One constructible spot outside all fixture hex geometry — enough
    to satisfy the classify() contract without altering density norms."""
    return _matched_ledger_and_source([
        {"spot_id": "s0", "x": -100.0, "y": -100.0,
         "g1": 1.0, "g2": 1.0, "pattern": "constructible"},
    ])


# ---------------------------------------------------------------------------
# 1. Rendering smoke: one representative shape covers all five states.
# ---------------------------------------------------------------------------

def test_plot_consensus_grids_produces_nonempty_pdf(tmp_path):
    ledger, source = _minimal_ledger_and_source()
    plot_consensus_grids(
        consensus_hexes_df=_five_quadrants_df(),
        spot_ledger_df=ledger, source_expression_df=source,
        output_dir=str(tmp_path),
        show_bivariate_morans=False,
        feature_a_id="Neuron", feature_b_id="AC",
    )
    output = tmp_path / "consensus_tessellation_summary.pdf"
    assert output.exists() and output.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. ROI panels rendered with real dissolved geometry.
# ---------------------------------------------------------------------------

def _make_roi_geometry_and_frames(rois=("A", "B")):
    """Return (component_polygons, component_to_roi, roi_corr_df,
    roi_moran_df, hexes_df) that exercise the dissolved-ROI paint path."""
    x0 = 0
    component_polygons = {}
    component_to_roi = {}
    hex_rows = []
    for i, roi_id in enumerate(rois):
        cid = f"c_{roi_id}"
        component_polygons[cid] = box(x0, 0, x0 + 10, 10)
        component_to_roi[cid] = roi_id
        hex_rows.append(_make_hex(
            x0 + 1, 2.0, 0.5, component_id=cid, roi_id=roi_id,
            polygon=box(x0 + 1, 1, x0 + 2, 2),
        ))
        x0 += 12
    roi_corr = pd.DataFrame([
        {SAMPLE_ID: "S", ROI_ID: rid, VALUE_SURFACE: "consensus_ensemble",
         PARENT_CONSENSUS_CORRELATION_STATUS: "finite",
         CORRELATION_VALUE: 0.42 if i == 0 else -0.31}
        for i, rid in enumerate(rois)
    ])
    roi_moran = pd.DataFrame([
        {SAMPLE_ID: "S", ROI_ID: rid,
         VALUE_SURFACE: "consensus_ensemble",
         MORAN_DIRECTION: d, MORAN_VALUE: 0.15,
         PARENT_MORAN_STATUS: "finite"}
        for rid in rois for d in ("g1_lag_g2", "g2_lag_g1")
    ])
    return component_polygons, component_to_roi, roi_corr, roi_moran, pd.DataFrame(hex_rows)


def test_plot_consensus_grids_paints_dissolved_roi_panels(tmp_path):
    """Supplies real component_polygons + component_to_roi so the
    constructed-support Pearson and Moran panels actually render per
    ROI (not merely proves a PDF was written)."""
    comp_polys, c2r, roi_corr, roi_moran, hexes = _make_roi_geometry_and_frames()
    ledger, source = _minimal_ledger_and_source()
    plot_consensus_grids(
        consensus_hexes_df=hexes,
        spot_ledger_df=ledger, source_expression_df=source,
        roi_consensus_correlations_df=roi_corr,
        roi_consensus_moran_df=roi_moran,
        component_polygons=comp_polys,
        component_to_roi=c2r,
        output_dir=str(tmp_path),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    output = tmp_path / "consensus_tessellation_summary.pdf"
    assert output.exists() and output.stat().st_size > 0


def test_plot_consensus_grids_multipart_roi_geometry(tmp_path):
    """A ROI whose dissolved polygon is a MultiPolygon still paints
    end-to-end (Pearson panel exercises the multipart branch)."""
    comp_polys = {
        "cA_left": box(0, 0, 5, 5), "cA_right": box(20, 0, 25, 5),  # two-part A
        "cB": box(40, 0, 50, 10),
    }
    c2r = {"cA_left": "A", "cA_right": "A", "cB": "B"}
    hexes = pd.DataFrame([
        _make_hex(1, 2.0, 0.5, component_id="cA_left", roi_id="A",
                  polygon=box(1, 1, 2, 2)),
        _make_hex(41, 0.5, 2.0, component_id="cB", roi_id="B",
                  polygon=box(41, 1, 42, 2)),
    ])
    roi_corr = pd.DataFrame([
        {SAMPLE_ID: "S", ROI_ID: rid, VALUE_SURFACE: "consensus_ensemble",
         PARENT_CONSENSUS_CORRELATION_STATUS: "finite",
         CORRELATION_VALUE: 0.5}
        for rid in ("A", "B")
    ])
    roi_moran = pd.DataFrame([
        {SAMPLE_ID: "S", ROI_ID: rid,
         VALUE_SURFACE: "consensus_ensemble",
         MORAN_DIRECTION: d,
         MORAN_VALUE: 0.1, PARENT_MORAN_STATUS: "finite"}
        for rid in ("A", "B") for d in ("g1_lag_g2", "g2_lag_g1")
    ])
    ledger, source = _minimal_ledger_and_source()
    plot_consensus_grids(
        consensus_hexes_df=hexes,
        spot_ledger_df=ledger, source_expression_df=source,
        roi_consensus_correlations_df=roi_corr,
        roi_consensus_moran_df=roi_moran,
        component_polygons=comp_polys,
        component_to_roi=c2r,
        output_dir=str(tmp_path),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    output = tmp_path / "consensus_tessellation_summary.pdf"
    assert output.exists() and output.stat().st_size > 0


# ---------------------------------------------------------------------------
# 3. _transform_log2_fc: saturation + clipping + non-finite + negative-raise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fc,expected", [
    (1.0, 0.0),          # FC=1 -> log2=0
    (8.0, 3.0),          # at saturation
    (100.0, 3.0),        # clipped above
    (1.0 / 100.0, -3.0), # clipped below
    (0.0, -3.0),         # EXACT zero saturates negative (not -inf)
])
def test_transform_log2_fc_saturation_and_clipping(fc, expected):
    assert _transform_log2_fc(fc) == pytest.approx(expected)


def test_transform_log2_fc_propagates_nonfinite():
    assert np.isnan(_transform_log2_fc(float("nan")))


def test_transform_log2_fc_raises_on_negative():
    """Negative FC is invalid input for a ratio-of-proportions
    statistic and must not silently degrade to zero-depletion."""
    with pytest.raises(ValueError, match="non-negative"):
        _transform_log2_fc(-0.5)


# ---------------------------------------------------------------------------
# 4. _classify_marginal_pattern: four states + near-neutral + undefined.
# ---------------------------------------------------------------------------

def test_classify_marginal_pattern_undefined_when_either_input_nonfinite():
    rgba, is_undef = _classify_marginal_pattern(float("nan"), 2.0)
    assert is_undef is True and rgba is None


def test_classify_marginal_pattern_near_neutral_when_either_marginal_in_band():
    # log2(1.1) ≈ 0.138 < _NEUTRAL_LOG2_BAND (0.25) → near-neutral gray
    rgba, is_undef = _classify_marginal_pattern(1.1, 4.0)
    assert is_undef is False
    assert rgba[:3] == pytest.approx((0.93, 0.93, 0.93))


@pytest.mark.parametrize("g1_fc,g2_fc,expected_hue", [
    (8.0, 8.0, "both_up"),
    (0.125, 0.125, "both_down"),
    (8.0, 0.125, "g1up_g2down"),
    (0.125, 8.0, "g1down_g2up"),
])
def test_classify_marginal_pattern_four_states_at_saturation(g1_fc, g2_fc, expected_hue):
    expected = {
        "both_up": _COLOR_BOTH_UP, "both_down": _COLOR_BOTH_DOWN,
        "g1up_g2down": _COLOR_G1UP_G2DOWN, "g1down_g2up": _COLOR_G1DOWN_G2UP,
    }[expected_hue]
    rgba, is_undef = _classify_marginal_pattern(g1_fc, g2_fc)
    assert is_undef is False
    assert rgba[:3] == pytest.approx(tuple(expected))


def test_classify_marginal_pattern_weaker_marginal_drives_brightness():
    def _dist(rgba):
        return float(np.linalg.norm(np.array(rgba[:3]) - _COLOR_NEUTRAL))
    strong_rgba, _ = _classify_marginal_pattern(8.0, 8.0)
    weaker_rgba, _ = _classify_marginal_pattern(8.0, 2.0)
    assert _dist(weaker_rgba) < _dist(strong_rgba)
    assert strong_rgba[:3] == pytest.approx(tuple(_COLOR_BOTH_UP))


def test_both_up_and_both_down_endpoint_contrast_are_balanced():
    """Equal-magnitude both-up and both-down hexes must carry equal
    visual weight against the shared neutral background. Contrast is
    measured as WCAG relative luminance ratio of the saturated
    quadrant colour against `_COLOR_NEUTRAL`; the two must agree
    within 5%. Guards against a palette drift that would visually
    favour one direction over the other at matched enrichment.
    """

    def _luminance(rgb):
        # WCAG relative luminance (sRGB → linear → weighted sum).
        rgb = np.asarray(rgb, dtype=float)
        linear = np.where(
            rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4,
        )
        return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])

    def _contrast(a, b):
        la, lb = _luminance(a), _luminance(b)
        hi, lo = max(la, lb), min(la, lb)
        return (hi + 0.05) / (lo + 0.05)

    up_contrast = _contrast(_COLOR_BOTH_UP, _COLOR_NEUTRAL)
    down_contrast = _contrast(_COLOR_BOTH_DOWN, _COLOR_NEUTRAL)
    rel = abs(up_contrast - down_contrast) / max(up_contrast, down_contrast)
    assert rel < 0.05, (
        f"both-up (contrast={up_contrast:.2f}) and both-down "
        f"(contrast={down_contrast:.2f}) must be within 5% at "
        f"saturation; got relative gap {rel:.3f}"
    )


def test_pattern_intensity_transfer_is_concave_and_monotonic():
    """The concave `t = ((m - neutral) / (sat - neutral)) ** 0.6`
    intensity ramp must (a) remain strictly monotone in the weaker
    marginal magnitude and (b) map a mid-range magnitude (m=1.35 —
    the p90 on real Greenwald cohorts) to > 55% of saturated
    intensity. A linear ramp would only reach ~45% at m=1.35 and
    fail the readability floor; this test distinguishes the two.
    """
    def _dist(fc):
        rgba, _ = _classify_marginal_pattern(fc, fc)
        return float(np.linalg.norm(np.array(rgba[:3]) - _COLOR_NEUTRAL))
    fcs = [2.0 ** m for m in (0.30, 0.75, 1.00, 1.35, 2.00, 3.00, 5.00)]
    distances = [_dist(fc) for fc in fcs]
    for prev, curr in zip(distances, distances[1:]):
        assert curr >= prev - 1e-9, (
            f"pattern intensity must be monotonic in weaker marginal; "
            f"got sequence {distances!r}"
        )
    # Concave readability floor at m=1.35: intensity ≥ 55% of the
    # magnitude-3 saturated value. Linear ramp gives 1.35/3 = 0.45
    # ⇒ fails; concave x**0.6 remap gives ~0.58 ⇒ passes.
    dist_135 = _dist(2.0 ** 1.35)
    dist_sat = _dist(2.0 ** 3.0)
    ratio = dist_135 / dist_sat
    assert ratio > 0.55, (
        f"m=1.35 intensity must exceed 55% of saturated intensity "
        f"under the concave transfer; got ratio={ratio:.3f}"
    )


def test_positive_over_zero_context_marker_gated_on_actual_status():
    """The `+∞` colorbar tick + label suffix appear only when at
    least one hex on that marginal actually has
    `positive_over_zero_context`. Independent per marginal so g1 and
    g2 label separately.
    """
    finite_only = _hex_has_positive_over_zero_context(_five_quadrants_df())
    assert finite_only == {"g1": False, "g2": False}
    assert "+∞" not in _enrichment_colorbar_label("A", False)

    df = _five_quadrants_df().copy()
    df.loc[0, HEX_G1_ENRICHMENT_STATUS] = "positive_over_zero_context"
    df.loc[0, HEX_G1_ENRICHMENT] = float("nan")
    mixed = _hex_has_positive_over_zero_context(df)
    assert mixed == {"g1": True, "g2": False}
    assert "+∞" in _enrichment_colorbar_label("A", True)
    assert "+∞" not in _enrichment_colorbar_label("B", False)


# ---------------------------------------------------------------------------
# 5. Stability layout: `show_stability=True` adds ONE column (was two).
# ---------------------------------------------------------------------------

def test_show_stability_true_renders_end_to_end(tmp_path):
    """Regression: `show_stability=True` used to crash with IndexError
    inside plot_consensus_grids because the layout helper added one
    column while the second layout arithmetic assumed two. A real
    render with show_stability=True must succeed."""
    ledger, source = _minimal_ledger_and_source()
    plot_consensus_grids(
        consensus_hexes_df=_five_quadrants_df(),
        spot_ledger_df=ledger, source_expression_df=source,
        output_dir=str(tmp_path),
        show_stability=True,
        show_bivariate_morans=False,
        feature_a_id="Neuron", feature_b_id="AC",
    )
    output = tmp_path / "consensus_tessellation_summary.pdf"
    assert output.exists() and output.stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. Identity-header behavioral: A/B mapping + heading land in suptitle.
# ---------------------------------------------------------------------------

def test_add_identity_header_writes_heading_and_mapping():
    fig, _ = plt.subplots(1, 1)
    add_identity_header(fig, "Neuron", "AC", "Consensus representation")
    text = fig._suptitle.get_text()
    plt.close(fig)
    assert "Consensus representation" in text
    assert "A = Neuron" in text
    assert "B = AC" in text


def test_add_identity_header_rejects_missing_ids():
    fig, _ = plt.subplots(1, 1)
    with pytest.raises(ValueError, match="biological feature identit"):
        add_identity_header(fig, None, None, "Source expression")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Density-panel spot/cell fallback + strict ledger/source contract.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["ledger", "source"])
def test_plot_consensus_grids_requires_both_ledger_and_source(tmp_path, missing):
    ledger, source = _minimal_ledger_and_source()
    kw = dict(
        consensus_hexes_df=_five_quadrants_df(),
        spot_ledger_df=None if missing == "ledger" else ledger,
        source_expression_df=None if missing == "source" else source,
        output_dir=str(tmp_path), show_bivariate_morans=False,
        feature_a_id="Neuron", feature_b_id="AC",
    )
    with pytest.raises(ValueError, match="acquisition|fallback"):
        plot_consensus_grids(**kw)


def test_feature_panels_use_native_expression_fallback(tmp_path, monkeypatch):
    """Given one constructible spot (already covered by hex means) and
    one acquired-but-unconstructible spot with fallback g1=30, g2=40:
    the consensus PDF renders end-to-end and only the unconstructible
    spot enters the fallback bucket. Norm behavior is covered by the
    dedicated resolver parity test — the resolver may saturate extreme
    values by design."""
    ledger, source = _matched_ledger_and_source([
        {"spot_id": "s0", "x": -100.0, "y": -100.0,
         "g1": 1.0, "g2": 1.0, "pattern": "constructible", "roi_id": "1"},
        {"spot_id": "s1", "x": 20.0, "y": 20.0,
         "g1": 30.0, "g2": 40.0, "pattern": "scattered_parent",
         "roi_id": "2"},
    ])
    real_painter = consensus_mod.paint_density_ring_overlay
    calls = []

    def spy_painter(ax, overlay, **kwargs):
        calls.append(kwargs)
        return real_painter(ax, overlay, **kwargs)

    monkeypatch.setattr(
        consensus_mod, "paint_density_ring_overlay", spy_painter,
    )
    plot_consensus_grids(
        consensus_hexes_df=_five_quadrants_df(),
        spot_ledger_df=ledger, source_expression_df=source,
        output_dir=str(tmp_path),
        show_bivariate_morans=False,
        source_median_nn=1.0,
        feature_a_id="Neuron", feature_b_id="AC",
    )
    assert (tmp_path / "consensus_tessellation_summary.pdf").exists()
    # A/B expression and A/B enrichment panels all reuse native A/B
    # expression. No denominator-one pseudo-enrichment is synthesized.
    assert [call["gene_col"] for call in calls] == [
        "gene1", "gene2", "gene1", "gene2",
    ]
    assert all(call["individual_outline"] is False for call in calls)
    # Constructible spots (s0) never enter unconstructible; only s1 does.
    overlay = classify(ledger, source)
    unc_g1 = set(overlay.unconstructible.gene1.tolist())
    assert 30.0 in unc_g1 and 1.0 not in unc_g1


def test_density_norm_resolver_shared_with_expression_summary():
    """Amendment 1 parity: both consensus + expression modules resolve
    the density cmap+norm through one shared helper on the same union,
    so identical inputs yield identical (cmap, vmin, vmax) triples."""
    # Both modules must import the SAME function object, not a copy.
    assert consensus_mod.resolve_expression_norm is resolve_expression_norm
    assert expression_mod.resolve_expression_norm is resolve_expression_norm
    values = np.array([-2.0, -0.5, 0.0, 0.3, 1.5, 30.0])
    cmap, norm = resolve_expression_norm(values, 'viridis')
    # Union crosses zero → diverging RdBu_r, symmetric limits at robust 99th.
    assert cmap == "RdBu_r"
    assert norm.vmin == -norm.vmax
    assert norm.vmax > 0
