"""Support-provenance diagnostic panel.

Three scientific states rendered on the sample geometry:

1. **No observation**: figure background (white).
2. **Observed, native-statistic only**: coloured fill in the parent-ROI
   colour, with a red outer contour showing that this observed support
   is unavailable to tessellation / consensus aggregation.
3. **Observed, tessellation-aggregatable**: coloured fill in the
   parent-ROI colour, with the normal ROI boundary line.

Contiguous fragments of unconstructible spots merge naturally through
overlapping spot-support footprints; disconnected fragments stay
disconnected (no hulls, no interpolation).

The legend + text summary sit outside the plot on the right. Cause
codes (A / B / C / D / F) are collapsed to user-facing labels only in
the count summary; the ledger + schema documentation retain the full
truth-table vocabulary.
"""
from __future__ import annotations

from tilescope.spatial.spacing import median_nn_distance_from_df
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Patch


from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from tilescope.cols import GENE1, GENE2
from tilescope.spatial import ComponentPolygonDict
from tilescope.viz.painters import draw_polygon_boundaries
from tilescope.viz.support_overlay import (
    CAUSE_FILTER_DROPPED, CAUSE_INELIGIBLE_REGION, CAUSE_OUTSIDE_DOMAIN,
    CAUSE_SCATTERED_PARENT, SPOT_SUPPORT_RADIUS_FACTOR, SupportOverlay,
    annotate_spots_with_parent_roi, classify, grouped_spot_support_parts,
)


_CONSTRUCTIBLE_ALPHA = 0.42
_UNCONSTRUCTIBLE_ALPHA = 0.42
_UNCONSTRUCTIBLE_EDGE = "#C0392B"        # red contour for excluded support
_UNCONSTRUCTIBLE_EDGE_WIDTH = 1.2

# User-facing labels for the three cause groups that surface in the
# count summary. A / B collapse into "small disconnected groups"; D
# gets its own label if non-zero (rare in practice). Zero-count causes
# are omitted from the figure per the presentation contract.
_CAUSE_USER_LABEL: Dict[str, str] = {
    CAUSE_FILTER_DROPPED:    "Small disconnected groups",
    CAUSE_SCATTERED_PARENT:  "Spatially dispersed ROIs",
    CAUSE_INELIGIBLE_REGION: "Local regions unable to support tessellation",
    CAUSE_OUTSIDE_DOMAIN:    "Observed support outside the computed domain",
}
_CAUSE_SUMMARY_ORDER = (
    CAUSE_FILTER_DROPPED,
    CAUSE_SCATTERED_PARENT,
    CAUSE_INELIGIBLE_REGION,
    CAUSE_OUTSIDE_DOMAIN,
)


def plot_support_provenance(
    *,
    spot_ledger_df: pd.DataFrame,
    source_expression_df: pd.DataFrame,
    component_polygons: Optional[ComponentPolygonDict],
    component_to_roi: Optional[Dict[str, str]],
    canvas,
    output_path: str,
    source_median_nn: Optional[float] = None,
    feature_a_col: str = GENE1,
    feature_b_col: str = GENE2,
) -> Optional[SupportOverlay]:
    """Render `support_provenance.pdf` and return the classified overlay."""
    if spot_ledger_df is None or source_expression_df is None:
        return None
    overlay = classify(
        spot_ledger_df, source_expression_df,
        gene1_col=feature_a_col, gene2_col=feature_b_col,
    )
    if source_median_nn is None:
        source_median_nn = median_nn_distance_from_df(source_expression_df, "x", "y")
    median_nn = source_median_nn
    parent_palette = _build_parent_palette(component_to_roi or {}, spot_ledger_df)
    parents_present = _parents_actually_shown(
        overlay, spot_ledger_df, component_polygons or {}, component_to_roi or {},
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Observed spatial support")

    _draw_constructible_polygons(
        ax, component_polygons or {}, component_to_roi or {}, parent_palette,
    )
    _draw_unconstructible_fragments(
        ax, overlay, spot_ledger_df, source_expression_df, parent_palette,
        median_nn=median_nn,
    )
    if component_polygons:
        draw_polygon_boundaries(ax, component_polygons)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")

    _draw_right_panel(
        fig, ax, overlay, parent_palette, parents_present,
        component_polygons or {}, median_nn,
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return overlay


# ------------------------------------------------------------------ #
# Parent-ROI palette (avoids reddish/orange hues to keep the red
# unconstructible contour + orange ROI boundary channels distinct).
# ------------------------------------------------------------------ #

# tab10 with reddish/orange slots (1, 3) skipped.
_TAB10_NON_RED_ORDER = (0, 2, 4, 5, 6, 8, 9, 7)


def _build_parent_palette(
    component_to_roi: Dict[str, str], spot_ledger_df: pd.DataFrame,
) -> Dict[str, tuple]:
    parents = set(component_to_roi.values())
    if "roi_id" in spot_ledger_df.columns:
        parents.update(str(r) for r in spot_ledger_df["roi_id"].dropna().unique())
    parents = sorted(parents, key=lambda r: (len(str(r)), str(r)))
    if not parents:
        return {}
    cmap = plt.get_cmap("tab10")
    palette: Dict[str, tuple] = {}
    for i, roi in enumerate(parents):
        idx = _TAB10_NON_RED_ORDER[i % len(_TAB10_NON_RED_ORDER)]
        palette[str(roi)] = tuple(cmap(idx)[:3])
    return palette


def _parents_actually_shown(
    overlay: SupportOverlay,
    spot_ledger_df: pd.DataFrame,
    component_polygons: ComponentPolygonDict,
    component_to_roi: Dict[str, str],
) -> list:
    """Return parent ROIs that receive any coloured fill on the panel,
    in the order used by the legend (numeric roi_id where possible)."""
    parents = set()
    for cid in component_polygons:
        p = component_to_roi.get(str(cid))
        if p is not None:
            parents.add(str(p))
    if "roi_id" in spot_ledger_df.columns:
        parents.update(str(r) for r in spot_ledger_df["roi_id"].dropna().unique())
    def _key(r: str):
        try:
            return (0, int(r))
        except (TypeError, ValueError):
            return (1, r)
    return sorted(parents, key=_key)


# ------------------------------------------------------------------ #
# Constructible component polygons.
# ------------------------------------------------------------------ #


def _draw_constructible_polygons(
    ax, component_polygons: ComponentPolygonDict,
    component_to_roi: Dict[str, str],
    parent_palette: Dict[str, tuple],
) -> None:
    for cid, poly in component_polygons.items():
        parent = component_to_roi.get(str(cid))
        colour = parent_palette.get(str(parent))
        if colour is None:
            continue
        for part in _iter_polygon_parts(poly):
            coords = np.asarray(part.exterior.coords)
            ax.add_patch(PathPatch(
                Path(coords), facecolor=colour, edgecolor="none",
                alpha=_CONSTRUCTIBLE_ALPHA,
            ))


def _iter_polygon_parts(poly) -> Iterable[Polygon]:
    if isinstance(poly, MultiPolygon):
        return [g for g in poly.geoms if not g.is_empty]
    if isinstance(poly, Polygon) and not poly.is_empty:
        return [poly]
    return []


# ------------------------------------------------------------------ #
# Unconstructible fragments: per-parent unioned spot-support discs with
# red outer contour. Disconnected fragments stay disconnected.
# ------------------------------------------------------------------ #


def _draw_unconstructible_fragments(
    ax, overlay: SupportOverlay,
    spot_ledger_df: pd.DataFrame,
    source_expression_df: pd.DataFrame,
    parent_palette: Dict[str, tuple],
    *, median_nn: Optional[float],
) -> None:
    if overlay.unconstructible.empty or median_nn is None:
        return

    unc = annotate_spots_with_parent_roi(
        overlay.unconstructible, spot_ledger_df, source_expression_df,
        output_col="_parent",
    )
    grouped = grouped_spot_support_parts(
        unc, group_col="_parent", median_nn=median_nn,
        radius_factor=SPOT_SUPPORT_RADIUS_FACTOR,
    )
    for parent, parts in grouped.items():
        colour = parent_palette.get(str(parent))
        if colour is None:
            continue
        for part in parts:
            coords = np.asarray(part.exterior.coords)
            ax.add_patch(PathPatch(
                Path(coords), facecolor=colour, edgecolor="none",
                alpha=_UNCONSTRUCTIBLE_ALPHA,
            ))
            ax.add_patch(PathPatch(
                Path(coords), facecolor="none",
                edgecolor=_UNCONSTRUCTIBLE_EDGE,
                linewidth=_UNCONSTRUCTIBLE_EDGE_WIDTH,
            ))


# ------------------------------------------------------------------ #
# Right-side panel: legend + explanatory text + count summary.
# ------------------------------------------------------------------ #


def _draw_right_panel(fig, ax, overlay: SupportOverlay,
                      parent_palette: Dict[str, tuple],
                      parents_present, component_polygons,
                      median_nn: Optional[float]) -> None:

    handles = []
    for parent in parents_present:
        colour = parent_palette.get(str(parent))
        if colour is None:
            continue
        handles.append(Patch(
            facecolor=(*colour, _CONSTRUCTIBLE_ALPHA),
            edgecolor="none",
            label=f"ROI {parent}",
        ))
    handles.append(Patch(
        facecolor="#DDDDDD", edgecolor=_UNCONSTRUCTIBLE_EDGE,
        linewidth=_UNCONSTRUCTIBLE_EDGE_WIDTH,
        label="Red contour — observed support unavailable to tessellation",
    ))
    handles.append(Patch(
        facecolor="white", edgecolor="#888888", linewidth=0.6,
        label="White — no observed acquisition",
    ))

    legend = ax.legend(
        handles=handles,
        bbox_to_anchor=(1.03, 1.0), loc="upper left",
        borderaxespad=0.0, frameon=True, fontsize=9,
        title="Parent ROI colour",
    )
    legend.get_frame().set_linewidth(0.4)

    explanation = (
        "All coloured areas contain observed spots and contribute to\n"
        "native spot-level ROI statistics. Red outlines mark observed\n"
        "support unavailable to tessellation and consensus aggregation.\n"
        "White denotes no observed acquisition. Geometric holes contain\n"
        "no observed spots and are excluded from constructed-grain analyses."
    )
    fig.text(
        0.72, 0.42, explanation,
        fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#FAFAFA", edgecolor="#CCCCCC", linewidth=0.4),
    )

    summary = _build_summary_text(overlay)
    geometry_summary = _build_geometry_summary_text(
        component_polygons, median_nn,
    )
    if geometry_summary:
        summary = f"{summary}\n\n{geometry_summary}" if summary else geometry_summary
    if summary:
        fig.text(
            0.72, 0.18, summary,
            fontsize=9, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#FAFAFA", edgecolor="#CCCCCC", linewidth=0.4),
        )


def _build_summary_text(overlay: SupportOverlay) -> str:
    total = sum(overlay.counts.get(c, 0) for c in _CAUSE_SUMMARY_ORDER)
    if total == 0:
        return ""
    lines = [f"Observed spots unavailable to tessellation: {total}", ""]
    for cause in _CAUSE_SUMMARY_ORDER:
        n = overlay.counts.get(cause, 0)
        if n == 0:
            continue
        lines.append(f"  {_CAUSE_USER_LABEL[cause]}: {n} spots")
    return "\n".join(lines)


def _build_geometry_summary_text(component_polygons, median_nn) -> str:
    hole_areas = [
        float(Polygon(ring).area)
        for geometry in component_polygons.values()
        for part in _iter_polygon_parts(geometry)
        for ring in part.interiors
    ]
    if not hole_areas or median_nn is None:
        return f"Constructed-domain holes: {len(hole_areas):,}"
    footprint_area = np.pi * (
        SPOT_SUPPORT_RADIUS_FACTOR * float(median_nn)
    ) ** 2
    median_relative_area = float(np.median(hole_areas)) / footprint_area
    return (
        f"Constructed-domain holes: {len(hole_areas):,}\n"
        f"Median hole size: {median_relative_area:.3g} spot footprints"
    )
