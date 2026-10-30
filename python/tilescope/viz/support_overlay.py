"""Support-overlay primitives for the visualization layer.

Classifies each spot or cell via the extended spot_ledger truth
table and provides native-value fills, unavailable-statistic rings,
and unioned connected-support contours shared by every renderer.
"""
from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, SOURCE_ROW_ID

CAUSE_FILTER_DROPPED_SPLIT = "filter_dropped_split"          # ledger pattern A
CAUSE_FILTER_DROPPED_FILTER = "filter_dropped_filter"        # ledger pattern B
CAUSE_SCATTERED_PARENT = "scattered_parent"                  # ledger pattern C
CAUSE_OUTSIDE_DOMAIN = "outside_retained_domain"             # ledger pattern D
CAUSE_INELIGIBLE_REGION = "ineligible_region"                # ledger pattern F
CONSTRUCTIBLE = "constructible"                              # ledger pattern E

# Four public cause groups shown on the provenance panel. A + B collapse
# into "filter-dropped small component" since they share biological
# meaning; the truth-table itself keeps them separate for auditability.
CAUSE_FILTER_DROPPED = "filter_dropped"

CAUSE_ORDER: Tuple[str, ...] = (
    CAUSE_FILTER_DROPPED,
    CAUSE_SCATTERED_PARENT,
    CAUSE_OUTSIDE_DOMAIN,
    CAUSE_INELIGIBLE_REGION,
)

# Okabe-Ito subset (orange dropped: reserved for ROI boundaries).
CAUSE_PALETTE: Dict[str, str] = {
    CAUSE_FILTER_DROPPED:    "#56B4E9",  # sky blue
    CAUSE_SCATTERED_PARENT:  "#009E73",  # bluish green
    CAUSE_OUTSIDE_DOMAIN:    "#D55E00",  # vermillion
    CAUSE_INELIGIBLE_REGION: "#CC79A7",  # reddish purple
}

CAUSE_LABEL: Dict[str, str] = {
    CAUSE_FILTER_DROPPED:    "Filter-dropped small component (A/B)",
    CAUSE_SCATTERED_PARENT:  "Scattered parent ROI (C)",
    CAUSE_OUTSIDE_DOMAIN:    "Retained component, outside its domain (D)",
    CAUSE_INELIGIBLE_REGION: "Tessellation-ineligible region (F)",
}

# Ring styling: density + statistic panels share the same charcoal ring;
# statistic panels additionally paint a thin light halo below it.
RING_EDGECOLOR = "#333333"
RING_HALO_COLOR = "#FFFFFF"
RING_HALO_ALPHA = 0.85
RING_LINEWIDTH = 0.55
RING_HALO_LINEWIDTH = 1.4

# Radius used by the canonical observed-support footprint. Overlapping
# footprints merge so a connected spot group receives one outer contour.
SPOT_SUPPORT_RADIUS_FACTOR = 0.60

# Constructed-but-undefined polygon fill (statistic panels + zero-obs tiles).
UNDEFINED_FILL = "#BBBBBB"
UNDEFINED_EDGE = "#666666"
UNDEFINED_EDGE_LINEWIDTH = 0.35
UNDEFINED_LEGEND_LABEL = "Constructed support; statistic undefined"


class LedgerClassificationError(ValueError):
    """Raised when the spot_ledger / source_expression cannot be reconciled
    or a truth-table row does not match any of the six declared patterns.
    """


@dataclass(frozen=True)
class SupportOverlay:
    """Classified spot bundle for one dataset."""

    unconstructible: pd.DataFrame        # x, y, cause, gene1, gene2
    constructible: pd.DataFrame          # x, y, gene1, gene2
    counts: Dict[str, int]               # per-public-cause + "constructible"


def classify(
    spot_ledger_df: pd.DataFrame,
    source_expression_df: pd.DataFrame,
    *,
    x_col: str = COORD_X,
    y_col: str = COORD_Y,
    gene1_col: str = GENE1,
    gene2_col: str = GENE2,
    spot_index_col: str = "spot_index",
    source_row_id_col: str = SOURCE_ROW_ID,
) -> SupportOverlay:
    """Classify every spot or cell by ledger truth-table pattern.

    Requires the ledger's `spot_index` set to equal the source frame's
    `source_row_id` set exactly. Any mismatch, duplicate, or truth-table
    row that does not match one of the six declared patterns raises
    :class:`LedgerClassificationError`.
    """
    if spot_ledger_df is None or source_expression_df is None:
        raise LedgerClassificationError(
            "classify: spot_ledger_df and source_expression_df are both required"
        )

    _assert_id_sets_equal_and_unique(
        spot_ledger_df, source_expression_df,
        spot_index_col=spot_index_col,
        source_row_id_col=source_row_id_col,
    )

    left = spot_ledger_df[[
        spot_index_col, "component_status", "exclusion_reason",
        "parent_scattered", "component_id",
        "domain_region_id", "tessellation_eligible",
    ]].copy()
    right = source_expression_df[[
        source_row_id_col, x_col, y_col, gene1_col, gene2_col,
    ]].rename(columns={source_row_id_col: spot_index_col})
    merged = left.merge(
        right, on=spot_index_col, how="outer", validate="one_to_one",
    )

    causes = [_row_to_pattern(row) for row in merged.itertuples(index=False)]
    merged["_pattern"] = causes

    # Collapse A + B truth-table patterns into a single public cause.
    def _to_public(pattern: str) -> str:
        if pattern in (CAUSE_FILTER_DROPPED_SPLIT, CAUSE_FILTER_DROPPED_FILTER):
            return CAUSE_FILTER_DROPPED
        return pattern

    merged["_cause"] = merged["_pattern"].map(_to_public)

    constructible = merged.loc[
        merged["_cause"] == CONSTRUCTIBLE,
        [x_col, y_col, gene1_col, gene2_col],
    ].reset_index(drop=True)
    unconstructible = merged.loc[
        merged["_cause"] != CONSTRUCTIBLE,
        [x_col, y_col, gene1_col, gene2_col, "_cause"],
    ].rename(columns={"_cause": "cause"}).reset_index(drop=True)

    counts: Dict[str, int] = {c: int((merged["_cause"] == c).sum()) for c in CAUSE_ORDER}
    counts[CONSTRUCTIBLE] = int(len(constructible))
    return SupportOverlay(
        unconstructible=unconstructible,
        constructible=constructible,
        counts=counts,
    )


def _assert_id_sets_equal_and_unique(
    spot_ledger_df: pd.DataFrame,
    source_expression_df: pd.DataFrame,
    *,
    spot_index_col: str,
    source_row_id_col: str,
) -> None:
    """Both frames must key on exactly the same unique id set."""
    if spot_index_col not in spot_ledger_df.columns:
        raise LedgerClassificationError(
            f"spot_ledger_df missing required column {spot_index_col!r}"
        )
    if source_row_id_col not in source_expression_df.columns:
        raise LedgerClassificationError(
            f"source_expression_df missing required column {source_row_id_col!r}"
        )
    ledger_ids = spot_ledger_df[spot_index_col]
    src_ids = source_expression_df[source_row_id_col]
    if ledger_ids.duplicated().any():
        dup = sorted(ledger_ids[ledger_ids.duplicated()].tolist())
        raise LedgerClassificationError(
            f"spot_ledger_df: duplicate {spot_index_col} values: {dup!r}"
        )
    if src_ids.duplicated().any():
        dup = sorted(src_ids[src_ids.duplicated()].tolist())
        raise LedgerClassificationError(
            f"source_expression_df: duplicate {source_row_id_col} values: {dup!r}"
        )
    ledger_set = set(ledger_ids)
    src_set = set(src_ids)
    if ledger_set != src_set:
        missing_in_ledger = sorted(src_set - ledger_set)
        missing_in_source = sorted(ledger_set - src_set)
        raise LedgerClassificationError(
            "spot_index / source_row_id sets disagree — "
            f"in source only: {missing_in_ledger!r}; "
            f"in ledger only: {missing_in_source!r}"
        )


def _row_to_pattern(row) -> str:
    """Map one merged row to one of the six declared truth-table patterns.

    Any other combination raises. Patterns per
    docs/CSV_EXPORT_CONTRACT.md "spot ledger truth table":

    - A_split_dropped:  component_status = small_connected_component,
                        exclusion_reason = below_split_min_component_spots,
                        component_id null, region null, eligibility null.
    - B_filter_dropped: component_status = small_connected_component,
                        exclusion_reason = below_filter_min_component_spots,
                        component_id present, region null, eligibility null.
    - C_scattered_parent: component_status = retained_component,
                          parent_scattered True, region null, eligibility null.
    - D_retained_outside_domain: retained_component, not scattered,
                                 region null, eligibility null.
    - E_eligible_region: retained_component, not scattered, region present,
                         tessellation_eligible True.
    - F_ineligible_region: retained_component, not scattered, region present,
                           tessellation_eligible False.
    """
    cs = getattr(row, "component_status")
    er = getattr(row, "exclusion_reason", None)
    cid_null = _is_null(getattr(row, "component_id"))
    ps = bool(getattr(row, "parent_scattered"))
    rid_null = _is_null(getattr(row, "domain_region_id"))
    te = getattr(row, "tessellation_eligible")
    te_null = _is_null(te)

    if cs == "small_connected_component":
        if er == "below_split_min_component_spots" \
                and cid_null and rid_null and te_null:
            return CAUSE_FILTER_DROPPED_SPLIT
        if er == "below_filter_min_component_spots" \
                and not cid_null and rid_null and te_null:
            return CAUSE_FILTER_DROPPED_FILTER
        raise LedgerClassificationError(
            f"small_connected_component row does not match A or B pattern: "
            f"exclusion_reason={er!r}, component_id_null={cid_null}, "
            f"domain_region_id_null={rid_null}, tessellation_eligible_null={te_null}"
        )
    if cs != "retained_component":
        raise LedgerClassificationError(
            f"unknown component_status={cs!r} — expected "
            f"'retained_component' or 'small_connected_component'"
        )

    # retained_component: C / D / E / F
    if ps and rid_null and te_null:
        return CAUSE_SCATTERED_PARENT
    if not ps and rid_null and te_null:
        return CAUSE_OUTSIDE_DOMAIN
    if not ps and not rid_null and te_null is False and _is_true(te):
        return CONSTRUCTIBLE
    if not ps and not rid_null and te_null is False and _is_false(te):
        return CAUSE_INELIGIBLE_REGION
    raise LedgerClassificationError(
        f"retained_component row does not match C/D/E/F pattern: "
        f"parent_scattered={ps}, domain_region_id_null={rid_null}, "
        f"tessellation_eligible={te!r}, tessellation_eligible_null={te_null}"
    )


def _is_null(value) -> bool:
    return bool(pd.isna(value))


def _is_true(value) -> bool:
    return not pd.isna(value) and bool(value) is True


def _is_false(value) -> bool:
    return not pd.isna(value) and bool(value) is False


def paint_density_ring_overlay(
    ax,
    overlay: SupportOverlay,
    *,
    gene_col: str,
    cmap: str,
    median_nn: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm=None,
    radius_scale: float = 0.45,
    individual_outline: bool = True,
) -> None:
    """Paint observed-unconstructible spots with native-expression fill
    plus a charcoal ring on a density panel.
    """
    unc = overlay.unconstructible
    if unc.empty:
        return
    values = unc[gene_col].to_numpy(dtype=float)
    s = _marker_size(ax, median_nn * radius_scale)
    scatter_kw = dict(c=values, s=s, cmap=cmap, marker="o", rasterized=False)
    if individual_outline:
        scatter_kw.update(
            linewidths=RING_LINEWIDTH, edgecolors=RING_EDGECOLOR,
        )
    else:
        scatter_kw.update(linewidths=0, edgecolors="none")
    if norm is not None:
        scatter_kw['norm'] = norm
    else:
        scatter_kw['vmin'] = vmin
        scatter_kw['vmax'] = vmax
    ax.scatter(unc["x"].to_numpy(), unc["y"].to_numpy(), **scatter_kw)


def grouped_spot_support_parts(
    spots: pd.DataFrame,
    *,
    group_col: str,
    median_nn: float,
    radius_factor: float = SPOT_SUPPORT_RADIUS_FACTOR,
) -> Dict[str, Tuple[Polygon, ...]]:
    """Union spot/cell footprints per group and return exterior parts.

    Adjacent footprints merge; disconnected fragments remain separate.
    No convex hull or interpolation is introduced.
    """
    required = {"x", "y", group_col}
    missing = sorted(required - set(spots.columns))
    if missing:
        raise ValueError(
            f"grouped_spot_support_parts missing columns: {missing!r}"
        )
    radius = float(median_nn) * float(radius_factor)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"grouped_spot_support_parts requires a positive finite radius; "
            f"got {radius!r}"
        )
    grouped: Dict[str, Tuple[Polygon, ...]] = {}
    for group, sub in spots.groupby(group_col, sort=True):
        discs = [
            Point(float(row.x), float(row.y)).buffer(radius, quad_segs=24)
            for row in sub[["x", "y"]].itertuples(index=False)
        ]
        union = unary_union(discs)
        if isinstance(union, Polygon):
            parts: Iterable[Polygon] = (union,)
        elif isinstance(union, MultiPolygon):
            parts = (part for part in union.geoms if not part.is_empty)
        else:
            parts = ()
        grouped[str(group)] = tuple(parts)
    return grouped


def annotate_spots_with_parent_roi(
    spots: pd.DataFrame,
    spot_ledger_df: pd.DataFrame,
    source_expression_df: pd.DataFrame,
    *,
    output_col: str = "_parent_roi",
) -> pd.DataFrame:
    """Attach parent ROI identity to classified spots via source identity."""
    ledger_roi = spot_ledger_df.set_index("spot_index")["roi_id"].astype(str)
    source_xy = source_expression_df.set_index("source_row_id")[["x", "y"]]
    lookup = source_xy.join(ledger_roi, how="inner")
    xy_to_roi = {
        (float(row.x), float(row.y)): str(row.roi_id)
        for row in lookup.itertuples()
    }
    out = spots.copy()
    out[output_col] = [
        xy_to_roi.get((float(row.x), float(row.y)))
        for row in out.itertuples()
    ]
    return out.dropna(subset=[output_col])


def paint_grouped_spot_support_contours(
    ax,
    spots: pd.DataFrame,
    *,
    group_col: str,
    colors: Mapping[str, object],
    median_nn: float,
    linewidth: float,
    halo_color: Optional[str] = None,
    halo_linewidth: Optional[float] = None,
) -> None:
    """Stroke one exterior contour around each connected spot group."""
    grouped = grouped_spot_support_parts(
        spots, group_col=group_col, median_nn=median_nn,
    )
    for group, parts in grouped.items():
        color = colors.get(group)
        if color is None:
            continue
        for part in parts:
            path = Path(np.asarray(part.exterior.coords))
            if halo_color is not None and halo_linewidth is not None:
                ax.add_patch(PathPatch(
                    path, facecolor="none", edgecolor=halo_color,
                    linewidth=halo_linewidth,
                ))
            ax.add_patch(PathPatch(
                path, facecolor="none", edgecolor=color,
                linewidth=linewidth,
            ))


def paint_statistic_ring_overlay(
    ax,
    overlay: SupportOverlay,
    *,
    median_nn: float,
    radius_scale: float = 0.45,
) -> None:
    """Paint observed-unconstructible spots with a hollow charcoal
    ring (thin light halo underneath) on a statistic panel.
    """
    unc = overlay.unconstructible
    if unc.empty:
        return
    x = unc["x"].to_numpy()
    y = unc["y"].to_numpy()
    s = _marker_size(ax, median_nn * radius_scale)
    ax.scatter(
        x, y, s=s, facecolors="none",
        edgecolors=RING_HALO_COLOR, linewidths=RING_HALO_LINEWIDTH,
        alpha=RING_HALO_ALPHA, marker="o", rasterized=False,
    )
    ax.scatter(
        x, y, s=s, facecolors="none",
        edgecolors=RING_EDGECOLOR, linewidths=RING_LINEWIDTH,
        marker="o", rasterized=False,
    )


def _marker_size(ax, radius_world: float) -> float:
    """Convert a world-space radius to matplotlib scatter `s` (area in points^2)."""
    fig = ax.figure
    p = ax.transData.transform([[0.0, 0.0], [radius_world, 0.0]])
    dx_px = abs(p[1, 0] - p[0, 0])
    dx_pt = dx_px * 72.0 / fig.dpi
    return (dx_pt * 2.0) ** 2

def unavailable_legend_handle():
    """Shared legend entry: observed spot with aggregate statistic unavailable."""
    return Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="none", markeredgecolor=RING_EDGECOLOR,
        markeredgewidth=RING_LINEWIDTH, markersize=8,
        label="Observed spot; aggregate statistic unavailable",
    )


def constructed_undefined_legend_handle():
    """Shared legend entry: polygon constructed but statistic undefined at grain."""
    return Patch(
        facecolor=UNDEFINED_FILL, edgecolor=UNDEFINED_EDGE,
        linewidth=UNDEFINED_EDGE_LINEWIDTH,
        label=UNDEFINED_LEGEND_LABEL,
    )
