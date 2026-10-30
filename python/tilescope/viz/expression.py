"""Source-spot ROI-analysis view: A/B expression + per-ROI correlation.

Layout is 1x3 when no ROI is canonically scattered and 2x3 otherwise; the
bottom row shows scattered-ROI (spatial_support_class == "scattered") content.
Bottom A/B panels share the top-row cmap/norm; Pearson panels use fixed [-1, +1].
"""
from matplotlib.colors import to_rgba


import logging
from typing import Dict, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tilescope.cols import GENE1, GENE2, ROI_ID
from tilescope.schema.helpers import extract_coordinates
from tilescope.spatial import ComponentPolygonDict
from tilescope.spatial.spacing import median_nn_distance
from tilescope.viz.geometry_helpers import (
    dissolved_roi_polygons, paint_roi_value,
)
from tilescope.viz.painters import (
    PEARSON_CMAP, PEARSON_NORM, SpotSplatPainter, add_identity_header,
    draw_polygon_boundaries, pearson_scalar_mappable,
)
from tilescope.viz.color_norms import resolve_expression_norm
from tilescope.viz.support_overlay import (
    SupportOverlay, UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH, UNDEFINED_FILL,
    classify, constructed_undefined_legend_handle, paint_density_ring_overlay,
    paint_grouped_spot_support_contours,
)


_ROI_REFERENCE_COLOR = "#B0B0B0"
_ROI_REFERENCE_LINEWIDTH = 0.6
_BOTTOM_FILL_SIZE = 60
_BOTTOM_GROUP_LINEWIDTH = 1.2
_IDENTITY_CONTOUR_HALO_COLOR = "#333333"
_IDENTITY_CONTOUR_HALO_LINEWIDTH = 2.2
# High-chroma fluorescents readable over RdBu-diverging expression fields
# and white backgrounds; cycles when > 6 scattered ROIs.
_SCATTERED_IDENTITY_PALETTE: tuple = (
    "#39FF14", "#B026FF", "#FFF200", "#00FF9C", "#FF4FD8", "#B6FF00",
)
_SUPPORT_CLASS_COL = "spatial_support_class"


def plot_expression_summary(source_expression_df: pd.DataFrame,
                            correlation_map: Dict,
                            component_to_roi: Dict[str, str],
                            roi_support_status_df: Optional[pd.DataFrame] = None,
                            output_dir: str = 'output',
                            cmap1: str = 'viridis',
                            cmap2: str = 'viridis',
                            component_polygons: Optional[ComponentPolygonDict] = None,
                            canvas: Optional['Canvas'] = None,
                            spot_ledger_df: Optional[pd.DataFrame] = None,
                            source_median_nn: Optional[float] = None,
                            feature_a_col: str = GENE1,
                            feature_b_col: str = GENE2,
                            *,
                            feature_a_id: str,
                            feature_b_id: str) -> None:
    """Source-spot ROI-analysis conditional 1x3 / 2x3 figure."""
    if source_expression_df is None or source_expression_df.empty:
        raise ValueError(
            "plot_expression_summary requires a non-empty source_expression_df"
        )
    logger = logging.getLogger(__name__)

    src_coords = extract_coordinates(source_expression_df)
    src_g1 = source_expression_df[feature_a_col].to_numpy(dtype=float)
    src_g2 = source_expression_df[feature_b_col].to_numpy(dtype=float)
    # Diverging cmap when data crosses zero, viridis for strictly positive;
    # 1st-99th percentile bounds resist outlier stretching.
    cmap1, norm_a = resolve_expression_norm(src_g1, cmap1)
    cmap2, norm_b = resolve_expression_norm(src_g2, cmap2)

    # Resolve shared inputs before creating the figure so the layout can be
    # sized from has_scattered.
    roi_polygons = dissolved_roi_polygons(component_polygons, component_to_roi)
    constructible_roi_ids = set(roi_polygons.keys())
    scattered_roi_ids = _resolve_scattered_roi_ids(
        roi_support_status_df=roi_support_status_df,
        correlation_map=correlation_map,
        constructible_roi_ids=constructible_roi_ids,
    )
    has_scattered = bool(scattered_roi_ids)
    roi_identity_colors = _roi_identity_color_map(scattered_roi_ids)
    scattered_ids_set = frozenset(str(r) for r in scattered_roi_ids)
    scattered_source_subframe = source_expression_df.loc[
        source_expression_df[ROI_ID].astype(str).isin(scattered_ids_set)
    ]
    overlay = _classify_or_none(
        spot_ledger_df, source_expression_df,
        feature_a_col=feature_a_col, feature_b_col=feature_b_col,
    )

    nrows = 2 if has_scattered else 1
    fig_height = 24 if has_scattered else 12
    fig, axes = plt.subplots(
        nrows, 3, figsize=(36, fig_height),
        constrained_layout=True, squeeze=False,
    )
    painter = SpotSplatPainter()
    _apply_bounds(axes.flatten(), src_coords, component_polygons)

    global_median_nn = source_median_nn
    if global_median_nn is None and len(src_coords) >= 2:
        global_median_nn = median_nn_distance(src_coords)
    if global_median_nn is not None:
        logger.info(f"[EXPR] Source-frame median_nn={global_median_nn:.3f}")

    painter.draw(axes[0, 0], src_coords, src_g1, component_polygon=None,
                 cmap=cmap1, norm=norm_a, canvas=canvas,
                 median_nn_override=global_median_nn)
    painter.draw(axes[0, 1], src_coords, src_g2, component_polygon=None,
                 cmap=cmap2, norm=norm_b, canvas=canvas,
                 median_nn_override=global_median_nn)
    if overlay is not None and global_median_nn is not None:
        paint_density_ring_overlay(
            axes[0, 0], overlay, gene_col=feature_a_col,
            cmap=cmap1, norm=norm_a, median_nn=global_median_nn,
        )
        paint_density_ring_overlay(
            axes[0, 1], overlay, gene_col=feature_b_col,
            cmap=cmap2, norm=norm_b, median_nn=global_median_nn,
        )
    # Constructed-support boundaries drawn last so they sit above fills.
    draw_polygon_boundaries(axes[0, 0], component_polygons)
    draw_polygon_boundaries(axes[0, 1], component_polygons)
    axes[0, 0].set_title("A expression - all spots or cells"); axes[0, 0].axis('off')
    axes[0, 1].set_title("B expression - all spots or cells"); axes[0, 1].axis('off')

    pearson_sm = pearson_scalar_mappable()
    any_undef_constructed = _paint_constructed_correlation_panel(
        axes[0, 2], roi_polygons, correlation_map, pearson_sm,
    )
    axes[0, 2].set_title("Correlation - spatial ROIs")
    axes[0, 2].set_aspect('equal', adjustable='box'); axes[0, 2].axis('off')

    any_undef_scattered = False
    if has_scattered:
        scattered_coords = extract_coordinates(scattered_source_subframe)
        scattered_roi_labels = scattered_source_subframe[ROI_ID].astype(str).to_numpy()
        _paint_scattered_value_panel(
            axes[1, 0], scattered_coords,
            values=scattered_source_subframe[feature_a_col].to_numpy(dtype=float),
            cmap=cmap1, norm=norm_a,
            roi_ids=scattered_roi_labels, roi_identity_colors=roi_identity_colors,
            roi_polygons=roi_polygons, median_nn=global_median_nn,
        )
        _paint_scattered_value_panel(
            axes[1, 1], scattered_coords,
            values=scattered_source_subframe[feature_b_col].to_numpy(dtype=float),
            cmap=cmap2, norm=norm_b,
            roi_ids=scattered_roi_labels, roi_identity_colors=roi_identity_colors,
            roi_polygons=roi_polygons, median_nn=global_median_nn,
        )
        per_spot_pearson = _per_spot_pearson(
            scattered_source_subframe, correlation_map,
        )
        any_undef_scattered = _paint_scattered_value_panel(
            axes[1, 2], scattered_coords, values=per_spot_pearson,
            cmap=PEARSON_CMAP, norm=PEARSON_NORM,
            roi_ids=scattered_roi_labels, roi_identity_colors=roi_identity_colors,
            roi_polygons=roi_polygons, median_nn=global_median_nn,
        )
        axes[1, 0].set_title("A expression - scattered ROIs")
        axes[1, 1].set_title("B expression - scattered ROIs")
        axes[1, 2].set_title("Correlation - scattered ROIs")
        for ax in (axes[1, 0], axes[1, 1], axes[1, 2]):
            ax.axis('off')

    # Each colorbar spans a column so top and bottom share the same scale.
    cbar_a = list(axes[:, 0])
    cbar_b = list(axes[:, 1])
    cbar_p = list(axes[:, 2])
    fig.colorbar(
        _scalar_mappable(cmap1, norm_a), ax=cbar_a,
        orientation='vertical', label='A mean expression', shrink=0.7,
    )
    fig.colorbar(
        _scalar_mappable(cmap2, norm_b), ax=cbar_b,
        orientation='vertical', label='B mean expression', shrink=0.7,
    )
    fig.colorbar(
        pearson_sm, ax=cbar_p, orientation='vertical', shrink=0.7,
        label='Pearson r (ROI, spots or cells)',
    )

    if any_undef_constructed or any_undef_scattered:
        axes[0, 2].legend(
            handles=[constructed_undefined_legend_handle()],
            loc='lower right', fontsize=9, framealpha=0.9,
        )
    if has_scattered:
        fig.legend(
            handles=_scattered_categorical_legend_handles(
                scattered_roi_ids, roi_identity_colors,
            ),
            loc='outside lower center',
            ncol=min(len(scattered_roi_ids), 10),
            fontsize=9, framealpha=0.9,
            title="Scattered ROI identity (outline colour)",
        )

    add_identity_header(fig, feature_a_id, feature_b_id, "Source expression")
    output_path = f"{output_dir}/expression_summary.pdf"
    plt.savefig(output_path)
    logger.info(f"[VIZ] Saved: {output_path}")
    plt.close(fig)


def _apply_bounds(axes, src_coords, component_polygons):
    x_mins, y_mins = [float(np.min(src_coords[:, 0]))], [float(np.min(src_coords[:, 1]))]
    x_maxs, y_maxs = [float(np.max(src_coords[:, 0]))], [float(np.max(src_coords[:, 1]))]
    if component_polygons:
        for poly in component_polygons.values():
            b = poly.bounds
            x_mins.append(b[0]); y_mins.append(b[1])
            x_maxs.append(b[2]); y_maxs.append(b[3])
    x_min, y_min = min(x_mins), min(y_mins)
    x_max, y_max = max(x_maxs), max(y_maxs)
    _range = max(x_max - x_min, y_max - y_min)
    margin = 0.05 * _range
    xlim = (x_min - margin, x_max + margin)
    ylim = (y_min - margin, y_max + margin)
    for ax in axes:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')


def _classify_or_none(
    spot_ledger_df, source_expression_df, *, feature_a_col, feature_b_col,
) -> Optional[SupportOverlay]:
    if spot_ledger_df is None:
        return None
    return classify(
        spot_ledger_df, source_expression_df,
        gene1_col=feature_a_col, gene2_col=feature_b_col,
    )


def _scalar_mappable(cmap: str, norm):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm




def _parent_corr_or_nan(correlation_map: Dict, parent_roi_id) -> float:
    if parent_roi_id is None or parent_roi_id not in correlation_map:
        return float("nan")
    try:
        return float(correlation_map[parent_roi_id])
    except (TypeError, ValueError):
        return float("nan")


def _scattered_roi_ids_from_status(
    roi_support_status_df: pd.DataFrame,
) -> list:
    """Return ROI ids with spatial_support_class == "scattered". Raises when
    the column is absent; never consults correlation_map or the spot ledger."""
    if _SUPPORT_CLASS_COL not in roi_support_status_df.columns:
        raise ValueError(
            f"roi_support_status_df missing column {_SUPPORT_CLASS_COL!r}; "
            "expression viz cannot partition ROIs without the canonical class."
        )
    scattered = roi_support_status_df.loc[
        roi_support_status_df[_SUPPORT_CLASS_COL].astype(str) == "scattered",
        ROI_ID,
    ].astype(str)
    return sorted(set(scattered), key=str)


def _resolve_scattered_roi_ids(
    *, roi_support_status_df: Optional[pd.DataFrame],
    correlation_map: Dict, constructible_roi_ids: set,
) -> list:
    """Read scattered ROI ids from roi_support_status_df. Fallback
    (correlation_map minus constructible) fires ONLY when the frame is None;
    an empty frame means "no scattered ROIs" and unaccounted entries raise."""
    if roi_support_status_df is None:
        return sorted(
            {str(r) for r in correlation_map} - constructible_roi_ids,
            key=str,
        )
    scattered = set(_scattered_roi_ids_from_status(roi_support_status_df))
    unaccounted = (
        {str(r) for r in correlation_map} - constructible_roi_ids - scattered
    )
    if unaccounted:
        raise ValueError(
            f"correlation_map contains ROI(s) {sorted(unaccounted)!r} that "
            "are neither in component_polygons nor marked scattered in "
            "roi_support_status_df; missing geometry must not silently "
            "reclassify a ROI as scattered."
        )
    return sorted(scattered, key=str)


def _paint_constructed_correlation_panel(
    ax, roi_polygons: Dict, correlation_map: Dict, pearson_sm,
) -> bool:
    """Paint each constructible ROI on its dissolved polygon with its Pearson r.
    Returns True when any ROI rendered as undefined-gray."""
    any_undefined = False
    for roi_id, poly in roi_polygons.items():
        r = _parent_corr_or_nan(correlation_map, roi_id)
        is_undef = paint_roi_value(
            ax, poly, None if not np.isfinite(r) else r,
            scalar_mappable=pearson_sm,
            undefined_fill=UNDEFINED_FILL,
            undefined_edge=UNDEFINED_EDGE,
            undefined_edge_linewidth=UNDEFINED_EDGE_LINEWIDTH,
        )
        any_undefined = any_undefined or is_undef
    _label_dissolved_rois(ax, roi_polygons, correlation_map)
    return any_undefined


def _label_dissolved_rois(ax, roi_polygons, correlation_map):
    for roi_id, poly in roi_polygons.items():
        r = _parent_corr_or_nan(correlation_map, roi_id)
        r_str = "n/a" if not np.isfinite(r) else f"{r:.2f}"
        rep = poly.representative_point()
        ax.text(
            float(rep.x), float(rep.y), f"ROI {roi_id}, r={r_str}",
            ha='center', va='center', fontsize=9, color='white',
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="black", alpha=0.5, edgecolor="none"),
        )


def _roi_identity_color_map(scattered_roi_ids) -> Dict[str, tuple]:
    """Categorical colour per scattered ROI id, shared across the three
    bottom panels. Assigned by sorted ROI id so mapping is stable across renders."""
    if not scattered_roi_ids:
        return {}
    sorted_ids = sorted({str(r) for r in scattered_roi_ids})
    return {
        rid: to_rgba(_SCATTERED_IDENTITY_PALETTE[
            idx % len(_SCATTERED_IDENTITY_PALETTE)
        ])
        for idx, rid in enumerate(sorted_ids)
    }


def _per_spot_pearson(
    scattered_subframe: pd.DataFrame, correlation_map: Dict,
) -> np.ndarray:
    if scattered_subframe.empty:
        return np.array([])
    return np.array([
        _parent_corr_or_nan(correlation_map, str(r))
        for r in scattered_subframe[ROI_ID].astype(str).to_numpy()
    ], dtype=float)


def _paint_scattered_value_panel(
    ax, coords: np.ndarray, *,
    values: np.ndarray, cmap, norm,
    roi_ids: np.ndarray, roi_identity_colors: Dict[str, tuple],
    roi_polygons: Dict, median_nn: Optional[float],
) -> bool:
    """Scattered-ROI value fill plus grouped identity contours.
    Returns True when any value is not finite."""
    draw_polygon_boundaries(
        ax, roi_polygons,
        color=_ROI_REFERENCE_COLOR, linewidth=_ROI_REFERENCE_LINEWIDTH,
        include_interiors=False,
    )
    if coords.shape[0] == 0:
        return False
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    any_undefined = bool((~finite_mask).any())
    if finite_mask.any():
        ax.scatter(
            coords[finite_mask, 0], coords[finite_mask, 1],
            c=values[finite_mask], cmap=cmap, norm=norm,
            edgecolors='none', s=_BOTTOM_FILL_SIZE, marker='o',
        )
    if any_undefined:
        ax.scatter(
            coords[~finite_mask, 0], coords[~finite_mask, 1],
            facecolors=UNDEFINED_FILL, edgecolors='none',
            s=_BOTTOM_FILL_SIZE, marker='o',
        )
    if median_nn is not None:
        support = pd.DataFrame({
            "x": coords[:, 0], "y": coords[:, 1], "_roi": roi_ids,
        })
        paint_grouped_spot_support_contours(
            ax, support, group_col="_roi", colors=roi_identity_colors,
            median_nn=median_nn, linewidth=_BOTTOM_GROUP_LINEWIDTH,
            halo_color=_IDENTITY_CONTOUR_HALO_COLOR,
            halo_linewidth=_IDENTITY_CONTOUR_HALO_LINEWIDTH,
        )
    return any_undefined


def _scattered_categorical_legend_handles(scattered_roi_ids, roi_identity_colors):
    return [
        mlines.Line2D(
            [], [], marker='o', linestyle='none',
            markerfacecolor='white',
            markeredgecolor=roi_identity_colors[str(roi_id)],
            markeredgewidth=1.6, markersize=10,
            label=f"ROI {roi_id}",
        )
        for roi_id in scattered_roi_ids
    ]
