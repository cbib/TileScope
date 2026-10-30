"""Tile density visualization: per-replicate Voronoi tile fills + unconstructible-spot ring overlay."""
from tilescope.spatial.spacing import median_nn_distance_from_df
from tilescope.viz.painters import draw_polygon_boundaries


import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional
from shapely import wkt as _wkt

from tilescope.spatial import ComponentPolygonDict
from tilescope.cols import (
    GENE1, GENE1_DENSITY, GENE2, GENE2_DENSITY, TILE_ID, TILE_GEOMETRY_WKT,
    COMPONENT_ID, REPLICATE_INDEX,
)
from tilescope.defaults import EPSILON
from tilescope.viz.geometry_helpers import (
    summarize_geom, classify_drop, clean_if_needed,
    iter_polygons, patch_from_polygon,
)
from tilescope.viz.painters import add_identity_header
from tilescope.viz.support_overlay import (
    UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH, UNDEFINED_FILL,
    classify, constructed_undefined_legend_handle,
    paint_density_ring_overlay,
    unavailable_legend_handle,
)

log = logging.getLogger(__name__)


def plot_tile_densities(tessellation_tiles_df: pd.DataFrame,
                       replicate_index: int = 0,
                       cmap: str = 'viridis',
                       output_dir: str = 'output',
                       component_polygons: Optional[ComponentPolygonDict] = None,
                       canvas: Optional['Canvas'] = None,
                       precision: Optional[float] = None,
                       d1: Optional[float] = None,
                       spot_ledger_df: Optional[pd.DataFrame] = None,
                       source_expression_df: Optional[pd.DataFrame] = None,
                       source_median_nn: Optional[float] = None,
                       feature_a_col: str = GENE1,
                       feature_b_col: str = GENE2,
                       *,
                       feature_a_id: str,
                       feature_b_id: str) -> None:
    """Per-replicate Voronoi tile fills; overlay unconstructible spots with charcoal rings."""
    replicate_slice = tessellation_tiles_df[
        tessellation_tiles_df[REPLICATE_INDEX] == replicate_index
    ]
    if replicate_slice.empty:
        log.warning(
            "[VIZ] No tiles for replicate_index=%d", replicate_index,
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title("A expression")
    axes[1].set_title("B expression")

    # Global colour normalisation across the whole replicate.
    norm1 = mcolors.Normalize(
        vmin=replicate_slice[GENE1_DENSITY].min(),
        vmax=replicate_slice[GENE1_DENSITY].max(),
    )
    norm2 = mcolors.Normalize(
        vmin=replicate_slice[GENE2_DENSITY].min(),
        vmax=replicate_slice[GENE2_DENSITY].max(),
    )
    scalarMap1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap)
    scalarMap2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap)

    # min_area = 1e-4 * d1^2 (tiny fraction of typical tile area).
    if d1 is not None:
        min_area = 1e-4 * d1 * d1
    else:
        min_area = EPSILON

    log.info(
        "\n[VIZ] Drawing tiles for replicate %s (precision=%s, min_area=%.3g)",
        replicate_index, precision if precision else "None", min_area,
    )

    drop = {"drawn": 0, "invalid": 0, "empty": 0, "too_small": 0, "fixed": 0, "undefined": 0}
    total_tiles = int(len(replicate_slice))

    for component_id, comp_slice in replicate_slice.groupby(COMPONENT_ID, sort=False):
        log.debug(
            "[VIZ] component %s: rendering %d tiles", component_id, len(comp_slice),
        )
        for row in comp_slice.itertuples(index=False):
            tile_id = int(getattr(row, TILE_ID))
            g1_dens = getattr(row, GENE1_DENSITY)
            g2_dens = getattr(row, GENE2_DENSITY)
            poly = _wkt.loads(getattr(row, TILE_GEOMETRY_WKT))
            reason = classify_drop(poly, min_area)
            if reason != "ok":
                if reason == "invalid":
                    poly, fixed, fix_reason = clean_if_needed(
                        poly, precision=precision, simplify_tol=0.0,
                    )
                    reason2 = classify_drop(poly, min_area)
                    if fixed:
                        drop["fixed"] += 1
                    if reason2 != "ok":
                        drop[reason2] += 1
                        log.debug(
                            "[TILE:SKIP] %s -> %s (after clean: %s)   %s",
                            tile_id, reason, reason2, summarize_geom(poly),
                        )
                        continue
                else:
                    drop[reason] += 1
                    log.debug(
                        "[TILE:SKIP] %s -> %s   %s",
                        tile_id, reason, summarize_geom(poly),
                    )
                    continue

            # Four-state contract: constructed zero-observation tiles
            # render as neutral gray with dark edge, not skipped.
            undefined = bool(pd.isna(g1_dens) or pd.isna(g2_dens))
            if undefined:
                face1 = face2 = UNDEFINED_FILL
                edgecolor = UNDEFINED_EDGE
                linewidth = UNDEFINED_EDGE_LINEWIDTH
                drop["undefined"] += 1
            else:
                face1 = scalarMap1.to_rgba(g1_dens)
                face2 = scalarMap2.to_rgba(g2_dens)
                edgecolor = "black"
                linewidth = 0.3
            for part in iter_polygons(poly):
                p1 = patch_from_polygon(part, facecolor=face1, edgecolor=edgecolor, linewidth=linewidth)
                p2 = patch_from_polygon(part, facecolor=face2, edgecolor=edgecolor, linewidth=linewidth)
                axes[0].add_patch(p1)
                axes[1].add_patch(p2)
            if not undefined:
                drop["drawn"] += 1

    log.info("    Tiles: drawn=%d/%d | undefined=%d | invalid=%d empty=%d too_small=%d fixed=%d",
             drop["drawn"], total_tiles, drop["undefined"], drop["invalid"], drop["empty"],
             drop["too_small"], drop["fixed"])

    # Autoscale BEFORE the ring overlay: `paint_density_ring_overlay` sizes
    # its markers via `ax.transData`, which returns nonsense on the default
    # 0..1 axes and produces sample-covering blobs.
    for ax in axes:
        ax.autoscale_view()
        ax.set_aspect('equal')

    legend_handles = []
    if spot_ledger_df is not None and source_expression_df is not None:
        overlay = classify(
            spot_ledger_df, source_expression_df,
            gene1_col=feature_a_col, gene2_col=feature_b_col,
        )
        if source_median_nn is None:
            source_median_nn = median_nn_distance_from_df(source_expression_df, "x", "y")
        median_nn = source_median_nn
        if median_nn is not None and not overlay.unconstructible.empty:
            paint_density_ring_overlay(
                axes[0], overlay, gene_col=feature_a_col,
                vmin=float(replicate_slice[GENE1_DENSITY].min()),
                vmax=float(replicate_slice[GENE1_DENSITY].max()),
                cmap=cmap, median_nn=median_nn,
            )
            paint_density_ring_overlay(
                axes[1], overlay, gene_col=feature_b_col,
                vmin=float(replicate_slice[GENE2_DENSITY].min()),
                vmax=float(replicate_slice[GENE2_DENSITY].max()),
                cmap=cmap, median_nn=median_nn,
            )
            legend_handles.append(unavailable_legend_handle())
    if drop["undefined"] > 0:
        legend_handles.append(constructed_undefined_legend_handle())
    if legend_handles:
        for ax in axes:
            ax.legend(handles=legend_handles, loc="lower right",
                      fontsize=8, framealpha=0.9)

    if component_polygons:
        for ax in axes:
            draw_polygon_boundaries(ax, component_polygons)

    for ax in axes:
        ax.axis('off')

    fig.colorbar(scalarMap1, ax=axes[0], orientation='vertical',
                 label='A mean expression', shrink=0.8)
    fig.colorbar(scalarMap2, ax=axes[1], orientation='vertical',
                 label='B mean expression', shrink=0.8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    add_identity_header(
        fig, feature_a_id, feature_b_id,
        f"Tessellation replicate {replicate_index + 1}",
    )
    os.makedirs(output_dir, exist_ok=True)

    plot_name = f"tessellation_densities_{replicate_index + 1}.pdf"
    output_path = f"{output_dir}/{plot_name}"
    plt.savefig(output_path)
    log.info("[VIZ] Saved: %s", output_path)
    plt.close()
