"""Consensus grid visualization on canonical consensus surfaces."""
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tilescope.spatial.spacing import median_nn_distance_from_df


import logging
import os
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt as _wkt

from tilescope.cols import (
    COMPONENT_ID, CORRELATION_VALUE, GENE1, GENE1_DENSITY_ENSEMBLE, GENE1_DENSITY_STABILITY,
    GENE2,
    GENE2_DENSITY_ENSEMBLE, GENE2_DENSITY_STABILITY,
    HEX_G1_ENRICHMENT, HEX_G1_ENRICHMENT_STATUS,
    HEX_G2_ENRICHMENT, HEX_G2_ENRICHMENT_STATUS,
    HEX_GEOMETRY_WKT,
    MORAN_DIRECTION, MORAN_VALUE, PARENT_CONSENSUS_CORRELATION_STATUS,
    PARENT_MORAN_STATUS, ROI_ID, SAMPLE_ID, VALUE_SURFACE,
)

from tilescope.metrics.enrichment import EnrichmentRatioStatus
from tilescope.spatial import ComponentPolygonDict
from tilescope.viz.geometry_helpers import (
    dissolved_roi_polygons, iter_polygons, paint_roi_value, patch_from_polygon,
)
from tilescope.viz.painters import (
    add_identity_header, draw_polygon_boundaries, pearson_scalar_mappable,
)
from tilescope.viz.color_norms import resolve_expression_norm
from tilescope.viz.support_overlay import (
    UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH, UNDEFINED_FILL,
    annotate_spots_with_parent_roi, classify,
    constructed_undefined_legend_handle,
    paint_grouped_spot_support_contours,
    paint_density_ring_overlay, unavailable_legend_handle,
)

log = logging.getLogger(__name__)

_CONSENSUS_ENSEMBLE_SURFACE = "consensus_ensemble"
_MORAN_G1_LAG_G2_KEY = "g1_lag_g2"
_MORAN_G2_LAG_G1_KEY = "g2_lag_g1"
_ENSEMBLE_SURFACE = "consensus_ensemble"
_FINITE_STATUS = "finite"

# Display constants locked to keep PDFs comparable across runs.

# log2FC = 3 = 8x enrichment or 1/8x depletion; exact-zero FCs saturate at -3.0.
_SATURATION_LOG2 = 3.0

# Display dead-zone (FC in ~[0.84, 1.19]); NOT a statistical significance threshold.
_NEUTRAL_LOG2_BAND = 0.25

# Four-state categorical palette (Okabe-Ito, colorblind-safe). _COLOR_BOTH_UP
# tuned so endpoint contrast matches _COLOR_BOTH_DOWN (~4.45 each).
_COLOR_BOTH_UP = np.array([193, 65, 24]) / 255.0       # red (#C14118)
_COLOR_BOTH_DOWN = np.array([0, 114, 178]) / 255.0     # blue
_COLOR_G1UP_G2DOWN = np.array([0, 158, 115]) / 255.0   # green
_COLOR_G1DOWN_G2UP = np.array([204, 121, 167]) / 255.0 # purple
_COLOR_NEUTRAL = np.array([0.93, 0.93, 0.93])          # very light gray


def _transform_log2_fc(fc: float) -> float:
    """Clip log2(fc) to +/- _SATURATION_LOG2; exact-zero FC saturates at -sat.

    Raises ValueError on negative FC (invalid for a ratio-of-proportions statistic).
    """
    if not np.isfinite(fc):
        return float("nan")
    if fc == 0.0:
        return -_SATURATION_LOG2
    if fc < 0.0:
        raise ValueError(
            f"_transform_log2_fc: fold-change must be non-negative "
            f"(got {fc!r}); a ratio of non-negative proportions cannot "
            "yield a negative value."
        )
    return float(np.clip(np.log2(fc), -_SATURATION_LOG2, _SATURATION_LOG2))


def _classify_marginal_pattern(g1_fc: float, g2_fc: float):
    """Return (rgba, is_undefined) for the combined-pattern panel.

    Categorical hue by quadrant; lightness modulated by min(|e1|, |e2|).
    """
    if not (np.isfinite(g1_fc) and np.isfinite(g2_fc)):
        return None, True

    e1 = _transform_log2_fc(g1_fc)
    e2 = _transform_log2_fc(g2_fc)

    if abs(e1) < _NEUTRAL_LOG2_BAND or abs(e2) < _NEUTRAL_LOG2_BAND:
        return (*_COLOR_NEUTRAL, 1.0), False

    if e1 > 0 and e2 > 0:
        base = _COLOR_BOTH_UP
    elif e1 < 0 and e2 < 0:
        base = _COLOR_BOTH_DOWN
    elif e1 > 0 and e2 < 0:
        base = _COLOR_G1UP_G2DOWN
    else:
        base = _COLOR_G1DOWN_G2UP

    # Concave remap so mid-range magnitudes reach readable saturation
    # instead of clustering near 0.4 under a linear ramp.
    m = min(abs(e1), abs(e2))
    x = float(np.clip(
        (m - _NEUTRAL_LOG2_BAND)
        / (_SATURATION_LOG2 - _NEUTRAL_LOG2_BAND),
        0.0, 1.0,
    ))
    t = x ** 0.6
    rgb = base * t + _COLOR_NEUTRAL * (1.0 - t)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0), False


def _create_consensus_layout(show_stability: bool, show_bivariate_morans: bool):
    n_cols = 3
    if show_stability:
        n_cols += 1
    if show_bivariate_morans:
        n_cols += 1

    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 16))

    axes[0, 0].set_title("A expression")
    axes[0, 1].set_title("B expression")
    axes[0, 2].set_title("Correlation")
    axes[1, 0].set_title("A enrichment")
    axes[1, 1].set_title("B enrichment")
    axes[1, 2].set_title("Enrichment pattern")

    if show_stability:
        axes[0, 3].set_title("A stability")
        axes[1, 3].set_title("B stability")

    if show_bivariate_morans:
        axes[0, n_cols - 1].set_title("Spatial association A ← B")
        axes[1, n_cols - 1].set_title("Spatial association B ← A")

    return fig, axes


def _finite(values):
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _create_scalar_mappables(values, show_stability, show_bivariate_morans):
    """Build the scalar mappables for expression, enrichment, Pearson and Moran panels."""
    cmap_g1, norm_g1 = resolve_expression_norm(values['g1_means'], 'viridis')
    cmap_g2, norm_g2 = resolve_expression_norm(values['g2_means'], 'viridis')
    norm_enrich_log2 = mcolors.TwoSlopeNorm(
        vmin=-_SATURATION_LOG2, vcenter=0.0, vmax=_SATURATION_LOG2,
    )
    mappables = {
        'g1': plt.cm.ScalarMappable(norm=norm_g1, cmap=cmap_g1),
        'g2': plt.cm.ScalarMappable(norm=norm_g2, cmap=cmap_g2),
        'g1_enrich': plt.cm.ScalarMappable(norm=norm_enrich_log2, cmap='RdBu_r'),
        'g2_enrich': plt.cm.ScalarMappable(norm=norm_enrich_log2, cmap='RdBu_r'),
        'roi_pearson': pearson_scalar_mappable(),
        'g1_cv': None, 'g2_cv': None,
        'i_g1_lag_g2': None, 'i_g2_lag_g1': None,
    }

    if show_stability:
        finite_g1_cvs = _finite(values['g1_cvs'])
        finite_g2_cvs = _finite(values['g2_cvs'])
        max_g1_cv = np.percentile(finite_g1_cvs, 95) if finite_g1_cvs.size else 0.5
        max_g2_cv = np.percentile(finite_g2_cvs, 95) if finite_g2_cvs.size else 0.5
        mappables['g1_cv'] = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=0.0, vmax=max(0.5, max_g1_cv)), cmap='YlOrRd',
        )
        mappables['g2_cv'] = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=0.0, vmax=max(0.5, max_g2_cv)), cmap='YlOrRd',
        )

    if show_bivariate_morans:
        # Fixed [-1, +1] range across every PDF so cross-PDF colour
        # comparisons stay consistent; Moran I is bounded near +/-1.
        norm_morans = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        mappables['i_g1_lag_g2'] = plt.cm.ScalarMappable(norm=norm_morans, cmap='RdBu_r')
        mappables['i_g2_lag_g1'] = plt.cm.ScalarMappable(norm=norm_morans, cmap='RdBu_r')

    return mappables


_ALLOWED_MORAN_DIRECTIONS = frozenset({_MORAN_G1_LAG_G2_KEY, _MORAN_G2_LAG_G1_KEY})


def _index_moran_by_roi(
    roi_consensus_moran_df: Optional[pd.DataFrame],
) -> dict:
    """Return {roi_id: {'g1_lag_g2': v|None, 'g2_lag_g1': v|None}} on ensemble."""
    if roi_consensus_moran_df is None or roi_consensus_moran_df.empty:
        return {}
    for required in (
        ROI_ID, MORAN_DIRECTION, MORAN_VALUE, PARENT_MORAN_STATUS,
        VALUE_SURFACE,
    ):
        if required not in roi_consensus_moran_df.columns:
            raise ValueError(
                f"roi_consensus_moran_df missing required column: {required}"
            )
    filtered = roi_consensus_moran_df[
        roi_consensus_moran_df[VALUE_SURFACE].astype(str)
        == _CONSENSUS_ENSEMBLE_SURFACE
    ]
    if filtered.empty:
        return {}
    unknown = set(filtered[MORAN_DIRECTION].astype(str)) - _ALLOWED_MORAN_DIRECTIONS
    if unknown:
        raise ValueError(
            f"roi_consensus_moran_df contains disallowed moran_direction "
            f"value(s): {sorted(unknown)!r}"
        )
    key_dup = filtered.duplicated([ROI_ID, MORAN_DIRECTION])
    if key_dup.any():
        offenders = filtered.loc[
            key_dup, [ROI_ID, MORAN_DIRECTION]
        ].to_dict(orient="records")
        raise ValueError(
            f"roi_consensus_moran_df has duplicate (roi_id, moran_direction) "
            f"rows on the ensemble surface: {offenders!r}"
        )
    out: dict = {}
    for _, row in filtered.iterrows():
        roi = str(row[ROI_ID])
        direction = str(row[MORAN_DIRECTION])
        entry = out.setdefault(
            roi, {_MORAN_G1_LAG_G2_KEY: None, _MORAN_G2_LAG_G1_KEY: None},
        )
        if str(row[PARENT_MORAN_STATUS]) == _FINITE_STATUS:
            entry[direction] = float(row[MORAN_VALUE])
        else:
            entry[direction] = None
    return out


def _gather_normalization_values(hexes_df: pd.DataFrame,
                                 show_stability: bool,
                                 *,
                                 g1_fallback_values: np.ndarray,
                                 g2_fallback_values: np.ndarray) -> dict:
    values = {
        'g1_means': np.concatenate([
            hexes_df[GENE1_DENSITY_ENSEMBLE].to_numpy(dtype=float),
            g1_fallback_values,
        ]),
        'g2_means': np.concatenate([
            hexes_df[GENE2_DENSITY_ENSEMBLE].to_numpy(dtype=float),
            g2_fallback_values,
        ]),
        'g1_cvs': [], 'g2_cvs': [],
    }
    if show_stability:
        values['g1_cvs'] = hexes_df[GENE1_DENSITY_STABILITY].dropna().to_numpy(dtype=float)
        values['g2_cvs'] = hexes_df[GENE2_DENSITY_STABILITY].dropna().to_numpy(dtype=float)
    return values


def _to_float(value) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _rgba_or_undefined(mappable, value):
    v = _to_float(value)
    if mappable is not None and np.isfinite(v):
        return mappable.to_rgba(v), 'black', 0.3
    return UNDEFINED_FILL, UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH


_POSITIVE_OVER_ZERO_CONTEXT = (
    EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value
)
_ENRICHMENT_FINITE_STATUS = EnrichmentRatioStatus.FINITE.value


def _log2_style_by_status(mappable, fc_value, status_value):
    """Marginal-enrichment cell style routed by per-ratio status.

    positive_over_zero_context maps to top-of-scale (+inf shares +sat colour);
    other non-finite statuses render as constructed-undefined.
    """
    status = _status_str(status_value)
    if status == _ENRICHMENT_FINITE_STATUS:
        v = _to_float(fc_value)
        if not np.isfinite(v):
            return UNDEFINED_FILL, UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH
        return mappable.to_rgba(_transform_log2_fc(v)), 'black', 0.3
    if status == _POSITIVE_OVER_ZERO_CONTEXT:
        return mappable.to_rgba(_SATURATION_LOG2), 'black', 0.3
    return UNDEFINED_FILL, UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH


def _pattern_style_by_status(g1_fc, g1_status, g2_fc, g2_status):
    """Combined-pattern style. Classifies iff both marginals are finite or
    positive_over_zero_context (fed as saturating +inf); else undefined."""
    g1_v = _finite_or_saturated_fc(g1_fc, g1_status)
    g2_v = _finite_or_saturated_fc(g2_fc, g2_status)
    if g1_v is None or g2_v is None:
        return UNDEFINED_FILL, UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH
    rgba, is_undef = _classify_marginal_pattern(g1_v, g2_v)
    if is_undef:
        return UNDEFINED_FILL, UNDEFINED_EDGE, UNDEFINED_EDGE_LINEWIDTH
    return rgba, 'black', 0.3


def _status_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _finite_or_saturated_fc(fc_value, status_value):
    """Return the FC when finite, a +sat sentinel for positive_over_zero_context,
    or None to route to undefined."""
    status = _status_str(status_value)
    if status == _ENRICHMENT_FINITE_STATUS:
        v = _to_float(fc_value)
        return v if np.isfinite(v) else None
    if status == _POSITIVE_OVER_ZERO_CONTEXT:
        # Maps back through _transform_log2_fc to exactly +_SATURATION_LOG2.
        return float(2.0 ** _SATURATION_LOG2)
    return None


def _index_roi_consensus_pearson(
    roi_consensus_correlations_df: Optional[pd.DataFrame], sample_id: str,
) -> dict:
    """Return {roi_id: pearson_r | None (non-finite status)}."""
    if roi_consensus_correlations_df is None or roi_consensus_correlations_df.empty:
        return {}
    for required in (SAMPLE_ID, ROI_ID, VALUE_SURFACE,
                     PARENT_CONSENSUS_CORRELATION_STATUS, CORRELATION_VALUE):
        if required not in roi_consensus_correlations_df.columns:
            raise ValueError(
                f"roi_consensus_correlations_df missing column: {required}"
            )
    df = roi_consensus_correlations_df[
        (roi_consensus_correlations_df[VALUE_SURFACE].astype(str) == _ENSEMBLE_SURFACE)
        & (roi_consensus_correlations_df[SAMPLE_ID].astype(str) == sample_id)
    ]
    dup = df[df.duplicated(subset=[ROI_ID], keep=False)][ROI_ID]
    if len(dup):
        raise ValueError(
            f"roi_consensus_correlations_df has duplicate (sample_id={sample_id!r}, "
            f"roi_id, value_surface={_ENSEMBLE_SURFACE!r}) rows: "
            f"{sorted(set(str(r) for r in dup))!r}"
        )
    out: dict = {}
    for _, row in df.iterrows():
        roi = str(row[ROI_ID])
        if str(row[PARENT_CONSENSUS_CORRELATION_STATUS]) == _FINITE_STATUS:
            out[roi] = float(row[CORRELATION_VALUE])
        else:
            out[roi] = None
    return out


def _assert_roi_row_coverage(
    expected_roi_ids, indexed, source_name: str,
) -> None:
    """Fail loud if any expected ROI has no row; silent gray-painting hides
    missing rows that should be investigated."""
    missing = sorted(set(expected_roi_ids) - set(indexed.keys()))
    if missing:
        raise ValueError(
            f"{source_name}: missing rows for ROI(s) {missing!r}. "
            f"Every ROI with hex support must have a row (typed status "
            f"marks non-finite values; row absence is not permitted)."
        )


def _draw_hex_panels(hexes_df: pd.DataFrame,
                     axes,
                     mappables: dict,
                     show_stability: bool) -> None:
    total = len(hexes_df)
    drawn = 0
    dropped_geom = 0
    log.info("[VIZ] Drawing %d consensus hexagons across %d components",
             total, hexes_df[COMPONENT_ID].nunique())

    for row in hexes_df.itertuples(index=False):
        polygon = _wkt.loads(getattr(row, HEX_GEOMETRY_WKT))
        if not polygon.is_valid or polygon.is_empty:
            dropped_geom += 1
            continue

        g1_density = _to_float(getattr(row, GENE1_DENSITY_ENSEMBLE))
        g2_density = _to_float(getattr(row, GENE2_DENSITY_ENSEMBLE))
        g1_enrich = getattr(row, HEX_G1_ENRICHMENT)
        g2_enrich = getattr(row, HEX_G2_ENRICHMENT)
        g1_status = getattr(row, HEX_G1_ENRICHMENT_STATUS)
        g2_status = getattr(row, HEX_G2_ENRICHMENT_STATUS)

        panel_styles = [
            (axes[0, 0], *_rgba_or_undefined(mappables['g1'], g1_density)),
            (axes[0, 1], *_rgba_or_undefined(mappables['g2'], g2_density)),
            # Pearson panel (axes[0, 2]) painted per ROI, not per hex.
            (axes[1, 0], *_log2_style_by_status(
                mappables['g1_enrich'], g1_enrich, g1_status,
            )),
            (axes[1, 1], *_log2_style_by_status(
                mappables['g2_enrich'], g2_enrich, g2_status,
            )),
            (axes[1, 2], *_pattern_style_by_status(
                g1_enrich, g1_status, g2_enrich, g2_status,
            )),
        ]

        if show_stability:
            s_g1 = _to_float(getattr(row, GENE1_DENSITY_STABILITY))
            s_g2 = _to_float(getattr(row, GENE2_DENSITY_STABILITY))
            panel_styles.extend([
                (axes[0, 3], *_rgba_or_undefined(mappables['g1_cv'], s_g1)),
                (axes[1, 3], *_rgba_or_undefined(mappables['g2_cv'], s_g2)),
            ])

        for part in iter_polygons(polygon):
            for ax, facecolor, edgecolor, linewidth in panel_styles:
                ax.add_patch(patch_from_polygon(
                    part, facecolor=facecolor, edgecolor=edgecolor,
                    linewidth=linewidth,
                ))
        drawn += 1

    log.info("    Hexagons: drawn=%d/%d | invalid_geometry=%d",
             drawn, total, dropped_geom)


def _draw_dissolved_roi_panel(ax, roi_polygons: dict, roi_value: dict,
                              mappable) -> None:
    """Paint each ROI once with its Pearson / Moran value; non-finite values
    render as the shared constructed-undefined style."""
    for roi_id, polygon in roi_polygons.items():
        paint_roi_value(
            ax, polygon, roi_value.get(roi_id),
            scalar_mappable=mappable,
            undefined_fill=UNDEFINED_FILL,
            undefined_edge=UNDEFINED_EDGE,
            undefined_edge_linewidth=UNDEFINED_EDGE_LINEWIDTH,
        )


_LABEL_MIN_AREA_FRAC = 0.01


def _draw_roi_boundaries_and_labels(axes, n_cols_layout, component_polygons,
                                    show_biv_morans, roi_polygons, show_stability):
    # Hex-grain panels get component boundaries; ROI-grain panels get dissolved outlines.
    hex_panels = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
    if show_stability:
        hex_panels.extend([axes[0, 3], axes[1, 3]])
    roi_panels = [axes[0, 2]]
    if show_biv_morans:
        moran_col = n_cols_layout - 1
        roi_panels.extend([axes[0, moran_col], axes[1, moran_col]])

    for ax in hex_panels:
        draw_polygon_boundaries(
            ax, component_polygons, include_interiors=False,
        )
    for ax in roi_panels:
        draw_polygon_boundaries(ax, roi_polygons, include_interiors=False)

    for i in range(2):
        for j in range(n_cols_layout):
            ax = axes[i, j]
            ax.autoscale_view()
            ax.set_aspect('equal')
            ax.axis('off')

    if not roi_polygons:
        return

    label_bbox = dict(boxstyle="round,pad=0.3", facecolor="black",
                      alpha=0.5, edgecolor="none")
    max_area = max((float(poly.area) for poly in roi_polygons.values()), default=0.0)
    threshold = max_area * _LABEL_MIN_AREA_FRAC
    for roi_id, poly in roi_polygons.items():
        if float(poly.area) < threshold:
            continue
        rep = poly.representative_point()
        for ax in roi_panels:
            ax.text(
                float(rep.x), float(rep.y), f"ROI {roi_id}",
                ha="center", va="center", fontsize=10,
                color="white", weight="bold", bbox=label_bbox,
            )


def _enrichment_colorbar_label(marginal: str, has_positive_over_zero: bool) -> str:
    """Label the enrichment colorbar; mention +inf only when it fires."""
    if has_positive_over_zero:
        return f"{marginal} log2 fold enrichment (capped ±3; +∞ shares top colour)"
    return f"{marginal} log2 fold enrichment (capped ±3)"


def _annotate_positive_over_zero_marker(cbar, has_positive_over_zero: bool) -> None:
    """Relabel top tick as '3 / +inf' only when at least one hex is
    positive_over_zero_context, to avoid signalling a state that never fired."""
    if not has_positive_over_zero:
        return
    existing = list(cbar.get_ticks())
    if _SATURATION_LOG2 not in existing:
        existing = sorted(set(existing + [_SATURATION_LOG2]))
    labels = [
        f"{_SATURATION_LOG2:g} / +∞" if float(t) == _SATURATION_LOG2
        else f"{float(t):g}"
        for t in existing
    ]
    cbar.set_ticks(existing)
    cbar.set_ticklabels(labels)


def _hex_has_positive_over_zero_context(consensus_hexes_df: pd.DataFrame) -> dict:
    """Report per-marginal whether any hex is positive_over_zero_context, so the
    colorbar only shows the +inf marker when the state actually fires."""
    pos = _POSITIVE_OVER_ZERO_CONTEXT
    return {
        "g1": (
            HEX_G1_ENRICHMENT_STATUS in consensus_hexes_df.columns
            and (
                consensus_hexes_df[HEX_G1_ENRICHMENT_STATUS]
                .fillna("").astype(str) == pos
            ).any()
        ),
        "g2": (
            HEX_G2_ENRICHMENT_STATUS in consensus_hexes_df.columns
            and (
                consensus_hexes_df[HEX_G2_ENRICHMENT_STATUS]
                .fillna("").astype(str) == pos
            ).any()
        ),
    }


def _add_colorbars(fig, axes, mappables, n_cols_layout,
                   show_stability, show_biv_morans, *,
                   g1_has_positive_over_zero: bool = False,
                   g2_has_positive_over_zero: bool = False):
    fig.colorbar(mappables['g1'], ax=axes[0, 0], orientation='vertical',
                 label='A mean expression', shrink=0.8)
    fig.colorbar(mappables['g2'], ax=axes[0, 1], orientation='vertical',
                 label='B mean expression', shrink=0.8)
    fig.colorbar(mappables['roi_pearson'], ax=axes[0, 2], orientation='vertical',
                 label='Pearson r (ROI)', shrink=0.8)
    cb_g1 = fig.colorbar(
        mappables['g1_enrich'], ax=axes[1, 0], orientation='vertical',
        label=_enrichment_colorbar_label("A", g1_has_positive_over_zero),
        shrink=0.8,
    )
    _annotate_positive_over_zero_marker(cb_g1, g1_has_positive_over_zero)
    cb_g2 = fig.colorbar(
        mappables['g2_enrich'], ax=axes[1, 1], orientation='vertical',
        label=_enrichment_colorbar_label("B", g2_has_positive_over_zero),
        shrink=0.8,
    )
    _annotate_positive_over_zero_marker(cb_g2, g2_has_positive_over_zero)
    _add_pattern_color_key(axes[1, 2])

    if show_stability and mappables['g1_cv'] is not None:
        fig.colorbar(mappables['g1_cv'], ax=axes[0, 3], orientation='vertical',
                     label='A SD/IQR', shrink=0.8)
        fig.colorbar(mappables['g2_cv'], ax=axes[1, 3], orientation='vertical',
                     label='B SD/IQR', shrink=0.8)

    if show_biv_morans:
        fig.colorbar(mappables['i_g1_lag_g2'], ax=axes[0, n_cols_layout - 1],
                     orientation='vertical', label='Moran I (A ← B)', shrink=0.8)
        fig.colorbar(mappables['i_g2_lag_g1'], ax=axes[1, n_cols_layout - 1],
                     orientation='vertical', label='Moran I (B ← A)', shrink=0.8)


def _add_pattern_color_key(ax):
    """Pattern hue and weaker-marginal magnitude beside the panel."""

    base_colors_bottom_to_top = [
        None,
        _COLOR_BOTH_DOWN,
        _COLOR_G1DOWN_G2UP,
        _COLOR_G1UP_G2DOWN,
        _COLOR_BOTH_UP,
    ]
    labels_bottom_to_top = [
        'near neutral',
        'A↓ B↓',
        'A↓ B↑',
        'A↑ B↓',
        'A↑ B↑',
    ]
    strength = np.linspace(0.0, 1.0, 64)
    rows = []
    for base in base_colors_bottom_to_top:
        if base is None:
            row = np.repeat(_COLOR_NEUTRAL[None, :], len(strength), axis=0)
        else:
            row = (
                _COLOR_NEUTRAL[None, :]
                + strength[:, None] * (base[None, :] - _COLOR_NEUTRAL[None, :])
            )
        rows.append(row)

    divider = make_axes_locatable(ax)
    key_ax = divider.append_axes('right', size='5%', pad=0.08)
    key_ax.imshow(
        np.concatenate(rows, axis=0)[:, None, :], origin='lower', aspect='auto',
        extent=(0.0, 1.0, 0.0, 5.0), interpolation='nearest',
    )
    key_ax.set_yticks(np.arange(5) + 0.5)
    key_ax.set_yticklabels(labels_bottom_to_top, fontsize=8)
    key_ax.yaxis.tick_right()
    key_ax.set_xticks([])
    key_ax.tick_params(axis='both', length=0, pad=3)
    key_ax.hlines(np.arange(6), 0.0, 1.0, colors='#666666', linewidth=0.4)
    for spine in key_ax.spines.values():
        spine.set_linewidth(0.5)


def plot_consensus_grids(*,
                         consensus_hexes_df: pd.DataFrame,
                         spot_ledger_df: pd.DataFrame,
                         source_expression_df: pd.DataFrame,
                         roi_consensus_moran_df: Optional[pd.DataFrame] = None,
                         roi_consensus_correlations_df: Optional[pd.DataFrame] = None,
                         output_dir: str = 'output',
                         component_polygons: Optional[ComponentPolygonDict] = None,
                         component_to_roi: Optional[dict] = None,
                         canvas: Optional['Canvas'] = None,
                         show_stability: bool = False,
                         show_bivariate_morans: bool = True,
                         source_median_nn: Optional[float] = None,
                         feature_a_col: str = GENE1,
                         feature_b_col: str = GENE2,
                         feature_a_id: str,
                         feature_b_id: str) -> None:
    """Render the multi-panel consensus figure from canonical surfaces."""
    if consensus_hexes_df is None or consensus_hexes_df.empty:
        raise ValueError("plot_consensus_grids requires non-empty consensus_hexes_df")
    if spot_ledger_df is None:
        raise ValueError(
            "plot_consensus_grids requires spot_ledger_df; without it the "
            "renderer cannot distinguish acquisition holes from "
            "acquired-but-unconstructible spots."
        )
    if source_expression_df is None:
        raise ValueError(
            "plot_consensus_grids requires source_expression_df; without it "
            "the density-panel fallback cannot be computed."
        )
    for required in (GENE1_DENSITY_ENSEMBLE, GENE2_DENSITY_ENSEMBLE,
                     HEX_GEOMETRY_WKT, COMPONENT_ID, ROI_ID):
        if required not in consensus_hexes_df.columns:
            raise ValueError(f"consensus_hexes_df missing column: {required}")

    roi_moran_index = _index_moran_by_roi(roi_consensus_moran_df)

    sample_ids = consensus_hexes_df[SAMPLE_ID].astype(str).unique() \
        if SAMPLE_ID in consensus_hexes_df.columns else []
    sample_id = str(sample_ids[0]) if len(sample_ids) == 1 else ""
    roi_pearson_by_roi = (
        _index_roi_consensus_pearson(roi_consensus_correlations_df, sample_id)
        if sample_id else {}
    )

    hex_roi_ids = set(consensus_hexes_df[ROI_ID].astype(str).unique())
    roi_panels_requested = bool(component_polygons) and bool(component_to_roi)

    # Fail loud when a ROI panel is requested but its source is missing;
    # silent gray would hide pipeline breakage.
    if roi_panels_requested:
        if roi_consensus_correlations_df is None or roi_consensus_correlations_df.empty:
            raise ValueError(
                "roi_consensus_correlations_df is required to populate the "
                "ROI correlation panel when component polygons are provided; "
                "got None or empty."
            )
        _assert_roi_row_coverage(
            hex_roi_ids, roi_pearson_by_roi, "roi_consensus_correlations_df",
        )
        if show_bivariate_morans:
            if (
                roi_consensus_moran_df is None
                or roi_consensus_moran_df.empty
            ):
                raise ValueError(
                    "roi_consensus_moran_df is required to populate the "
                    "Moran panels when component polygons are provided and "
                    "show_bivariate_morans=True; got None or empty."
                )
            _assert_roi_row_coverage(
                hex_roi_ids, roi_moran_index, "roi_consensus_moran_df",
            )
    show_biv_morans = show_bivariate_morans and bool(roi_moran_index)

    # Classify spots or cells before building norms so the density colour
    # scale spans both hex-ensemble values and fallback spots.
    overlay = classify(
        spot_ledger_df, source_expression_df,
        gene1_col=feature_a_col, gene2_col=feature_b_col,
    )
    g1_fallback = overlay.unconstructible[feature_a_col].to_numpy(dtype=float)
    g2_fallback = overlay.unconstructible[feature_b_col].to_numpy(dtype=float)

    fig, axes = _create_consensus_layout(show_stability, show_biv_morans)
    # Read width from the actual axes grid so it cannot drift from the
    # layout helper's own arithmetic.
    n_cols_layout = axes.shape[1]

    values = _gather_normalization_values(
        consensus_hexes_df, show_stability,
        g1_fallback_values=g1_fallback,
        g2_fallback_values=g2_fallback,
    )
    mappables = _create_scalar_mappables(
        values, show_stability, show_biv_morans,
    )

    _draw_hex_panels(consensus_hexes_df, axes, mappables, show_stability)

    roi_polygons = dissolved_roi_polygons(component_polygons, component_to_roi)
    if roi_polygons:
        _draw_dissolved_roi_panel(
            axes[0, 2], roi_polygons, roi_pearson_by_roi, mappables['roi_pearson'],
        )
        if show_biv_morans:
            moran_col = n_cols_layout - 1
            moran_g1_lag_g2 = {
                r: e[_MORAN_G1_LAG_G2_KEY] for r, e in roi_moran_index.items()
            }
            moran_g2_lag_g1 = {
                r: e[_MORAN_G2_LAG_G1_KEY] for r, e in roi_moran_index.items()
            }
            _draw_dissolved_roi_panel(
                axes[0, moran_col], roi_polygons, moran_g1_lag_g2,
                mappables['i_g1_lag_g2'],
            )
            _draw_dissolved_roi_panel(
                axes[1, moran_col], roi_polygons, moran_g2_lag_g1,
                mappables['i_g2_lag_g1'],
            )

    _draw_roi_boundaries_and_labels(
        axes, n_cols_layout, component_polygons,
        show_biv_morans, roi_polygons, show_stability,
    )

    has_pos_over_zero = _hex_has_positive_over_zero_context(consensus_hexes_df)
    _add_colorbars(
        fig, axes, mappables, n_cols_layout, show_stability, show_biv_morans,
        g1_has_positive_over_zero=has_pos_over_zero["g1"],
        g2_has_positive_over_zero=has_pos_over_zero["g2"],
    )

    availability_handles = []
    if source_median_nn is None:
        source_median_nn = median_nn_distance_from_df(source_expression_df, "x", "y")
    median_nn = source_median_nn
    if median_nn is not None and not overlay.unconstructible.empty:
        # Bottom-row enrichment fill reuses the top-row expression cmap/norm
        # deliberately: enrichment is never synthesized at spot grain.
        paint_density_ring_overlay(
            axes[0, 0], overlay, gene_col=feature_a_col,
            cmap=mappables['g1'].get_cmap().name, norm=mappables['g1'].norm,
            median_nn=median_nn, individual_outline=False,
        )
        paint_density_ring_overlay(
            axes[0, 1], overlay, gene_col=feature_b_col,
            cmap=mappables['g2'].get_cmap().name, norm=mappables['g2'].norm,
            median_nn=median_nn, individual_outline=False,
        )
        paint_density_ring_overlay(
            axes[1, 0], overlay, gene_col=feature_a_col,
            cmap=mappables['g1'].get_cmap().name, norm=mappables['g1'].norm,
            median_nn=median_nn, individual_outline=False,
        )
        paint_density_ring_overlay(
            axes[1, 1], overlay, gene_col=feature_b_col,
            cmap=mappables['g2'].get_cmap().name, norm=mappables['g2'].norm,
            median_nn=median_nn, individual_outline=False,
        )

        # Group by parent ROI so touching support from distinct biological
        # ROIs never merges into a single contour.
        grouped_unc = annotate_spots_with_parent_roi(
            overlay.unconstructible, spot_ledger_df, source_expression_df,
        )
        contour_colors = {
            str(roi): "#333333"
            for roi in grouped_unc["_parent_roi"].astype(str).unique()
        }
        for i in range(2):
            for j in range(n_cols_layout):
                paint_grouped_spot_support_contours(
                    axes[i, j], grouped_unc, group_col="_parent_roi",
                    colors=contour_colors, median_nn=median_nn,
                    linewidth=0.55,
                )
        availability_handles.append(unavailable_legend_handle())
    availability_handles.append(constructed_undefined_legend_handle())

    fig.legend(
        handles=availability_handles, loc='lower center',
        ncol=len(availability_handles),
        bbox_to_anchor=(0.5, 0.0), fontsize=9, framealpha=0.9,
        title="Availability (all panels)",
    )

    add_identity_header(fig, feature_a_id, feature_b_id, "Consensus representation")
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.93, bottom=0.06,
        wspace=0.15, hspace=0.15,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "consensus_tessellation_summary.pdf")
    plt.savefig(output_path)
    log.info(f"[VIZ] Saved: {output_path}")
    plt.close(fig)
