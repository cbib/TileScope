"""
HTML summary report generator for TileScope analysis results.

Minimal-valid template: run metadata, embedded visualization panels
(when available), and dynamic file downloads. Rich parent-ROI tables are
outside this compact summary; the retired `summary_cross_scale.csv` is not
rendered.
"""
import base64
from io import BytesIO


import html
import json
import logging
from pathlib import Path

import pandas as pd

from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.reporters.html_template import build_html_header, build_html_footer
from tilescope.reporters.spatial_support_sensitivity import (
    SupportSensitivityResult,
    load_spatial_support_results,
)
from tilescope.schema.helpers import natural_sort_key

logger = logging.getLogger(__name__)

# Check if pdf2image is available for PDF conversion
try:
    from pdf2image import convert_from_path
    PDF_CONVERSION_AVAILABLE = True
except ImportError:
    PDF_CONVERSION_AVAILABLE = False
    logger.warning("[HTML Report] pdf2image not available - visual panels will not be embedded")


def _coverage_table_html(metadata: dict) -> str:
    """Inner HTML for the sample-level coverage table (no section header)."""
    population = metadata.get("population", {})
    return f"""
    <div style="background: #ecf0f1; padding: 15px; border-radius: 8px;">
        <table style="border-collapse: collapse; font-size: 0.9em;">
        <tr><td style="padding:4px 12px;">Input biological ROIs</td>
            <td style="padding:4px 12px;text-align:right;"><strong>{population.get('biological_roi_count', 0):,}</strong></td></tr>
        <tr><td style="padding:4px 12px;">Input spots</td>
            <td style="padding:4px 12px;text-align:right;">{population.get('source_spot_count', 0):,}</td></tr>
        <tr><td style="padding:4px 12px;">Spatially supported components</td>
            <td style="padding:4px 12px;text-align:right;">{population.get('spatially_supported_component_count', 0):,}</td></tr>
        <tr><td style="padding:4px 12px;">Spots included in spatial geometry</td>
            <td style="padding:4px 12px;text-align:right;">{population.get('spatially_supported_spot_count', 0):,}</td></tr>
        <tr><td style="padding:4px 12px;">Tessellated components</td>
            <td style="padding:4px 12px;text-align:right;">{population.get('tessellated_component_count', 0):,}</td></tr>
        <tr><td style="padding:4px 12px;">Spots included in tessellation</td>
            <td style="padding:4px 12px;text-align:right;">{population.get('tessellated_spot_count', 0):,}</td></tr>
        </table>
    </div>
    """


def _build_component_details_section(metadata: dict) -> str:
    """Per-component provenance table: collapsed by default.

    Reader-facing columns: Biological ROI, Spatial component, Spots,
    Processing status (Tessellated / Geometry only), Mean tiles per
    tessellation, Consensus regions.

    Rows are ordered by orchestrator (natural sort on roi_id, then
    component_id). Multi-component parents are visually separated by
    a top border on the first row of each parent group.
    """
    component_stats = metadata.get("component_stats", [])
    population = metadata.get("population", {})
    n_total = int(population.get("spatially_supported_component_count", len(component_stats)))
    n_tessellated = int(population.get("tessellated_component_count", 0))

    if not component_stats:
        return (
            "<details><summary>Spatial component details — no spatially supported components</summary>"
            "<p><em>All biological ROIs were classified <code>scattered</code>; no per-component "
            "rows to report.</em></p></details>"
        )

    rows_html = []
    prev_roi = None
    for s in component_stats:
        roi_id = s["roi_id"]
        component_id = s["component_id"]
        n_spots = int(s["n_spots"])
        tessellated = int(s.get("n_tessellations", 0)) > 0
        status = "Tessellated" if tessellated else "Geometry only"
        mean_tiles = f"{float(s['avg_tiles_per_tess']):.1f}" if tessellated else "—"
        n_hexes = int(s.get("n_consensus_hexagons", 0))
        n_hexes_display = f"{n_hexes:,}" if n_hexes > 0 else "—"

        first_of_group = roi_id != prev_roi
        row_style = "border-top:2px solid #bdc3c7;" if (first_of_group and prev_roi is not None) else ""
        status_style = "color:#27ae60;" if tessellated else "color:#7f8c8d;"

        rows_html.append(
            f"<tr style='{row_style}'>"
            f"<td style='padding:6px 12px;'>{roi_id}</td>"
            f"<td style='padding:6px 12px;'>{component_id}</td>"
            f"<td style='padding:6px 12px;text-align:right;'>{n_spots:,}</td>"
            f"<td style='padding:6px 12px;{status_style}'>{status}</td>"
            f"<td style='padding:6px 12px;text-align:right;'>{mean_tiles}</td>"
            f"<td style='padding:6px 12px;text-align:right;'>{n_hexes_display}</td>"
            f"</tr>"
        )
        prev_roi = roi_id

    return f"""
    <details>
    <summary style="cursor:pointer;font-weight:600;padding:8px 0;">
        Construction provenance — {n_total} spatial components; {n_tessellated} tessellated
    </summary>
    <p style="font-size: 0.9em;">One row per spatially supported component. Sibling components
    (same biological ROI) share a colour boundary; the first row of each parent group is
    marked by a horizontal rule.</p>
    <table style="border-collapse: collapse; margin-bottom: 20px; font-size: 0.9em;">
        <thead>
            <tr style="background:#34495e;color:white;">
                <th style="padding:6px 12px;text-align:left;">Biological ROI</th>
                <th style="padding:6px 12px;text-align:left;">Spatial component</th>
                <th style="padding:6px 12px;text-align:right;">Spots</th>
                <th style="padding:6px 12px;text-align:left;">Processing status</th>
                <th style="padding:6px 12px;text-align:right;">Mean tiles per tessellation</th>
                <th style="padding:6px 12px;text-align:right;">Consensus regions</th>
            </tr>
        </thead>
        <tbody>{''.join(rows_html)}</tbody>
    </table>
    </details>
    """


def _build_settings_provenance_section(metadata: dict) -> str:
    """Enrichment thresholds + eligibility thresholds.

    Collapsed by default: reference material, not scientific results.
    """
    thresholds = metadata.get("thresholds", {})
    gates = metadata.get("gates", {})
    identity = metadata.get("feature_identity", {})
    g1_id, g2_id = require_semantic_feature_ids(
        identity.get("feature_g1_id"),
        identity.get("feature_g2_id"),
        owner="HTML report metadata",
    )
    # Escape before interpolation: feature ids arrive from user input.
    g1_label = html.escape(g1_id)
    g2_label = html.escape(g2_id)

    thresholds_html = f"""
    <h4 style="margin-top: 0; color: #2c3e50;">Enrichment thresholds</h4>
    <div style="font-family: monospace; font-size: 0.9em;">
        HIGH (≥ {thresholds.get('q_high_percentile', 0)*100:.0f}%):
        {g1_label} ≥ {thresholds.get('gene1_high_threshold', 0):.4f},
        {g2_label} ≥ {thresholds.get('gene2_high_threshold', 0):.4f}<br>
        LOW  (≤ {thresholds.get('q_low_percentile', 0)*100:.0f}%):
        {g1_label} ≤ {thresholds.get('gene1_low_threshold', 0):.4f},
        {g2_label} ≤ {thresholds.get('gene2_low_threshold', 0):.4f}
    </div>
    """

    gates_html = f"""
    <h4 style="color: #2c3e50;">Eligibility thresholds</h4>
    <table style="border-collapse: collapse; font-size: 0.9em; font-family: monospace;">
    <tr><td style="padding:4px 12px;">min_roi_spots</td>
        <td style="padding:4px 12px;text-align:right;">{gates.get('min_roi_spots', 0)}</td></tr>
    <tr><td style="padding:4px 12px;">min_component_spots</td>
        <td style="padding:4px 12px;text-align:right;">{gates.get('min_component_spots', 0)}</td></tr>
    <tr><td style="padding:4px 12px;">minimum tile occupancy floor</td>
        <td style="padding:4px 12px;text-align:right;">{gates.get('min_spots_per_tile', 0)}</td></tr>
    <tr><td style="padding:4px 12px;">min_context_spots</td>
        <td style="padding:4px 12px;text-align:right;">{gates.get('min_context_spots', 0)}</td></tr>
    </table>
    """

    return f"""
    <details>
    <summary style="cursor:pointer;font-weight:600;padding:8px 0;">
        Analysis settings
    </summary>
    <div style="background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 12px 0;">
        {thresholds_html}
        {gates_html}
    </div>
    </details>
    """


def _embed_pdf_pages(
    pdf_path: Path,
    title: str,
    caption: str,
    *,
    collapsible: bool = False,
    max_width_px: int = 1200,
) -> str:
    """Embed a PDF's pages as inline PNGs, or fall back to a link.

    When ``collapsible=True``, wraps the section in a top-level
    ``<details>`` block so the pages render only on demand.
    ``max_width_px`` caps the rendered image width in the report.
    """
    if not pdf_path.exists():
        return ""
    body_pieces: list[str] = []
    if PDF_CONVERSION_AVAILABLE:
        try:
            images = convert_from_path(str(pdf_path), dpi=150)
            logger.info(f"[HTML Report] Converted {len(images)} pages from {pdf_path.name}")
            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="PNG", optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                body_pieces.append(
                    "<div style=\"margin: 20px 0; page-break-inside: avoid;\">"
                    f"<img src=\"data:image/png;base64,{img_str}\" "
                    f"style=\"width: 100%; max-width: {max_width_px}px; "
                    "height: auto; border: 1px solid #ddd; border-radius: 4px;\">"
                    "</div>"
                )
        except Exception as e:
            logger.warning(f"[HTML Report] Failed to embed {pdf_path.name}: {e}")
            body_pieces.append(
                f'<p>⚠️ Could not embed panels. '
                f'<a href="{pdf_path.name}">View PDF directly</a></p>'
            )
    else:
        body_pieces.append(f'<p><a href="{pdf_path.name}">📊 View {title} PDF</a></p>')
    caption_html = f"<p><em>{caption}</em></p>\n" if caption else ""
    body = caption_html + "\n".join(body_pieces)
    if collapsible:
        return (
            f"<details><summary style=\"font-size:1.3em;font-weight:600;\">"
            f"{title}</summary>{body}</details>"
        )
    return f"<h2>{title}</h2>\n{body}"


def _orientation_panel_body(output_dir: Path, max_width_px: int) -> str:
    """Bare image markup for support_provenance.pdf (no h2, no caption).

    Empty string if the PDF is absent or pdf2image is unavailable.
    """
    pdf_path = output_dir / "support_provenance.pdf"
    if not pdf_path.exists() or not PDF_CONVERSION_AVAILABLE:
        return ""
    try:
        images = convert_from_path(str(pdf_path), dpi=150)
        pieces = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            pieces.append(
                "<div style=\"margin: 0 0 12px 0; page-break-inside: avoid;\">"
                f"<img src=\"data:image/png;base64,{img_str}\" "
                f"style=\"width: 100%; max-width: {max_width_px}px; "
                "height: auto; border: 1px solid #ddd; border-radius: 4px;\">"
                "</div>"
            )
        return "\n".join(pieces)
    except Exception as e:
        logger.warning(f"[HTML Report] Failed to embed support_provenance.pdf: {e}")
        return (
            f'<p>⚠️ Could not embed panel. '
            f'<a href="{pdf_path.name}">View PDF directly</a></p>'
        )


def _build_coverage_and_orientation_section(
    metadata: dict | None, output_dir: Path,
) -> str:
    """Analysis coverage table + observed-support orientation panel.

    When both are present, they render side by side (orientation panel on
    the left, capped at ~600 px; coverage table on the right). The panel
    uses the existing support_provenance.pdf; the caption is small so the
    combined block stays compact.
    """
    orientation_html = _orientation_panel_body(output_dir, max_width_px=600)
    coverage_html = _coverage_table_html(metadata) if metadata else ""
    if not orientation_html and not coverage_html:
        return ""

    caption = (
        '<p style="font-size:0.85em;margin:6px 0 0 0;"><em>'
        'Location of the numbered biological ROIs and the observed '
        'regions contributing to spatial analysis.'
        '</em></p>'
    )

    if orientation_html and coverage_html:
        return (
            "<h2>Analysis coverage</h2>\n"
            '<div style="display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;">\n'
            '  <div style="flex:0 1 620px;min-width:280px;">'
            f"{orientation_html}{caption}"
            "</div>\n"
            '  <div style="flex:1 1 340px;min-width:280px;">'
            f"{coverage_html}"
            "</div>\n"
            "</div>"
        )
    if orientation_html:
        return f"<h2>Observed spatial support</h2>\n{orientation_html}{caption}"
    return f"<h2>Analysis coverage</h2>\n{coverage_html}"


def _build_visual_panels_section(output_dir: Path) -> str:
    """Embed the expression summary (collapsible) and the consensus visual."""
    parts = [
        _embed_pdf_pages(
            output_dir / "expression_summary.pdf",
            "Expression summary",
            "Per-ROI expression, joint highs, and support geometry.",
            collapsible=True,
        ),
        _embed_pdf_pages(
            output_dir / "consensus_tessellation_summary.pdf",
            "Consensus & tessellation visual summary",
            "Spatial patterns across spatially eligible components: gene "
            "expression, enrichment, correlations, and bivariate Moran's I.",
        ),
    ]
    return "\n".join(p for p in parts if p)


_SPATIAL_SUPPORT_CLASS_ORDER = (
    "scattered",
    "mixed",
    "fully_connected",
    "fully_supported_multicomponent",
)

_CLASS_DESCRIPTION = {
    "scattered": (
        "No connected region contains enough observations for spatial "
        "analysis; tessellation, consensus and Moran's I are unavailable."
    ),
    "mixed": (
        "The ROI contains both analysable connected regions and fragments "
        "that are too small; only the analysable regions contribute to "
        "geometry-based results."
    ),
    "fully_connected": (
        "Every spot lies in ONE geometry-eligible sample-support component."
    ),
    "fully_supported_multicomponent": (
        "Every spot lies in a geometry-eligible sample-support component; "
        "there are multiple such components (legitimate disconnected biology)."
    ),
}


def _build_spatial_support_section(output_dir: Path) -> str:
    """Per-biological-ROI spatial-support classification table.

    Reads `roi_support_status.csv` from `output_dir`. Every source
    ROI_ID appears in the table; scattered ROIs are NOT silently
    dropped from the report just because they have no polygon.
    """
    csv_path = output_dir / "roi_support_status.csv"
    if not csv_path.exists():
        return ""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"[HTML Report] Failed to load roi_support_status.csv: {e}")
        return ""
    if df.empty:
        return ""

    # Class-count summary.
    counts = df["spatial_support_class"].value_counts().to_dict()
    total = int(df.shape[0])
    counts_row = "".join(
        f'<td style="padding: 6px 12px; text-align: right;">{counts.get(cls, 0)}</td>'
        for cls in _SPATIAL_SUPPORT_CLASS_ORDER
    )
    counts_header = "".join(
        f'<th style="padding: 6px 12px; text-align: right;">{cls}</th>'
        for cls in _SPATIAL_SUPPORT_CLASS_ORDER
    )

    # Per-class textual description.
    description_items = "".join(
        f'<li><strong>{cls}</strong> — {_CLASS_DESCRIPTION[cls]}</li>'
        for cls in _SPATIAL_SUPPORT_CLASS_ORDER
        if counts.get(cls, 0) > 0
    )

    # Per-ROI table, natural ROI ordering (integer-aware).
    df_sorted = df.assign(
        _roi_key=df["roi_id"].map(natural_sort_key),
    ).sort_values(by="_roi_key").drop(columns="_roi_key")

    thead = (
        "<tr style='background:#34495e;color:white;'>"
        "<th style='padding:6px 10px;text-align:left;'>ROI</th>"
        "<th style='padding:6px 10px;text-align:right;'>input spots</th>"
        "<th style='padding:6px 10px;text-align:right;'>support components</th>"
        "<th style='padding:6px 10px;text-align:right;'>Qualifying support components</th>"
        "<th style='padding:6px 10px;text-align:right;'>Spots in qualifying components</th>"
        "<th style='padding:6px 10px;text-align:right;'>largest support component</th>"
        "<th style='padding:6px 10px;text-align:left;'>spatial support class</th>"
        "</tr>"
    )
    body_rows = []
    for _, row in df_sorted.iterrows():
        cls = row["spatial_support_class"]
        cls_style = "color:#c0392b;font-weight:bold;" if cls == "scattered" else ""
        body_rows.append(
            f"<tr>"
            f"<td style='padding:6px 10px;'>{row['roi_id']}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{int(row['source_spots'])}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{int(row['support_component_count'])}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{int(row['support_qualified_component_count'])}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{int(row['support_qualified_spots'])}</td>"
            f"<td style='padding:6px 10px;text-align:right;'>{int(row['largest_support_component_spots'])}</td>"
            f"<td style='padding:6px 10px;text-align:left;{cls_style}'>{cls}</td>"
            f"</tr>"
        )
    body = "".join(body_rows)

    return f"""
<h2>Spatial support by ROI</h2>
<p>TileScope checks whether each ROI contains connected regions large
enough for spatial analysis. All ROIs are shown here. If no region
passes this check, expression level results may still be available,
but tessellation, consensus and Moran's I are not. The table shows how
many connected regions were analysed and how many observations they
contain.</p>

<h3>Class counts (of {total} biological ROIs)</h3>
<table style="border-collapse:collapse;margin-bottom:12px;">
<thead><tr style="background:#34495e;color:white;">{counts_header}</tr></thead>
<tbody><tr>{counts_row}</tr></tbody>
</table>
<ul style="font-size:0.9em;">{description_items}</ul>

<h3>Per-ROI status</h3>
<table style="border-collapse:collapse;margin-bottom:20px;font-size:0.9em;">
<thead>{thead}</thead>
<tbody>{body}</tbody>
</table>
"""


def _fmt_support_value(
    value: object,
    statistic: str,
    status: str = "",
) -> str:
    """Format a canonical raw value on its natural, reader-facing scale."""
    if status == "positive_over_zero_context":
        return "∞"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not pd.notna(number):
        return "—"
    if statistic == "enrichment":
        return f"{number:.2f}×"
    return f"{number:+.2f}"


def _fmt_tessellation_interval(result: SupportSensitivityResult) -> str:
    q25 = result.tessellation_q25_raw
    q75 = result.tessellation_q75_raw
    if q25 is None or q75 is None:
        return "—"
    left = _fmt_support_value(q25, result.statistic)
    right = _fmt_support_value(q75, result.statistic)
    return f"{left} to {right}"


def _support_measure_label(result: SupportSensitivityResult) -> str:
    if result.statistic == "correlation":
        return "Within-ROI correlation"
    if result.statistic == "enrichment":
        return "High-value overlap"
    if result.moran_direction == "g1_lag_g2":
        return f"{result.g1_id} near {result.g2_id}"
    return f"{result.g2_id} near {result.g1_id}"


def _build_spatial_support_sensitivity_section(output_dir: Path) -> str:
    """Compact biological reading of sensitivity to spatial support."""
    try:
        results = load_spatial_support_results(output_dir)
    except Exception as e:
        logger.warning(f"[HTML Report] Failed to interpret spatial support: {e}")
        return ""
    if not results:
        return ""

    parts = [
        "<h2>Sensitivity to spatial support</h2>",
        "<p>TileScope checks whether each spatial relationship keeps the "
        "same interpretation when observations are grouped in different "
        "ways.</p>",
        "<p><strong>Stable</strong> means the result stays consistent. "
        "<strong>Sensitive to zoning</strong> means alternative groupings "
        "give meaningfully different results. <strong>Direction changes</strong> "
        "means the observed spots and consensus map show opposite clear "
        "directions.</p>",
    ]
    table_head = (
        "<thead><tr><th>Measure</th><th>Reading</th>"
        "<th>Observed spots</th>"
        "<th>Alternative groupings<br><small>middle 50%</small></th>"
        "<th>Consensus map</th></tr></thead>"
    )

    by_roi: dict[str, list[SupportSensitivityResult]] = {}
    for result in results:
        by_roi.setdefault(result.roi_id, []).append(result)
    roi_ids = sorted(by_roi, key=natural_sort_key)
    for roi_id in roi_ids:
        roi_results = by_roi[roi_id]
        first = roi_results[0]
        g1 = html.escape(first.g1_id)
        g2 = html.escape(first.g2_id)
        roi_label = html.escape(str(roi_id))
        header = f"ROI {roi_label} ({g1} × {g2})"
        rows = []
        for result in roi_results:
            measure = html.escape(_support_measure_label(result))
            reading = html.escape(result.reading.value)
            source = _fmt_support_value(
                result.source_raw, result.statistic, result.source_status,
            )
            tessellations = _fmt_tessellation_interval(result)
            consensus = _fmt_support_value(
                result.consensus_raw, result.statistic, result.consensus_status,
            )
            rows.append(
                f"<tr><td>{measure}</td><td><strong>{reading}</strong></td>"
                f"<td>{source}</td><td>{tessellations}</td>"
                f"<td>{consensus}</td></tr>"
            )
        parts.append(
            f"<details><summary>{header}</summary>"
            f"<table>{table_head}<tbody>{''.join(rows)}</tbody></table>"
            "</details>"
        )
    return "\n".join(parts)


_HEADLINE_OUTPUTS = (
    "roi_support_status.csv",
    "roi_correlations.csv",
    "roi_enrichment.csv",
    "roi_tessellation_correlations.csv",
    "roi_tessellation_enrichment.csv",
    "roi_replicate_correlations.csv",
    "roi_replicate_enrichment.csv",
    "roi_consensus_correlations.csv",
    "roi_consensus_enrichment.csv",
    "roi_source_moran.csv",
    "roi_tessellation_moran.csv",
    "roi_consensus_moran.csv",
    "roi_cross_grain_sensitivity.csv",
    "expression_summary.pdf",
    "consensus_tessellation_summary.pdf",
    "support_provenance.pdf",
)


def _build_files_section(output_dir: Path) -> str:
    """Headline parent-ROI outputs + collapsed 'All generated files'.

    Headline row lists the parent-ROI CSVs and main visual summaries
    for readers coming to the report first. Everything else in the
    output directory (per-component CSVs, tessellation-density PDFs,
    effective_config.json, run_metadata.json) sits inside a collapsed
    <details> so an unfamiliar reader is not overwhelmed by a 20+
    file list.

    `analysis_report.html` is excluded from both lists; the report
    doesn't link to itself, and pre/post-write timing would otherwise
    make its presence in Downloads nondeterministic between runs.
    """
    present = {p.name for p in output_dir.iterdir() if p.is_file()}
    present.discard("analysis_report.html")
    headline = [name for name in _HEADLINE_OUTPUTS if name in present]
    others = sorted(present - set(_HEADLINE_OUTPUTS))

    def _icon(name: str) -> str:
        if name.endswith(".pdf"):
            return "📊"
        if name.endswith(".csv"):
            return "📄"
        if name.endswith(".html"):
            return "🌐"
        return "📎"

    headline_html = "".join(
        f'<div class="link-section">{_icon(n)} <a href="{n}">{n}</a></div>\n'
        for n in headline
    )
    others_html = "".join(
        f'<div class="link-section">{_icon(n)} <a href="{n}">{n}</a></div>\n'
        for n in others
    )

    # Two-estimand clarification. Reader-facing wording avoids the
    # internal "sibling-component" phrasing: for TileScope-unaware
    # readers, "all tessellation tiles in that ROI" is more accurate
    # and matches what the summary CSV actually reports (median +
    # Q25–Q75 across replicates).
    clarification = (
        '<p style="font-size:0.9em;"><em>'
        '<code>expression_summary.pdf</code> shows correlation calculated '
        'from all spots or cells in each biological ROI. For ROIs with '
        'spatial support, <code>roi_tessellation_correlations.csv</code> '
        'summarizes the median and Q25–Q75 of correlations recalculated '
        'from all tessellation tiles in that ROI, separately within each '
        'replicate.'
        '</em></p>'
    )

    return f"""
<h2>Downloads</h2>
{clarification}
<div class="links">
{headline_html}
</div>

<details>
<summary style="cursor:pointer;font-weight:600;padding:8px 0;">
    All generated files ({len(others)} additional files)
</summary>
<div class="links">
{others_html}
</div>
</details>
"""


def generate_html_report(output_dir: Path) -> Path:
    """Generate the TileScope HTML summary report.

    The report contains a header, optional run-metadata block, embedded
    consensus visualization panels (if the PDF exists), and a dynamic file
    downloads section. Rich per-ROI parent-grain tables are outside this
    compact summary.

    """
    output_dir = Path(output_dir)

    metadata: dict = {}
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info("[HTML Report] Loaded run metadata")
        except Exception as e:
            logger.warning(f"[HTML Report] Failed to load run metadata: {e}")

    html_parts = [
        build_html_header(),
        "<h1>TileScope Analysis Report</h1>",
        (
            f"<p><em>Generated from: {output_dir.name}</em>"
            " &middot; "
            "<a href=\"https://github.com/cbib/TileScope/blob/main/docs/"
            "HOW_TO_READ_THE_REPORT.md\">How to read this report</a></p>"
        ),
    ]
    # Section order, biology-first, with the ROI orientation map at the
    # top so numbered ROIs are readable in every section below it.
    #   1. Analysis coverage + Observed spatial support
    #      (compact banner with orientation panel on the left of the
    #       coverage table)
    #   2. Expression summary               (collapsible)
    #   3. Consensus / tessellation visual  (visible)
    #   4. Sensitivity to spatial support   (compact per-ROI interpretation)
    #   5. Detailed support and provenance tables
    #        - Spatial support by ROI       (visible detail)
    #        - Spatial component details    (collapsed)
    #        - Analysis settings + provenance (collapsed)
    #   6. Downloads                        (headline visible; rest collapsed)
    html_parts.append(_build_coverage_and_orientation_section(metadata, output_dir))
    html_parts.append(_build_visual_panels_section(output_dir))
    html_parts.append(_build_spatial_support_sensitivity_section(output_dir))
    html_parts.append(_build_spatial_support_section(output_dir))
    if metadata:
        html_parts.append(_build_component_details_section(metadata))
        html_parts.append(_build_settings_provenance_section(metadata))
    html_parts.append(_build_files_section(output_dir))
    html_parts.append(build_html_footer())

    html_content = '\n'.join(html_parts)
    html_path = output_dir / "analysis_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"[HTML Report] Generated: {html_path}")
    return html_path
