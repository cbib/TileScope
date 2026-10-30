"""Semantic tests for the spatial-support interpretation and HTML."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tilescope.reporters.html_summary import (
    _build_spatial_support_sensitivity_section,
)
from tilescope.reporters.spatial_support_sensitivity import (
    SpatialSupportReading,
    classify_spatial_support,
)


def test_spatial_support_readings_keep_zoning_and_direction_separate():
    classify = classify_spatial_support
    assert classify(
        statistic="correlation",
        source_effect=-0.40,
        tessellation_effect=-0.30,
        consensus_effect=-0.50,
        tessellation_effect_iqr=0.08,
        n_tessellations=10,
    ) == SpatialSupportReading.STABLE
    assert classify(
        statistic="correlation",
        source_effect=-0.30,
        tessellation_effect=-0.10,
        consensus_effect=-0.20,
        tessellation_effect_iqr=0.50,
        n_tessellations=10,
    ) == SpatialSupportReading.ZONING_SENSITIVE
    assert classify(
        statistic="correlation",
        source_effect=0.30,
        tessellation_effect=0.28,
        consensus_effect=0.35,
        tessellation_effect_iqr=0.22,
        n_tessellations=10,
    ) == SpatialSupportReading.ZONING_SENSITIVE
    assert classify(
        statistic="correlation",
        source_effect=-0.30,
        tessellation_effect=0.40,
        consensus_effect=0.60,
        tessellation_effect_iqr=0.20,
        n_tessellations=10,
    ) == SpatialSupportReading.DIRECTION_CHANGE

    # A source/tessellation disagreement alone is neither a consensus
    # direction change nor zoning sensitivity when the tessellations agree.
    assert classify(
        statistic="correlation",
        source_effect=0.30,
        tessellation_effect=-0.40,
        consensus_effect=0.50,
        tessellation_effect_iqr=0.08,
        n_tessellations=10,
    ) == SpatialSupportReading.NO_CLEAR_PATTERN
    assert classify(
        statistic="correlation",
        source_effect=0.03,
        tessellation_effect=0.02,
        consensus_effect=0.04,
        tessellation_effect_iqr=0.06,
        n_tessellations=10,
    ) == SpatialSupportReading.NO_CLEAR_PATTERN
    assert classify(
        statistic="correlation",
        source_effect=0.30,
        tessellation_effect=None,
        consensus_effect=None,
        tessellation_effect_iqr=None,
        n_tessellations=0,
    ) == SpatialSupportReading.NOT_ASSESSABLE


def _cross_row(
    statistic: str,
    target_surface: str,
    source_raw: float,
    target_raw: float,
    *,
    channel: str = "",
    direction: str = "",
    tessellation_iqr: float | None = None,
) -> dict:
    if statistic == "enrichment":
        source_effect = source_raw / (1.0 + source_raw) - 0.5
        target_effect = target_raw / (1.0 + target_raw) - 0.5
    else:
        source_effect = source_raw
        target_effect = target_raw
    return {
        "sample_id": 0,
        "roi_id": 1,
        "g1_id": "A<script>",
        "g2_id": "B",
        "statistic": statistic,
        "channel": channel,
        "moran_direction": direction,
        "reference_surface": "source_native",
        "target_surface": target_surface,
        "reference_raw_value": source_raw,
        "reference_raw_status": "finite",
        "reference_effect_value": source_effect,
        "target_raw_value": target_raw,
        "target_raw_status": "finite",
        "target_effect_value": target_effect,
        "tessellation_effect_iqr": tessellation_iqr,
        "n_comparable_replicates": 10 if tessellation_iqr is not None else None,
    }


def test_html_is_compact_biological_summary(tmp_path: Path):
    rows = []
    lanes = (
        ("correlation", "", "", -0.30, 0.40, 0.60, 0.10),
        ("enrichment", "joint_high", "", 0.20, 0.30, 0.40, 0.02),
        ("moran", "", "g1_lag_g2", -0.30, -0.10, -0.20, 0.50),
        ("moran", "", "g2_lag_g1", 0.30, 0.25, 0.40, 0.05),
    )
    for statistic, channel, direction, source, tessellation, consensus, iqr in lanes:
        rows.append(_cross_row(
            statistic, "tessellation_median", source, tessellation,
            channel=channel, direction=direction, tessellation_iqr=iqr,
        ))
        rows.append(_cross_row(
            statistic, "consensus_ensemble", source, consensus,
            channel=channel, direction=direction,
        ))
    pd.DataFrame(rows).to_csv(
        tmp_path / "roi_cross_grain_sensitivity.csv", index=False,
    )
    pd.DataFrame({
        "roi_id": [1] * 4,
        "parent_replicate_correlation_status": ["finite"] * 4,
        "correlation_value": [0.30, 0.35, 0.45, 0.50],
    }).to_csv(tmp_path / "roi_replicate_correlations.csv", index=False)
    pd.DataFrame({
        "roi_id": [1] * 4,
        "joint_high_enrichment_status": ["finite"] * 4,
        "joint_high_enrichment_fc": [0.20, 0.25, 0.30, 0.35],
    }).to_csv(tmp_path / "roi_replicate_enrichment.csv", index=False)
    pd.DataFrame({
        "roi_id": [1, 1],
        "moran_direction": ["g1_lag_g2", "g2_lag_g1"],
        "parent_moran_status": ["finite", "finite"],
        "moran_value_p25_across_replicates": [-0.40, 0.20],
        "moran_value_p75_across_replicates": [0.10, 0.30],
        "n_finite_replicates": [10, 10],
    }).to_csv(tmp_path / "roi_tessellation_moran.csv", index=False)

    rendered = _build_spatial_support_sensitivity_section(tmp_path)

    assert "Sensitivity to spatial support" in rendered
    assert "Observed spots" in rendered
    assert "Alternative groupings" in rendered
    assert "middle 50%" in rendered
    assert "Consensus map" in rendered
    assert "Within-ROI correlation" in rendered
    assert "High-value overlap" in rendered
    assert "A&lt;script&gt; near B" in rendered
    assert "B near A&lt;script&gt;" in rendered
    assert "Direction changes" in rendered
    assert "Sensitive to zoning" in rendered
    assert "Stable" in rendered
    assert "<script>" not in rendered

    # Retired technical presentation is not exposed in the biological report.
    assert "Cross-grain sensitivity" not in rendered
    assert "|Δ effect|" not in rendered
    assert "Transition" not in rendered
    assert "Comparison" not in rendered
    assert "Random-tessellation diagnostic" not in rendered
