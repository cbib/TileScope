"""Biologist-facing interpretation of spatial-support sensitivity.

The canonical ``roi_cross_grain_sensitivity.csv`` remains the continuous
scientific record.  This module adds a compact report interpretation using the
same distinction made in the MAUP literature:

* zoning sensitivity is variation across alternative boundaries at a fixed
  configured scale;
* direction change is a spot/cell versus consensus-ensemble comparison.

References:

* Jelinski & Wu (1996): https://doi.org/10.1007/BF02447512
* Fotheringham & Wong (1991): https://doi.org/10.1068/a231025
* Zormpas et al. (2023), spatial transcriptomics:
  https://doi.org/10.1016/j.cell.2023.11.003
* Duque et al. (2018), a specialised univariate MAUP test:
  https://doi.org/10.1371/journal.pone.0207377

The effect floors below are pragmatic display safeguards, not significance
thresholds or literature-defined universal cutoffs.  They prevent tiny changes
around the null from being presented as stable biological directions or
direction changes.  Continuous values remain available in the exported CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tilescope.analyzers.parent_roi_outcomes import (
    ParentMoranStatus,
    ParentReplicateCorrelationStatus,
)
from tilescope.cols import (
    CORRELATION_VALUE,
    GENE1_ENRICHMENT,
    GENE1_ENRICHMENT_STATUS,
    GENE2_ENRICHMENT,
    GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC,
    JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC,
    JOINT_LOW_ENRICHMENT_STATUS,
    MORAN_DIRECTION,
    MORAN_VALUE_P25_ACROSS_REPLICATES,
    MORAN_VALUE_P75_ACROSS_REPLICATES,
    PARENT_MORAN_STATUS,
    PARENT_REPLICATE_CORRELATION_STATUS,
    ROI_ID,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus
from tilescope.schema.helpers import natural_sort_key


class SpatialSupportReading(str, Enum):
    STABLE = "Stable"
    ZONING_SENSITIVE = "Sensitive to zoning"
    DIRECTION_CHANGE = "Direction changes"
    NO_CLEAR_PATTERN = "No clear pattern"
    NOT_ASSESSABLE = "Not assessable"


# Operational display floors on the null-centred effect scale.  These are
# deliberately local to the report interpretation and do not alter or annotate
# any canonical analysis output.
_PRESENCE_FLOOR = {
    "correlation": 0.15,
    "enrichment": 0.08,
    "moran": 0.10,
}
_MIN_TESSELLATIONS = 3

_HEADLINE_LANES = (
    ("correlation", "", ""),
    ("enrichment", "joint_high", ""),
    ("moran", "", "g1_lag_g2"),
    ("moran", "", "g2_lag_g1"),
)


@dataclass(frozen=True)
class SupportSensitivityResult:
    roi_id: str
    g1_id: str
    g2_id: str
    statistic: str
    channel: str
    moran_direction: str
    reading: SpatialSupportReading
    source_raw: Optional[float]
    source_status: str
    consensus_raw: Optional[float]
    consensus_status: str
    tessellation_q25_raw: Optional[float]
    tessellation_q75_raw: Optional[float]


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def classify_spatial_support(
    *,
    statistic: str,
    source_effect: object,
    tessellation_effect: object,
    consensus_effect: object,
    tessellation_effect_iqr: object,
    n_tessellations: int,
) -> SpatialSupportReading:
    """Assign one compact interpretation without conflating MAUP axes.

    A direction change is assessed only between spot/cell and consensus-
    ensemble effects.  Zoning sensitivity is assessed only from the spread
    across alternative tessellations.  Stable requires a clear common direction
    at all three reported supports.  Weak or unavailable evidence stays neutral.
    """
    source = _finite(source_effect)
    consensus = _finite(consensus_effect)
    if source is None or consensus is None:
        return SpatialSupportReading.NOT_ASSESSABLE

    floor = _PRESENCE_FLOOR[statistic]
    source_clear = abs(source) >= floor
    consensus_clear = abs(consensus) >= floor
    if source_clear and consensus_clear and np.sign(source) != np.sign(consensus):
        return SpatialSupportReading.DIRECTION_CHANGE

    tessellation = _finite(tessellation_effect)
    tessellation_iqr = _finite(tessellation_effect_iqr)
    if (
        n_tessellations >= _MIN_TESSELLATIONS
        and tessellation_iqr is not None
        and tessellation_iqr >= floor
    ):
        return SpatialSupportReading.ZONING_SENSITIVE

    evidence = [value for value in (source, tessellation, consensus)
                if value is not None]
    if not evidence or max(abs(value) for value in evidence) < floor:
        return SpatialSupportReading.NO_CLEAR_PATTERN

    if tessellation is None:
        return SpatialSupportReading.NOT_ASSESSABLE
    if all(abs(value) >= floor for value in (source, tessellation, consensus)):
        signs = {int(np.sign(value)) for value in (source, tessellation, consensus)}
        if len(signs) == 1:
            return SpatialSupportReading.STABLE

    return SpatialSupportReading.NO_CLEAR_PATTERN


def _normalise_optional_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _replicate_interval(
    correlation_replicates: pd.DataFrame,
    enrichment_replicates: pd.DataFrame,
    tessellation_moran: pd.DataFrame,
    roi_id: str,
    statistic: str,
    channel: str,
    moran_direction: str,
) -> tuple[Optional[float], Optional[float]]:
    """Return natural-scale tessellation quartiles for display."""
    if statistic == "correlation":
        frame = correlation_replicates
        if frame.empty:
            return None, None
        rows = frame[
            (frame[ROI_ID].astype(str) == roi_id)
            & (frame[PARENT_REPLICATE_CORRELATION_STATUS]
               == ParentReplicateCorrelationStatus.FINITE.value)
        ]
        values = pd.to_numeric(rows[CORRELATION_VALUE], errors="coerce").dropna()
        if values.empty:
            return None, None
        q25, q75 = np.quantile(values.to_numpy(dtype=float), [0.25, 0.75])
        return float(q25), float(q75)

    if statistic == "enrichment":
        value_col, status_col = {
            "g1": (GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS),
            "g2": (GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS),
            "joint_high": (JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS),
            "joint_low": (JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS),
        }[channel]
        frame = enrichment_replicates
        if frame.empty:
            return None, None
        rows = frame[
            (frame[ROI_ID].astype(str) == roi_id)
            & (frame[status_col] == EnrichmentRatioStatus.FINITE.value)
        ]
        values = pd.to_numeric(rows[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return None, None
        raw_q25, raw_q75 = np.quantile(values, [0.25, 0.75])
        return float(raw_q25), float(raw_q75)

    frame = tessellation_moran
    if frame.empty:
        return None, None
    rows = frame[
        (frame[ROI_ID].astype(str) == roi_id)
        & (frame[MORAN_DIRECTION] == moran_direction)
        & (frame[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value)
    ]
    if len(rows) != 1:
        return None, None
    row = rows.iloc[0]
    q25 = _finite(row[MORAN_VALUE_P25_ACROSS_REPLICATES])
    q75 = _finite(row[MORAN_VALUE_P75_ACROSS_REPLICATES])
    return q25, q75


def load_spatial_support_results(output_dir: Path) -> list[SupportSensitivityResult]:
    """Read one pair-run's canonical outputs and interpret headline lanes."""
    cross_path = output_dir / "roi_cross_grain_sensitivity.csv"
    if not cross_path.exists():
        return []
    cross = pd.read_csv(cross_path)
    if cross.empty:
        return []
    correlation_replicates = _read_optional(
        output_dir / "roi_replicate_correlations.csv"
    )
    enrichment_replicates = _read_optional(
        output_dir / "roi_replicate_enrichment.csv"
    )
    tessellation_moran = _read_optional(
        output_dir / "roi_tessellation_moran.csv"
    )

    results: list[SupportSensitivityResult] = []
    for roi_id in sorted(cross[ROI_ID].astype(str).unique(), key=natural_sort_key):
        roi_rows = cross[cross[ROI_ID].astype(str) == roi_id]
        first = roi_rows.iloc[0]
        for statistic, channel, direction in _HEADLINE_LANES:
            lane = roi_rows[
                (roi_rows["statistic"] == statistic)
                & (roi_rows["channel"].map(_normalise_optional_key) == channel)
                & (roi_rows["moran_direction"].map(_normalise_optional_key) == direction)
            ]
            tessellation = lane[lane["target_surface"] == "tessellation_median"]
            consensus = lane[lane["target_surface"] == "consensus_ensemble"]
            if len(tessellation) != 1 or len(consensus) != 1:
                continue
            tess_row = tessellation.iloc[0]
            consensus_row = consensus.iloc[0]
            raw_q25, raw_q75 = _replicate_interval(
                correlation_replicates,
                enrichment_replicates,
                tessellation_moran,
                roi_id,
                statistic,
                channel,
                direction,
            )
            n_replicates = _finite(tess_row["n_comparable_replicates"])
            reading = classify_spatial_support(
                statistic=statistic,
                source_effect=consensus_row["reference_effect_value"],
                tessellation_effect=tess_row["target_effect_value"],
                consensus_effect=consensus_row["target_effect_value"],
                tessellation_effect_iqr=tess_row["tessellation_effect_iqr"],
                n_tessellations=int(n_replicates or 0),
            )
            results.append(SupportSensitivityResult(
                roi_id=roi_id,
                g1_id=str(first["g1_id"]),
                g2_id=str(first["g2_id"]),
                statistic=statistic,
                channel=channel,
                moran_direction=direction,
                reading=reading,
                source_raw=_finite(consensus_row["reference_raw_value"]),
                source_status=str(consensus_row["reference_raw_status"]),
                consensus_raw=_finite(consensus_row["target_raw_value"]),
                consensus_status=str(consensus_row["target_raw_status"]),
                tessellation_q25_raw=raw_q25,
                tessellation_q75_raw=raw_q75,
            ))
    return results
