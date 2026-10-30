"""Sample/feature authorities for spot/cell parent-ROI marginals."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from tilescope.cols import ROI_ID
from tilescope.metrics.pair_stats_core import PairStatsCore


@dataclass(frozen=True)
class ParentRoiSupportIndex:
    """Canonical target/context source-row positions, once per sample."""

    roi_ids: tuple[str, ...]
    target_positions_by_roi: Mapping[str, tuple[int, ...]]
    context_positions_by_roi: Mapping[str, tuple[int, ...]]
    n_source_rows: int


@dataclass(frozen=True)
class FeatureParentRoiMarginal:
    """One feature's reusable marginal statistics for one parent ROI."""

    target_sum: Optional[float]
    target_high: Optional[int]
    target_low: Optional[int]
    context_sum: Optional[float]
    context_high: Optional[int]
    context_low: Optional[int]
    log2fc: Optional[float]
    cles: Optional[float]


@dataclass(frozen=True)
class FeatureParentRoiProjection:
    """Per-feature spot/cell values and parent-ROI marginal calculations."""

    feature_name: str
    source_values: np.ndarray
    support_index: ParentRoiSupportIndex
    marginals_by_roi: Mapping[str, FeatureParentRoiMarginal]
    min_roi_spots: int
    min_context_spots: int

    def __post_init__(self) -> None:
        values = np.asarray(self.source_values, dtype=float)
        if values.ndim != 1 or values.size != self.support_index.n_source_rows:
            raise ValueError(
                f"FeatureParentRoiProjection({self.feature_name!r}): source "
                f"shape {values.shape!r} disagrees with support size "
                f"{self.support_index.n_source_rows}"
            )
        if values.flags.writeable:
            values = np.array(values, dtype=float, copy=True)
            values.flags.writeable = False
        object.__setattr__(self, "source_values", values)
        if set(self.marginals_by_roi) != set(self.support_index.roi_ids):
            raise ValueError(
                f"FeatureParentRoiProjection({self.feature_name!r}): ROI "
                "marginal keys disagree with the support index"
            )
        if not isinstance(self.marginals_by_roi, MappingProxyType):
            object.__setattr__(
                self, "marginals_by_roi",
                MappingProxyType(dict(self.marginals_by_roi)),
            )

    def target_values(self, roi_id: str) -> np.ndarray:
        return self.source_values[
            list(self.support_index.target_positions_by_roi[str(roi_id)])
        ]


def build_parent_roi_support_index(
    source_expression_df: pd.DataFrame,
) -> ParentRoiSupportIndex:
    """Resolve the legacy sort-index target/context order once per sample."""
    if ROI_ID not in source_expression_df.columns:
        raise ValueError(f"parent ROI support index requires {ROI_ID!r}")
    positions = pd.DataFrame(
        {
            "_position": np.arange(len(source_expression_df), dtype=np.int64),
            "_roi": source_expression_df[ROI_ID].astype(str).to_numpy(),
        },
        index=source_expression_df.index,
    )
    roi_ids = tuple(sorted(positions["_roi"].unique()))
    target = {}
    context = {}
    for roi_id in roi_ids:
        target[roi_id] = tuple(
            positions.loc[positions["_roi"] == roi_id]
            .sort_index()["_position"]
            .astype(int)
            .tolist()
        )
        context[roi_id] = tuple(
            positions.loc[positions["_roi"] != roi_id]
            .sort_index()["_position"]
            .astype(int)
            .tolist()
        )
    return ParentRoiSupportIndex(
        roi_ids=roi_ids,
        target_positions_by_roi=MappingProxyType(target),
        context_positions_by_roi=MappingProxyType(context),
        n_source_rows=len(source_expression_df),
    )


def build_feature_parent_roi_projection(
    *,
    feature_name: str,
    source_values: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    support_index: ParentRoiSupportIndex,
    gene_min_value: float,
    min_roi_spots: int,
    min_context_spots: int,
) -> FeatureParentRoiProjection:
    """Compute sums/counts/logFC/CLES once for one feature x parent ROI."""
    values = np.asarray(source_values, dtype=float)
    if values.ndim != 1 or values.size != support_index.n_source_rows:
        raise ValueError(
            f"feature {feature_name!r} spot/cell values disagree with parent ROI "
            "support"
        )
    core = PairStatsCore()
    marginals = {}
    for roi_id in support_index.roi_ids:
        target_positions = support_index.target_positions_by_roi[roi_id]
        context_positions = support_index.context_positions_by_roi[roi_id]
        n_target = len(target_positions)
        n_context = len(context_positions)
        if n_target < min_roi_spots:
            marginals[roi_id] = FeatureParentRoiMarginal(
                None, None, None, None, None, None, None, None,
            )
            continue
        target_values = values[list(target_positions)]
        target_sum = float(target_values.sum())
        target_high = int((target_values > high_threshold).sum())
        target_low = int((target_values < low_threshold).sum())
        if n_context < min_context_spots:
            marginals[roi_id] = FeatureParentRoiMarginal(
                target_sum, target_high, target_low,
                None, None, None, None, None,
            )
            continue
        context_values = values[list(context_positions)]
        marginals[roi_id] = FeatureParentRoiMarginal(
            target_sum=target_sum,
            target_high=target_high,
            target_low=target_low,
            context_sum=float(context_values.sum()),
            context_high=int((context_values > high_threshold).sum()),
            context_low=int((context_values < low_threshold).sum()),
            log2fc=float(core.log2fc(
                target_values, context_values, gene_min_value,
            )),
            cles=float(core.cles(target_values, context_values)),
        )
    return FeatureParentRoiProjection(
        feature_name=feature_name,
        source_values=values,
        support_index=support_index,
        marginals_by_roi=MappingProxyType(marginals),
        min_roi_spots=int(min_roi_spots),
        min_context_spots=int(min_context_spots),
    )
