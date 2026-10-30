"""
Pure functions for global threshold computation.

ARCHITECTURE:
=============
- Pure functions: compute_global_thresholds() with no hidden state
- Immutable dataclass: PercentileThresholds
- Single source of truth: compute_global_thresholds()

USAGE PATTERN:
==============
    thresholds = compute_global_thresholds(
        gene1_values=expression_df[GENE1].values,
        gene2_values=expression_df[GENE2].values,
        q_percentile=0.75,
        q_low_percentile=0.25,
        min_points_threshold=20
    )
    # Returns immutable PercentileThresholds dataclass

⚠️  DO NOT:
- Compute thresholds per-ROI (breaks enrichment semantics - "high" must be consistent)
- Use np.percentile() directly on gene data

THRESHOLD vs BASELINE SEPARATION:
==================================
- Thresholds (GLOBAL): Define "high expression" consistently (75th percentile across dataset)
- Baselines (PER-ROI): Define "normal" for each ROI (proportion high in its context/other ROIs)

Enrichment compares: (ROI proportion) / (ROI's context proportion) using GLOBAL threshold

SCALE NORMALIZATION:
====================
For log2fc scale normalization parameters, see metrics/normalization.py (ScaleNormalizationParams).
"""

from typing import Dict
import numpy as np
from dataclasses import dataclass

from tilescope.defaults import (
    DEFAULT_MIN_TILE_SPOTS,
    DEFAULT_Q_PERCENTILE,
    DEFAULT_Q_LOW_PERCENTILE
)


@dataclass(frozen=True)
class PercentileThresholds:
    """
    Immutable container for global analysis thresholds.
    Computed ONCE across entire dataset to ensure all analysis units use
    consistent definitions of "high" and "low" expression.
    DOES NOT contain baseline proportions - those must be computed per-ROI
    from context data to answer: "Is this unit enriched vs other ROIs?"
    """
    q_g1_threshold: float  # Gene1 high percentile value (e.g., 75th percentile)
    q_g2_threshold: float  # Gene2 high percentile value
    q_percentile: float    # High percentile used (e.g., 0.75)

    # Low thresholds (e.g., 25th percentile for co-depletion analysis)
    q_g1_low_threshold: float  # Gene1 low percentile value (e.g., 25th percentile)
    q_g2_low_threshold: float  # Gene2 low percentile value
    q_low_percentile: float    # Low percentile used (e.g., 0.25)

    # Count threshold (sample size filtering)
    min_points_threshold: int = DEFAULT_MIN_TILE_SPOTS  # Minimum spots per tile/hexagon (Voronoi + consensus)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict format - threshold values only (no baseline proportions)."""
        return {
            'q_g1_threshold': self.q_g1_threshold,
            'q_g2_threshold': self.q_g2_threshold,
            'q_percentile': self.q_percentile,
            'q_g1_low_threshold': self.q_g1_low_threshold,
            'q_g2_low_threshold': self.q_g2_low_threshold,
            'q_low_percentile': self.q_low_percentile,
        }

    def __repr__(self) -> str:
        """User-friendly representation."""
        pct_high = int(self.q_percentile * 100)
        pct_low = int(self.q_low_percentile * 100)

        return (
            f"PercentileThresholds(high={pct_high}%, low={pct_low}%, min_spots={self.min_points_threshold})\n"
            f"  HIGH (≥ {pct_high}%):\n"
            f"    Gene1 ≥ {self.q_g1_threshold:.2f}\n"
            f"    Gene2 ≥ {self.q_g2_threshold:.2f}\n"
            f"  LOW (≤ {pct_low}%):\n"
            f"    Gene1 ≤ {self.q_g1_low_threshold:.2f}\n"
            f"    Gene2 ≤ {self.q_g2_low_threshold:.2f}\n"
            f"  COUNT FILTER:\n"
            f"    Min spots per tile/hexagon: {self.min_points_threshold}"
        )


def compute_global_thresholds(
    gene1_values: np.ndarray,
    gene2_values: np.ndarray,
    q_percentile: float = DEFAULT_Q_PERCENTILE,
    q_low_percentile: float = DEFAULT_Q_LOW_PERCENTILE,
    min_points_threshold: int = DEFAULT_MIN_TILE_SPOTS
) -> PercentileThresholds:
    """Compute global thresholds from entire dataset (high and low percentiles)."""
    if len(gene1_values) != len(gene2_values):
        raise ValueError(
            "Gene arrays must have same length: "
            f"{len(gene1_values)} != {len(gene2_values)}"
        )
    high_pct = q_percentile * 100
    low_pct = q_low_percentile * 100

    # Return immutable dataclass (threshold values only, no baselines)
    return PercentileThresholds(
        q_g1_threshold=float(np.percentile(gene1_values, high_pct)),
        q_g2_threshold=float(np.percentile(gene2_values, high_pct)),
        q_percentile=q_percentile,
        q_g1_low_threshold=float(np.percentile(gene1_values, low_pct)),
        q_g2_low_threshold=float(np.percentile(gene2_values, low_pct)),
        q_low_percentile=q_low_percentile,
        min_points_threshold=min_points_threshold
    )


# Export dataclass and pure function for use in other modules
__all__ = ['PercentileThresholds', 'compute_global_thresholds']
