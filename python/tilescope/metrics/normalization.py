"""
Scale-specific normalization parameters for log2 fold change computation.

SEPARATION OF CONCERNS:
=======================
- threshold_manager.py: Defines "high"/"low" thresholds for enrichment classification
- metrics/normalization.py: Defines shift parameters for log2fc effect size scaling
- metrics/enrichment.py: Pure functions for computing enrichment ratios and log2fc

ARCHITECTURE:
=============
- Immutable dataclass: ScaleNormalizationParams
- Scale-specific parameters: spot vs density (Voronoi tiles + consensus hexagons)
- Unified field names: NORM_GENE1_MIN, NORM_GENE2_MIN (from defaults.py)

USAGE PATTERN:
==============
    # Compute normalization params once across entire dataset
    norm_params = ScaleNormalizationParams(
        gene1_min_spot=float(expression_df[GENE1].min()),
        gene2_min_spot=float(expression_df[GENE2].min()),
        gene1_min_density=float(all_tile_densities[GENE1_DENSITY].min()),
        gene2_min_density=float(all_tile_densities[GENE2_DENSITY].min())
    )

    # Use in log2fc computation (spot scale)
    spot_params = norm_params.to_dict_spot()
    log2fc_g1 = compute_log2fc(mu_roi, mu_ctx, **spot_params)

    # Use in log2fc computation (density scale: Voronoi + consensus)
    density_params = norm_params.to_dict_density()
    log2fc_g1 = compute_log2fc(mu_roi, mu_ctx, **density_params)

LOG2FC FORMULA:
===============
    log2fc = log2((mu_roi + lambda) / (mu_ctx + lambda))
    where lambda = max(eps, -gene_min_scale + eps)

    The shift parameter lambda prevents:
    - Division by zero when mu_ctx = 0
    - Log of zero when mu_roi = 0
    - Negative values in log domain
"""

from typing import Dict
from dataclasses import dataclass

from tilescope.defaults import NORM_GENE1_MIN, NORM_GENE2_MIN


@dataclass(frozen=True)
class ScaleNormalizationParams:
    """
    Immutable container for scale-specific normalization parameters for log2fc computation.

    Computed ONCE across entire dataset to ensure consistent shift parameters
    for log2 fold change calculation at different spatial scales.

    SEPARATION OF CONCERNS:
    - PercentileThresholds (threshold_manager.py): Define "high"/"low" for enrichment classification
    - ScaleNormalizationParams (metrics/normalization.py): Define shift parameters for log2fc effect sizes

    USAGE: log2fc = log2((mu_roi + lam) / (mu_ctx + lam))
           where lam = max(eps, -gene_min_scale + eps)

    FIELD NAMES: Unified to NORM_GENE1_MIN and NORM_GENE2_MIN
    VALUES: Scale-specific (spot vs density) accessed via to_dict_spot() / to_dict_density()
    """
    gene1_min_spot: float       # Minimum spot expression for gene1 across dataset
    gene2_min_spot: float       # Minimum spot expression for gene2 across dataset
    gene1_min_density: float    # Minimum tile density for gene1 (across all tessellations: Voronoi + consensus)
    gene2_min_density: float    # Minimum tile density for gene2 (across all tessellations: Voronoi + consensus)

    def to_dict_spot(self) -> Dict[str, float]:
        """Convert to dict format for spot-level compute functions (unified field names)."""
        return {
            NORM_GENE1_MIN: self.gene1_min_spot,
            NORM_GENE2_MIN: self.gene2_min_spot,
        }

    def to_dict_density(self) -> Dict[str, float]:
        """Convert to dict format for density-level compute functions (unified field names)."""
        return {
            NORM_GENE1_MIN: self.gene1_min_density,
            NORM_GENE2_MIN: self.gene2_min_density,
        }

    def __repr__(self) -> str:
        """User-friendly representation."""
        return (
            f"ScaleNormalizationParams(\n"
            f"  SPOT SCALE:\n"
            f"    Gene1 min: {self.gene1_min_spot:.4f}\n"
            f"    Gene2 min: {self.gene2_min_spot:.4f}\n"
            f"  DENSITY SCALE (Voronoi + Consensus):\n"
            f"    Gene1 min: {self.gene1_min_density:.4f}\n"
            f"    Gene2 min: {self.gene2_min_density:.4f}\n"
            f")"
        )


# Export for use in other modules
__all__ = ['ScaleNormalizationParams']
