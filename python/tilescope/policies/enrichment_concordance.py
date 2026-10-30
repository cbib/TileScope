"""
Individual gene enrichment classification policy.

Classifies individual gene enrichment status (enriched/depleted/neutral/unknown)
based on enrichment ratio thresholds (FC and CLES).
"""

from dataclasses import dataclass

import numpy as np

from tilescope.enums import GeneVerdict
from tilescope.defaults import (
    DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD,
    DEFAULT_FC_WEAK_ENRICHMENT_THRESHOLD,
    DEFAULT_FC_WEAK_DEPLETION_THRESHOLD,
    DEFAULT_FC_STRONG_DEPLETION_THRESHOLD,
    DEFAULT_CLES_STRONG_ENRICHMENT_THRESHOLD,
    DEFAULT_CLES_WEAK_ENRICHMENT_THRESHOLD,
    DEFAULT_CLES_WEAK_DEPLETION_THRESHOLD,
    DEFAULT_CLES_STRONG_DEPLETION_THRESHOLD,
)


@dataclass(frozen=True)
class ConcordancePolicy:
    """Policy for classifying individual gene enrichment with 5-level granularity.

    Classifies FC and CLES into: strong_enriched, weak_enriched, neutral, weak_depleted, strong_depleted.
    Provides strength-aware verdicts for concordance checking.
    """
    # FC thresholds (5-level classification)
    fc_strong_enrich_threshold: float = DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD  # FC >= 1.5 → strong enriched
    fc_weak_enrich_threshold: float = DEFAULT_FC_WEAK_ENRICHMENT_THRESHOLD      # FC >= 1.2 → weak enriched
    fc_weak_deplete_threshold: float = DEFAULT_FC_WEAK_DEPLETION_THRESHOLD      # FC <= 0.8 → weak depleted
    fc_strong_deplete_threshold: float = DEFAULT_FC_STRONG_DEPLETION_THRESHOLD  # FC <= 0.67 → strong depleted

    # CLES thresholds (5-level classification)
    cles_strong_enrich_threshold: float = DEFAULT_CLES_STRONG_ENRICHMENT_THRESHOLD  # CLES >= 0.6 → strong enriched
    cles_weak_enrich_threshold: float = DEFAULT_CLES_WEAK_ENRICHMENT_THRESHOLD      # CLES >= 0.55 → weak enriched
    cles_weak_deplete_threshold: float = DEFAULT_CLES_WEAK_DEPLETION_THRESHOLD      # CLES <= 0.45 → weak depleted
    cles_strong_deplete_threshold: float = DEFAULT_CLES_STRONG_DEPLETION_THRESHOLD  # CLES <= 0.4 → strong depleted

    def classify_fc(self, enrichment_ratio: float) -> str:
        """Classify FC enrichment with 5-level granularity."""
        if np.isnan(enrichment_ratio) or not np.isfinite(enrichment_ratio):
            return GeneVerdict.UNKNOWN.value

        if enrichment_ratio >= self.fc_strong_enrich_threshold:
            return GeneVerdict.STRONG_ENRICHED.value
        elif enrichment_ratio >= self.fc_weak_enrich_threshold:
            return GeneVerdict.WEAK_ENRICHED.value
        elif enrichment_ratio <= self.fc_strong_deplete_threshold:
            return GeneVerdict.STRONG_DEPLETED.value
        elif enrichment_ratio <= self.fc_weak_deplete_threshold:
            return GeneVerdict.WEAK_DEPLETED.value
        else:
            return GeneVerdict.NEUTRAL.value

    def classify_cles(self, cles: float) -> str:
        """Classify CLES with 5-level granularity."""
        if cles is None or np.isnan(cles) or not np.isfinite(cles):
            return GeneVerdict.UNKNOWN.value

        if cles >= self.cles_strong_enrich_threshold:
            return GeneVerdict.STRONG_ENRICHED.value
        elif cles >= self.cles_weak_enrich_threshold:
            return GeneVerdict.WEAK_ENRICHED.value
        elif cles <= self.cles_strong_deplete_threshold:
            return GeneVerdict.STRONG_DEPLETED.value
        elif cles <= self.cles_weak_deplete_threshold:
            return GeneVerdict.WEAK_DEPLETED.value
        else:
            return GeneVerdict.NEUTRAL.value

    def format_enrichment_direction(self,
                                    gene1_enrichment: float,
                                    gene2_enrichment: float) -> str:
        """Format individual gene verdicts as 'g1:{status},g2:{status}'."""
        g1_status = self.classify_fc(gene1_enrichment)
        g2_status = self.classify_fc(gene2_enrichment)

        return f"g1:{g1_status},g2:{g2_status}"

    def __repr__(self) -> str:
        """User-friendly representation."""
        return (
            f"ConcordancePolicy(\n"
            f"  FC: strong_enrich>={self.fc_strong_enrich_threshold}, weak_enrich>={self.fc_weak_enrich_threshold}, "
            f"weak_deplete<={self.fc_weak_deplete_threshold}, strong_deplete<={self.fc_strong_deplete_threshold}\n"
            f"  CLES: strong_enrich>={self.cles_strong_enrich_threshold}, weak_enrich>={self.cles_weak_enrich_threshold}, "
            f"weak_deplete<={self.cles_weak_deplete_threshold}, strong_deplete<={self.cles_strong_deplete_threshold}\n"
            f")"
        )


# Default policy instance for convenience
DEFAULT_CONCORDANCE_POLICY = ConcordancePolicy()
