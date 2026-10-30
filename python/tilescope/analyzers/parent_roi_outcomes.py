"""Typed availability statuses for canonical ROI-grain outputs."""
from __future__ import annotations

from enum import Enum


class ParentCorrelationStatus(str, Enum):
    FINITE = "finite"
    INSUFFICIENT_ROI_SPOTS = "insufficient_roi_spots"
    ZERO_GENE1_VARIANCE = "zero_gene1_variance"
    ZERO_GENE2_VARIANCE = "zero_gene2_variance"
    ZERO_GENE1_AND_GENE2_VARIANCE = "zero_gene1_and_gene2_variance"
    UNDEFINED_CORRELATION_OTHER = "undefined_correlation_other"


class ParentEnrichmentStatus(str, Enum):
    FINITE = "finite"
    INSUFFICIENT_ROI_SPOTS = "insufficient_roi_spots"
    INSUFFICIENT_CONTEXT_SPOTS = "insufficient_context_spots"
    CONTEXT_PROPORTION_ZERO = "context_proportion_zero"


class ParentEffectSizeStatus(str, Enum):
    FINITE = "finite"
    INSUFFICIENT_ROI_SPOTS = "insufficient_roi_spots"
    INSUFFICIENT_CONTEXT_SPOTS = "insufficient_context_spots"


class ParentTessellationCorrelationStatus(str, Enum):
    FINITE = "finite"
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_VALID_REPLICATES = "no_valid_replicates"


class ParentReplicateCorrelationStatus(str, Enum):
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_CONTRIBUTING_COMPONENTS = "no_contributing_components"
    INCOMPLETE_SUPPORT_COVERAGE = "incomplete_support_coverage"
    INSUFFICIENT_TILES = "insufficient_tiles"
    ZERO_GENE1_VARIANCE = "zero_gene1_variance"
    ZERO_GENE2_VARIANCE = "zero_gene2_variance"
    ZERO_GENE1_AND_GENE2_VARIANCE = "zero_gene1_and_gene2_variance"
    FINITE = "finite"
    UNDEFINED_CORRELATION_OTHER = "undefined_correlation_other"


class ParentReplicateEnrichmentStatus(str, Enum):
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    INCOMPLETE_SUPPORT_COVERAGE = "incomplete_support_coverage"
    NO_ELIGIBLE_CONTEXT = "no_eligible_context"
    INCOMPLETE_CONTEXT_COVERAGE = "incomplete_context_coverage"
    # Eligibility gates passed and context tiles were contributed, but
    # every context tile's paired density was non-finite (NaN): a
    # different fact from "no eligible context ever existed".
    ALL_CONTEXT_TILES_NON_FINITE = "all_context_tiles_non_finite"
    CONTEXT_PROPORTION_ZERO = "context_proportion_zero"
    FINITE = "finite"
    UNDEFINED_ENRICHMENT_OTHER = "undefined_enrichment_other"


class ParentReplicateEffectSizeStatus(str, Enum):
    FINITE = "finite"
    INSUFFICIENT_TARGET_UNITS = "insufficient_target_units"
    INSUFFICIENT_CONTEXT_UNITS = "insufficient_context_units"
    NOT_COMPUTED = "not_computed"


class ParentTessellationEnrichmentStatus(str, Enum):
    FINITE = "finite"
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_VALID_REPLICATES = "no_valid_replicates"
    PARTIAL_RATIO_AVAILABILITY = "partial_ratio_availability"


class ParentTessellationEffectSizeStatus(str, Enum):
    FINITE = "finite"
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_VALID_REPLICATES = "no_valid_replicates"


class ParentConsensusCorrelationStatus(str, Enum):
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_CONSENSUS_HEXES = "no_consensus_hexes"
    INCOMPLETE_SUPPORT_COVERAGE = "incomplete_support_coverage"
    INSUFFICIENT_FINITE_HEXES = "insufficient_finite_hexes"
    ZERO_GENE1_VARIANCE = "zero_gene1_variance"
    ZERO_GENE2_VARIANCE = "zero_gene2_variance"
    ZERO_GENE1_AND_GENE2_VARIANCE = "zero_gene1_and_gene2_variance"
    FINITE = "finite"
    UNDEFINED_CORRELATION_OTHER = "undefined_correlation_other"


class ParentConsensusEnrichmentStatus(str, Enum):
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_CONSENSUS_HEXES = "no_consensus_hexes"
    INCOMPLETE_SUPPORT_COVERAGE = "incomplete_support_coverage"
    NO_ELIGIBLE_CONTEXT = "no_eligible_context"
    INCOMPLETE_CONTEXT_COVERAGE = "incomplete_context_coverage"
    CONTEXT_PROPORTION_ZERO = "context_proportion_zero"
    FINITE = "finite"


class ParentConsensusEffectSizeStatus(str, Enum):
    FINITE = "finite"
    INSUFFICIENT_TARGET_HEXES = "insufficient_target_hexes"
    INSUFFICIENT_CONTEXT_HEXES = "insufficient_context_hexes"
    INCOMPLETE_SUPPORT_COVERAGE = "incomplete_support_coverage"
    INCOMPLETE_CONTEXT_COVERAGE = "incomplete_context_coverage"
    NOT_COMPUTED = "not_computed"


class ParentMoranStatus(str, Enum):
    """Generic Moran availability enum shared by every ROI-grain Moran surface."""
    FINITE = "finite"
    INSUFFICIENT_UNITS = "insufficient_units"
    ZERO_VARIANCE_A = "zero_variance_a"
    ZERO_VARIANCE_B = "zero_variance_b"
    ZERO_VARIANCE_A_AND_B = "zero_variance_a_and_b"
    NO_ADJACENCY_EDGES = "no_adjacency_edges"
    INSUFFICIENT_ADJACENCY_SUPPORT = "insufficient_adjacency_support"
    NO_SPATIAL_DISCRIMINATION = "no_spatial_discrimination"
    MORAN_DISABLED = "moran_disabled"
    NO_TESSELLATION_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
    NO_CONSENSUS_HEXES = "no_consensus_hexes"
    INCOMPLETE_SUPPORT_COVERAGE = "incomplete_support_coverage"
    NO_VALID_REPLICATES = "no_valid_replicates"
    UNDEFINED = "undefined"


_CORRELATION_REASON_MEMBER = {
    "gene1_constant": "ZERO_GENE1_VARIANCE",
    "gene2_constant": "ZERO_GENE2_VARIANCE",
    "both_constant": "ZERO_GENE1_AND_GENE2_VARIANCE",
}


def _correlation_status_from_reason(reason, status_type):
    member = _CORRELATION_REASON_MEMBER.get(
        getattr(reason, "value", str(reason)),
        "UNDEFINED_CORRELATION_OTHER",
    )
    return status_type[member]


def parent_correlation_status_from_reason(reason) -> ParentCorrelationStatus:
    return _correlation_status_from_reason(reason, ParentCorrelationStatus)


def parent_replicate_correlation_status_from_reason(
    reason,
) -> ParentReplicateCorrelationStatus:
    return _correlation_status_from_reason(
        reason, ParentReplicateCorrelationStatus,
    )


def parent_consensus_correlation_status_from_reason(
    reason,
) -> ParentConsensusCorrelationStatus:
    return _correlation_status_from_reason(
        reason, ParentConsensusCorrelationStatus,
    )


def parent_replicate_enrichment_status_from_partition(
    status,
) -> ParentReplicateEnrichmentStatus:
    value = getattr(status, "value", str(status))
    return {
        "finite": ParentReplicateEnrichmentStatus.FINITE,
        "context_proportion_zero": (
            ParentReplicateEnrichmentStatus.CONTEXT_PROPORTION_ZERO
        ),
        # Distinct from NO_ELIGIBLE_CONTEXT: upstream eligibility
        # gates passed and context tiles were contributed, but every
        # paired density was non-finite (NaN). Preserve the
        # distinction: "no eligible context ever existed" and "context
        # tiles were built but their densities failed pairwise
        # finiteness" are different facts.
        "insufficient_context_units": (
            ParentReplicateEnrichmentStatus.ALL_CONTEXT_TILES_NON_FINITE
        ),
    }.get(value, ParentReplicateEnrichmentStatus.UNDEFINED_ENRICHMENT_OTHER)


def parent_replicate_effect_size_status_from_partition(
    status,
) -> ParentReplicateEffectSizeStatus:
    return {
        "finite": ParentReplicateEffectSizeStatus.FINITE,
        "insufficient_target_units": (
            ParentReplicateEffectSizeStatus.INSUFFICIENT_TARGET_UNITS
        ),
        "insufficient_context_units": (
            ParentReplicateEffectSizeStatus.INSUFFICIENT_CONTEXT_UNITS
        ),
    }[getattr(status, "value", str(status))]
