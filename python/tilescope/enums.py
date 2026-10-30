"""Categorical constants for live statistic and geometry contracts.

Primary API: Type-safe Enum classes
===================================
Use Enum.MEMBER.value at I/O boundaries (DataFrames, dicts, Pandera validators).

Example::

    from tilescope.enums import CorrelationMode

    def analyze(mode: str = CorrelationMode.PEARSON.value):
        if mode == CorrelationMode.PEARSON.value:
            ...

Utility Collections
===================
Derived collections for iteration/validation (NOT aliases):
- TESSELLATION_METHODS: tuple of all GridMethod values
- GENE_VERDICT_LABELS: tuple of all GeneVerdict values

These appear in data *cells* but are NOT DataFrame column names
(column names live in cols.py).
"""

from enum import Enum
from typing import Final, Tuple


# =============================================================================
# CORRELATION / SCORING MODES
# =============================================================================

class CorrelationMode(str, Enum):
    """Correlation computation modes"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    NAIVE = "naive"
    ADAPTIVE = "adaptive"


# =============================================================================
# TESSELLATION / GRID METHODS
# =============================================================================

class GridMethod(str, Enum):
    """Spatial tessellation/grid generation methods"""
    VORONOI = "voronoi"
    HEXAGONAL = "hexagonal"


TESSELLATION_METHODS: Tuple[str, ...] = tuple(g.value for g in GridMethod)


# =============================================================================
# GENE ENRICHMENT VERDICTS
# =============================================================================

class GeneVerdict(str, Enum):
    """Individual gene enrichment verdicts (5-level granular classification).

    Provides strength-aware verdicts for enrichment metrics (FC, CLES, LOG2FC).
    Direction and strength are encoded in single verdict for concordance checking.
    """
    STRONG_ENRICHED = "strong_enriched"
    WEAK_ENRICHED = "weak_enriched"
    NEUTRAL = "neutral"
    WEAK_DEPLETED = "weak_depleted"
    STRONG_DEPLETED = "strong_depleted"
    UNKNOWN = "unknown"


GENE_VERDICT_LABELS: Tuple[str, ...] = tuple(v.value for v in GeneVerdict)


# =============================================================================
# ENRICHMENT DIRECTION PARSING CONSTANTS (string literals, NOT enums)
# =============================================================================

ENRICHMENT_G1_PREFIX: Final[str] = "g1:"
ENRICHMENT_G2_PREFIX: Final[str] = "g2:"
ENRICHMENT_PAIR_SEPARATOR: Final[str] = ","
ENRICHMENT_VALUE_SEPARATOR: Final[str] = ":"


# =============================================================================
# ENRICHMENT DIRECTION PARSING UTILITIES
# =============================================================================

def parse_enrichment_direction(direction_str: str) -> dict:
    """Parse enrichment_direction string into gene verdicts dictionary.

    Parses format "g1:enriched,g2:depleted" into {'g1': 'enriched', 'g2': 'depleted'}.

    Args:
        direction_str: Enrichment direction string (e.g., "g1:enriched,g2:depleted")

    Returns:
        Dictionary with 'g1' and 'g2' keys mapping to verdict strings.
        Returns {'g1': 'unknown', 'g2': 'unknown'} for invalid/missing input.

    Example:
        >>> parse_enrichment_direction("g1:enriched,g2:depleted")
        {'g1': 'enriched', 'g2': 'depleted'}
    """
    # Handle invalid input
    if not direction_str or direction_str == "unknown" or not isinstance(direction_str, str):
        return {'g1': GeneVerdict.UNKNOWN.value, 'g2': GeneVerdict.UNKNOWN.value}

    # Parse "g1:enriched,g2:depleted" format
    parts = direction_str.split(ENRICHMENT_PAIR_SEPARATOR)
    result = {}

    for part in parts:
        if ENRICHMENT_VALUE_SEPARATOR not in part:
            continue
        gene, verdict = part.split(ENRICHMENT_VALUE_SEPARATOR, 1)
        result[gene.strip()] = verdict.strip()

    # Ensure both genes are present (fill missing with unknown)
    if 'g1' not in result:
        result['g1'] = GeneVerdict.UNKNOWN.value
    if 'g2' not in result:
        result['g2'] = GeneVerdict.UNKNOWN.value

    return result


def get_gene_verdict(direction_str: str, gene: str) -> str:
    """Extract verdict for specific gene from enrichment_direction string.

    Args:
        direction_str: Enrichment direction string (e.g., "g1:enriched,g2:depleted")
        gene: Gene identifier ('g1' or 'g2')

    Returns:
        Verdict string (enriched/depleted/neutral/unknown)

    Example:
        >>> get_gene_verdict("g1:enriched,g2:depleted", "g1")
        'enriched'
    """
    verdicts = parse_enrichment_direction(direction_str)
    return verdicts.get(gene, GeneVerdict.UNKNOWN.value)


__all__ = [
    # Enums (primary API)
    "CorrelationMode",
    "GridMethod",
    "GeneVerdict",
    # Utility collections and helpers (derived from Enums)
    "TESSELLATION_METHODS",
    "GENE_VERDICT_LABELS",

    # Enrichment parsing constants
    "ENRICHMENT_G1_PREFIX",
    "ENRICHMENT_G2_PREFIX",
    "ENRICHMENT_PAIR_SEPARATOR",
    "ENRICHMENT_VALUE_SEPARATOR",

    # Enrichment parsing utilities
    "parse_enrichment_direction",
    "get_gene_verdict",
]
