"""Semantic checks for public enum wire values and parsing."""
from tilescope.enums import (
    CorrelationMode,
    GENE_VERDICT_LABELS,
    GridMethod,
    GeneVerdict,
    TESSELLATION_METHODS,
    get_gene_verdict,
    parse_enrichment_direction,
)


def test_enum_wire_vocabularies_and_derived_collections():
    expected = {
        CorrelationMode: ("pearson", "spearman", "naive", "adaptive"),
        GridMethod: ("voronoi", "hexagonal"),
        GeneVerdict: (
            "strong_enriched", "weak_enriched", "neutral",
            "weak_depleted", "strong_depleted", "unknown",
        ),
    }
    for enum_type, wire_values in expected.items():
        assert tuple(member.value for member in enum_type) == wire_values

    assert TESSELLATION_METHODS == expected[GridMethod]
    assert GENE_VERDICT_LABELS == expected[GeneVerdict]


def test_enrichment_direction_parsing_handles_complete_partial_and_invalid():
    assert parse_enrichment_direction("g1:enriched,g2:depleted") == {
        "g1": "enriched", "g2": "depleted",
    }
    assert parse_enrichment_direction("g1:neutral") == {
        "g1": "neutral", "g2": "unknown",
    }
    assert get_gene_verdict("", "g1") == "unknown"
