"""Semantic feature-identity validation.

``gene1`` and ``gene2`` are reserved computational columns in TileScope's
canonical input schema. They are never valid biological labels.
"""

from __future__ import annotations

from tilescope.cols import GENE1, GENE2


_RESERVED_COMPUTATIONAL_IDS = frozenset({GENE1, GENE2})


def require_semantic_feature_ids(
    feature_a_id: object,
    feature_b_id: object,
    *,
    owner: str,
) -> tuple[str, str]:
    """Return stripped semantic IDs or raise on absent/generic identities."""
    resolved: list[str] = []
    for label, value in (
        ("feature_a_id", feature_a_id),
        ("feature_b_id", feature_b_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{owner}: {label} must be a non-empty biological feature "
                "identity. Both feature_a_id and feature_b_id are required."
            )
        resolved.append(value.strip())

    generic = [value for value in resolved if value in _RESERVED_COMPUTATIONAL_IDS]
    if generic:
        raise ValueError(
            f"{owner}: {generic!r} are reserved computational column names, "
            "not biological feature identities. Supply the real names via "
            "feature_a_id and feature_b_id."
        )
    return resolved[0], resolved[1]


__all__ = ["require_semantic_feature_ids"]
