"""Contract tests for the tessellation diagnostic dataclasses.

Semantic scenarios only. Input-class validation is parametrised
compactly. Geometry containment, region assignment, and reporter
serialisation live in their own owning layers.
"""
from __future__ import annotations

import pytest

from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus,
    RegionIneligibilityReason,
    RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _region(**overrides) -> RegionTessellationEligibilityDiagnostic:
    base = dict(
        component_id="A",
        domain_region_id=0,
        n_component_spots=5,
        region_area=1.0,
        ineligibility_reason=None,
    )
    base.update(overrides)
    return RegionTessellationEligibilityDiagnostic(**base)


def _region_replicate(**overrides) -> RegionReplicateTessellationDiagnostic:
    base = dict(
        component_id="A",
        domain_region_id=0,
        replicate_index=0,
        n_tiles=5,
        region_status=RegionExecutionStatus.CONTRIBUTED,
    )
    base.update(overrides)
    return RegionReplicateTessellationDiagnostic(**base)


# ---------------------------------------------------------------------------
# Region eligibility — one reason only
# ---------------------------------------------------------------------------


def test_region_eligibility_flag_is_derived():
    assert _region(ineligibility_reason=None).is_tessellation_eligible is True
    ineligible = _region(
        ineligibility_reason=RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN,
    )
    assert ineligible.is_tessellation_eligible is False


def test_region_ineligibility_reason_enum_has_one_member():
    # Guards against re-introducing correlation-based region ineligibility;
    # the pooled estimand keeps 1- and 2-tile regions as contributors.
    assert list(RegionIneligibilityReason) == [
        RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN,
    ]


def test_region_ineligibility_reason_must_be_enum_or_none():
    with pytest.raises(ValueError, match="ineligibility_reason"):
        _region(ineligibility_reason="not-an-enum")


@pytest.mark.parametrize(
    "field,bad_value,pattern",
    [
        ("component_id", "", "component_id"),
        ("domain_region_id", -1, "domain_region_id"),
        ("n_component_spots", -1, "n_component_spots"),
        ("region_area", 0.0, "region_area"),
        ("region_area", float("nan"), "region_area"),
    ],
)
def test_region_scalar_validation(field, bad_value, pattern):
    with pytest.raises(ValueError, match=pattern):
        _region(**{field: bad_value})


# ---------------------------------------------------------------------------
# Region × replicate execution — no density-failure status
# ---------------------------------------------------------------------------


def test_region_execution_enum_membership():
    # Pins the closed set. Programming failures (missing columns /
    # invalid indices) propagate as exceptions — no catch-all status.
    # `TILE_NO_POLYGONAL_CONTENT` covers the only reclip-failure case
    # under restricted-cell semantics: a tile that ends up with no
    # positive-area polygonal support.
    assert list(RegionExecutionStatus) == [
        RegionExecutionStatus.CONTRIBUTED,
        RegionExecutionStatus.NO_TILES,
        RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT,
    ]


def test_contributed_region_permits_positive_tile_count():
    n_tiles = 1
    diag = _region_replicate(
        n_tiles=n_tiles, region_status=RegionExecutionStatus.CONTRIBUTED,
    )
    assert diag.n_tiles == n_tiles


def test_contributed_region_requires_at_least_one_tile():
    with pytest.raises(ValueError, match="CONTRIBUTED requires n_tiles>=1"):
        _region_replicate(n_tiles=0, region_status=RegionExecutionStatus.CONTRIBUTED)


def test_no_tiles_region_pins_zero_count():
    diag = _region_replicate(n_tiles=0, region_status=RegionExecutionStatus.NO_TILES)
    assert diag.n_tiles == 0
    with pytest.raises(ValueError, match="NO_TILES requires n_tiles==0"):
        _region_replicate(n_tiles=1, region_status=RegionExecutionStatus.NO_TILES)


def test_tile_no_polygonal_content_pins_zero_tile_count():
    # `n_tiles` is documented as CONTRIBUTED tiles, so a failed
    # region-replicate must report zero contributed tiles.
    diag = _region_replicate(
        n_tiles=0,
        region_status=RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT,
    )
    assert diag.n_tiles == 0
    for bad in (1, 5):
        with pytest.raises(
            ValueError,
            match="TILE_NO_POLYGONAL_CONTENT requires n_tiles==0",
        ):
            _region_replicate(
                n_tiles=bad,
                region_status=RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT,
            )


def test_region_replicate_status_must_be_enum():
    with pytest.raises(ValueError, match="region_status"):
        _region_replicate(region_status="contributed")
