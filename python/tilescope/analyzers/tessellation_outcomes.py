"""Analyzer-outcome diagnostics emitted by the tessellation pipeline.

Two orthogonal axes kept separate throughout:

- **Region execution**: one row per (component, region, replicate).
  Records how many tiles the region produced and whether it
  contributed those tiles to the component-replicate tile pool or
  hard-failed with no positive-area polygonal support after
  clipping (`TILE_NO_POLYGONAL_CONTENT`).
- **Region eligibility**: one row per (component, region). Fixed
  before any replicate runs.

Every string-valued field is a `str`-valued `Enum` (or `Optional`
of one) so consumers cannot store an unknown code.

Pooled Pearson correlation is not emitted at component grain. The
canonical grain is ROI (`roi_replicate_correlations_df`, built by
the parent-ROI spatial analyzer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Closed status enums
# ---------------------------------------------------------------------------


class RegionIneligibilityReason(str, Enum):
    """Reason a domain region is ineligible for tessellation.

    A region is *eligible* iff its diagnostic's
    `ineligibility_reason` is `None`. An empty string is not a valid
    encoding and is rejected by the validator.

    The single ineligibility reason is the construction floor: a
    region with fewer spots than `min_spots_per_tile` cannot form
    even a single valid tile, so no tessellation is attempted. The
    pooled-correlation estimand accepts region contributions with
    1 or 2 tiles once tessellation succeeds, so there is no
    "correlation impossible at region grain" reason; reliability
    of the pooled correlation is a separate downstream question
    evaluated on the realised tile count.
    """
    N_SPOTS_BELOW_TILE_MIN = "n_spots_below_min_spots_per_tile"


class RegionExecutionStatus(str, Enum):
    """Region-level execution outcome for one (component, region, replicate).

    Three-way partition:

    - `CONTRIBUTED`: every emitted tile has positive-area polygonal
      support (Polygon OR MultiPolygon). Restricted Voronoi cells on
      a non-convex domain may be multipart under restricted-cell
      semantics (one seed, one identity, one point
      assignment) and does not degrade the outcome.
    - `NO_TILES`: the tessellation engine emitted zero tiles.
    - `TILE_NO_POLYGONAL_CONTENT`: at least one emitted tile had no
      positive-area polygonal support after clipping (points/lines
      only, or empty). Integrity failure: the tessellator assigned
      spots to a tile with no spatial support. The region-replicate
      contributes nothing, and the correlation status escalates to
      `INCOMPLETE_REGION_COVERAGE` on the pooled correlation.

    Programming-contract failures (missing columns, invalid point
    indices, engine crashes) propagate as exceptions; they do NOT
    map to a generic catch-all status.
    """
    CONTRIBUTED = "contributed"
    NO_TILES = "no_tiles"
    TILE_NO_POLYGONAL_CONTENT = "tile_no_polygonal_content"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _require_nonempty(value: str, field: str, owner: str) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(
            f"{owner}: {field}={value!r} must be a non-empty string."
        )


def _require_ge(value: int, threshold: int, field: str, owner: str) -> None:
    if value < threshold:
        raise ValueError(
            f"{owner}: {field}={value} must be >= {threshold}."
        )


def _require_finite_positive(value: float, field: str, owner: str) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(
            f"{owner}: {field}={value!r} must be finite and > 0."
        )


def _require_optional_enum(value, enum_cls, field: str, owner: str) -> None:
    if value is None:
        return
    if not isinstance(value, enum_cls):
        raise ValueError(
            f"{owner}: {field}={value!r} must be None or a "
            f"{enum_cls.__name__} member (got type {type(value).__name__})."
        )


def _require_enum(value, enum_cls, field: str, owner: str) -> None:
    if not isinstance(value, enum_cls):
        raise ValueError(
            f"{owner}: {field}={value!r} must be a {enum_cls.__name__} "
            f"member (got type {type(value).__name__})."
        )


# ---------------------------------------------------------------------------
# Region eligibility (fixed before any replicate runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionTessellationEligibilityDiagnostic:
    """Per (component, region). `is_tessellation_eligible` is derived."""
    component_id: str
    domain_region_id: int
    n_component_spots: int
    region_area: float
    ineligibility_reason: Optional[RegionIneligibilityReason]

    def __post_init__(self) -> None:
        owner = "RegionTessellationEligibilityDiagnostic"
        _require_nonempty(self.component_id, "component_id", owner)
        _require_ge(self.domain_region_id, 0, "domain_region_id", owner)
        _require_ge(self.n_component_spots, 0, "n_component_spots", owner)
        _require_finite_positive(self.region_area, "region_area", owner)
        _require_optional_enum(
            self.ineligibility_reason, RegionIneligibilityReason,
            "ineligibility_reason", owner,
        )

    @property
    def is_tessellation_eligible(self) -> bool:
        return self.ineligibility_reason is None


# ---------------------------------------------------------------------------
# Region × replicate execution outcome (no correlation at this grain)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionReplicateTessellationDiagnostic:
    """Per (component, region, replicate). Region-level execution only.

    `n_tiles` is the number of tiles the region contributed to the
    component-replicate tile pool. Regions with 1 or 2 valid tiles
    are `CONTRIBUTED`; the ≥3-tile requirement applies after
    pooling, not per region.
    """
    component_id: str
    domain_region_id: int
    replicate_index: int
    n_tiles: int
    region_status: RegionExecutionStatus

    def __post_init__(self) -> None:
        owner = "RegionReplicateTessellationDiagnostic"
        _require_nonempty(self.component_id, "component_id", owner)
        _require_ge(self.domain_region_id, 0, "domain_region_id", owner)
        _require_ge(self.replicate_index, 0, "replicate_index", owner)
        _require_ge(self.n_tiles, 0, "n_tiles", owner)
        _require_enum(
            self.region_status, RegionExecutionStatus, "region_status", owner,
        )
        if self.region_status is RegionExecutionStatus.CONTRIBUTED:
            if self.n_tiles < 1:
                raise ValueError(
                    f"{owner}: region_status=CONTRIBUTED requires "
                    f"n_tiles>=1, got n_tiles={self.n_tiles}."
                )
        elif self.region_status is RegionExecutionStatus.NO_TILES:
            if self.n_tiles != 0:
                raise ValueError(
                    f"{owner}: region_status=NO_TILES requires "
                    f"n_tiles==0, got n_tiles={self.n_tiles}."
                )
        elif self.region_status is RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT:
            # The region-replicate contributed no valid tile; `n_tiles`
            # is documented as contributed tiles, so it must be zero.
            if self.n_tiles != 0:
                raise ValueError(
                    f"{owner}: region_status=TILE_NO_POLYGONAL_CONTENT "
                    f"requires n_tiles==0, got n_tiles={self.n_tiles}."
                )
