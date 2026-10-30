"""Pydantic models for TileScope configuration.

Two-part architecture:

- ``RunConfigModel``: the user-configurable fields.
- ``ResolvedPolicyModel``: the frozen method-policy values plus
  ``policy_version`` and deterministic ``policy_hash``.
- ``ResolvedRunConfig``: the composed envelope threaded through the
  analysis pipeline. Analyzers read ``resolved.run_config.<field>`` for
  user-configurable values and ``resolved.resolved_policy.<field>`` for
  frozen policy values.

Frozen policy values are derived from packaged defaults; they are
never accepted from user JSON or ``--set`` overrides (see
``settings.loader`` for rejection). The split makes the public
configuration surface unmistakable and prevents drift between the
"tuning knob" and "resolved method definition" concepts."""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_CEILING, Decimal
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tilescope.defaults import ANALYSIS_POLICY_VERSION
from tilescope.spatial.source_spot_adjacency import (
    XENIUM_DELAUNAY_LOCAL_FACTOR,
)


_FROZEN_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


def resolve_min_tess_per_hex(coverage_fraction: float, n: int) -> int:
    """Resolve the integer coverage floor via decimal arithmetic so
    values like ``0.14 * 50`` do not slip a hex above the ceiling
    through IEEE-754 rounding (``math.ceil(0.14 * 50) == 8``, but
    ``Decimal('0.14') * 50 == Decimal('7.00')``). ``str(float)`` uses
    Python's shortest-round-trip repr, so the Decimal reflects the
    number the user wrote in JSON."""
    product = Decimal(str(coverage_fraction)) * Decimal(n)
    return int(product.to_integral_value(rounding=ROUND_CEILING))


# ---------------------------------------------------------------------------
# User-configurable submodels: belong to RunConfigModel.
# ---------------------------------------------------------------------------


class TessellationModel(BaseModel):
    """User-configurable ensemble size and observation-scale support."""
    model_config = _FROZEN_MODEL_CONFIG

    n: int = Field(..., ge=1, description="Number of tessellations per ROI")
    target_observations_per_tile: int = Field(
        ..., ge=1,
        description="Nominal spot or cell observations per initial Voronoi tile",
    )
    min_observations_per_tile: int = Field(
        ..., ge=1,
        description="Minimum spot or cell observations retained in a Voronoi tile",
    )
    min_component_observations: int = Field(
        ..., ge=1,
        description="Minimum spot or cell observations in a retained component",
    )

    @property
    def voronoi_fraction(self) -> float:
        """Internal seed fraction derived from the user-facing target."""
        return 1.0 / float(self.target_observations_per_tile)

    @model_validator(mode="after")
    def check_observation_scale(self) -> "TessellationModel":
        if self.min_observations_per_tile > self.target_observations_per_tile:
            raise ValueError(
                "tessellation.min_observations_per_tile must be <= "
                "tessellation.target_observations_per_tile"
            )
        nominal_seeds = max(
            1,
            round(
                self.min_component_observations
                / self.target_observations_per_tile
            ),
        )
        if nominal_seeds < 3:
            raise ValueError(
                "tessellation.min_component_observations must support at "
                "least three nominal tiles under round(n_observations / "
                "target_observations_per_tile); got "
                f"{nominal_seeds}"
            )
        return self


class BivariateMoransUserModel(BaseModel):
    """Reporting-only visibility switch. Method-policy fields
    (``compute``, ``k_neighbors``) are frozen in
    :class:`ResolvedPolicyModel`."""
    model_config = _FROZEN_MODEL_CONFIG

    visualize: bool = Field(
        True,
        description="Add bivariate Moran's I panels to consensus plot (when computed)",
    )


class ConsensusUserModel(BaseModel):
    """User-configurable consensus eligibility knobs. The integer floor
    ``min_tess_per_hex`` is resolved from ``min_tess_coverage_fraction``
    and ``tessellation.n`` and lives in :class:`ResolvedPolicyModel`."""
    model_config = _FROZEN_MODEL_CONFIG

    min_spots_per_hex: int = Field(..., ge=1, description="Minimum spots per hexagon")
    min_tess_coverage_fraction: float = Field(
        2.0 / 3.0, gt=0.0, le=1.0,
        description=(
            "Minimum tessellation coverage fraction; resolves to "
            "min_tess_per_hex = ceil(fraction * tessellation.n)"
        ),
    )
    bivariate_morans: BivariateMoransUserModel = Field(
        default_factory=BivariateMoransUserModel,
        description="Bivariate Moran's I reporting toggle",
    )


class RoiFiltersUserModel(BaseModel):
    """User-configurable biological ROI eligibility."""
    model_config = _FROZEN_MODEL_CONFIG

    min_roi_spots: int = Field(..., ge=1, description="Minimum source-labelled spots per biological ROI (pooled across sibling components)")
    min_context_spots: int = Field(..., ge=0, description="Minimum spots in context")


class VisualizationModel(BaseModel):
    """Reporting-only visualization options."""
    model_config = _FROZEN_MODEL_CONFIG

    show_stability: bool = Field(
        False,
        description="Show A/B stability panels (SD/IQR of per-hex ensemble densities) - use CSV summaries by default",
    )


class TechnologyOptionsModel(BaseModel):
    """Optional technology-specific overrides.

    Technology still selects the graph method. This narrow override changes
    only Xenium's local Delaunay pruning scale; omitting it uses the validated
    core default recorded in ``resolved_policy``.
    """
    model_config = _FROZEN_MODEL_CONFIG

    xenium_local_factor: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Optional Xenium Delaunay edge limit relative to the larger "
            "endpoint nearest-neighbour distance; omit for the core default"
        ),
    )


class RunConfigModel(BaseModel):
    """User-configurable fields. Frozen method-policy values
    live in :class:`ResolvedPolicyModel` and are composed alongside
    this model by :class:`ResolvedRunConfig`."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    technology: Literal["visium", "xenium"] = Field(
        "visium",
        description="Acquisition technology selecting the source-support graph",
    )
    technology_options: TechnologyOptionsModel = Field(
        default_factory=TechnologyOptionsModel,
    )
    tessellation: TessellationModel
    consensus: ConsensusUserModel
    roi_filters: RoiFiltersUserModel
    visualization: VisualizationModel = Field(default_factory=VisualizationModel)
    permutations: int = Field(..., ge=0, description="Number of permutations for statistical testing")
    seed: int = Field(42, description="Random seed for reproducibility")
    output_dir: str = Field(..., description="Output directory path")
    verbose: bool = Field(False, description="Enable debug logging")


# ---------------------------------------------------------------------------
# Frozen policy submodels: belong to ResolvedPolicyModel.
# ---------------------------------------------------------------------------


class ThresholdsModel(BaseModel):
    """Frozen high/low expression percentiles for the standard-v3 policy."""
    model_config = _FROZEN_MODEL_CONFIG

    q_high: Literal[0.75] = 0.75
    q_low: Literal[0.25] = 0.25


class SourceAdjacencyPolicyModel(BaseModel):
    """Resolved, technology-selected source-neighbour construction."""
    model_config = _FROZEN_MODEL_CONFIG

    method: Literal["fixed_radius", "locally_pruned_delaunay"] = "fixed_radius"
    radius_factor: Optional[float] = 1.5
    local_factor: Optional[float] = None
    global_cap_factor: Optional[float] = None

    @model_validator(mode="after")
    def check_method_parameters(self) -> "SourceAdjacencyPolicyModel":
        if self.method == "fixed_radius":
            if self.radius_factor is None or self.radius_factor <= 0:
                raise ValueError("fixed_radius requires radius_factor > 0")
            if self.local_factor is not None or self.global_cap_factor is not None:
                raise ValueError(
                    "fixed_radius does not accept Delaunay pruning parameters"
                )
        else:
            if self.radius_factor is not None:
                raise ValueError(
                    "locally_pruned_delaunay does not accept radius_factor"
                )
            if self.local_factor is None or self.local_factor <= 0:
                raise ValueError(
                    "locally_pruned_delaunay requires local_factor > 0"
                )
            if self.global_cap_factor is None or self.global_cap_factor <= 0:
                raise ValueError(
                    "locally_pruned_delaunay requires global_cap_factor > 0"
                )
        return self


def source_adjacency_policy_for_technology(
    technology: Literal["visium", "xenium"],
    *,
    xenium_local_factor: Optional[float] = None,
) -> SourceAdjacencyPolicyModel:
    """Resolve the graph policy selected by ``technology``.

    Xenium's method and global bridge cap remain core policy. The local
    pruning scale has a validated default but may be overridden explicitly
    for a dataset geometry audit.
    """
    if technology == "visium":
        if xenium_local_factor is not None:
            raise ValueError(
                "technology_options.xenium_local_factor is only valid when "
                "technology='xenium'"
            )
        return SourceAdjacencyPolicyModel(
            method="fixed_radius", radius_factor=1.5,
            local_factor=None, global_cap_factor=None,
        )
    if technology == "xenium":
        local_factor = (
            XENIUM_DELAUNAY_LOCAL_FACTOR
            if xenium_local_factor is None
            else float(xenium_local_factor)
        )
        return SourceAdjacencyPolicyModel(
            method="locally_pruned_delaunay", radius_factor=None,
            local_factor=local_factor, global_cap_factor=4.0,
        )
    raise ValueError(f"Unsupported technology: {technology!r}")


class BivariateMoransPolicyModel(BaseModel):
    """Frozen bivariate Moran's I method (always computed, k=6)."""
    model_config = _FROZEN_MODEL_CONFIG

    compute: Literal[True] = True
    k_neighbors: Literal[6] = 6


class AdaptiveOverlapModel(BaseModel):
    """Frozen adaptive minimum-overlap policy."""
    model_config = _FROZEN_MODEL_CONFIG

    small_roi_max_spots: Literal[150] = 150
    medium_roi_max_spots: Literal[400] = 400
    small_roi_overlap: Literal[0.04] = 0.04
    medium_roi_overlap: Literal[0.05] = 0.05
    large_roi_overlap: Literal[0.07] = 0.07


class ConsensusPolicyModel(BaseModel):
    """Frozen consensus method-policy values. ``min_tess_per_hex`` is
    resolved per run from ``run_config.consensus.min_tess_coverage_fraction``
    and ``run_config.tessellation.n``."""
    model_config = _FROZEN_MODEL_CONFIG

    min_tess_per_hex: int = Field(
        ..., ge=1,
        description="Resolved integer floor: ceil(min_tess_coverage_fraction * tessellation.n)",
    )
    occupancy_safety: Literal[1.2] = 1.2
    area_sizing_method: Literal["winsor_mean"] = "winsor_mean"
    winsor_limits: tuple[Literal[0.0], Literal[0.98]] = (0.0, 0.98)
    base_area_quantile: Literal[0.5] = 0.5
    max_hex_area_fraction_of_roi: Literal[0.25] = 0.25
    min_hex_roi_overlap_fraction: None = None
    bivariate_morans: BivariateMoransPolicyModel = Field(
        default_factory=BivariateMoransPolicyModel
    )
    adaptive_overlap: AdaptiveOverlapModel = Field(
        default_factory=AdaptiveOverlapModel
    )


class ComponentDomainRadiiModel(BaseModel):
    """Frozen standard-v3 bounded-support radius factors."""
    model_config = _FROZEN_MODEL_CONFIG

    component_support_radius_factor: Literal[0.6] = 0.6
    acquisition_support_radius_factor: Literal[0.6] = 0.6
    ineligible_spot_exclusion_radius_factor: Literal[0.6] = 0.6


class RadialShellPolicyModel(BaseModel):
    """Frozen component-grain radial shell cross-correlogram policy.

    Shell edges are in units of the component's median nearest-neighbour
    spacing (dimensionless). A shell emits a finite `shell_symmetric`
    only when its eligible-anchor count is `>= min_anchors_per_shell`
    AND its eligible-anchor fraction is `>= min_anchor_fraction`; both
    floors must hold. Fewer knobs than the sister pilot: `quad_segs`
    is retired because the polygon-coverage QC matrix is not exported."""
    model_config = _FROZEN_MODEL_CONFIG

    shell_edges_normalized: tuple[
        Literal[0.5], Literal[1.5], Literal[2.5],
        Literal[4.5], Literal[8.5],
    ] = (0.5, 1.5, 2.5, 4.5, 8.5)
    min_anchors_per_shell: Literal[10] = 10
    min_anchor_fraction: Literal[0.5] = 0.5


class ResolvedPolicyModel(BaseModel):
    """The frozen method-policy values plus ``policy_version`` and
    deterministic ``policy_hash``. Never accepted from user JSON or
    ``--set``; constructed from packaged defaults and validated by
    the policy registry in :mod:`tilescope.settings.loader`."""
    model_config = _FROZEN_MODEL_CONFIG

    analysis_policy_version: Literal["standard-v3"] = ANALYSIS_POLICY_VERSION
    thresholds: ThresholdsModel = Field(default_factory=ThresholdsModel)
    source_adjacency: SourceAdjacencyPolicyModel = Field(
        default_factory=SourceAdjacencyPolicyModel
    )
    tile_value_aggregation: Literal["mean_per_source_observation"] = (
        "mean_per_source_observation"
    )
    consensus: ConsensusPolicyModel = Field(
        ...,
        description=(
            "Resolved consensus policy including "
            "min_tess_per_hex = ceil(min_tess_coverage_fraction * tessellation.n)"
        ),
    )
    component_domain_radii: ComponentDomainRadiiModel = Field(
        default_factory=ComponentDomainRadiiModel
    )
    radial_shell: RadialShellPolicyModel = Field(
        default_factory=RadialShellPolicyModel
    )

    @property
    def policy_hash(self) -> str:
        """Deterministic hash of the canonical policy serialization
        (excluding this property itself). Same policy content -> same
        hash regardless of Python-side attribute order."""
        return compute_policy_hash(self)


def compute_policy_hash(policy: "ResolvedPolicyModel") -> str:
    """SHA-256 (first 16 hex chars) of the canonical
    ``json.dumps(sort_keys=True, separators=(",", ":"))`` form of the
    policy's model_dump. Stable across Python versions."""
    canonical = json.dumps(
        policy.model_dump(), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ResolvedRunConfig(BaseModel):
    """Composed envelope: user-configurable ``run_config`` + frozen
    ``resolved_policy``. This is what :func:`load_run_config` returns
    and what every analyzer receives. The two-part shape mirrors the
    ``effective_config.json`` schema."""
    model_config = _FROZEN_MODEL_CONFIG

    run_config: RunConfigModel
    resolved_policy: ResolvedPolicyModel

    @model_validator(mode="after")
    def check_technology_policy(self) -> "ResolvedRunConfig":
        expected = source_adjacency_policy_for_technology(
            self.run_config.technology,
            xenium_local_factor=(
                self.run_config.technology_options.xenium_local_factor
            ),
        )
        if self.resolved_policy.source_adjacency != expected:
            raise ValueError(
                "resolved spot/cell adjacency policy does not match technology "
                f"{self.run_config.technology!r}"
            )
        return self

    @model_validator(mode="after")
    def check_min_tess_per_hex_matches_fraction(self) -> "ResolvedRunConfig":
        """Envelope-level guard against stale resolved integer floors.
        ``model_copy(update=...)`` bypasses the loader's resolver, so
        every constructed envelope reasserts the identity."""
        expected = resolve_min_tess_per_hex(
            self.run_config.consensus.min_tess_coverage_fraction,
            self.run_config.tessellation.n,
        )
        actual = self.resolved_policy.consensus.min_tess_per_hex
        if actual != expected:
            raise ValueError(
                "resolved consensus.min_tess_per_hex "
                f"({actual}) does not match ceil("
                f"consensus.min_tess_coverage_fraction * tessellation.n) "
                f"= {expected}"
            )
        return self


# ---------------------------------------------------------------------------
# Frozen-policy path registry: auto-derived from ResolvedPolicyModel.
# ---------------------------------------------------------------------------


def _walk_model_paths(model_cls, prefix: str = "") -> list[str]:
    """Enumerate dotted leaf paths of a Pydantic model class."""
    paths: list[str] = []
    for name, field in model_cls.model_fields.items():
        ann = field.annotation
        child = f"{prefix}.{name}" if prefix else name
        try:
            is_submodel = isinstance(ann, type) and issubclass(ann, BaseModel)
        except TypeError:
            is_submodel = False
        if is_submodel:
            paths.extend(_walk_model_paths(ann, child))
        else:
            paths.append(child)
    return paths


FROZEN_POLICY_PATHS: frozenset[str] = frozenset(
    _walk_model_paths(ResolvedPolicyModel)
)
"""The dotted paths that must not appear on user-facing configs.
Auto-derived from ``ResolvedPolicyModel``; adding a policy field
automatically extends the rejection registry."""


# ---------------------------------------------------------------------------
# Dataset schema (unchanged from prior model).
# ---------------------------------------------------------------------------


class DatasetSchemaModel(BaseModel):
    """Dataset column mapping schema."""
    model_config = ConfigDict(extra="forbid")

    columns: Dict[str, str] = Field(
        ..., description="Source column → canonical column mapping"
    )

    @field_validator("columns")
    @classmethod
    def unique_targets(cls, m: Dict[str, str]) -> Dict[str, str]:
        """Ensure no duplicate target columns in mapping."""
        inv: Dict[str, str] = {}
        for src, dst in m.items():
            if dst in inv:
                raise ValueError(f"Duplicate target mapping for {dst!r}: {inv[dst]!r} and {src!r}")
            inv[dst] = src
        return m


__all__ = [
    # User-configurable
    "TessellationModel",
    "BivariateMoransUserModel",
    "ConsensusUserModel",
    "RoiFiltersUserModel",
    "VisualizationModel",
    "TechnologyOptionsModel",
    "RunConfigModel",
    # Frozen policy
    "ThresholdsModel",
    "SourceAdjacencyPolicyModel",
    "source_adjacency_policy_for_technology",
    "BivariateMoransPolicyModel",
    "AdaptiveOverlapModel",
    "ConsensusPolicyModel",
    "ComponentDomainRadiiModel",
    "RadialShellPolicyModel",
    "ResolvedPolicyModel",
    "compute_policy_hash",
    "resolve_min_tess_per_hex",
    # Envelope
    "ResolvedRunConfig",
    # Registry
    "FROZEN_POLICY_PATHS",
    # Dataset
    "DatasetSchemaModel",
]
