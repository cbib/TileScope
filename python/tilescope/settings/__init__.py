"""TileScope configuration system.

Two-part architecture:

- ``RunConfigModel``: the user-configurable fields.
- ``ResolvedPolicyModel``: the frozen method-policy values + version + hash.
- ``ResolvedRunConfig``: the composed envelope threaded through the
  analysis pipeline.

Usage:

    from tilescope.settings import load_run_config

    resolved = load_run_config("user_configs/run.config.json")
    n = resolved.run_config.tessellation.n
    q_high = resolved.resolved_policy.thresholds.q_high
"""

from tilescope.settings.loader import (
    ConfigMigrationError,
    load_run_config,
    load_dataset_schema,
    parse_set_overrides,
)
from tilescope.settings.models import (
    # User-configurable
    BivariateMoransUserModel,
    ConsensusUserModel,
    DatasetSchemaModel,
    RoiFiltersUserModel,
    RunConfigModel,
    TessellationModel,
    TechnologyOptionsModel,
    VisualizationModel,
    # Frozen policy
    AdaptiveOverlapModel,
    BivariateMoransPolicyModel,
    ComponentDomainRadiiModel,
    ConsensusPolicyModel,
    FROZEN_POLICY_PATHS,
    ResolvedPolicyModel,
    SourceAdjacencyPolicyModel,
    ThresholdsModel,
    compute_policy_hash,
    source_adjacency_policy_for_technology,
    # Envelope
    ResolvedRunConfig,
)

__all__ = [
    "load_run_config",
    "load_dataset_schema",
    "parse_set_overrides",
    "ConfigMigrationError",
    # User-configurable
    "RunConfigModel",
    "TessellationModel",
    "TechnologyOptionsModel",
    "ConsensusUserModel",
    "BivariateMoransUserModel",
    "RoiFiltersUserModel",
    "VisualizationModel",
    # Frozen policy
    "ResolvedPolicyModel",
    "ThresholdsModel",
    "SourceAdjacencyPolicyModel",
    "source_adjacency_policy_for_technology",
    "ConsensusPolicyModel",
    "BivariateMoransPolicyModel",
    "AdaptiveOverlapModel",
    "ComponentDomainRadiiModel",
    "FROZEN_POLICY_PATHS",
    "compute_policy_hash",
    # Envelope
    "ResolvedRunConfig",
    "DatasetSchemaModel",
]
