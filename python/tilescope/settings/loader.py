"""Configuration loader: merges JSON layers, rejects frozen/retired
paths, and returns the composed :class:`ResolvedRunConfig` envelope.

Precedence (later layers win):

  1. Packaged defaults from ``tilescope_defaults/run.config.json``
  2. Profile JSON (skipped when ``profile`` is None or empty)
  3. User JSON (``path`` argument)
  4. CLI flag overrides (``flag_overrides``)
  5. ``--set KEY=VALUE`` overrides (``set_overrides``)

The merged dict is then:

  a. Rejected on presence of any :data:`RETIRED_PATHS` leaf. This
     path fires first so retired fields with tailored migration hints
     (e.g. ``consensus.min_tess_per_hex`` to ``min_tess_coverage_fraction``)
     surface the actionable message rather than the generic
     frozen-policy one.
  b. Rejected on presence of any :data:`FROZEN_POLICY_PATHS` leaf.
     The value is checked as well; declared-value-equals-frozen-value
     is still a rejection because it perpetuates the transitional
     accepted-if-matches contract.
  c. Validated through :class:`RunConfigModel` (user-configurable fields).
  d. Composed alongside a :class:`ResolvedPolicyModel` whose
     ``consensus.min_tess_per_hex`` is resolved from the user-facing
     ``consensus.min_tess_coverage_fraction`` and ``tessellation.n``.
  e. Returned as :class:`ResolvedRunConfig`, which reasserts the
     resolver identity as a defensive envelope invariant.

See ``docs/CONFIGURATION.md`` for the full contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from tilescope.defaults import _PACKAGED_CONFIG, load_profile
from tilescope.settings.models import (
    FROZEN_POLICY_PATHS,
    ConsensusPolicyModel,
    DatasetSchemaModel,
    ResolvedPolicyModel,
    ResolvedRunConfig,
    RunConfigModel,
    resolve_min_tess_per_hex,
    source_adjacency_policy_for_technology,
)


class ConfigMigrationError(ValueError):
    """A retired or frozen configuration path requires migration."""


# ---------------------------------------------------------------------------
# Retired-path registry: historical fields that once existed on the
# user-facing model but have been deleted. Every entry carries a message
# consumed by the single presence-validator below.
# ---------------------------------------------------------------------------


RETIRED_PATHS: Dict[str, str] = {
    "policy": (
        "Configuration key 'policy' is retired. Enrichment confidence "
        "and its FDR controls were removed because the confidence score "
        "was not a p-value. Remove the policy block; correlation "
        "permutation count remains configured by 'permutations'."
    ),
    "hull": (
        "Configuration key 'hull' is retired. The alpha/mask "
        "polygonization pipeline was replaced by the "
        "bounded_spot_support_owned_acquisition geometry contract; "
        "its standard-v2 radius factors are frozen in the resolved "
        "policy. Remove 'hull' from your config."
    ),
    "hull_adaptive": (
        "Configuration key 'hull_adaptive' is retired. The alpha/mask "
        "polygonization pipeline was replaced by the "
        "bounded_spot_support_owned_acquisition geometry contract. "
        "Remove 'hull_adaptive' from your config."
    ),
    "roi_filters.min_tessellation_component_spots": (
        "roi_filters.min_tessellation_component_spots is retired. "
        "Use tessellation.min_component_observations."
    ),
    "roi_filters.min_component_spots": (
        "roi_filters.min_component_spots is retired. Use "
        "tessellation.min_component_observations; the floor describes "
        "tessellation support rather than biological ROI eligibility."
    ),
    "knn": (
        "Configuration key 'knn' is retired. ROI components now use the "
        "same technology-selected spot/cell adjacency as source support, "
        "interfaces, and source Moran. Remove the knn block."
    ),
    "consensus.min_tess_per_hex": (
        "consensus.min_tess_per_hex is retired as user-facing "
        "configuration. Use consensus.min_tess_coverage_fraction "
        "(float in (0, 1]); TileScope resolves the integer floor as "
        "ceil(fraction * tessellation.n)."
    ),
    "consensus.use_density_for_radius": (
        "consensus.use_density_for_radius is retired. Density-aware "
        "hex-radius sizing is now the sole policy."
    ),
    "consensus.corr_bootstrap_n": (
        "consensus.corr_bootstrap_n is retired. It referred to a "
        "bootstrap that was never implemented."
    ),
    "consensus.reassign_uncovered_spots": (
        "consensus.reassign_uncovered_spots is retired. The uncovered-"
        "spot reassignment policy was never invoked."
    ),
    "consensus.reassign_max_distance_factor": (
        "consensus.reassign_max_distance_factor is retired. Removed with "
        "consensus.reassign_uncovered_spots."
    ),
    "consensus.reassign_min_unassigned_fraction": (
        "consensus.reassign_min_unassigned_fraction is retired. Removed "
        "with consensus.reassign_uncovered_spots."
    ),
    "tessellation.voronoi_fraction": (
        "tessellation.voronoi_fraction is retired as user-facing "
        "configuration. Use "
        "tessellation.target_observations_per_tile; TileScope derives "
        "the reciprocal seed fraction internally."
    ),
    "tessellation.min_spots_per_tile": (
        "tessellation.min_spots_per_tile is retired as user-facing "
        "configuration. Use tessellation.min_observations_per_tile."
    ),
    "visualization.show_discordant": (
        "visualization.show_discordant is retired. The consensus panel "
        "is a four-state categorical view; all four states are always "
        "rendered."
    ),
    "visualization.concordant_metric": (
        "visualization.concordant_metric is retired. The consensus "
        "panel now uses a locked near-neutral band and |log2 FC|=3 "
        "saturation."
    ),
    "visualization.significance_threshold": (
        "visualization.significance_threshold is retired. Consensus "
        "panels no longer expose a p-value-based significance overlay."
    ),
}


# ---------------------------------------------------------------------------
# Path helpers.
# ---------------------------------------------------------------------------


def _walk_leaves(obj: Any, prefix: str = "") -> List[str]:
    """Yield dotted paths for every leaf in a nested dict."""
    out: List[str] = []
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            child = _walk_leaves(v, path)
            if child:
                out.extend(child)
            else:
                out.append(path)  # empty container is itself a leaf
        else:
            out.append(path)
    return out


def _has_path(obj: Any, dotted: str) -> bool:
    """Return True iff the dotted path exists in obj (any value)."""
    current: Any = obj
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True
# ---------------------------------------------------------------------------
# Consolidated presence-validator.
# ---------------------------------------------------------------------------


def _reject_frozen_and_retired(config: Dict[str, Any]) -> None:
    """Single path-presence validator: rejects any leaf whose dotted
    path is in :data:`RETIRED_PATHS` (fires first so tailored migration
    hints reach the user) or in :data:`FROZEN_POLICY_PATHS`."""
    retired_violations = [
        p for p in RETIRED_PATHS if _has_path(config, p)
    ]
    if retired_violations:
        messages = "\n  - ".join(
            RETIRED_PATHS[p] for p in retired_violations
        )
        raise ConfigMigrationError(
            f"The following configuration paths are retired:\n  - {messages}"
        )
    frozen_violations = sorted(
        p for p in FROZEN_POLICY_PATHS if _has_path(config, p)
    )
    if frozen_violations:
        raise ConfigMigrationError(
            "The following configuration paths are frozen method-policy "
            "values and must not appear in user configs (they are recorded "
            "in effective_config.json under `resolved_policy`; a value "
            "matching the frozen default is still a rejection because "
            "declared-but-frozen would perpetuate the transitional "
            f"accepted-if-matches contract): {frozen_violations!r}. "
            "Remove them from your config."
        )


# ---------------------------------------------------------------------------
# JSON I/O + merge + --set parsing (unchanged behaviour).
# ---------------------------------------------------------------------------


def _read_json(path: Optional[str]) -> Dict[str, Any]:
    """Read and parse JSON file. Filters ``_``-prefixed metadata keys."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    return {k: v for k, v in config.items() if not k.startswith("_")}


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with overlay taking precedence."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_set_overrides(set_args: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``--set key.path=value`` arguments into a nested dict."""
    if not set_args:
        return {}
    result: Dict[str, Any] = {}
    for arg in set_args:
        if "=" not in arg:
            raise ValueError(f"Invalid --set argument (missing '='): {arg}")
        key_path, value_str = arg.split("=", 1)
        keys = key_path.split(".")
        value: Any = value_str
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        elif value_str.lower() == "null":
            value = None
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    pass  # keep as string
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def load_run_config(
    path: Optional[str] = None,
    *,
    profile: Optional[str] = None,
    flag_overrides: Optional[Dict[str, Any]] = None,
    set_overrides: Optional[Dict[str, Any]] = None,
) -> ResolvedRunConfig:
    """Load runtime configuration and return the composed envelope.

    Precedence: ``--set`` > CLI flags > user JSON > profile JSON >
    packaged defaults. Any presence of a frozen-policy or retired path
    raises :class:`ConfigMigrationError`."""
    flag_overrides = flag_overrides or {}
    set_overrides = set_overrides or {}

    merged: Dict[str, Any] = dict(_PACKAGED_CONFIG)
    if profile:
        try:
            merged = deep_merge(merged, load_profile(profile))
        except FileNotFoundError as e:
            raise ValueError(
                f"Profile '{profile}' not found. Available profiles: "
                "exploratory, standard, debugging"
            ) from e
    user_json = _read_json(path)
    if user_json:
        merged = deep_merge(merged, user_json)
    if flag_overrides:
        merged = deep_merge(merged, flag_overrides)
    if set_overrides:
        merged = deep_merge(merged, set_overrides)

    _reject_frozen_and_retired(merged)

    try:
        run_config = RunConfigModel.model_validate(merged)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}") from e

    min_tess_per_hex = resolve_min_tess_per_hex(
        run_config.consensus.min_tess_coverage_fraction,
        run_config.tessellation.n,
    )
    resolved_policy = ResolvedPolicyModel(
        source_adjacency=source_adjacency_policy_for_technology(
            run_config.technology,
            xenium_local_factor=(
                run_config.technology_options.xenium_local_factor
            ),
        ),
        consensus=ConsensusPolicyModel(min_tess_per_hex=min_tess_per_hex),
    )
    return ResolvedRunConfig(
        run_config=run_config, resolved_policy=resolved_policy
    )


def load_dataset_schema(path: Optional[str] = None) -> Optional[DatasetSchemaModel]:
    """Load dataset schema configuration."""
    data = _read_json(path)
    if not data:
        return None
    try:
        return DatasetSchemaModel.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid dataset.schema.json: {e}") from e


__all__ = [
    "ConfigMigrationError",
    "RETIRED_PATHS",
    "load_run_config",
    "load_dataset_schema",
    "parse_set_overrides",
    "deep_merge",
]
