"""CORE-CONFIG-03 contract: user-configurable RunConfigModel + frozen
ResolvedPolicyModel + composed ResolvedRunConfig envelope; loader
rejects declared frozen paths even when the value equals the frozen
default; policy hash is deterministic."""
from __future__ import annotations

import json

import pytest

from tilescope.settings import (
    ConfigMigrationError,
    FROZEN_POLICY_PATHS,
    ResolvedPolicyModel,
    ResolvedRunConfig,
    RunConfigModel,
    compute_policy_hash,
    load_run_config,
    parse_set_overrides,
)
from tilescope.settings.models import (
    ConsensusPolicyModel,
    resolve_min_tess_per_hex,
)


def _make_policy(min_tess_per_hex: int = 7) -> ResolvedPolicyModel:
    """Construct a ResolvedPolicyModel for tests. ``min_tess_per_hex``
    is a required, per-run resolved field, so no bare default exists."""
    return ResolvedPolicyModel(
        consensus=ConsensusPolicyModel(min_tess_per_hex=min_tess_per_hex),
    )


# --------------------------------------------------------------------------- #
# 1. Public model contains only the 12 user-configurable fields.              #
# --------------------------------------------------------------------------- #


USER_CONFIGURABLE_LEAVES = frozenset({
    "technology",
    "technology_options.xenium_local_factor",
    "tessellation.n",
    "tessellation.target_observations_per_tile",
    "tessellation.min_observations_per_tile",
    "tessellation.min_component_observations",
    "consensus.min_spots_per_hex",
    "consensus.min_tess_coverage_fraction",
    "consensus.bivariate_morans.visualize",
    "roi_filters.min_roi_spots",
    "roi_filters.min_context_spots",
    "visualization.show_stability",
    "permutations",
    "seed",
    "output_dir",
    "verbose",
})


def _walk_leaves(model_dump):
    def rec(d, prefix=""):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from rec(v, path)
            else:
                yield path
    yield from rec(model_dump)


def test_run_config_dump_yields_exact_user_leaves():
    r = load_run_config()
    leaves = set(_walk_leaves(r.run_config.model_dump()))
    assert leaves == USER_CONFIGURABLE_LEAVES


def test_resolved_policy_dump_covers_frozen_registry():
    """Every dotted path in FROZEN_POLICY_PATHS appears as a leaf in
    the resolved-policy dump, and vice versa. Set equality is the
    invariant; the current cardinality is derived from that."""
    policy = _make_policy()
    leaves = set(_walk_leaves(policy.model_dump()))
    assert leaves == set(FROZEN_POLICY_PATHS)


# --------------------------------------------------------------------------- #
# 2. Loader rejects every frozen path even when value equals default.         #
# --------------------------------------------------------------------------- #


def _resolve_path(dump: dict, dotted: str):
    """Walk a dotted path through a nested dict."""
    current = dump
    for k in dotted.split("."):
        current = current[k]
    return current


def _nest(path: str, value):
    keys = path.split(".")
    result = value
    for k in reversed(keys):
        result = {k: result}
    return result


# Derived from the single policy authority so this rejection matrix
# cannot drift from FROZEN_POLICY_PATHS or from the frozen defaults.
# consensus.min_tess_per_hex is dual-mapped: it lives on
# ResolvedPolicyModel but is also a retired user-facing path with a
# tailored migration hint that fires ahead of the generic frozen
# message. It is covered by the dedicated retirement test below.
_POLICY_DUMP = _make_policy().model_dump()
_FROZEN_DEFAULT_PAIRS = sorted(
    (path, _resolve_path(_POLICY_DUMP, path))
    for path in FROZEN_POLICY_PATHS
    if path != "consensus.min_tess_per_hex"
)


def test_every_declared_frozen_path_is_rejected_even_at_its_default():
    for path, value in _FROZEN_DEFAULT_PAIRS:
        with pytest.raises(ConfigMigrationError, match="frozen method-policy"):
            load_run_config(set_overrides=_nest(path, value))


# --------------------------------------------------------------------------- #
# 3. Policy hash is deterministic from canonical policy serialization.        #
# --------------------------------------------------------------------------- #


def test_policy_hash_is_deterministic():
    policy_a = _make_policy()
    policy_b = _make_policy()
    assert compute_policy_hash(policy_a) == compute_policy_hash(policy_b)
    assert policy_a.policy_hash == policy_b.policy_hash
    assert len(policy_a.policy_hash) == 16  # 16 hex chars


def test_policy_hash_varies_with_resolved_min_tess_per_hex():
    """Coverage-fraction resolution enters the policy hash, so two runs
    that resolve to different min_tess_per_hex values are provenance-
    distinct even under identical frozen defaults."""
    assert compute_policy_hash(_make_policy(min_tess_per_hex=7)) != (
        compute_policy_hash(_make_policy(min_tess_per_hex=8))
    )


# --------------------------------------------------------------------------- #
# 4. Envelope threads user + policy together.                                  #
# --------------------------------------------------------------------------- #


def test_resolved_run_config_carries_both_halves():
    r = load_run_config()
    assert isinstance(r, ResolvedRunConfig)
    assert isinstance(r.run_config, RunConfigModel)
    assert isinstance(r.resolved_policy, ResolvedPolicyModel)


@pytest.mark.parametrize(
    "fraction,n,expected",
    [
        (2.0 / 3.0, 1, 1),
        (2.0 / 3.0, 3, 2),
        (2.0 / 3.0, 10, 7),
        (2.0 / 3.0, 20, 14),
        (0.5, 2, 1),
        (0.2, 5, 1),
        (1.0, 10, 10),
    ],
)
def test_min_tess_per_hex_resolved_from_coverage_fraction(fraction, n, expected):
    r = load_run_config(set_overrides={
        "tessellation": {"n": n},
        "consensus": {"min_tess_coverage_fraction": fraction},
    })
    assert r.resolved_policy.consensus.min_tess_per_hex == expected


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_min_tess_coverage_fraction_rejects_out_of_bounds(bad):
    with pytest.raises(ValueError, match="min_tess_coverage_fraction"):
        load_run_config(set_overrides={
            "consensus": {"min_tess_coverage_fraction": bad},
        })


@pytest.mark.parametrize(
    "fraction,n,expected",
    [
        # IEEE-754 landmines: float(fraction) * n rounds slightly above the
        # exact product, so math.ceil returns 8. Decimal(str(fraction)) * n
        # returns exactly 7 and ceil to 7.
        (0.14, 50, 7),
        (0.07, 100, 7),
        (0.28, 25, 7),
        # Exact-boundary cases the resolver must not round up.
        (0.5, 4, 2),
        (0.25, 4, 1),
        (1.0, 10, 10),
    ],
)
def test_resolve_min_tess_per_hex_handles_ieee754_boundary_cases(
    fraction, n, expected,
):
    """Decimal-string arithmetic avoids the trap where a mathematically
    exact product like 0.14*50=7 becomes 7.000000000000001 under float
    multiplication, sending math.ceil to 8."""
    assert resolve_min_tess_per_hex(fraction, n) == expected


def test_resolved_run_config_rejects_stale_min_tess_per_hex():
    """model_copy(update=...) bypasses the loader's resolver, so
    ResolvedRunConfig reasserts the identity as an envelope invariant
    on every direct construction."""
    r = load_run_config(set_overrides={
        "tessellation": {"n": 10},
        "consensus": {"min_tess_coverage_fraction": 2.0 / 3.0},
    })
    assert r.resolved_policy.consensus.min_tess_per_hex == 7
    stale_policy = r.resolved_policy.model_copy(update={
        "consensus": ConsensusPolicyModel(min_tess_per_hex=2),
    })
    with pytest.raises(ValueError, match="does not match ceil"):
        ResolvedRunConfig(
            run_config=r.run_config, resolved_policy=stale_policy,
        )


def test_user_facing_min_tess_per_hex_is_retired():
    """The old integer field is now a retired path with a migration hint
    pointing to consensus.min_tess_coverage_fraction."""
    with pytest.raises(
        ConfigMigrationError, match="min_tess_coverage_fraction"
    ):
        load_run_config(set_overrides={
            "consensus": {"min_tess_per_hex": 2},
        })


def test_technology_selects_internal_source_graph_policy():
    visium = load_run_config()
    xenium = load_run_config(set_overrides={"technology": "xenium"})
    assert visium.resolved_policy.source_adjacency.method == "fixed_radius"
    assert xenium.resolved_policy.source_adjacency.method == (
        "locally_pruned_delaunay"
    )
    assert xenium.resolved_policy.source_adjacency.local_factor == 2.5


def test_xenium_local_factor_is_optional_and_user_overridable():
    xenium = load_run_config(set_overrides={
        "technology": "xenium",
        "technology_options": {"xenium_local_factor": 2.0},
    })
    assert xenium.resolved_policy.source_adjacency.local_factor == 2.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_xenium_local_factor_rejects_nonpositive_values(bad):
    with pytest.raises(ValueError, match="xenium_local_factor"):
        load_run_config(set_overrides={
            "technology": "xenium",
            "technology_options": {"xenium_local_factor": bad},
        })


def test_xenium_local_factor_rejected_for_visium():
    with pytest.raises(ValueError, match="only valid.*xenium"):
        load_run_config(set_overrides={
            "technology": "visium",
            "technology_options": {"xenium_local_factor": 2.0},
        })


def test_historical_knn_component_config_is_retired():
    with pytest.raises(ConfigMigrationError, match="knn.*retired"):
        load_run_config(set_overrides={"knn": {"k": 5}})


def test_tessellation_scale_must_support_three_nominal_tiles():
    with pytest.raises(ValueError, match="at least three nominal tiles"):
        load_run_config(set_overrides={
            "tessellation": {
                "target_observations_per_tile": 15,
                "min_observations_per_tile": 9,
                "min_component_observations": 30,
            }
        })


# --------------------------------------------------------------------------- #
# 5. Profiles contain only user-configurable fields.                          #
# --------------------------------------------------------------------------- #


def test_packaged_profiles_load():
    for profile in ("exploratory", "standard", "debugging"):
        assert isinstance(load_run_config(profile=profile), ResolvedRunConfig)
