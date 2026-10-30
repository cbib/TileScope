"""Regression tests for unified profile=None semantics across the public API.

Before the fix, `make_run_config(profile=None)` skipped the profile layer
while `run_tilescope_from_csv(profile=None)` applied the 'standard' profile.
This test suite guards the unified convention: None or "" means no profile;
a named profile means apply that profile.
"""
import pytest

from tilescope.api import make_run_config
from tilescope.settings import load_run_config


def _model(cfg):
    # Cheap comparison: dump to dict of the fields profiles actually change.
    d = cfg.run_config.model_dump()
    return {
        "tess_n": d["tessellation"]["n"],
        "min_roi_spots": d["roi_filters"]["min_roi_spots"],
        "min_context_spots": d["roi_filters"]["min_context_spots"],
        "min_tess_coverage_fraction": d["consensus"]["min_tess_coverage_fraction"],
    }


def test_make_run_config_none_matches_empty_string():
    # None and "" are aliases: both skip the profile layer.
    assert _model(make_run_config(profile=None)) == _model(make_run_config(profile=""))


def test_make_run_config_default_is_no_profile():
    # No argument at all == profile=None.
    assert _model(make_run_config()) == _model(make_run_config(profile=None))


def test_make_run_config_none_differs_from_standard():
    # Standard profile changes tessellation.n (3 -> 10) and other fields.
    # This proves profile=None does NOT silently apply "standard".
    assert _model(make_run_config(profile=None)) != _model(make_run_config(profile="standard"))


def test_load_run_config_default_is_no_profile():
    # Settings-layer default now matches the public API: no profile.
    assert _model(load_run_config()) == _model(load_run_config(profile=None))
    assert _model(load_run_config()) == _model(load_run_config(profile=""))


def test_load_run_config_standard_differs_from_none():
    assert _model(load_run_config(profile=None)) != _model(load_run_config(profile="standard"))


def test_named_profile_still_applies():
    # The named-profile path must still work end-to-end.
    standard = _model(make_run_config(profile="standard"))
    assert standard["tess_n"] == 10  # standard_profile.json bumps this from 3


# Packaged-profile invariants — the (n, fraction, resolved floor) triple
# should stay in lockstep. Any change to profile JSON that shifts either
# axis must be intentional; catching drift here beats catching it in
# corpus provenance.
@pytest.mark.parametrize(
    "profile,expected_n,expected_fraction,expected_min_tess",
    [
        (None,          3,  2.0 / 3.0, 2),
        ("standard",    10, 2.0 / 3.0, 7),
        ("exploratory", 3,  2.0 / 3.0, 2),
        ("debugging",   2,  0.5,       1),
    ],
)
def test_profile_resolves_coverage_triple(
    profile, expected_n, expected_fraction, expected_min_tess,
):
    cfg = load_run_config(profile=profile)
    assert cfg.run_config.tessellation.n == expected_n
    assert (
        cfg.run_config.consensus.min_tess_coverage_fraction == expected_fraction
    )
    assert cfg.resolved_policy.consensus.min_tess_per_hex == expected_min_tess
