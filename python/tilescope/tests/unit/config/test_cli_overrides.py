"""Regression tests for CLI override detection.

Guards the fix for the argparse `default != DEFAULT_*` bug: previously the CLI
compared parsed values to packaged defaults, so a user could not explicitly
override a profile back to the packaged default via the CLI. The fix uses
`default=None` for override flags and includes them in the override dict only
when the user actually supplied them.
"""
import argparse
from pathlib import Path

from tilescope.tilescope import build_run_cli_overrides
from tilescope.defaults import DEFAULT_N_TESSELLATIONS


def _ns(**kwargs) -> argparse.Namespace:
    # Namespace with every override flag defaulting to None (matches parser).
    defaults = dict(
        n_tessellations=None,
        min_roi_spots=None,
        seed=None, output=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_no_flags_gives_empty_overrides():
    assert build_run_cli_overrides(_ns()) == {}


def test_single_flag_produces_nested_key():
    result = build_run_cli_overrides(_ns(n_tessellations=7))
    assert result == {'tessellation': {'n': 7}}


def test_multiple_flags_across_sections():
    result = build_run_cli_overrides(_ns(
        n_tessellations=5, min_roi_spots=100, seed=123,
    ))
    assert result == {
        'tessellation': {'n': 5},
        'roi_filters': {'min_roi_spots': 100},
        'seed': 123,
    }


def test_flag_equal_to_packaged_default_is_still_an_override():
    # This is the whole point of the fix: passing --n-tessellations equal to
    # the packaged default must override a profile that set a different value.
    result = build_run_cli_overrides(_ns(n_tessellations=DEFAULT_N_TESSELLATIONS))
    assert result == {'tessellation': {'n': DEFAULT_N_TESSELLATIONS}}


def test_seed_equal_to_legacy_default_42_is_override():
    # Old code special-cased seed with `!= 42`, which silently dropped it.
    result = build_run_cli_overrides(_ns(seed=42))
    assert result == {'seed': 42}


def test_output_equal_to_legacy_default_is_override():
    result = build_run_cli_overrides(_ns(output=Path('tilescope_output')))
    assert result == {'output_dir': 'tilescope_output'}
