#!/usr/bin/env python3
"""Example: Using TileScope as a Python library (programmatic API).

This script demonstrates the programmatic API for integration with other tools
like R/Seurat via reticulate. The API is designed for library use, not CLI.

For CLI usage, use tilescope.py instead.
"""

from __future__ import annotations

import traceback


import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tilescope.api.core import (
    run_tilescope_from_csv,
    run_tilescope_from_df,
    make_run_config,
    validate_run_config,
    validate_tilescope_dataframe,
    __version__,
)


def example_1_basic_csv_analysis():
    """Example 1: Basic CSV analysis with default settings."""
    print("=" * 70)
    print("Example 1: Basic CSV Analysis")
    print("=" * 70)

    # Use test fixture for demonstration
    csv_path = "tests/fixtures/grid_square_hole_and_circle.csv"

    results = run_tilescope_from_csv(
        input_csv=csv_path,
        profile="debugging",  # Fast profile for demonstration
        output_dir="tests/tests_output/example_output_1/",
        export=True,
        html_report=True,
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    print(f"\n✓ Analysis complete!")
    print(f"  - Parent ROIs: {len(results['roi_enrichment'])}")
    print(f"  - Consensus hexagons: {len(results.get('consensus_hexes_df', []))}")
    print(f"  - Output directory: tests/tests_output/example_output_1/")


def example_2_custom_configuration():
    """Example 2: Custom configuration with overrides."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)

    # Create custom config with profile + overrides
    config = make_run_config(
        profile="debugging",
        overrides={
            "tessellation": {
                "n": 3
            }
        },
        seed=42,
        verbose=True
    )

    print(f"\n✓ Config created:")
    print(f"  - Profile: debugging")
    print(f"  - Tessellations: {config.run_config.tessellation.n}")
    print(f"  - High threshold: {config.resolved_policy.thresholds.q_high}")
    print(f"  - Seed: {config.run_config.seed}")

    # Run analysis with custom config
    csv_path = "tests/fixtures/grid_square_hole_and_circle.csv"

    results = run_tilescope_from_csv(
        input_csv=csv_path,
        run_config=config,
        output_dir="tests/tests_output/example_output_2/",
        export=False,  # Skip export for speed
        return_minimal=True,  # Return only summary tables (memory efficient)
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    print(f"\n✓ Analysis complete with custom settings!")
    print(f"  - ROI enrichment: {len(results['roi_enrichment'])} rows")
    print(f"  - ROI consensus correlations: {len(results['roi_consensus_correlations_df'])} rows")


def example_3_dataframe_input():
    """Example 3: Analyze DataFrame directly (no CSV)."""
    print("\n" + "=" * 70)
    print("Example 3: DataFrame Input")
    print("=" * 70)

    # Load CSV into DataFrame
    csv_path = "tests/fixtures/grid_square_hole_and_circle.csv"
    df = pd.read_csv(csv_path)

    print(f"\n✓ DataFrame loaded:")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")

    # Validate DataFrame format
    validate_tilescope_dataframe(df)
    print(f"  - ✓ DataFrame format is valid")

    # Create config and run analysis
    config = make_run_config(profile="debugging")

    results = run_tilescope_from_df(
        df=df,
        config=config,
        output_dir="tests/tests_output/example_output_3/",
        export=False,
        return_minimal=True,
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    print(f"\n✓ Analysis complete from DataFrame!")
    print(f"  - ROI enrichment: {len(results['roi_enrichment'])} rows")


def example_4_config_validation():
    """Example 4: Config validation and inspection."""
    print("\n" + "=" * 70)
    print("Example 4: Config Validation")
    print("=" * 70)

    # Create config from partial dict (merges with defaults)
    config_dict = {
        "tessellation": {"n": 5},
        "consensus": {"min_tess_coverage_fraction": 0.6},
    }

    print(f"\n✓ Input config dict:")
    print(f"  {config_dict}")

    # Validate and merge with defaults
    config = validate_run_config(config_dict)

    print(f"\n✓ Validated config (merged with defaults):")
    print(f"  - Tessellations: {config.run_config.tessellation.n}")
    print(f"  - Analysis policy: {config.resolved_policy.analysis_policy_version}")


def example_5_minimal_mode_for_r():
    """Example 5: Minimal return mode (optimized for R integration)."""
    print("\n" + "=" * 70)
    print("Example 5: Minimal Mode (R Integration)")
    print("=" * 70)

    csv_path = "tests/fixtures/grid_square_hole_and_circle.csv"

    # Run with return_minimal=True (excludes heavy objects)
    results = run_tilescope_from_csv(
        input_csv=csv_path,
        profile="debugging",
        output_dir="tests/tests_output/example_output_5/",
        export=False,
        return_minimal=True,  # Memory efficient for R
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    print(f"\n✓ Minimal results returned:")
    print(f"  - Keys: {list(results.keys())}")
    print(f"  - roi_enrichment: {type(results['roi_enrichment']).__name__} ({len(results['roi_enrichment'])} rows)")
    print(f"  - roi_correlations: {type(results['roi_correlations']).__name__} ({len(results['roi_correlations'])} rows)")
    print(f"  - run_metadata: {type(results['run_metadata']).__name__}")

    print(f"\n✓ Heavy objects excluded (memory efficient):")
    print(f"  - analysis_df: Not included")
    print(f"  - canvas: Not included")


def main():
    """Run all examples."""
    print(f"TileScope Programmatic API Examples (v{__version__})")
    print("=" * 70)

    try:
        example_1_basic_csv_analysis()
        example_2_custom_configuration()
        example_3_dataframe_input()
        example_4_config_validation()
        example_5_minimal_mode_for_r()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        print("\nFor R integration via reticulate:")
        print("  api <- reticulate::import('api')")
        print("  results <- api$run_tilescope_from_csv('data.csv', profile='standard', feature_a_id='Neuron', feature_b_id='AC')")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
