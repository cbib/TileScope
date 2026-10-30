"""TileScope Programmatic API.

This package provides programmatic access to TileScope functionality for
integration with other tools (e.g., R/Seurat via reticulate).

For CLI usage, use the tilescope.py script in the parent directory.

Main Functions:
    - run_tilescope_from_csv: Analyze CSV file
    - run_tilescope_from_df: Analyze pandas DataFrame
    - make_run_config: Create configuration from profile + overrides
    - validate_run_config: Validate configuration dict
    - validate_tilescope_dataframe: Validate DataFrame format

Example:
    from tilescope.api import run_tilescope_from_csv

    results = run_tilescope_from_csv(
        "data.csv", profile="standard",
        feature_a_id="Neuron", feature_b_id="AC",
    )
"""

from tilescope.api.core import (
    run_tilescope_from_csv,
    run_tilescope_from_df,
    make_run_config,
    validate_run_config,
    validate_tilescope_dataframe,
    __version__,
)

__all__ = [
    "run_tilescope_from_csv",
    "run_tilescope_from_df",
    "make_run_config",
    "validate_run_config",
    "validate_tilescope_dataframe",
    "__version__",
]
