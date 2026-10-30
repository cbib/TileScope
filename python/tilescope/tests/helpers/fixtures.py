"""Shared test fixture helpers."""

from __future__ import annotations

from tilescope.tilescope_io import read_expression_csv
from tilescope.settings import load_dataset_schema
from tilescope.spatial import make_canvas
from tilescope.metrics.enrichment_thresholds import compute_global_thresholds
from tilescope.cols import GENE1, GENE2
from tilescope.settings import load_run_config


from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import pytest


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture_dataframe(
    fixture_name: str,
    *,
    rename_gene_columns: bool = True,
) -> pd.DataFrame:
    """Load a CSV fixture from tests/fixtures with optional gene column normalization."""
    path = FIXTURES_DIR / fixture_name
    df = pd.read_csv(path)
    if rename_gene_columns and "gene1" in df.columns:
        df = df.rename(columns={"gene1": "gene_1", "gene2": "gene_2"})
    return df


def load_fixture(fixture_name: str, schema_path: Optional[str] = None) -> pd.DataFrame:
    """Load test fixture from tests/fixtures/ using TileScope's expression CSV reader.

    Validates column schema and applies canonical transforms.
    Use this for loading test fixtures in a production-like way.
    """

    fixture_path = FIXTURES_DIR / fixture_name

    dataset_schema = None
    if schema_path:
        dataset_schema = load_dataset_schema(schema_path)

    return read_expression_csv(fixture_path, dataset_schema=dataset_schema)


def coord_fixture_paths(names: Optional[Sequence[str]] = None) -> List[Path]:
    """Return absolute paths for common coordinate fixtures."""
    default_names: Tuple[str, ...] = (
        "test_int_coords.csv",
        "test_float_coords.csv",
        "test_float_scaled_offset.csv",
    )
    selected = names if names is not None else default_names
    return [FIXTURES_DIR / name for name in selected]


def load_worldspace_dataset(fixture_path: Path):
    """Load an expression CSV and return (df, canvas, config, thresholds)."""

    df = read_expression_csv(fixture_path)
    canvas = make_canvas(df)
    # Coverage fraction 0.2 preserves the legacy min_tess=2 floor under
    # standard-profile tessellation.n=10 (ceil(0.2 * 10) = 2).
    config = load_run_config(
        profile="standard",
        set_overrides={
        'output_dir': 'tests/tests_output',
        'consensus': {
            'min_tess_coverage_fraction': 0.2,
        },
    },
    )
    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1].values,
        gene2_values=df[GENE2].values,
    )
    return df, canvas, config, thresholds


# =============================================================================
# Pytest fixtures (shared across geometry tests)
# =============================================================================

@pytest.fixture(params=coord_fixture_paths())
def fixture_path(request):
    """Parametrize over int/float/scaled coordinate fixtures."""
    return request.param


@pytest.fixture
def loaded_data(fixture_path):
    """Load expression data and compute coordinate space."""
    return load_worldspace_dataset(Path(fixture_path))
