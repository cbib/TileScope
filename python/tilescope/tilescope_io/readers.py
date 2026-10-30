"""
Centralized data reading utilities for TileScope.

Provides standardized data input functions with schema validation.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Union, Optional

from tilescope.cols import (
    COORD_X,
    COORD_Y,
    GENE1,
    GENE2,
    ROI_ID
)

logger = logging.getLogger(__name__)


def read_expression_csv(file_path: Union[str, Path], dataset_schema: Optional[object] = None) -> pd.DataFrame:
    """Read and validate user-provided expression CSV."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Expression file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read expression CSV from {file_path}: {e}")

    # Determine column mapping (schema overrides default)
    if dataset_schema:
        # Use schema mapping: source column → canonical column
        column_mapping = dataset_schema.columns
        required_canonical_cols = {GENE1, GENE2, ROI_ID, COORD_X, COORD_Y}

        # Check that all canonical columns are mapped
        mapped_canonical_cols = set(column_mapping.values())
        missing_canonical = required_canonical_cols - mapped_canonical_cols
        if missing_canonical:
            raise ValueError(
                f"Dataset schema missing mappings for canonical columns: {missing_canonical}\n"
                f"Schema maps: {column_mapping}"
            )

        # Invert mapping: canonical → source
        inv_mapping = {canonical: source for source, canonical in column_mapping.items()}

        # Check that source columns exist in the file
        required_source_cols = [inv_mapping.get(canonical) for canonical in required_canonical_cols]
        missing_cols = [col for col in required_source_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required source columns: {missing_cols}\n"
                f"Expected (from schema): {required_source_cols}\n"
                f"Found columns: {list(df.columns)}"
            )

        # Apply mapping: rename source columns to canonical
        # Build rename dict: source → canonical for all mapped columns
        rename_dict = column_mapping.copy()
        logger.info("Applying dataset schema mapping:")
        for source, canonical in sorted(column_mapping.items()):
            logger.info(f"  {source} → {canonical}")

        df = df.rename(columns=rename_dict)
    else:
        # Default mapping (backward compatible)
        required_cols = [GENE1, GENE2, ROI_ID, COORD_X, COORD_Y]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Expected columns: {required_cols}\n"
                f"Found columns: {list(df.columns)}"
            )

        # Rename to canonical columns
        df = df.rename(columns={
            GENE1: GENE1,
            GENE2: GENE2,
            ROI_ID: ROI_ID  # Already uses constant
        })

    if df.empty:
        raise ValueError("Expression data is empty")

    # Validate data types
    if not pd.api.types.is_numeric_dtype(df[COORD_X]) or not pd.api.types.is_numeric_dtype(df[COORD_Y]):
        raise ValueError("Coordinates (x, y) must be numeric")

    if not pd.api.types.is_numeric_dtype(df[GENE1]) or not pd.api.types.is_numeric_dtype(df[GENE2]):
        raise ValueError(f"Gene expression columns must be numeric")

    # Ensure ROI IDs are strings for consistency
    df[ROI_ID] = df[ROI_ID].astype(str)

    # Convert coordinates to float (normalize int grids to continuous space)
    # Canvas-based: all coordinates flow through ℝ² pipeline
    df[COORD_X] = df[COORD_X].astype(float)
    df[COORD_Y] = df[COORD_Y].astype(float)

    logger.info(f"Loaded {len(df)} spots across {df[ROI_ID].nunique()} ROIs")
    logger.info(f"ROI IDs: {sorted(df[ROI_ID].unique())}")

    return df
