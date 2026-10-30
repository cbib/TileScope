"""Generic DataFrame-to-CSV I/O.

Reporters own column selection, formatting, and schema validation. This
module only creates the destination directory and serializes the frame.
"""
from pathlib import Path

import pandas as pd


def export_dataframe_csv(
    df: pd.DataFrame,
    output_path: str,
    *,
    index: bool = False,
) -> str:
    """Write a reporter-prepared frame and return its path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    return output_path
