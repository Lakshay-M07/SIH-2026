"""Preprocessing loaders for FIRMS and geospatial datasets.

Maintains the schema-alignment boundary for Person A's Day 2 handoff.
"""

from pathlib import Path
import pandas as pd
from .schema_firms import validate_firms_columns, FIRMS_SCHEMA


def load_firms_parquet(path):
    """Loads and validates a FIRMS Parquet file against the schema contract.

    Args:
        path (str or Path): Path to the parquet file.

    Returns:
        pd.DataFrame: Validated FIRMS DataFrame with standard column names.

    Raises:
        FileNotFoundError: If the file does not exist (waiting on Person A handoff).
        ValueError: If required schema columns are missing or malformed.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"FIRMS Parquet file not found at '{file_path}'. "
            f"Waiting on Person A's Day 2 ingestion handoff."
        )

    df = pd.read_parquet(file_path)

    is_valid, missing = validate_firms_columns(df.columns)
    if not is_valid:
        raise ValueError(
            f"Malformed FIRMS Parquet format: Missing required contract fields: {missing}. "
            f"Expected schema: {list(FIRMS_SCHEMA.keys())}"
        )

    # Standardize column aliases if present
    rename_dict = {}
    if "lat" in df.columns and "latitude" not in df.columns:
        rename_dict["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        rename_dict["lon"] = "longitude"

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df
