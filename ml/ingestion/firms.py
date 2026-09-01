"""NASA FIRMS fire detection ingestion."""

import argparse
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_firms_data(
    source: str = "VIIRS_SNPP_NRT",
    bbox: str = "70,20,90,35",
    days: int = 1,
) -> pd.DataFrame:
    """Fetch FIRMS detections for a geographic bounding box."""

    if not FIRMS_MAP_KEY:
        raise RuntimeError("FIRMS_MAP_KEY is missing from .env")

    url = f"{BASE_URL}/{FIRMS_MAP_KEY}/{source}/{bbox}/{days}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return pd.read_csv(StringIO(response.text))


def save_firms_data(df: pd.DataFrame) -> Path:
    """Save FIRMS detections to the project's data directory."""

    output_dir = PROJECT_ROOT / "ml" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "firms_latest.csv"
    df.to_csv(output_file, index=False)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NASA FIRMS fire detections."
    )

    parser.add_argument(
        "--bbox",
        default="70,20,90,35",
        help="Bounding box: west,south,east,north",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of recent days to fetch",
    )

    parser.add_argument(
        "--source",
        default="VIIRS_SNPP_NRT",
        help="FIRMS data source",
    )

    args = parser.parse_args()

    print("Fetching NASA FIRMS data...")
    print(f"Source: {args.source}")
    print(f"Bounding box: {args.bbox}")
    print(f"Days: {args.days}")

    df = fetch_firms_data(
        source=args.source,
        bbox=args.bbox,
        days=args.days,
    )

    output_file = save_firms_data(df)

    print(f"Downloaded {len(df)} detections.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()