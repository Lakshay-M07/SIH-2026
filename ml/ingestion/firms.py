"""NASA FIRMS data ingestion module for THERMOGRID.

Fetches Near-Real-Time (NRT) active thermal hotspot detections from the
NASA FIRMS REST API (VIIRS / MODIS) or syncs local orbital observations.
"""

import argparse
import csv
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default coverage: Indian Subcontinent [West, South, East, North]
DEFAULT_BBOX = "67,6,99,37"
DEFAULT_SOURCE = "VIIRS_SNPP_NRT"
NASA_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def get_map_key() -> str:
    """Retrieve NASA FIRMS MAP_KEY from environment or .env."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key

    # Check root, server, or client .env
    for env_path in [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "server" / ".env",
        PROJECT_ROOT / "client" / ".env",
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FIRMS_MAP_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    return ""


def fetch_firms_nrt(
    map_key: str,
    bbox: str = DEFAULT_BBOX,
    source: str = DEFAULT_SOURCE,
    day_range: int = 1,
    timeout: int = 20,
) -> list[dict]:
    """Query the NASA FIRMS Area API and parse CSV responses."""
    url = f"{NASA_FIRMS_BASE_URL}/{map_key}/{source}/{bbox}/{day_range}"
    print(f"🛰️  Querying NASA FIRMS API ({source}, range={day_range}d)...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "THERMOGRID-Geospatial-Intelligence/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode("utf-8")
            if "Invalid MAP_KEY" in content or "Unauthorized" in content:
                raise ValueError("Invalid NASA FIRMS MAP_KEY supplied.")

            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            print(f"✅ Successfully ingested {len(rows)} detections from NASA FIRMS.")
            return rows
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"NASA FIRMS API HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"NASA FIRMS Network Connection Error: {e.reason}")


def save_firms_csv(records: list[dict], target_path: Path) -> Path:
    """Save records to CSV with standard schema."""
    if not records:
        print("⚠️  No records to save.")
        return target_path

    fieldnames = list(records[0].keys())
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"💾 Saved {len(records)} records to {target_path}")
    return target_path


def main() -> None:
    """CLI execution."""
    parser = argparse.ArgumentParser(description="NASA FIRMS Ingestion CLI")
    parser.add_argument("--key", help="NASA FIRMS MAP_KEY", default="")
    parser.add_argument("--bbox", help="Bounding box west,south,east,north", default=DEFAULT_BBOX)
    parser.add_argument("--source", help="Satellite instrument source", default=DEFAULT_SOURCE)
    parser.add_argument("--days", help="Day range (1-10)", type=int, default=1)
    parser.add_argument("--output", help="Output file path", default=str(DATA_DIR / "firms_latest.csv"))
    args = parser.parse_args()

    map_key = args.key or get_map_key()
    out_file = Path(args.output)

    if not map_key:
        print("ℹ️  No FIRMS_MAP_KEY configured.")
        print("   Checking existing dataset in repository...")
        if out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"   Using existing baseline dataset: {len(rows)} records in {out_file.name}")
            print(f"   To fetch live directly from NASA, get a free key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
            return
        else:
            print("❌ No existing dataset and no MAP_KEY provided.")
            sys.exit(1)

    try:
        records = fetch_firms_nrt(map_key, args.bbox, args.source, args.days)
        save_firms_csv(records, out_file)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

