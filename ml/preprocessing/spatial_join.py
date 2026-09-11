"""Spatial join and infrastructure-distance feature extraction utilities.

Derived from SIH26162 Roadmap (Day 3 — Person B deliverable):
Calculates nearest-neighbor distances (in meters) from FIRMS thermal hotspots
to key industrial infrastructure types:
- distance_to_industrial_area
- distance_to_power_plant
- distance_to_refinery
- distance_to_mine
- distance_to_factory
- distance_to_major_road

Dependency:
    Consumes ml/data/external/osm_industrial.geojson produced by Person C (Day 5).
    Person A's ml/ingestion/osm.py pipeline is the upstream generator.

Anti-Leakage Rule (Roadmap Page 6, 8, 13):
These OSM infrastructure distances are strictly MODEL INPUTS, never target labels.
"""

import json
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# The 6 locked feature column names for Day 6 feature engineering contract
OSM_DISTANCE_COLUMNS = [
    "distance_to_industrial_area",
    "distance_to_power_plant",
    "distance_to_refinery",
    "distance_to_mine",
    "distance_to_factory",
    "distance_to_major_road",
]

DEFAULT_OSM_PATH = Path("ml/data/external/osm_industrial.geojson")
EARTH_RADIUS_METERS = 6371000.0


def haversine_np(lat1, lon1, lat2, lon2):
    """Computes great-circle distances in meters between arrays of coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_METERS * c


def distance_to_nearest(hotspot_df, facility_df, facility_type):
    """Calculates distance in meters from each hotspot to the nearest facility of a given type.

    Args:
        hotspot_df (pd.DataFrame or gpd.GeoDataFrame): DataFrame with 'latitude' and 'longitude'.
        facility_df (pd.DataFrame or gpd.GeoDataFrame): Facility DataFrame with 'facility_type',
                                                        'latitude', 'longitude'.
        facility_type (str): Facility category to measure against.

    Returns:
        pd.Series: Distance in meters (float64) for each hotspot.
    """
    sub = facility_df[facility_df["facility_type"] == facility_type]

    if sub.empty:
        # If no facility of this type exists in context, return a standard max distance fallback
        return pd.Series(999999.0, index=hotspot_df.index, dtype="float64")

    h_lats = hotspot_df["latitude"].values[:, np.newaxis]
    h_lons = hotspot_df["longitude"].values[:, np.newaxis]

    f_lats = sub["latitude"].values[np.newaxis, :]
    f_lons = sub["longitude"].values[np.newaxis, :]

    distances = haversine_np(h_lats, h_lons, f_lats, f_lons)
    min_distances = np.min(distances, axis=1)

    return pd.Series(min_distances, index=hotspot_df.index, dtype="float64")


def attach_osm_distance_features(hotspot_df, facility_gdf=None):
    """Attaches all six OSM infrastructure distance columns to a hotspot DataFrame.

    Contract:
    - distance_to_industrial_area
    - distance_to_power_plant
    - distance_to_refinery
    - distance_to_mine
    - distance_to_factory
    - distance_to_major_road
    """
    if facility_gdf is None:
        if not DEFAULT_OSM_PATH.exists():
            raise FileNotFoundError(
                f"OSM dataset not found at {DEFAULT_OSM_PATH}. "
                f"This file is produced by Person A's OSM ingestion pipeline (ml/ingestion/osm.py) "
                f"and deposited by Person C into ml/data/external/ (expected Day 5)."
            )
        if gpd is not None:
            facility_gdf = gpd.read_file(DEFAULT_OSM_PATH)
        else:
            with open(DEFAULT_OSM_PATH, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)
            rows = []
            for feat in geojson_data.get("features", []):
                coords = feat.get("geometry", {}).get("coordinates", [0, 0])
                props = feat.get("properties", {})
                rows.append({
                    "name": props.get("name", ""),
                    "facility_type": props.get("facility_type", ""),
                    "latitude": coords[1],
                    "longitude": coords[0],
                })
            facility_gdf = pd.DataFrame(rows)

    # Ensure latitude and longitude columns exist
    df = hotspot_df.copy()
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = df["lat"]
    if "longitude" not in df.columns and "lon" in df.columns:
        df["longitude"] = df["lon"]

    # Calculate distance for each of the 6 categories
    type_map = {
        "distance_to_industrial_area": "industrial_area",
        "distance_to_power_plant": "power_plant",
        "distance_to_refinery": "refinery",
        "distance_to_mine": "mine",
        "distance_to_factory": "factory",
        "distance_to_major_road": "major_road",
    }

    for col_name, fac_type in type_map.items():
        df[col_name] = distance_to_nearest(df, facility_gdf, fac_type)

    return df


def run_test_fixture():
    """Runs test against mock FIRMS parquet fixture and verified OSM data."""
    print("Running spatial join test fixture...")
    fixture_path = Path("ml/preprocessing/fixtures/mock_firms_sample.parquet")
    if not fixture_path.exists():
        raise FileNotFoundError(f"Mock fixture not found at {fixture_path}")

    hotspots = pd.read_parquet(fixture_path)
    print(f"Loaded {len(hotspots)} mock hotspots.")

    enriched = attach_osm_distance_features(hotspots)

    # Validation
    for col in OSM_DISTANCE_COLUMNS:
        assert col in enriched.columns, f"Missing expected column: {col}"
        assert not enriched[col].isnull().any(), f"Null values found in {col}"
        assert (enriched[col] >= 0).all(), f"Negative distances found in {col}"

    print("\n--- Successfully Enriched Hotspots with OSM Distances (meters) ---")
    display_cols = ["latitude", "longitude", "frp"] + OSM_DISTANCE_COLUMNS
    print(enriched[display_cols].to_string(index=False))
    print("\nAll 6 OSM distance feature columns successfully verified.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="OSM Spatial Join Distance Features")
    parser.add_argument(
        "--test-fixture",
        action="store_true",
        help="Run spatial join against mock FIRMS sample fixture",
    )
    args = parser.parse_args()

    if args.test_fixture:
        sys.exit(run_test_fixture())
    else:
        print("Usage: python -m ml.preprocessing.spatial_join --test-fixture")


if __name__ == "__main__":
    main()
