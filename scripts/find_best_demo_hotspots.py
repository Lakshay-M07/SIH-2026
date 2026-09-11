from pathlib import Path
from datetime import timedelta

import pandas as pd
from pystac_client import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIRMS_FILE = PROJECT_ROOT / "ml" / "data" / "firms_latest.csv"
SELECTION_FILE = PROJECT_ROOT / "ml" / "data" / "firms_demo_selection.csv"

STAC_URL = "https://earth-search.aws.element84.com/v1/"
COLLECTION = "sentinel-2-l2a"


def main():
    firms = pd.read_csv(FIRMS_FILE)
    selected = pd.read_csv(SELECTION_FILE)

    # Coordinates already selected for the current 20-row plan.
    existing = set(
        zip(
            selected["latitude"].round(5),
            selected["longitude"].round(5),
        )
    )

    catalog = Client.open(STAC_URL)

    candidates = []

    print("Searching remaining FIRMS observations...\n")

    for index, row in firms.iterrows():

        lat = float(row["latitude"])
        lon = float(row["longitude"])

        key = (round(lat, 5), round(lon, 5))

        if key in existing:
            continue

        date = pd.Timestamp(row["acq_date"]).date()

        start = date - timedelta(days=10)
        end = date + timedelta(days=10)

        search = catalog.search(
            collections=[COLLECTION],
            bbox=[
                lon - 0.05,
                lat - 0.05,
                lon + 0.05,
                lat + 0.05,
            ],
            datetime=(
                f"{start.isoformat()}/"
                f"{end.isoformat()}"
            ),
            max_items=10,
        )

        items = list(search.items())

        if not items:
            continue

        # Best available scene = lowest cloud cover.
        items.sort(
            key=lambda item: item.properties.get(
                "eo:cloud_cover",
                100,
            )
        )

        scene = items[0]

        cloud = scene.properties.get(
            "eo:cloud_cover",
            100,
        )

        candidates.append(
            {
                "source_row": index,
                "latitude": lat,
                "longitude": lon,
                "acq_date": row["acq_date"],
                "acq_time": row["acq_time"],
                "confidence": row["confidence"],
                "frp": row["frp"],
                "scene": scene.id,
                "cloud": cloud,
            }
        )

    result = pd.DataFrame(candidates)

    if result.empty:
        print("No additional Sentinel candidates found.")
        return

    result = result.sort_values(
        ["cloud", "frp"],
        ascending=[True, False],
    )

    print(
        "Best additional candidates:\n"
    )

    print(
        result.head(20).to_string(
            index=False
        )
    )

    output = (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "sentinel_candidate_hotspots.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"Saved {len(result)} candidates to:"
    )
    print(output)


if __name__ == "__main__":
    main()