from pathlib import Path
import pandas as pd
from pystac_client import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTION_FILE = (
    PROJECT_ROOT / "ml" / "data" / "firms_demo_selection.csv"
)

STAC_URL = "https://earth-search.aws.element84.com/v1/"
COLLECTION = "sentinel-2-l2a"


def main():
    df = pd.read_csv(SELECTION_FILE)
    catalog = Client.open(STAC_URL)

    print("Checking missing observations...\n")

    for _, row in df.iterrows():

        demo_id = int(row["demo_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        date = pd.Timestamp(row["acq_date"])

        output = (
            PROJECT_ROOT
            / "ml"
            / "data"
            / "images"
            / "demo_chips"
            / f"demo_{demo_id:03d}_"
            f"{lat:.5f}_{lon:.5f}.tif"
        )

        # Already generated — skip it.
        if output.exists():
            continue

        start = (date - pd.Timedelta("10D")).strftime("%Y-%m-%d")
        end = (date + pd.Timedelta("10D")).strftime("%Y-%m-%d")

        search = catalog.search(
            collections=[COLLECTION],
            bbox=[
                lon - 0.05,
                lat - 0.05,
                lon + 0.05,
                lat + 0.05,
            ],
            datetime=f"{start}/{end}",
            max_items=10,
        )

        items = list(search.items())

        print(
            f"{demo_id:02d}  "
            f"{lat:.5f}, {lon:.5f}  "
            f"({row['acq_date']})"
        )

        if not items:
            print("    NO SCENE\n")
            continue

        items.sort(
            key=lambda item: item.properties.get(
                "eo:cloud_cover",
                100,
            )
        )

        for item in items[:3]:
            print(
                "   ",
                item.id,
                "cloud=",
                item.properties.get(
                    "eo:cloud_cover"
                ),
            )

        print()


if __name__ == "__main__":
    main()