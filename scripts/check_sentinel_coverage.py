from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pystac_client import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIRMS_FILE = PROJECT_ROOT / "ml" / "data" / "firms_latest.csv"

STAC_URL = "https://earth-search.aws.element84.com/v1/"
COLLECTION = "sentinel-2-l2a"

DATE_WINDOW_DAYS = 5


def main():
    df = pd.read_csv(FIRMS_FILE)

    catalog = Client.open(STAC_URL)

    covered = 0

    print(f"Checking {len(df)} FIRMS hotspots...\n")

    for index, row in df.iterrows():

        latitude = row["latitude"]
        longitude = row["longitude"]

        firms_date = datetime.strptime(
            str(row["acq_date"]),
            "%Y-%m-%d",
        ).date()

        start_date = firms_date - timedelta(
            days=DATE_WINDOW_DAYS
        )

        end_date = firms_date + timedelta(
            days=DATE_WINDOW_DAYS
        )

        search = catalog.search(
            collections=[COLLECTION],
            bbox=[
                longitude - 0.01,
                latitude - 0.01,
                longitude + 0.01,
                latitude + 0.01,
            ],
            datetime=(
                f"{start_date.isoformat()}/"
                f"{end_date.isoformat()}"
            ),
            max_items=5,
        )

        items = list(search.items())

        if items:
            covered += 1

            items.sort(
                key=lambda item: item.properties.get(
                    "eo:cloud_cover",
                    100,
                )
            )

            scene = items[0]

            print(
                f"{index + 1:02d}  "
                f"{latitude:.5f}, {longitude:.5f}  ->  "
                f"{scene.id}  "
                f"(cloud={scene.properties.get('eo:cloud_cover')})"
            )

        else:
            print(
                f"{index + 1:02d}  "
                f"{latitude:.5f}, {longitude:.5f}  ->  "
                f"NO SCENE"
            )

    print()
    print(
        f"Sentinel scenes found: "
        f"{covered}/{len(df)}"
    )


if __name__ == "__main__":
    main()