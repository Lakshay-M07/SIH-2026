"""Generate final Sentinel-2 evidence chips from selected FIRMS observations."""

from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.sentinel import (  # noqa: E402
    find_scene,
    create_multispectral_chip,
)


SELECTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "firms_demo_selection.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "images"
    / "demo_chips"
)


def main():
    df = pd.read_csv(SELECTION_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Generating final chips for {len(df)} "
        "selected FIRMS observations...\n"
    )

    successful = 0

    for _, row in df.iterrows():

        demo_id = int(row["demo_id"])
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        firms_date = str(row["acq_date"])

        print(
            f"[{demo_id:02d}/{len(df):02d}] "
            f"{latitude:.5f}, {longitude:.5f}"
        )

        start_date = (
            pd.Timestamp(firms_date)
            - pd.Timedelta("5D")
        ).strftime("%Y-%m-%d")

        end_date = (
            pd.Timestamp(firms_date)
            + pd.Timedelta("5D")
        ).strftime("%Y-%m-%d")

        scene = find_scene(
            latitude,
            longitude,
            start_date=start_date,
            end_date=end_date,
        )

        if scene is None:
            print("  No Sentinel scene found.\n")
            continue

        chip_name = (
            f"demo_{demo_id:03d}_"
            f"{latitude:.5f}_"
            f"{longitude:.5f}.tif"
        )

        chip_path = OUTPUT_DIR / chip_name

        print(f"  Scene: {scene.id}")
        print("  Creating 7-band chip...")

        create_multispectral_chip(
            scene,
            latitude,
            longitude,
            chip_path,
        )

        metadata = {
            "demo_id": demo_id,
            "chip": chip_name,

            "firms": {
                "latitude": latitude,
                "longitude": longitude,
                "acq_date": firms_date,
                "acq_time": int(row["acq_time"]),
                "satellite": str(row["satellite"]),
                "instrument": str(row["instrument"]),
                "confidence": str(row["confidence"]),
                "frp": float(row["frp"]),
            },

            "sentinel": {
                "scene_id": scene.id,
                "acquisition_datetime": (
                    scene.datetime.isoformat()
                    if scene.datetime
                    else None
                ),
                "cloud_cover": scene.properties.get(
                    "eo:cloud_cover"
                ),
            },

            "chip": {
                "width": 224,
                "height": 224,
                "bands": [
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "B11",
                    "B12",
                    "SCL",
                ],
            },

            "source": (
                "NASA FIRMS + "
                "Earth Search Sentinel-2 L2A COG"
            ),
        }

        metadata_path = chip_path.with_suffix(".json")

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        successful += 1

        print("  Chip created.\n")

    print(
        f"Completed: {successful}/{len(df)} chips"
    )


if __name__ == "__main__":
    main()