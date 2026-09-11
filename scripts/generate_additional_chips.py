"""Generate additional Sentinel-2 chips from selected FIRMS observations."""

from pathlib import Path
import json
import sys

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the project root importable so that:
# from ml.ingestion.sentinel import ...
# works when this script is executed from /scripts.
sys.path.insert(0, str(PROJECT_ROOT))


from ml.ingestion.sentinel import (  # noqa: E402
    find_scene,
    create_multispectral_chip,
)


# ---------------------------------------------------------
# Input / output
# ---------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "firms_final_selection.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "images"
    / "demo_chips"
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Selection file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Generating {len(df)} additional Sentinel chips...\n"
    )

    successful = 0

    for _, row in df.iterrows():

        demo_id = int(row["demo_id"])

        latitude = float(row["latitude"])
        longitude = float(row["longitude"])

        firms_date = str(row["acq_date"])

        # -------------------------------------------------
        # Output filename
        # -------------------------------------------------

        chip_name = (
            f"final_{demo_id:03d}_"
            f"{latitude:.5f}_"
            f"{longitude:.5f}.tif"
        )

        chip_path = OUTPUT_DIR / chip_name

        metadata_path = chip_path.with_suffix(".json")

        print(
            f"[{demo_id:02d}/{len(df):02d}] "
            f"{latitude:.5f}, {longitude:.5f}"
        )

        # -------------------------------------------------
        # Search window
        # -------------------------------------------------

        start_date = (
            pd.Timestamp(firms_date)
            - pd.Timedelta(10, unit="D")
        ).strftime("%Y-%m-%d")

        end_date = (
            pd.Timestamp(firms_date)
            + pd.Timedelta(10, unit="D")
        ).strftime("%Y-%m-%d")

        # -------------------------------------------------
        # Find Sentinel scene
        # -------------------------------------------------

        scene = find_scene(
            latitude,
            longitude,
            start_date=start_date,
            end_date=end_date,
        )

        if scene is None:
            print("  No Sentinel scene found.\n")
            continue

        print(f"  Scene: {scene.id}")

        # -------------------------------------------------
        # Create multispectral chip
        # -------------------------------------------------

        print("  Creating 7-band chip...")

        create_multispectral_chip(
            scene,
            latitude,
            longitude,
            chip_path,
        )

        # -------------------------------------------------
        # Create metadata
        # -------------------------------------------------

        metadata = {
            "demo_id": demo_id,

            "chip": chip_name,

            "firms": {
                "latitude": latitude,
                "longitude": longitude,
                "acq_date": firms_date,
                "acq_time": int(row["acq_time"]),
                "satellite": "VIIRS_SNPP_NRT",
                "instrument": "VIIRS",
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

            "bands": [
                "B02",
                "B03",
                "B04",
                "B08",
                "B11",
                "B12",
                "SCL",
            ],

            "chip_size": [
                224,
                224,
            ],

            "source": (
                "NASA FIRMS + "
                "Earth Search Sentinel-2 L2A COG"
            ),
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        successful += 1

        print("  Chip created.")
        print()

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print(
        f"Completed: {successful}/{len(df)} chips"
    )


if __name__ == "__main__":
    main()