from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHIP_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "images"
    / "demo_chips"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "label_provenance.csv"
)


def main():
    rows = []

    # Only use the 20 final demo chips:
    # 10 demo_*.tif + 10 final_*.tif
    chip_files = sorted(
        list(CHIP_DIR.glob("demo_*.tif"))
        + list(CHIP_DIR.glob("final_*.tif"))
    )

    for chip_path in chip_files:

        metadata_path = chip_path.with_suffix(".json")

        if not metadata_path.exists():
            print(
                f"WARNING: Missing metadata: "
                f"{metadata_path.name}"
            )
            continue

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        firms = metadata["firms"]
        sentinel = metadata["sentinel"]

        rows.append(
            {
                "chip": chip_path.name,
                "demo_id": metadata["demo_id"],
                "latitude": firms["latitude"],
                "longitude": firms["longitude"],
                "firms_date": firms["acq_date"],
                "firms_time": firms["acq_time"],
                "firms_confidence": firms["confidence"],
                "firms_frp": firms["frp"],
                "sentinel_scene": sentinel["scene_id"],
                "sentinel_datetime": sentinel[
                    "acquisition_datetime"
                ],
                "sentinel_cloud_cover": sentinel[
                    "cloud_cover"
                ],
                "label": "fire_candidate",
                "label_source": "NASA FIRMS VIIRS",
                "image_source": (
                    "Earth Search Sentinel-2 L2A COG"
                ),
                "bands": (
                    "B02,B03,B04,B08,"
                    "B11,B12,SCL"
                ),
                "chip_size": "224x224",
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "demo_id"
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Created provenance table with "
        f"{len(df)} records."
    )

    print()
    print(
        df[
            [
                "chip",
                "demo_id",
                "firms_date",
                "firms_confidence",
                "firms_frp",
                "sentinel_scene",
                "sentinel_cloud_cover",
                "label",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()