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

PROVENANCE_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "label_provenance.csv"
)


def main():
    # -----------------------------------------------------
    # Find exactly the 20 new production chips
    # -----------------------------------------------------

    chip_files = sorted(
        list(CHIP_DIR.glob("demo_*.tif"))
        + list(CHIP_DIR.glob("final_*.tif"))
    )

    if len(chip_files) != 20:
        raise RuntimeError(
            f"Expected 20 production chips, "
            f"found {len(chip_files)}"
        )

    rows = []

    # -----------------------------------------------------
    # Assign unique IDs 1-20
    # -----------------------------------------------------

    for new_id, chip_path in enumerate(
        chip_files,
        start=1,
    ):

        metadata_path = chip_path.with_suffix(".json")

        if not metadata_path.exists():
            raise RuntimeError(
                f"Missing metadata for "
                f"{chip_path.name}"
            )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        # Update ID.
        metadata["demo_id"] = new_id

        # Preserve the actual chip filename.
        metadata["chip"] = chip_path.name

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        firms = metadata["firms"]
        sentinel = metadata["sentinel"]

        rows.append(
            {
                "chip": chip_path.name,
                "demo_id": new_id,

                "latitude": firms["latitude"],
                "longitude": firms["longitude"],

                "firms_date": firms["acq_date"],
                "firms_time": firms["acq_time"],
                "firms_confidence": firms["confidence"],
                "firms_frp": firms["frp"],

                "sentinel_scene": sentinel["scene_id"],
                "sentinel_datetime": (
                    sentinel["acquisition_datetime"]
                ),
                "sentinel_cloud_cover": (
                    sentinel["cloud_cover"]
                ),

                "label": "fire_candidate",
                "label_source": (
                    "NASA FIRMS VIIRS"
                ),

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

    # -----------------------------------------------------
    # Rebuild provenance
    # -----------------------------------------------------

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "demo_id"
    ).reset_index(drop=True)

    df.to_csv(
        PROVENANCE_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print(
        f"Production chips: {len(chip_files)}"
    )

    print(
        f"Unique IDs: {df['demo_id'].nunique()}"
    )

    print(
        f"Provenance records: {len(df)}"
    )

    print(
        f"Unique chip names: {df['chip'].nunique()}"
    )

    print()

    if (
        len(chip_files) == 20
        and df["demo_id"].nunique() == 20
        and len(df) == 20
        and df["chip"].nunique() == 20
    ):
        print("VALIDATION: PASS")
    else:
        print("VALIDATION: FAIL")

    print()
    print(
        f"Provenance saved to: "
        f"{PROVENANCE_FILE}"
    )


if __name__ == "__main__":
    main()