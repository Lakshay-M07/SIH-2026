from pathlib import Path
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "sentinel_candidate_hotspots.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "firms_final_selection.csv"
)


def distance_km(lat1, lon1, lat2, lon2):
    """Approximate distance between two coordinates."""
    lat_km = 111.0
    lon_km = 111.0 * math.cos(
        math.radians((lat1 + lat2) / 2)
    )

    return math.sqrt(
        ((lat1 - lat2) * lat_km) ** 2
        + ((lon1 - lon2) * lon_km) ** 2
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    # Prefer low cloud cover, then stronger FIRMS FRP.
    df = df.sort_values(
        ["cloud", "frp"],
        ascending=[True, False],
    ).reset_index(drop=True)

    selected = []

    # Require selected points to be reasonably separated.
    MIN_DISTANCE_KM = 25

    for _, row in df.iterrows():

        if not selected:
            selected.append(row)
            continue

        too_close = False

        for existing in selected:

            distance = distance_km(
                float(row["latitude"]),
                float(row["longitude"]),
                float(existing["latitude"]),
                float(existing["longitude"]),
            )

            if distance < MIN_DISTANCE_KM:
                too_close = True
                break

        if not too_close:
            selected.append(row)

        if len(selected) == 10:
            break

    result = pd.DataFrame(selected).copy()

    result.insert(
        0,
        "demo_id",
        range(1, len(result) + 1),
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Selected {len(result)} final additional observations."
    )

    print()

    print(
        result[
            [
                "demo_id",
                "latitude",
                "longitude",
                "acq_date",
                "confidence",
                "frp",
                "scene",
                "cloud",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()