from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "ml" / "data" / "firms_latest.csv"
OUTPUT_FILE = PROJECT_ROOT / "ml" / "data" / "firms_demo_selection.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Select 8 observations from each available date.
    # Within each date, spread the selection across
    # the dataset rather than taking adjacent detections.
    selected_parts = []

    for date, group in df.groupby("acq_date", sort=True):
        group = group.reset_index(drop=True)

        n = min(8, len(group))

        positions = (
            pd.Series(range(n))
            .mul(len(group) - 1)
            .div(max(n - 1, 1))
            .round()
            .astype(int)
        )

        selected = group.iloc[positions.tolist()]
        selected_parts.append(selected)

    selected_df = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # If fewer than 24 were available, that's okay.
    # Remove accidental exact duplicate coordinates.
    selected_df = selected_df.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time"]
    )

    selected_df.insert(
        0,
        "demo_id",
        range(1, len(selected_df) + 1),
    )

    selected_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Selected {len(selected_df)} "
        f"genuine FIRMS observations."
    )

    print()
    print(
        selected_df[
            [
                "demo_id",
                "latitude",
                "longitude",
                "acq_date",
                "acq_time",
                "confidence",
                "frp",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()