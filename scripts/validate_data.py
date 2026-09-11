# pyrefly: ignore-errors

from pathlib import Path
import csv
from datetime import datetime

import rasterio


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "ml" / "data"
IMAGES_DIR = DATA_DIR / "images"
CHIPS_DIR = IMAGES_DIR / "demo_chips"

FIRMS_FILE = DATA_DIR / "firms_latest.csv"
PROVENANCE_FILE = DATA_DIR / "label_provenance.csv"


# =============================================================================
# EXPECTED STRUCTURE
# =============================================================================

REQUIRED_DIRECTORIES = [
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    DATA_DIR / "external",
    IMAGES_DIR,
    CHIPS_DIR,
]

REQUIRED_FIRMS_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "bright_ti5",
    "frp",
    "daynight",
]

REQUIRED_PROVENANCE_COLUMNS = [
    "chip",
    "demo_id",
    "latitude",
    "longitude",
    "firms_date",
    "firms_time",
    "firms_confidence",
    "firms_frp",
    "sentinel_scene",
    "sentinel_datetime",
    "sentinel_cloud_cover",
    "label",
    "label_source",
    "image_source",
    "bands",
    "chip_size",
]

EXPECTED_CHIP_WIDTH = 224
EXPECTED_CHIP_HEIGHT = 224
EXPECTED_CHIP_BANDS = 7
EXPECTED_PRODUCTION_CHIPS = 20


# =============================================================================
# DIRECTORY VALIDATION
# =============================================================================

def check_directories() -> bool:
    """Check that all required project directories exist."""

    print("Checking data directories...")

    valid = True

    for directory in REQUIRED_DIRECTORIES:
        if directory.is_dir():
            print(f"[PASS] {directory.relative_to(ROOT)}")
        else:
            print(f"[FAIL] Missing: {directory.relative_to(ROOT)}")
            valid = False

    return valid


# =============================================================================
# CSV READER
# =============================================================================

def read_csv_file(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file."""

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        header = reader.fieldnames or []
        rows = list(reader)

    return header, rows


# =============================================================================
# PROVENANCE SCHEMA
# =============================================================================

def check_provenance_schema() -> bool:
    """Validate the provenance CSV schema."""

    print("\nChecking label provenance...")

    if not PROVENANCE_FILE.is_file():
        print("[FAIL] label_provenance.csv is missing")
        return False

    try:
        header, _ = read_csv_file(PROVENANCE_FILE)
    except (OSError, csv.Error) as error:
        print(f"[FAIL] Could not read provenance file: {error}")
        return False

    if header == REQUIRED_PROVENANCE_COLUMNS:
        print("[PASS] Provenance schema is valid")
        return True

    print("[FAIL] Provenance schema is invalid")
    print(f"Expected: {REQUIRED_PROVENANCE_COLUMNS}")
    print(f"Found:    {header}")

    return False


# =============================================================================
# PROVENANCE ROWS
# =============================================================================

def check_provenance_rows() -> bool:
    """Validate provenance records and referenced chips."""

    print("\nChecking provenance rows...")

    if not PROVENANCE_FILE.is_file():
        print("[FAIL] Provenance file is missing")
        return False

    try:
        _, rows = read_csv_file(PROVENANCE_FILE)
    except (OSError, csv.Error) as error:
        print(f"[FAIL] Could not read provenance file: {error}")
        return False

    print(f"[INFO] Provenance records found: {len(rows)}")

    if not rows:
        print("[FAIL] No provenance records found")
        return False

    valid = True

    # Required fields
    for index, row in enumerate(rows, start=2):
        missing_fields = []

        for field in REQUIRED_PROVENANCE_COLUMNS:
            value = row.get(field, "")

            if value is None or not str(value).strip():
                missing_fields.append(field)

        if missing_fields:
            print(
                f"[FAIL] Row {index} missing: "
                + ", ".join(missing_fields)
            )
            valid = False

    if valid:
        print("[PASS] Required provenance values are present")

    # Demo IDs
    demo_ids = []

    for row in rows:
        demo_id = str(row.get("demo_id", "")).strip()

        if demo_id:
            demo_ids.append(demo_id)

    duplicate_ids = len(demo_ids) - len(set(demo_ids))

    if duplicate_ids == 0:
        print("[PASS] Demo IDs are unique")
    else:
        print(f"[FAIL] Duplicate demo IDs found: {duplicate_ids}")
        valid = False

    # Labels
    invalid_labels = 0

    for row in rows:
        label = str(row.get("label", "")).strip()

        if label != "fire_candidate":
            invalid_labels += 1

    if invalid_labels == 0:
        print("[PASS] Labels are marked as fire_candidate")
    else:
        print(
            f"[FAIL] {invalid_labels} invalid label(s) found"
        )
        valid = False

    # Chip existence
    missing_chips = []

    for row in rows:
        chip_name = str(row.get("chip", "")).strip()

        if not chip_name:
            continue

        chip_path = CHIPS_DIR / chip_name

        if not chip_path.is_file():
            missing_chips.append(chip_name)

    if not missing_chips:
        print("[PASS] All provenance chips exist")
    else:
        print(
            f"[FAIL] Missing provenance chips: "
            f"{len(missing_chips)}"
        )

        for chip_name in missing_chips:
            print(f"       {chip_name}")

        valid = False

    return valid


# =============================================================================
# FIRMS VALIDATION
# =============================================================================

def check_firms_data() -> bool:
    """Validate the NASA FIRMS CSV artifact."""

    print("\nChecking FIRMS data artifact...")

    if not FIRMS_FILE.is_file():
        print("[FAIL] firms_latest.csv is missing")
        return False

    try:
        header, rows = read_csv_file(FIRMS_FILE)
    except (OSError, csv.Error) as error:
        print(f"[FAIL] Could not read FIRMS CSV: {error}")
        return False

    print(f"[INFO] FIRMS rows: {len(rows)}")
    print(f"[INFO] FIRMS columns: {len(header)}")

    if not rows:
        print("[FAIL] FIRMS file contains no observations")
        return False

    # Required columns
    missing_columns = [
        column
        for column in REQUIRED_FIRMS_COLUMNS
        if column not in header
    ]

    if missing_columns:
        print(
            "[FAIL] Missing required FIRMS columns: "
            + ", ".join(missing_columns)
        )
        return False

    print("[PASS] Required FIRMS columns are present")

    valid = True

    # Numeric fields
    numeric_columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "scan",
        "track",
        "bright_ti5",
        "frp",
    ]

    for column in numeric_columns:
        invalid_count = 0

        for row in rows:
            value = str(row.get(column, "")).strip()

            try:
                float(value)
            except (ValueError, TypeError):
                invalid_count += 1

        if invalid_count > 0:
            print(
                f"[FAIL] {column}: "
                f"{invalid_count} non-numeric value(s)"
            )
            valid = False

    if valid:
        print("[PASS] Numeric FIRMS fields are valid")

    # Latitude / longitude
    latitude_valid = True
    longitude_valid = True

    for row in rows:
        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (ValueError, TypeError, KeyError):
            latitude_valid = False
            longitude_valid = False
            continue

        if not -90.0 <= latitude <= 90.0:
            latitude_valid = False

        if not -180.0 <= longitude <= 180.0:
            longitude_valid = False

    if latitude_valid:
        print("[PASS] Latitude values are within valid range")
    else:
        print("[FAIL] Invalid latitude value detected")
        valid = False

    if longitude_valid:
        print("[PASS] Longitude values are within valid range")
    else:
        print("[FAIL] Invalid longitude value detected")
        valid = False

    # Acquisition dates
    dates_valid = True

    for row in rows:
        date_value = str(row.get("acq_date", "")).strip()

        try:
            datetime.strptime(
                date_value,
                "%Y-%m-%d",
            )
        except ValueError:
            dates_valid = False

    if dates_valid:
        print("[PASS] Acquisition dates are valid")
    else:
        print("[FAIL] Invalid acquisition date detected")
        valid = False

    # Duplicate rows
    row_tuples = [
        tuple(row.get(column, "") for column in header)
        for row in rows
    ]

    duplicate_count = len(row_tuples) - len(set(row_tuples))

    if duplicate_count == 0:
        print("[PASS] No duplicate FIRMS rows detected")
    else:
        print(
            f"[INFO] Duplicate FIRMS rows detected: "
            f"{duplicate_count}"
        )

    # Additional columns
    extra_columns = [
        column
        for column in header
        if column not in REQUIRED_FIRMS_COLUMNS
    ]

    if extra_columns:
        print(
            "[INFO] Additional source columns preserved: "
            + ", ".join(extra_columns)
        )

    return valid


# =============================================================================
# SENTINEL-2 VALIDATION
# =============================================================================

def check_sentinel_chips() -> bool:
    """Validate production Sentinel-2 chips."""

    print("\nChecking Sentinel-2 chips...")

    tiff_files = sorted(
        list(CHIPS_DIR.glob("demo_*.tif"))
        + list(CHIPS_DIR.glob("final_*.tif"))
    )

    json_files = sorted(
        list(CHIPS_DIR.glob("demo_*.json"))
        + list(CHIPS_DIR.glob("final_*.json"))
    )

    print(f"[INFO] Production TIFF chips: {len(tiff_files)}")
    print(f"[INFO] Production JSON files: {len(json_files)}")

    valid = True

    # TIFF count
    if len(tiff_files) == EXPECTED_PRODUCTION_CHIPS:
        print(
            f"[PASS] {len(tiff_files)} "
            "Sentinel-2 TIFF chips found"
        )
    else:
        print(
            f"[FAIL] Expected "
            f"{EXPECTED_PRODUCTION_CHIPS} TIFF chips, "
            f"found {len(tiff_files)}"
        )
        valid = False

    # JSON count
    if len(json_files) == EXPECTED_PRODUCTION_CHIPS:
        print(
            f"[PASS] {len(json_files)} "
            "Sentinel-2 metadata files found"
        )
    else:
        print(
            f"[FAIL] Expected "
            f"{EXPECTED_PRODUCTION_CHIPS} metadata files, "
            f"found {len(json_files)}"
        )
        valid = False

    # TIFF validation
    chip_structure_valid = True

    for chip_path in tiff_files:
        try:
            with rasterio.open(chip_path) as src:
                width = src.width
                height = src.height
                bands = src.count

        except Exception as error:
            print(
                f"[FAIL] Could not read "
                f"{chip_path.name}: {error}"
            )
            chip_structure_valid = False
            continue

        if (
            width != EXPECTED_CHIP_WIDTH
            or height != EXPECTED_CHIP_HEIGHT
            or bands != EXPECTED_CHIP_BANDS
        ):
            print(
                f"[FAIL] {chip_path.name}: "
                f"{width}x{height}, {bands} bands"
            )
            chip_structure_valid = False

    if chip_structure_valid:
        print(
            "[PASS] All production chips are "
            "224x224 with 7 bands"
        )
    else:
        valid = False

    # Matching JSON metadata
    missing_metadata = []

    for chip_path in tiff_files:
        metadata_path = chip_path.with_suffix(".json")

        if not metadata_path.is_file():
            missing_metadata.append(metadata_path.name)

    if not missing_metadata:
        print("[PASS] All production chips have metadata")
    else:
        print(
            f"[FAIL] {len(missing_metadata)} "
            "chip(s) are missing metadata"
        )

        for metadata_name in missing_metadata:
            print(f"       {metadata_name}")

        valid = False

    return valid


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Run all Person C data validation checks."""

    print("=== SIH26162 Person C Data Validation ===\n")

    directory_ok = check_directories()
    provenance_schema_ok = check_provenance_schema()
    provenance_rows_ok = check_provenance_rows()
    firms_ok = check_firms_data()
    sentinel_ok = check_sentinel_chips()

    print("\n=== Validation Summary ===")

    if (
        directory_ok
        and provenance_schema_ok
        and provenance_rows_ok
        and firms_ok
        and sentinel_ok
    ):
        print("[PASS] Person C data foundation is valid")
        return 0

    print("[FAIL] Person C data foundation needs attention")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())