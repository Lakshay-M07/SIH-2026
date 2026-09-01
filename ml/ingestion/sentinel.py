"""Sentinel-2 ingestion for hotspot satellite evidence.

Person C workstream:
- Search Sentinel-2 L2A scenes using STAC.
- Access public Sentinel-2 Cloud-Optimized GeoTIFFs.
- Extract a 224x224 chip around a hotspot.
- Include RGB, NIR, SWIR and SCL information.
- Save traceability metadata.
"""

from pathlib import Path
import json

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform
from rasterio.windows import Window, transform as window_transform


STAC_URL = "https://earth-search.aws.element84.com/v1/"
COLLECTION = "sentinel-2-l2a"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "images"
    / "demo_chips"
)

CHIP_SIZE = 224


def find_scene(
    latitude,
    longitude,
    start_date,
    end_date,
    max_cloud_cover=20,
):
    """Find the lowest-cloud Sentinel-2 scene around a hotspot."""

    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=[COLLECTION],
        bbox=[
            longitude - 0.01,
            latitude - 0.01,
            longitude + 0.01,
            latitude + 0.01,
        ],
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_cover
            }
        },
        max_items=20,
    )

    items = list(search.items())

    if not items:
        return None

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100,
        )
    )

    return items[0]


def get_raster_coordinates(
    src,
    latitude,
    longitude,
):
    """Convert WGS84 latitude/longitude to the raster CRS."""

    x, y = transform(
        "EPSG:4326",
        src.crs,
        [longitude],
        [latitude],
    )

    return x[0], y[0]


def read_band(
    url,
    latitude,
    longitude,
    chip_size=CHIP_SIZE,
    resampling=Resampling.bilinear,
):
    """Read a 224x224 window around the hotspot."""

    with rasterio.open(url) as src:

        x, y = get_raster_coordinates(
            src,
            latitude,
            longitude,
        )

        row, col = src.index(x, y)

        half = chip_size // 2

        # Keep the window completely inside the raster.
        col_off = max(
            0,
            min(
                col - half,
                src.width - chip_size,
            ),
        )

        row_off = max(
            0,
            min(
                row - half,
                src.height - chip_size,
            ),
        )

        window = Window(
            col_off,
            row_off,
            chip_size,
            chip_size,
        )

        data = src.read(
            1,
            window=window,
            out_shape=(chip_size, chip_size),
            resampling=resampling,
        )

        return data, src.profile.copy(), window


def create_multispectral_chip(
    scene,
    latitude,
    longitude,
    output_path,
):
    """Create a 7-band Sentinel-2 evidence chip."""

    print("Reading B02 Blue...")
    blue, _, _ = read_band(
        scene.assets["blue"].href,
        latitude,
        longitude,
    )

    print("Reading B03 Green...")
    green, _, _ = read_band(
        scene.assets["green"].href,
        latitude,
        longitude,
    )

    print("Reading B04 Red...")
    red, red_profile, red_window = read_band(
        scene.assets["red"].href,
        latitude,
        longitude,
    )

    print("Reading B08 NIR...")
    nir, _, _ = read_band(
        scene.assets["nir"].href,
        latitude,
        longitude,
    )

    print("Reading B11 SWIR...")
    swir16, _, _ = read_band(
        scene.assets["swir16"].href,
        latitude,
        longitude,
    )

    print("Reading B12 SWIR...")
    swir22, _, _ = read_band(
        scene.assets["swir22"].href,
        latitude,
        longitude,
    )

    print("Reading SCL...")
    scl, _, _ = read_band(
        scene.assets["scl"].href,
        latitude,
        longitude,
        resampling=Resampling.nearest,
    )

    # Keep the seven bands together.
    chip = np.stack(
        [
            blue,
            green,
            red,
            nir,
            swir16,
            swir22,
            scl,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Georeferencing comes from the 10 m red-band grid.
    output_transform = window_transform(
        red_window,
        red_profile["transform"],
    )

    profile = red_profile.copy()

    profile.update(
        driver="GTiff",
        height=CHIP_SIZE,
        width=CHIP_SIZE,
        count=7,
        dtype=chip.dtype,
        transform=output_transform,
        compress="deflate",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(chip)

        dst.set_band_description(
            1,
            "B02_Blue",
        )

        dst.set_band_description(
            2,
            "B03_Green",
        )

        dst.set_band_description(
            3,
            "B04_Red",
        )

        dst.set_band_description(
            4,
            "B08_NIR",
        )

        dst.set_band_description(
            5,
            "B11_SWIR16",
        )

        dst.set_band_description(
            6,
            "B12_SWIR22",
        )

        dst.set_band_description(
            7,
            "SCL",
        )


def save_metadata(
    scene,
    latitude,
    longitude,
    chip_path,
):
    """Save traceability information for the chip."""

    metadata = {
        "chip": chip_path.name,
        "latitude": latitude,
        "longitude": longitude,
        "sentinel_scene": scene.id,
        "acquisition_datetime": (
            scene.datetime.isoformat()
            if scene.datetime
            else None
        ),
        "cloud_cover": scene.properties.get(
            "eo:cloud_cover"
        ),
        "source": (
            "Earth Search Sentinel-2 L2A COG"
        ),
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
            CHIP_SIZE,
            CHIP_SIZE,
        ],
    }

    metadata_path = chip_path.with_suffix(".json")

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_chip(
    latitude,
    longitude,
    start_date,
    end_date,
):
    """Find a scene and create one complete evidence chip."""

    scene = find_scene(
        latitude,
        longitude,
        start_date,
        end_date,
    )

    if scene is None:
        print(
            "No suitable Sentinel-2 scene found."
        )
        return None

    print()
    print("Scene:", scene.id)

    print(
        "Cloud cover:",
        scene.properties.get(
            "eo:cloud_cover"
        ),
    )

    print()

    output_path = (
        OUTPUT_DIR
        / "test_sentinel_multispectral.tif"
    )

    create_multispectral_chip(
        scene,
        latitude,
        longitude,
        output_path,
    )

    save_metadata(
        scene,
        latitude,
        longitude,
        output_path,
    )

    print()
    print(
        "Chip saved:",
        output_path,
    )

    print(
        "Metadata saved:",
        output_path.with_suffix(".json"),
    )

    return output_path


def main():
    """Create one test multispectral Sentinel-2 chip."""

    # Temporary coordinate used only to test
    # the Sentinel extraction pipeline.
    #
    # This will later be replaced with actual
    # FIRMS hotspot coordinates.

    latitude = 29.785
    longitude = 72.265

    create_chip(
        latitude=latitude,
        longitude=longitude,
        start_date="2026-08-20",
        end_date="2026-08-25",
    )


if __name__ == "__main__":
    main()