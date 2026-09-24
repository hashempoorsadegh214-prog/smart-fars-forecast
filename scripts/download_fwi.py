#!/usr/bin/env python3

"""
Download tomorrow's ECMWF FWI forecast from Copernicus GWIS WMS.

Input:
    fars.geojson

Outputs:
    data/fwi/fwi_latest.tif
    data/fwi/fwi_latest.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import rasterio
import requests
from rasterio.transform import from_bounds
from shapely.geometry import shape


WMS_URL = "https://maps.effis.emergency.copernicus.eu/gwis"
WMS_LAYER = "ecmwf.fwi"
WMS_VERSION = "1.1.1"

WIDTH = 2000
HEIGHT = 2000

MARGIN_DEGREES = 0.10

TIME_ZONE = "Asia/Tehran"


def load_bbox(geojson_path: Path) -> tuple[float, float, float, float]:
    with geojson_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    geojson_type = data.get("type")

    if geojson_type == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in data.get("features", [])
            if feature.get("geometry")
        ]

        if not geometries:
            raise ValueError("No geometry found in GeoJSON.")

        minx, miny, maxx, maxy = geometries[0].bounds

        for geometry in geometries[1:]:
            x1, y1, x2, y2 = geometry.bounds

            minx = min(minx, x1)
            miny = min(miny, y1)
            maxx = max(maxx, x2)
            maxy = max(maxy, y2)

    elif geojson_type == "Feature":
        minx, miny, maxx, maxy = shape(data["geometry"]).bounds

    else:
        minx, miny, maxx, maxy = shape(data).bounds

    return (
        minx - MARGIN_DEGREES,
        miny - MARGIN_DEGREES,
        maxx + MARGIN_DEGREES,
        maxy + MARGIN_DEGREES,
    )


def looks_like_error_document(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()

    head = response.content[:1000].lstrip().lower()

    return (
        "xml" in content_type
        or "html" in content_type
        or head.startswith(b"<?xml")
        or b"serviceexception" in head
        or b"exceptionreport" in head
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--boundary",
        default="fars.geojson",
    )

    parser.add_argument(
        "--output",
        default="data/fwi/fwi_latest.tif",
    )

    args = parser.parse_args()

    boundary_path = Path(args.boundary)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    iran_now = datetime.now(
        ZoneInfo(TIME_ZONE)
    )

    target_date = (
        iran_now.date()
        + timedelta(days=1)
    )

    bbox = load_bbox(boundary_path)

    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "FORMAT": "image/tiff",
        "TRANSPARENT": "TRUE",
        "SRS": "EPSG:4326",
        "BBOX": ",".join(
            f"{value:.8f}"
            for value in bbox
        ),
        "WIDTH": str(WIDTH),
        "HEIGHT": str(HEIGHT),
        "TIME": target_date.isoformat(),
    }

    print(
        "Downloading ECMWF FWI for:",
        target_date.isoformat(),
    )

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=180,
    )

    response.raise_for_status()

    if looks_like_error_document(response):
        raise RuntimeError(
            "GWIS returned an error document:\n"
            + response.text[:2000]
        )

    with tempfile.NamedTemporaryFile(
        suffix=".tif",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:

        temporary_file.write(
            response.content
        )

        temporary_path = Path(
            temporary_file.name
        )

    try:

        with rasterio.open(temporary_path) as source:

            if source.count != 1:
                raise RuntimeError(
                    "Expected a single-band numerical FWI raster, "
                    f"but GWIS returned {source.count} bands."
                )

            array = source.read(
                1,
                masked=True,
            )

            if array.count() == 0:
                raise RuntimeError(
                    "Downloaded FWI raster contains no valid pixels."
                )

            profile = source.profile.copy()

            if source.crs is None:
                profile.update(
                    crs="EPSG:4326"
                )

            if source.transform is None:
                profile.update(
                    transform=from_bounds(
                        *bbox,
                        WIDTH,
                        HEIGHT,
                    ),
                    width=WIDTH,
                    height=HEIGHT,
                )

            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as destination:

                destination.write(
                    source.read(1),
                    1,
                )

            metadata = {
                "source": "Copernicus GWIS / ECMWF",
                "wms_url": WMS_URL,
                "layer": WMS_LAYER,
                "target_date": target_date.isoformat(),
                "downloaded_at_iran": iran_now.isoformat(),
                "bbox": list(bbox),
                "crs": str(profile.get("crs")),
                "width": int(profile["width"]),
                "height": int(profile["height"]),
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
            }

        metadata_path = output_path.with_suffix(
            ".json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "Saved:",
            output_path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )


if __name__ == "__main__":
    main()
