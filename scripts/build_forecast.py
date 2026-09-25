
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Fars Forecast
Build continuous wildfire risk forecast for Fars province.

Scientific raster:
    data/output/fire_risk_latest.tif

Web visualization:
    web/generated/fire_risk_latest.png
    web/generated/fire_risk_latest.json

Important:
    Web smoothing is applied ONLY to the PNG visualization.
    The scientific GeoTIFF remains untouched.

Processing:
    Raw FWI
        -> NoData fill
        -> Bilinear resampling to Master Raster
        -> FWI normalization

    Fuel Master Raster
        -> percentile normalization

    DEM
        -> Bilinear resampling
        -> Aspect calculation

    Slope
        -> Bilinear resampling
        -> normalization 0-45 degrees

    Topography
        = slope_weight * slope_norm
        + aspect_weight * aspect_norm

    Final Risk
        = 100 *
          (
            fwi_weight * fwi_norm
            + fuel_weight * fuel_norm
            + topography_weight * topography
          )
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.fill import fillnodata
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union
from pyproj import Transformer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FWI = PROJECT_ROOT / "data" / "fwi" / "fwi_latest.tif"
DEFAULT_MASTER = PROJECT_ROOT / "fars_fire_fuel_hazard_60m.tif"
DEFAULT_DEM = PROJECT_ROOT / "dem_fars.tif"
DEFAULT_SLOPE = PROJECT_ROOT / "fars_slope_60m_light.tif"
DEFAULT_BOUNDARY = PROJECT_ROOT / "fars.geojson"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model_config.json"

DEFAULT_OUTPUT_TIF = (
    PROJECT_ROOT / "data" / "output" / "fire_risk_latest.tif"
)

DEFAULT_WEB_DIR = (
    PROJECT_ROOT / "web" / "generated"
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_FWI_FILL_DISTANCE = 20.0
DEFAULT_FWI_SMOOTHING_ITERATIONS = 1

# Web-only smoothing.
# This does NOT modify the scientific raster.
DEFAULT_WEB_SMOOTHING_RADIUS = 18.0

ALPHA_VALUE = 235


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Smart Fars wildfire forecast."
    )

    parser.add_argument(
        "--fwi",
        type=Path,
        default=DEFAULT_FWI
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER
    )

    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM
    )

    parser.add_argument(
        "--slope",
        type=Path,
        default=DEFAULT_SLOPE
    )

    parser.add_argument(
        "--boundary",
        type=Path,
        default=DEFAULT_BOUNDARY
    )

    parser.add_argument(
        "--output-tif",
        type=Path,
        default=DEFAULT_OUTPUT_TIF
    )

    parser.add_argument(
        "--web-dir",
        type=Path,
        default=DEFAULT_WEB_DIR
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def ensure_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------

def load_boundary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    geometries = []

    if data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                geometries.append(
                    shape(geometry)
                )

    elif data.get("type") == "Feature":
        geometry = data.get("geometry")
        if geometry:
            geometries.append(
                shape(geometry)
            )

    else:
        geometries.append(
            shape(data)
        )

    if not geometries:
        raise ValueError(
            f"No geometry found in {path}"
        )

    geometry = unary_union(
        geometries
    )

    if geometry.is_empty:
        raise ValueError(
            "Fars boundary geometry is empty."
        )

    return geometry, "EPSG:4326"


def reproject_geometry(
    geometry,
    source_crs,
    target_crs
):
    if str(source_crs) == str(target_crs):
        return geometry

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True
    )

    return shp_transform(
        transformer.transform,
        geometry
    )


# ---------------------------------------------------------------------
# Master Raster
# ---------------------------------------------------------------------

def load_master(path: Path):
    with rasterio.open(path) as src:

        data = src.read(1).astype(
            "float32"
        )

        profile = src.profile.copy()

        return {
            "data": data,
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "profile": profile,
        }


# ---------------------------------------------------------------------
# Generic raster reprojection
# ---------------------------------------------------------------------

def read_to_master_grid(
    source_path: Path,
    master,
    resampling=Resampling.bilinear
):
    with rasterio.open(source_path) as src:

        source = src.read(1).astype(
            "float32"
        )

        source_nodata = src.nodata

        if source_nodata is not None:

            invalid = (
                ~np.isfinite(source)
                |
                np.isclose(
                    source,
                    float(source_nodata)
                )
            )

        else:

            invalid = ~np.isfinite(
                source
            )

        source_work = source.copy()

        source_work[invalid] = -9999.0

        destination = np.full(
            (
                master["height"],
                master["width"]
            ),
            -9999.0,
            dtype="float32"
        )

        reproject(
            source=source_work,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-9999.0,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            dst_nodata=-9999.0,
            resampling=resampling
        )

    destination[
        ~np.isfinite(destination)
        |
        np.isclose(
            destination,
            -9999.0
        )
    ] = np.nan

    return destination


# ---------------------------------------------------------------------
# FWI NoData filling
# ---------------------------------------------------------------------

def fill_fwi_nodata(
    array,
    source_nodata,
    max_search_distance=DEFAULT_FWI_FILL_DISTANCE,
    smoothing_iterations=DEFAULT_FWI_SMOOTHING_ITERATIONS
):
    """
    Fill FWI holes before Bilinear resampling.

    mask:
        1 = valid source pixel
        0 = pixel that may be interpolated
    """

    fwi = np.asarray(
        array,
        dtype="float32"
    ).copy()

    valid = (
        np.isfinite(fwi)
        &
        (fwi >= 0.0)
    )

    if source_nodata is not None:

        valid &= ~np.isclose(
            fwi,
            float(source_nodata)
        )

    nodata_before = int(
        np.count_nonzero(
            ~valid
        )
    )

    if nodata_before == 0:

        return fwi, {
            "nodata_before": 0,
            "filled_pixels": 0,
            "nodata_after": 0,
            "max_search_distance": float(
                max_search_distance
            ),
            "smoothing_iterations": int(
                smoothing_iterations
            )
        }

    work = fwi.copy()

    work[~valid] = 0.0

    valid_mask = valid.astype(
        "uint8"
    )

    filled = fillnodata(
        image=work,
        mask=valid_mask,
        max_search_distance=float(
            max_search_distance
        ),
        smoothing_iterations=int(
            smoothing_iterations
        )
    ).astype("float32")

    filled[
        ~np.isfinite(filled)
        |
        (filled < 0.0)
    ] = np.nan

    filled_pixels = int(
        np.count_nonzero(
            (~valid)
            &
            np.isfinite(filled)
        )
    )

    nodata_after = int(
        np.count_nonzero(
            ~np.isfinite(filled)
        )
    )

    return filled, {
        "nodata_before": nodata_before,
        "filled_pixels": filled_pixels,
        "nodata_after": nodata_after,
        "max_search_distance": float(
            max_search_distance
        ),
        "smoothing_iterations": int(
            smoothing_iterations
        )
    }


def prepare_fwi(
    fwi_path: Path,
    master
):
    with rasterio.open(fwi_path) as src:

        raw = src.read(1).astype(
            "float32"
        )

        source_nodata = src.nodata

        source_transform = src.transform
        source_crs = src.crs

    print()
    print("FWI source:")
    print("  Size:", raw.shape)
    print("  CRS:", source_crs)
    print("  NoData:", source_nodata)

    filled, fill_stats = fill_fwi_nodata(
        raw,
        source_nodata
    )

    print(
        "  NoData before:",
        fill_stats["nodata_before"]
    )

    print(
        "  Filled:",
        fill_stats["filled_pixels"]
    )

    print(
        "  NoData after:",
        fill_stats["nodata_after"]
    )

    source_for_reproject = np.where(
        np.isfinite(filled),
        filled,
        -9999.0
    ).astype("float32")

    destination = np.full(
        (
            master["height"],
            master["width"]
        ),
        -9999.0,
        dtype="float32"
    )

    reproject(
        source=source_for_reproject,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=-9999.0,
        dst_transform=master["transform"],
        dst_crs=master["crs"],
        dst_nodata=-9999.0,
        resampling=Resampling.bilinear
    )

    destination[
        ~np.isfinite(destination)
        |
        np.isclose(
            destination,
            -9999.0
        )
    ] = np.nan

    return destination, fill_stats


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_linear(
    array,
    minimum,
    maximum
):
    result = np.full(
        array.shape,
        np.nan,
        dtype="float32"
    )

    valid = np.isfinite(
        array
    )

    if maximum <= minimum:
        raise ValueError(
            "Invalid normalization range."
        )

    if np.any(valid):

        result[valid] = (
            (
                np.clip(
                    array[valid],
                    minimum,
                    maximum
                )
                - minimum
            )
            /
            (maximum - minimum)
        ).astype("float32")

    return result


def normalize_percentile(
    array,
    mask,
    low_percentile,
    high_percentile
):
    valid = (
        np.isfinite(array)
        &
        mask
    )

    if not np.any(valid):
        raise ValueError(
            "No valid pixels for percentile normalization."
        )

    values = array[valid]

    low = float(
        np.percentile(
            values,
            low_percentile
        )
    )

    high = float(
        np.percentile(
            values,
            high_percentile
        )
    )

    if high <= low:
        raise ValueError(
            "Invalid percentile range."
        )

    result = np.full(
        array.shape,
        np.nan,
        dtype="float32"
    )

    result[valid] = (
        (
            np.clip(
                array[valid],
                low,
                high
            )
            - low
        )
        /
        (high - low)
    ).astype("float32")

    return result, low, high


# ---------------------------------------------------------------------
# Aspect
# ---------------------------------------------------------------------

def calculate_aspect_risk(
    dem,
    transform
):
    result = np.full(
        dem.shape,
        np.nan,
        dtype="float32"
    )

    valid = np.isfinite(
        dem
    )

    if not np.any(valid):
        return result

    dx = abs(
        float(transform.a)
    )

    dy = abs(
        float(transform.e)
    )

    if dx <= 0:
        dx = 1.0

    if dy <= 0:
        dy = 1.0

    safe_dem = dem.copy()

    # Use nearest valid edge values only for gradient calculation.
    # This avoids introducing a large artificial negative terrain value.
    if np.any(~np.isfinite(safe_dem)):

        valid_values = safe_dem[
            np.isfinite(safe_dem)
        ]

        if valid_values.size == 0:
            return result

        fill_value = float(
            np.median(valid_values)
        )

        safe_dem[
            ~np.isfinite(safe_dem)
        ] = fill_value

    dz_dy, dz_dx = np.gradient(
        safe_dem.astype("float64"),
        dy,
        dx
    )

    aspect = (
        np.degrees(
            np.arctan2(
                dz_dx,
                -dz_dy
            )
        )
        + 360.0
    ) % 360.0

    valid_aspect = (
        valid
        &
        np.isfinite(aspect)
    )

    north = (
        (aspect >= 315.0)
        |
        (aspect < 45.0)
    )

    east_west = (
        (
            (aspect >= 45.0)
            &
            (aspect < 135.0)
        )
        |
        (
            (aspect >= 225.0)
            &
            (aspect < 315.0)
        )
    )

    south = (
        (aspect >= 135.0)
        &
        (aspect < 225.0)
    )

    result[
        valid_aspect & north
    ] = 0.3

    result[
        valid_aspect & east_west
    ] = 0.6

    result[
        valid_aspect & south
    ] = 1.0

    return result


# ---------------------------------------------------------------------
# Topography
# ---------------------------------------------------------------------

def build_topography(
    dem,
    slope,
    transform,
    slope_max,
    slope_weight,
    aspect_weight
):
    slope_norm = normalize_linear(
        slope,
        0.0,
        slope_max
    )

    aspect_norm = calculate_aspect_risk(
        dem,
        transform
    )

    topography = (
        slope_weight * slope_norm
        +
        aspect_weight * aspect_norm
    )

    topography[
        ~np.isfinite(topography)
    ] = np.nan

    return (
        topography,
        slope_norm,
        aspect_norm
    )


# ---------------------------------------------------------------------
# Web image helpers
# ---------------------------------------------------------------------

def compute_web_size(
    width,
    height,
    max_dimension
):
    scale = min(
        1.0,
        float(max_dimension)
        /
        float(max(width, height))
    )

    return (
        max(
            1,
            int(round(width * scale))
        ),
        max(
            1,
            int(round(height * scale))
        )
    )


def box_blur_2d(
    array,
    radius
):
    """
    Pure NumPy box blur.

    Used only for the web visualization.

    It avoids adding scipy to the GitHub Actions environment.
    """

    radius = int(
        round(radius)
    )

    if radius <= 0:
        return array.astype(
            "float32"
        )

    radius = min(
        radius,
        64
    )

    src = np.asarray(
        array,
        dtype="float32"
    )

    padded = np.pad(
        src,
        (
            (radius, radius),
            (radius, radius)
        ),
        mode="edge"
    )

    cumulative = np.cumsum(
        padded,
        axis=0,
        dtype="float32"
    )

    cumulative = np.cumsum(
        cumulative,
        axis=1,
        dtype="float32"
    )

    cumulative = np.pad(
        cumulative,
        (
            (1, 0),
            (1, 0)
        ),
        mode="constant"
    )

    size = (
        2 * radius
        + 1
    )

    blurred = (
        cumulative[size:, size:]
        - cumulative[:-size, size:]
        - cumulative[size:, :-size]
        + cumulative[:-size, :-size]
    )

    blurred /= float(
        size * size
    )

    return blurred.astype(
        "float32"
    )


def smooth_web_risk(
    risk,
    valid_mask,
    radius
):
    """
    Smooth only the web representation.

    Weighted smoothing is used so that NoData/outside pixels
    do not pull the valid edge values toward zero.
    """

    if radius <= 0:
        return risk.astype(
            "float32"
        )

    valid_float = valid_mask.astype(
        "float32"
    )

    values = np.where(
        valid_mask,
        risk,
        0.0
    ).astype("float32")

    numerator = box_blur_2d(
        values,
        radius
    )

    denominator = box_blur_2d(
        valid_float,
        radius
    )

    smoothed = np.full(
        risk.shape,
        np.nan,
        dtype="float32"
    )

    valid_denominator = (
        denominator > 0.001
    )

    smoothed[
        valid_denominator
    ] = (
        numerator[
            valid_denominator
        ]
        /
        denominator[
            valid_denominator
        ]
    )

    smoothed = np.clip(
        smoothed,
        0.0,
        100.0
    )

    return smoothed


def risk_to_rgb(
    risk
):
    stops = np.array(
        [
            0.0,
            20.0,
            40.0,
            60.0,
            80.0,
            100.0
        ],
        dtype="float32"
    )

    colors = np.array(
        [
            [255, 245, 157],
            [253, 216, 53],
            [251, 140, 0],
            [229, 57, 53],
            [229, 57, 53],
            [136, 14, 79]
        ],
        dtype="float32"
    )

    rgb = np.zeros(
        (
            risk.shape[0],
            risk.shape[1],
            3
        ),
        dtype="uint8"
    )

    valid = np.isfinite(
        risk
    )

    if not np.any(valid):
        return rgb, valid

    values = np.clip(
        risk[valid],
        0.0,
        100.0
    )

    rgb[valid, 0] = np.interp(
        values,
        stops,
        colors[:, 0]
    ).astype("uint8")

    rgb[valid, 1] = np.interp(
        values,
        stops,
        colors[:, 1]
    ).astype("uint8")

    rgb[valid, 2] = np.interp(
        values,
        stops,
        colors[:, 2]
    ).astype("uint8")

    return rgb, valid


def calculate_web_bounds(
    transform,
    width,
    height,
    crs
):
    samples = 200

    xs = []
    ys = []

    for i in range(
        samples + 1
    ):

        t = (
            i /
            float(samples)
        )

        points = [
            (
                t * width,
                0.0
            ),
            (
                t * width,
                float(height)
            ),
            (
                0.0,
                t * height
            ),
            (
                float(width),
                t * height
            )
        ]

        for col, row in points:

            x, y = (
                transform *
                (col, row)
            )

            xs.append(x)
            ys.append(y)

    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = transformer.transform(
        np.asarray(xs),
        np.asarray(ys)
    )

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    valid = (
        np.isfinite(lon)
        &
        np.isfinite(lat)
    )

    if not np.any(valid):
        raise ValueError(
            "Could not calculate web bounds."
        )

    return [
        float(np.min(lon[valid])),
        float(np.min(lat[valid])),
        float(np.max(lon[valid])),
        float(np.max(lat[valid]))
    ]


# ---------------------------------------------------------------------
# Web PNG
# ---------------------------------------------------------------------

def create_web_png(
    risk,
    master,
    boundary_geometry,
    boundary_crs,
    web_dir,
    max_dimension,
    smoothing_radius
):
    from PIL import Image

    web_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    width = master["width"]
    height = master["height"]

    target_width, target_height = (
        compute_web_size(
            width,
            height,
            max_dimension
        )
    )

    print(
        f"  Web size: "
        f"{target_width} x "
        f"{target_height}"
    )

    # --------------------------------------------------------------
    # Resize risk field for web visualization.
    # This is NOT the scientific raster.
    # --------------------------------------------------------------

    risk_safe = np.where(
        np.isfinite(risk),
        risk,
        0.0
    ).astype("float32")

    risk_image = Image.fromarray(
        risk_safe,
        mode="F"
    )

    risk_web = risk_image.resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.BICUBIC
    )

    risk_web = np.asarray(
        risk_web,
        dtype="float32"
    )

    # Valid-data mask.
    valid_mask = np.isfinite(
        risk
    )

    valid_image = Image.fromarray(
        (
            valid_mask * 255
        ).astype("uint8"),
        mode="L"
    )

    valid_web_image = valid_image.resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.NEAREST
    )

    valid_web = (
        np.asarray(
            valid_web_image,
            dtype="uint8"
        )
        > 0
    )

    # --------------------------------------------------------------
    # WEB-ONLY SMOOTHING
    # --------------------------------------------------------------

    risk_web[
        ~valid_web
    ] = np.nan

    risk_web = smooth_web_risk(
        risk_web,
        valid_web,
        smoothing_radius
    )

    # --------------------------------------------------------------
    # Colorize smoothed web risk.
    # --------------------------------------------------------------

    rgb, finite = risk_to_rgb(
        risk_web
    )

    rgb_image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    # Slight final visual interpolation.
    rgb_image = rgb_image.resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.BICUBIC
    )

    rgb_final = np.asarray(
        rgb_image,
        dtype="uint8"
    )

    # --------------------------------------------------------------
    # Final affine transform
    # --------------------------------------------------------------

    sx = (
        float(width)
        /
        float(target_width)
    )

    sy = (
        float(height)
        /
        float(target_height)
    )

    target_transform = Affine(
        master["transform"].a * sx,
        master["transform"].b * sx,
        master["transform"].c,
        master["transform"].d * sy,
        master["transform"].e * sy,
        master["transform"].f
    )

    # --------------------------------------------------------------
    # Rasterize Fars boundary at final PNG size.
    # --------------------------------------------------------------

    geometry_master = reproject_geometry(
        boundary_geometry,
        boundary_crs,
        master["crs"]
    )

    province_mask = rasterize(
        [
            (
                mapping(
                    geometry_master
                ),
                1
            )
        ],
        out_shape=(
            target_height,
            target_width
        ),
        transform=target_transform,
        fill=0,
        all_touched=False,
        dtype="uint8"
    ).astype(bool)

    alpha = np.where(
        province_mask & finite,
        ALPHA_VALUE,
        0
    ).astype("uint8")

    rgba = np.dstack(
        [
            rgb_final,
            alpha
        ]
    )

    output_image = Image.fromarray(
        rgba,
        mode="RGBA"
    )

    png_path = (
        web_dir /
        "fire_risk_latest.png"
    )

    output_image.save(
        png_path,
        format="PNG",
        optimize=True
    )

    bounds = calculate_web_bounds(
        master["transform"],
        width,
        height,
        master["crs"]
    )

    metadata = {
        "image_size": [
            target_width,
            target_height
        ],
        "master_size": [
            width,
            height
        ],
        "bounds": bounds,
        "image_url":
            "generated/fire_risk_latest.png",
        "web": {
            "crs": "EPSG:4326",
            "bounds": bounds,
            "image_url":
                "generated/fire_risk_latest.png",
            "smoothing": {
                "enabled": smoothing_radius > 0,
                "method":
                    "web-only weighted box smoothing",
                "radius":
                    float(smoothing_radius)
            }
        },
        "master_crs": (
            master["crs"].to_string()
            if hasattr(
                master["crs"],
                "to_string"
            )
            else str(master["crs"])
        ),
        "alpha": ALPHA_VALUE
    }

    metadata_path = (
        web_dir /
        "fire_risk_latest.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    return (
        png_path,
        metadata_path,
        metadata
    )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def calculate_statistics(
    risk,
    province_mask
):
    valid = (
        np.isfinite(risk)
        &
        province_mask
    )

    if not np.any(valid):
        raise ValueError(
            "No valid risk cells inside Fars."
        )

    values = risk[valid]

    return {
        "valid_cells": int(
            values.size
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "std": float(
            np.std(values)
        ),
        "classes": {
            "very_low": int(
                np.count_nonzero(
                    values < 20
                )
            ),
            "low": int(
                np.count_nonzero(
                    (values >= 20)
                    &
                    (values < 40)
                )
            ),
            "moderate": int(
                np.count_nonzero(
                    (values >= 40)
                    &
                    (values < 60)
                )
            ),
            "high": int(
                np.count_nonzero(
                    (values >= 60)
                    &
                    (values < 80)
                )
            ),
            "critical": int(
                np.count_nonzero(
                    values >= 80
                )
            )
        }
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    print("=" * 70)
    print("SMART FARS FORECAST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Files
    # --------------------------------------------------------------

    ensure_file(
        args.fwi,
        "FWI raster"
    )

    ensure_file(
        args.master,
        "Master Raster"
    )

    ensure_file(
        args.dem,
        "DEM"
    )

    ensure_file(
        args.slope,
        "Slope raster"
    )

    ensure_file(
        args.boundary,
        "Fars boundary"
    )

    ensure_file(
        args.config,
        "Model config"
    )

    # --------------------------------------------------------------
    # Config
    # --------------------------------------------------------------

    config = load_json(
        args.config
    )

    weights = config.get(
        "weights",
        {}
    )

    fwi_weight = float(
        weights.get(
            "fwi",
            0.45
        )
    )

    fuel_weight = float(
        weights.get(
            "fuel",
            0.35
        )
    )

    topography_weight = float(
        weights.get(
            "topography",
            0.20
        )
    )

    if not np.isclose(
        fwi_weight
        +
        fuel_weight
        +
        topography_weight,
        1.0,
        atol=1e-6
    ):
        raise ValueError(
            "Model weights must sum to 1."
        )

    normalization = config.get(
        "normalization",
        {}
    )

    fuel_low_percentile = float(
        normalization.get(
            "fuel_percentile_low",
            1.0
        )
    )

    fuel_high_percentile = float(
        normalization.get(
            "fuel_percentile_high",
            99.0
        )
    )

    fwi_min = float(
        normalization.get(
            "fwi_min",
            0.0
        )
    )

    fwi_max = float(
        normalization.get(
            "fwi_max",
            100.0
        )
    )

    slope_max = float(
        normalization.get(
            "slope_max",
            45.0
        )
    )

    topography_config = config.get(
        "topography",
        {}
    )

    slope_weight = float(
        topography_config.get(
            "slope_weight",
            0.80
        )
    )

    aspect_weight = float(
        topography_config.get(
            "aspect_weight",
            0.20
        )
    )

    web_config = config.get(
        "web",
        {}
    )

    max_dimension = int(
        web_config.get(
            "max_dimension",
            3000
        )
    )

    web_smoothing_radius = float(
        web_config.get(
            "smoothing_radius",
            DEFAULT_WEB_SMOOTHING_RADIUS
        )
    )

    print()
    print("Model weights:")
    print(
        "  FWI:",
        fwi_weight
    )
    print(
        "  Fuel:",
        fuel_weight
    )
    print(
        "  Topography:",
        topography_weight
    )

    print()
    print(
        "Web smoothing radius:",
        web_smoothing_radius
    )

    # --------------------------------------------------------------
    # Master
    # --------------------------------------------------------------

    print()
    print("Loading Master Raster...")

    master = load_master(
        args.master
    )

    print(
        f"  Size: "
        f"{master['width']} x "
        f"{master['height']}"
    )

    print(
        "  CRS:",
        master["crs"]
    )

    print(
        "  Resolution:",
        abs(
            float(
                master["transform"].a
            )
        ),
        "x",
        abs(
            float(
                master["transform"].e
            )
        )
    )

    # --------------------------------------------------------------
    # Boundary
    # --------------------------------------------------------------

    print()
    print("Loading Fars boundary...")

    boundary_geometry, boundary_crs = (
        load_boundary(
            args.boundary
        )
    )

    boundary_master = reproject_geometry(
        boundary_geometry,
        boundary_crs,
        master["crs"]
    )

    province_mask = geometry_mask(
        [
            mapping(
                boundary_master
            )
        ],
        out_shape=(
            master["height"],
            master["width"]
        ),
        transform=master["transform"],
        invert=True,
        all_touched=False
    )

    print(
        "  Province cells:",
        int(
            np.count_nonzero(
                province_mask
            )
        )
    )

    # --------------------------------------------------------------
    # Fuel
    # --------------------------------------------------------------

    print()
    print("Preparing Fuel...")

    fuel = master["data"].copy()

    if master["nodata"] is not None:

        fuel[
            np.isclose(
                fuel,
                float(
                    master["nodata"]
                )
            )
        ] = np.nan

    fuel[
        ~np.isfinite(fuel)
    ] = np.nan

    fuel_norm, fuel_low, fuel_high = (
        normalize_percentile(
            fuel,
            province_mask,
            fuel_low_percentile,
            fuel_high_percentile
        )
    )

    # --------------------------------------------------------------
    # DEM
    # --------------------------------------------------------------

    print()
    print("Preparing DEM...")

    dem = read_to_master_grid(
        args.dem,
        master,
        resampling=Resampling.bilinear
    )

    dem[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------------
    # Slope
    # --------------------------------------------------------------

    print()
    print("Preparing Slope...")

    slope = read_to_master_grid(
        args.slope,
        master,
        resampling=Resampling.bilinear
    )

    slope[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------------
    # Topography
    # --------------------------------------------------------------

    print()
    print("Calculating Topography...")

    topography, slope_norm, aspect_norm = (
        build_topography(
            dem=dem,
            slope=slope,
            transform=master["transform"],
            slope_max=slope_max,
            slope_weight=slope_weight,
            aspect_weight=aspect_weight
        )
    )

    topography[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------------
    # FWI
    # --------------------------------------------------------------

    print()
    print("Preparing FWI...")

    fwi, fwi_fill_stats = (
        prepare_fwi(
            args.fwi,
            master
        )
    )

    fwi[
        ~province_mask
    ] = np.nan

    fwi_norm = normalize_linear(
        fwi,
        fwi_min,
        fwi_max
    )

    # --------------------------------------------------------------
    # Final risk
    # --------------------------------------------------------------

    print()
    print("Calculating wildfire risk...")

    valid = (
        province_mask
        &
        np.isfinite(fwi_norm)
        &
        np.isfinite(fuel_norm)
        &
        np.isfinite(topography)
    )

    risk = np.full(
        (
            master["height"],
            master["width"]
        ),
        np.nan,
        dtype="float32"
    )

    risk[valid] = (
        100.0
        *
        (
            fwi_weight
            *
            fwi_norm[valid]

            +

            fuel_weight
            *
            fuel_norm[valid]

            +

            topography_weight
            *
            topography[valid]
        )
    )

    risk = np.clip(
        risk,
        0.0,
        100.0
    )

    risk[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    statistics = calculate_statistics(
        risk,
        province_mask
    )

    print()
    print("Forecast statistics:")

    print(
        f"  Min    : "
        f"{statistics['min']:.3f}"
    )

    print(
        f"  Max    : "
        f"{statistics['max']:.3f}"
    )

    print(
        f"  Mean   : "
        f"{statistics['mean']:.3f}"
    )

    print(
        f"  Median : "
        f"{statistics['median']:.3f}"
    )

    # --------------------------------------------------------------
    # Save scientific raster
    # --------------------------------------------------------------

    args.output_tif.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    profile = master["profile"].copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
        width=master["width"],
        height=master["height"],
        transform=master["transform"],
        crs=master["crs"]
    )

    scientific_array = np.where(
        np.isfinite(risk),
        risk,
        -9999.0
    ).astype("float32")

    print()
    print("Writing scientific GeoTIFF...")

    with rasterio.open(
        args.output_tif,
        "w",
        **profile
    ) as dst:

        dst.write(
            scientific_array,
            1
        )

        dst.set_band_description(
            1,
            "Smart Fars Wildfire Risk (0-100)"
        )

    # --------------------------------------------------------------
    # Web PNG
    # --------------------------------------------------------------

    print()
    print("Building Web GIS PNG...")

    png_path, web_metadata_path, web_metadata = (
        create_web_png(
            risk=risk,
            master=master,
            boundary_geometry=boundary_geometry,
            boundary_crs=boundary_crs,
            web_dir=args.web_dir,
            max_dimension=max_dimension,
            smoothing_radius=web_smoothing_radius
        )
    )

    # --------------------------------------------------------------
    # Scientific metadata
    # --------------------------------------------------------------

    metadata = {
        "model":
            "Smart Fars Forecast",

        "risk_range":
            [0.0, 100.0],

        "formula":
            "100 * "
            "(0.45 * FWI_norm + "
            "0.35 * Fuel_norm + "
            "0.20 * Topography)",

        "weights": {
            "fwi":
                fwi_weight,
            "fuel":
                fuel_weight,
            "topography":
                topography_weight
        },

        "topography": {
            "slope_weight":
                slope_weight,
            "aspect_weight":
                aspect_weight,
            "slope_max":
                slope_max,
            "aspect_classes": {
                "south":
                    1.0,
                "east_west":
                    0.6,
                "north":
                    0.3
            }
        },

        "fwi_processing": {
            "raw_source":
                str(args.fwi),

            "fillnodata":
                fwi_fill_stats,

            "resampling_to_master":
                "bilinear",

            "normalization_min":
                fwi_min,

            "normalization_max":
                fwi_max
        },

        "fuel_normalization": {
            "method":
                "percentile",

            "low_percentile":
                fuel_low_percentile,

            "high_percentile":
                fuel_high_percentile,

            "low_value":
                fuel_low,

            "high_value":
                fuel_high
        },

        "master_grid": {
            "width":
                master["width"],

            "height":
                master["height"],

            "crs": (
                master["crs"].to_string()
                if hasattr(
                    master["crs"],
                    "to_string"
                )
                else str(
                    master["crs"]
                )
            ),

            "transform":
                list(
                    master["transform"]
                )[:6],

            "resolution": [
                abs(
                    float(
                        master["transform"].a
                    )
                ),
                abs(
                    float(
                        master["transform"].e
                    )
                )
            ]
        },

        "statistics":
            statistics,

        "web":
            web_metadata,

        "outputs": {
            "raster":
                str(args.output_tif),

            "web_png":
                str(png_path),

            "web_metadata":
                str(web_metadata_path)
        }
    }

    metadata_path = (
        args.output_tif.with_suffix(
            ".json"
        )
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "Scientific raster:",
        args.output_tif
    )

    print(
        "Web PNG:",
        png_path
    )

    print(
        "Web metadata:",
        web_metadata_path
    )

    print()
    print(
        "FWI:",
        "Raw -> fillnodata -> Bilinear -> Master -> normalization"
    )

    print(
        "Risk:",
        "Scientific raster kept unchanged"
    )

    print(
        "Web:",
        "Web-only smoothing radius =",
        web_smoothing_radius
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
