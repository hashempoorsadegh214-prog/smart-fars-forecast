
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Fars Forecast
Build continuous wildfire risk forecast for Fars province.

Processing chain:

Raw FWI
  -> NoData detection
  -> fillnodata
  -> Bilinear resampling to Master Raster
  -> FWI normalization

Fuel Master Raster
  -> percentile normalization

DEM
  -> Bilinear resampling
  -> Aspect calculation

Slope
  -> Bilinear resampling
  -> 0-45 degree normalization

Topography
  = 0.80 * Slope_norm + 0.20 * Aspect_norm

Final Risk
  = 100 * (
        0.45 * FWI_norm +
        0.35 * Fuel_norm +
        0.20 * Topography
    )

All calculations are masked to Fars province.

Outputs:
    data/output/fire_risk_latest.tif
    data/output/fire_risk_latest.json
    web/generated/fire_risk_latest.png
    web/generated/fire_risk_latest.json

The raw FWI file is NOT modified.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.fill import fillnodata
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as shp_transform

from pyproj import Transformer


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MASTER = PROJECT_ROOT / "fars_fire_fuel_hazard_60m.tif"
DEFAULT_DEM = PROJECT_ROOT / "dem_fars.tif"
DEFAULT_SLOPE = PROJECT_ROOT / "fars_slope_60m_light.tif"
DEFAULT_BOUNDARY = PROJECT_ROOT / "fars.geojson"

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model_config.json"

DEFAULT_FWI = PROJECT_ROOT / "data" / "fwi" / "fwi_latest.tif"

DEFAULT_OUTPUT_TIF = (
    PROJECT_ROOT / "data" / "output" / "fire_risk_latest.tif"
)

DEFAULT_WEB_DIR = (
    PROJECT_ROOT / "web" / "generated"
)

DEFAULT_FILL_DISTANCE = 20.0
DEFAULT_FILL_SMOOTHING = 1

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
        default=DEFAULT_FWI,
        help="Downloaded FWI GeoTIFF"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Model configuration JSON"
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help="Master/reference raster"
    )

    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        help="DEM raster"
    )

    parser.add_argument(
        "--slope",
        type=Path,
        default=DEFAULT_SLOPE,
        help="Slope raster"
    )

    parser.add_argument(
        "--boundary",
        type=Path,
        default=DEFAULT_BOUNDARY,
        help="Fars GeoJSON boundary"
    )

    parser.add_argument(
        "--output-tif",
        type=Path,
        default=DEFAULT_OUTPUT_TIF,
        help="Final risk GeoTIFF"
    )

    parser.add_argument(
        "--web-dir",
        type=Path,
        default=DEFAULT_WEB_DIR,
        help="Web output directory"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def ensure_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------

def load_boundary_geojson(path: Path):
    """
    Load Fars GeoJSON using stdlib json + shapely.
    No geopandas/fiona required.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    geometries = []

    if data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                geometries.append(shape(geometry))

    elif data.get("type") == "Feature":
        geometry = data.get("geometry")
        if geometry:
            geometries.append(shape(geometry))

    else:
        geometries.append(shape(data))

    if not geometries:
        raise ValueError(
            f"No geometry found in boundary file: {path}"
        )

    geometry = unary_union(geometries)

    if geometry.is_empty:
        raise ValueError("Boundary geometry is empty.")

    # GeoJSON is expected to be WGS84
    source_crs = "EPSG:4326"

    return geometry, source_crs


def reproject_geometry(geometry, source_crs, target_crs):
    if source_crs is None or target_crs is None:
        raise ValueError(
            "Source and target CRS are required."
        )

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
# Master grid
# ---------------------------------------------------------------------

def get_master_grid(master_path: Path):
    with rasterio.open(master_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        nodata = src.nodata
        master_data = src.read(1).astype("float32")

    if crs is None:
        raise ValueError("Master raster has no CRS.")

    return {
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "width": width,
        "height": height,
        "nodata": nodata,
        "data": master_data,
    }


def read_to_master_grid(
    source_path: Path,
    master,
    resampling=Resampling.bilinear,
):
    """
    Reproject/resample a source raster to the exact Master Raster grid.

    Matching:
        CRS
        width
        height
        transform
        pixel alignment
        extent
    """

    with rasterio.open(source_path) as src:

        source = src.read(1).astype("float32")

        source_nodata = src.nodata

        if source_nodata is not None:
            source_invalid = (
                ~np.isfinite(source)
                | np.isclose(source, source_nodata)
            )
        else:
            source_invalid = ~np.isfinite(source)

        source_work = source.copy()

        if np.any(source_invalid):
            source_work[source_invalid] = -9999.0

        destination = np.full(
            (master["height"], master["width"]),
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
            resampling=resampling,
        )

    destination = destination.astype("float32")

    destination[
        ~np.isfinite(destination)
        | np.isclose(destination, -9999.0)
    ] = np.nan

    return destination


# ---------------------------------------------------------------------
# FWI NoData filling
# ---------------------------------------------------------------------

def fill_fwi_nodata(
    fwi_array,
    source_nodata,
    max_search_distance=DEFAULT_FILL_DISTANCE,
    smoothing_iterations=DEFAULT_FILL_SMOOTHING,
):
    """
    Fill FWI NoData BEFORE Bilinear resampling.

    Valid FWI:
        finite
        >= 0
        not equal to source nodata

    Invalid FWI:
        NaN
        source nodata
        negative values

    The raw source raster is not modified.
    """

    fwi = np.asarray(
        fwi_array,
        dtype="float32"
    ).copy()

    valid = np.isfinite(fwi) & (fwi >= 0.0)

    if source_nodata is not None:
        valid &= ~np.isclose(
            fwi,
            float(source_nodata)
        )

    invalid_count_before = int((~valid).sum())

    if invalid_count_before == 0:
        return fwi, {
            "nodata_before": 0,
            "nodata_after": 0,
            "filled_pixels": 0,
        }

    # fillnodata uses:
    #   mask > 0  -> valid source pixels
    #   mask == 0 -> pixels to interpolate
    mask = valid.astype("uint8")

    # Invalid pixels must not contain NaN while running fillnodata.
    work = fwi.copy()
    work[~valid] = 0.0

    filled = fillnodata(
        image=work,
        mask=mask,
        max_search_distance=float(max_search_distance),
        smoothing_iterations=int(smoothing_iterations),
    ).astype("float32")

    # Restore invalid pixels that could not be filled.
    remaining_invalid = ~np.isfinite(filled)

    # A second check also catches negative or invalid values.
    remaining_invalid |= filled < 0.0

    filled_count = int(
        np.count_nonzero(
            valid & np.isfinite(filled)
        ) - np.count_nonzero(
            valid & np.isfinite(fwi)
        )
    )

    # More reliable filled-pixel count:
    filled_pixels = int(
        np.count_nonzero(
            (~valid) & np.isfinite(filled)
        )
    )

    filled[remaining_invalid] = np.nan

    invalid_count_after = int(
        np.count_nonzero(~np.isfinite(filled))
    )

    stats = {
        "nodata_before": invalid_count_before,
        "nodata_after": invalid_count_after,
        "filled_pixels": filled_pixels,
        "max_search_distance": float(max_search_distance),
        "smoothing_iterations": int(smoothing_iterations),
    }

    if invalid_count_after > 0:
        print(
            "WARNING: Some FWI NoData pixels remained after fillnodata: "
            f"{invalid_count_after}"
        )

    return filled, stats


def prepare_fwi_to_master(
    fwi_path: Path,
    master,
):
    """
    Raw FWI
        -> fill NoData
        -> save no temporary modified source
        -> Bilinear to Master
    """

    with rasterio.open(fwi_path) as src:
        fwi_raw = src.read(1).astype("float32")

        fwi_crs = src.crs
        fwi_transform = src.transform
        fwi_nodata = src.nodata

    print("Preparing FWI:")
    print(f"  Source CRS: {fwi_crs}")
    print(f"  Source size: {fwi_raw.shape}")
    print(f"  Source NoData: {fwi_nodata}")

    fwi_filled, fill_stats = fill_fwi_nodata(
        fwi_raw,
        fwi_nodata,
        max_search_distance=DEFAULT_FILL_DISTANCE,
        smoothing_iterations=DEFAULT_FILL_SMOOTHING,
    )

    # If some holes remain, keep them as NoData.
    # Bilinear will not use these pixels in interpolation.
    fwi_source = np.where(
        np.isfinite(fwi_filled),
        fwi_filled,
        -9999.0
    ).astype("float32")

    fwi_master = np.full(
        (master["height"], master["width"]),
        -9999.0,
        dtype="float32"
    )

    reproject(
        source=fwi_source,
        destination=fwi_master,
        src_transform=fwi_transform,
        src_crs=fwi_crs,
        src_nodata=-9999.0,
        dst_transform=master["transform"],
        dst_crs=master["crs"],
        dst_nodata=-9999.0,
        resampling=Resampling.bilinear,
    )

    fwi_master[
        ~np.isfinite(fwi_master)
        | np.isclose(fwi_master, -9999.0)
    ] = np.nan

    return fwi_master, fill_stats


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_linear(
    array,
    low,
    high,
):
    """
    Linear normalization to 0-1.
    """

    out = np.full(
        array.shape,
        np.nan,
        dtype="float32"
    )

    valid = np.isfinite(array)

    if not np.any(valid):
        return out

    if high <= low:
        raise ValueError(
            f"Invalid normalization range: low={low}, high={high}"
        )

    clipped = np.clip(
        array[valid],
        low,
        high
    )

    out[valid] = (
        (clipped - low)
        / (high - low)
    ).astype("float32")

    return out


def normalize_percentile(
    array,
    low_percentile=1.0,
    high_percentile=99.0,
    mask=None,
):
    """
    Robust percentile normalization to 0-1.
    """

    if mask is None:
        valid = np.isfinite(array)
    else:
        valid = (
            np.isfinite(array)
            & mask
        )

    if not np.any(valid):
        raise ValueError(
            "No valid cells available for percentile normalization."
        )

    values = array[valid]

    low_value = float(
        np.percentile(
            values,
            low_percentile
        )
    )

    high_value = float(
        np.percentile(
            values,
            high_percentile
        )
    )

    if high_value <= low_value:
        raise ValueError(
            "Fuel percentile range is invalid: "
            f"{low_value} -> {high_value}"
        )

    out = np.full(
        array.shape,
        np.nan,
        dtype="float32"
    )

    clipped = np.clip(
        array[valid],
        low_value,
        high_value
    )

    out[valid] = (
        (clipped - low_value)
        / (high_value - low_value)
    ).astype("float32")

    return out, low_value, high_value


# ---------------------------------------------------------------------
# Aspect / Topography
# ---------------------------------------------------------------------

def calculate_aspect_risk(
    dem,
    transform,
):
    """
    Convert terrain aspect into a directional risk score.

    South              = 1.0
    East / West        = 0.6
    North              = 0.3

    GDAL-style aspect:
        0   = North
        90  = East
        180 = South
        270 = West
    """

    aspect_risk = np.full(
        dem.shape,
        np.nan,
        dtype="float32"
    )

    valid = np.isfinite(dem)

    if not np.any(valid):
        return aspect_risk

    # Cell size from affine transform.
    dx = abs(float(transform.a))
    dy = abs(float(transform.e))

    if dx <= 0:
        dx = 1.0

    if dy <= 0:
        dy = 1.0

    # Gradient.
    dz_dy, dz_dx = np.gradient(
        np.where(valid, dem, np.nan).astype("float64"),
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
        & np.isfinite(aspect)
    )

    # North:
    # 315-360 and 0-45
    north = (
        (aspect >= 315.0)
        | (aspect < 45.0)
    )

    # East / West:
    # 45-135 and 225-315
    east_west = (
        (
            (aspect >= 45.0)
            & (aspect < 135.0)
        )
        |
        (
            (aspect >= 225.0)
            & (aspect < 315.0)
        )
    )

    # South:
    # 135-225
    south = (
        (aspect >= 135.0)
        & (aspect < 225.0)
    )

    aspect_risk[
        valid_aspect & north
    ] = 0.3

    aspect_risk[
        valid_aspect & east_west
    ] = 0.6

    aspect_risk[
        valid_aspect & south
    ] = 1.0

    return aspect_risk


def build_topography(
    dem,
    slope,
    transform,
    slope_max,
    slope_weight,
    aspect_weight,
):
    """
    Topography:
        0.80 * normalized slope
        +
        0.20 * aspect risk
    """

    slope_norm = normalize_linear(
        slope,
        0.0,
        slope_max
    )

    slope_norm = np.clip(
        slope_norm,
        0.0,
        1.0
    )

    aspect_norm = calculate_aspect_risk(
        dem,
        transform
    )

    topo = (
        slope_weight * slope_norm
        +
        aspect_weight * aspect_norm
    )

    topo[
        ~np.isfinite(topo)
    ] = np.nan

    return topo, slope_norm, aspect_norm


# ---------------------------------------------------------------------
# Web rendering
# ---------------------------------------------------------------------

def compute_web_size(
    width,
    height,
    max_dimension,
):
    scale = min(
        1.0,
        float(max_dimension) / float(
            max(width, height)
        )
    )

    out_width = max(
        1,
        int(round(width * scale))
    )

    out_height = max(
        1,
        int(round(height * scale))
    )

    return out_width, out_height


def calculate_web_bounds(
    transform,
    width,
    height,
    crs,
):
    """
    Convert all four raster edges to EPSG:4326
    so the Leaflet overlay gets reliable geographic bounds.
    """

    samples = 200

    xs = []
    ys = []

    for i in range(samples + 1):
        t = i / float(samples)

        points = [
            (t * width, 0.0),
            (t * width, float(height)),
            (0.0, t * height),
            (float(width), t * height),
        ]

        for col, row in points:
            x, y = transform * (
                col,
                row
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
        & np.isfinite(lat)
    )

    if not np.any(valid):
        raise ValueError(
            "Could not calculate valid web bounds."
        )

    return [
        float(np.min(lon[valid])),
        float(np.min(lat[valid])),
        float(np.max(lon[valid])),
        float(np.max(lat[valid])),
    ]


def risk_to_rgb(
    risk_0_100,
):
    """
    Continuous visualization.
    Underlying science raster remains continuous 0-100.

    Display gradient:
        0    -> very low
        20
        40
        60
        80
        100  -> critical
    """

    stops = np.array(
        [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        dtype="float32"
    )

    colors = np.array(
        [
            [255, 245, 157],
            [253, 216, 53],
            [251, 140, 0],
            [229, 57, 53],
            [229, 57, 53],
            [136, 14, 79],
        ],
        dtype="float32"
    )

    rgb = np.zeros(
        (*risk_0_100.shape, 3),
        dtype="uint8"
    )

    valid = np.isfinite(risk_0_100)

    if not np.any(valid):
        return rgb, valid

    values = np.clip(
        risk_0_100[valid],
        0.0,
        100.0
    )

    r = np.interp(
        values,
        stops,
        colors[:, 0]
    )

    g = np.interp(
        values,
        stops,
        colors[:, 1]
    )

    b = np.interp(
        values,
        stops,
        colors[:, 2]
    )

    rgb[valid, 0] = np.clip(
        r,
        0,
        255
    ).astype("uint8")

    rgb[valid, 1] = np.clip(
        g,
        0,
        255
    ).astype("uint8")

    rgb[valid, 2] = np.clip(
        b,
        0,
        255
    ).astype("uint8")

    return rgb, valid


def create_web_png(
    risk,
    master,
    boundary_geometry,
    boundary_crs,
    web_dir,
    max_dimension,
):
    """
    Create continuous gradient PNG for Leaflet.

    RGB can be bicubic-resized only for visualization.

    Province alpha mask is rasterized directly to final PNG dimensions,
    preventing color/alpha bleeding outside Fars.
    """

    from PIL import Image

    web_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    width = master["width"]
    height = master["height"]
    transform = master["transform"]
    crs = master["crs"]

    target_width, target_height = compute_web_size(
        width,
        height,
        max_dimension
    )

    rgb, valid_mask = risk_to_rgb(
        risk
    )

    rgb_image = Image.fromarray(
        rgb,
        mode="RGB"
    )

    rgb_resized = rgb_image.resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.BICUBIC
    )

    rgb_final = np.asarray(
        rgb_resized,
        dtype="uint8"
    )

    # Scale the geotransform to final image size.
    sx = float(width) / float(target_width)
    sy = float(height) / float(target_height)

    target_transform = Affine(
        transform.a * sx,
        transform.b * sx,
        transform.c,
        transform.d * sy,
        transform.e * sy,
        transform.f,
    )

    geometry_master = reproject_geometry(
        boundary_geometry,
        boundary_crs,
        crs
    )

    province_mask = rasterize(
        [(mapping(geometry_master), 1)],
        out_shape=(
            target_height,
            target_width
        ),
        transform=target_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    # Resize the valid-data mask without inventing validity.
    valid_image = Image.fromarray(
        (
            valid_mask * 255
        ).astype("uint8"),
        mode="L"
    )

    valid_resized = valid_image.resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.NEAREST
    )

    valid_final = (
        np.asarray(
            valid_resized,
            dtype="uint8"
        ) > 0
    )

    alpha = np.where(
        province_mask & valid_final,
        ALPHA_VALUE,
        0
    ).astype("uint8")

    rgba = np.dstack(
        [
            rgb_final,
            alpha
        ]
    )

    image = Image.fromarray(
        rgba,
        mode="RGBA"
    )

    png_path = web_dir / "fire_risk_latest.png"

    image.save(
        png_path,
        format="PNG",
        optimize=True,
    )

    bounds = calculate_web_bounds(
        transform,
        width,
        height,
        crs
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
        "web": {
            "crs": "EPSG:4326",
            "bounds": bounds
        },
        "master_crs": (
            crs.to_string()
            if hasattr(crs, "to_string")
            else str(crs)
        ),
        "alpha": ALPHA_VALUE,
    }

    metadata_path = (
        web_dir / "fire_risk_latest.json"
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

    return png_path, metadata_path, metadata


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def calculate_statistics(
    risk,
    province_mask,
):
    valid = (
        np.isfinite(risk)
        & province_mask
    )

    if not np.any(valid):
        raise ValueError(
            "No valid risk cells inside Fars."
        )

    values = risk[valid]

    counts = {
        "very_low": int(
            np.count_nonzero(
                values < 20
            )
        ),
        "low": int(
            np.count_nonzero(
                (values >= 20)
                & (values < 40)
            )
        ),
        "moderate": int(
            np.count_nonzero(
                (values >= 40)
                & (values < 60)
            )
        ),
        "high": int(
            np.count_nonzero(
                (values >= 60)
                & (values < 80)
            )
        ),
        "critical": int(
            np.count_nonzero(
                values >= 80
            )
        ),
    }

    return {
        "valid_cells": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "classes": counts,
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
    # Check files
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
        "Slope"
    )

    ensure_file(
        args.boundary,
        "Fars boundary"
    )

    ensure_file(
        args.config,
        "Model configuration"
    )

    # --------------------------------------------------------------
    # Load config
    # --------------------------------------------------------------

    config = read_json(
        args.config
    )

    weights = config.get(
        "weights",
        {}
    )

    fwi_weight = float(
        weights.get("fwi", 0.45)
    )

    fuel_weight = float(
        weights.get("fuel", 0.35)
    )

    topo_weight = float(
        weights.get("topography", 0.20)
    )

    weight_sum = (
        fwi_weight
        + fuel_weight
        + topo_weight
    )

    if not np.isclose(
        weight_sum,
        1.0,
        atol=1e-6
    ):
        raise ValueError(
            "Model weights must sum to 1. "
            f"Current sum = {weight_sum}"
        )

    normalization = config.get(
        "normalization",
        {}
    )

    fuel_percentile_low = float(
        normalization.get(
            "fuel_percentile_low",
            1.0
        )
    )

    fuel_percentile_high = float(
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

    topo_config = config.get(
        "topography",
        {}
    )

    slope_weight = float(
        topo_config.get(
            "slope_weight",
            0.80
        )
    )

    aspect_weight = float(
        topo_config.get(
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
            2048
        )
    )

    print()
    print("Model weights:")
    print(f"  FWI         = {fwi_weight}")
    print(f"  Fuel        = {fuel_weight}")
    print(f"  Topography  = {topo_weight}")

    # --------------------------------------------------------------
    # Master raster
    # --------------------------------------------------------------

    print()
    print("Loading Master Raster...")

    master = get_master_grid(
        args.master
    )

    print(
        f"  Size: {master['width']} x {master['height']}"
    )
    print(
        f"  CRS: {master['crs']}"
    )
    print(
        f"  Resolution: "
        f"{abs(master['transform'].a)} x "
        f"{abs(master['transform'].e)}"
    )

    # --------------------------------------------------------------
    # Boundary
    # --------------------------------------------------------------

    print()
    print("Loading Fars boundary...")

    boundary_geometry, boundary_crs = (
        load_boundary_geojson(
            args.boundary
        )
    )

    boundary_master = reproject_geometry(
        boundary_geometry,
        boundary_crs,
        master["crs"]
    )

    province_mask = geometry_mask(
        [mapping(boundary_master)],
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
        int(np.count_nonzero(province_mask))
    )

    # --------------------------------------------------------------
    # Fuel
    # --------------------------------------------------------------

    print()
    print("Preparing Fuel...")

    with rasterio.open(args.master) as fuel_src:
        fuel = fuel_src.read(1).astype(
            "float32"
        )

        fuel_nodata = fuel_src.nodata

    if fuel_nodata is not None:
        fuel[
            np.isclose(
                fuel,
                fuel_nodata
            )
        ] = np.nan

    fuel[
        ~np.isfinite(fuel)
    ] = np.nan

    fuel_norm, fuel_low_value, fuel_high_value = (
        normalize_percentile(
            fuel,
            low_percentile=fuel_percentile_low,
            high_percentile=fuel_percentile_high,
            mask=province_mask
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
            aspect_weight=aspect_weight,
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
        prepare_fwi_to_master(
            args.fwi,
            master
        )
    )

    fwi[
        ~province_mask
    ] = np.nan

    # Normalize FWI.
    fwi_norm = normalize_linear(
        fwi,
        fwi_min,
        fwi_max
    )

    # --------------------------------------------------------------
    # Final WLC
    # --------------------------------------------------------------

    print()
    print("Calculating final wildfire risk...")

    valid = (
        province_mask
        & np.isfinite(fwi_norm)
        & np.isfinite(fuel_norm)
        & np.isfinite(topography)
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
        * (
            fwi_weight * fwi_norm[valid]
            +
            fuel_weight * fuel_norm[valid]
            +
            topo_weight * topography[valid]
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

    if not np.any(np.isfinite(risk)):
        raise ValueError(
            "Final risk raster contains no valid cells."
        )

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
        f"  Min     : {statistics['min']:.3f}"
    )
    print(
        f"  Max     : {statistics['max']:.3f}"
    )
    print(
        f"  Mean    : {statistics['mean']:.3f}"
    )
    print(
        f"  Median  : {statistics['median']:.3f}"
    )
    print(
        f"  Valid   : {statistics['valid_cells']}"
    )

    # --------------------------------------------------------------
    # Save final raster
    # --------------------------------------------------------------

    args.output_tif.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_profile = master["profile"].copy()

    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
        nodata=-9999.0,
        width=master["width"],
        height=master["height"],
        transform=master["transform"],
        crs=master["crs"],
    )

    output_array = np.where(
        np.isfinite(risk),
        risk,
        -9999.0
    ).astype("float32")

    print()
    print("Writing final GeoTIFF...")

    with rasterio.open(
        args.output_tif,
        "w",
        **output_profile
    ) as dst:
        dst.write(
            output_array,
            1
        )

        dst.set_band_description(
            1,
            "Smart Fars Wildfire Risk (0-100)"
        )

    # --------------------------------------------------------------
    # Web output
    # --------------------------------------------------------------

    print()
    print("Building Web GIS output...")

    png_path, web_metadata_path, web_metadata = (
        create_web_png(
            risk=risk,
            master=master,
            boundary_geometry=boundary_geometry,
            boundary_crs=boundary_crs,
            web_dir=args.web_dir,
            max_dimension=max_dimension,
        )
    )

    # --------------------------------------------------------------
    # Final JSON metadata
    # --------------------------------------------------------------

    args.output_tif.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_metadata = {
        "model": "Smart Fars Forecast",
        "risk_range": [
            0.0,
            100.0
        ],
        "formula": (
            "100 * "
            "(0.45 * FWI_norm + "
            "0.35 * Fuel_norm + "
            "0.20 * Topography)"
        ),
        "weights": {
            "fwi": fwi_weight,
            "fuel": fuel_weight,
            "topography": topo_weight
        },
        "topography": {
            "slope_weight": slope_weight,
            "aspect_weight": aspect_weight,
            "slope_max": slope_max,
            "aspect_classes": {
                "south": 1.0,
                "east_west": 0.6,
                "north": 0.3
            }
        },
        "fwi_processing": {
            "raw_source": str(args.fwi),
            "fillnodata": fwi_fill_stats,
            "resampling_to_master": "bilinear",
            "normalization_min": fwi_min,
            "normalization_max": fwi_max
        },
        "fuel_normalization": {
            "method": "percentile",
            "low_percentile": fuel_percentile_low,
            "high_percentile": fuel_percentile_high,
            "low_value": fuel_low_value,
            "high_value": fuel_high_value
        },
        "master_grid": {
            "width": master["width"],
            "height": master["height"],
            "crs": (
                master["crs"].to_string()
                if hasattr(
                    master["crs"],
                    "to_string"
                )
                else str(master["crs"])
            ),
            "transform": list(
                master["transform"]
            )[:6],
            "resolution": [
                abs(float(master["transform"].a)),
                abs(float(master["transform"].e))
            ]
        },
        "statistics": statistics,
        "web": web_metadata,
        "outputs": {
            "raster": str(args.output_tif),
            "web_png": str(png_path),
            "web_metadata": str(
                web_metadata_path
            )
        }
    }

    metadata_path = (
        args.output_tif.with_suffix(".json")
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "Final raster:",
        args.output_tif
    )

    print(
        "Final metadata:",
        metadata_path
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
    print("FWI NoData:")
    print(
        f"  Before fill : "
        f"{fwi_fill_stats['nodata_before']}"
    )
    print(
        f"  Filled      : "
        f"{fwi_fill_stats['filled_pixels']}"
    )
    print(
        f"  Remaining   : "
        f"{fwi_fill_stats['nodata_after']}"
    )

    print()
    print("Processing chain:")
    print(
        "Raw FWI -> fillnodata -> Bilinear -> "
        "Master 60m -> FWI_norm -> WLC"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
