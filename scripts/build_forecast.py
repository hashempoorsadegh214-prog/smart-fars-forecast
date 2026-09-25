# -*- coding: utf-8 -*-

"""Smart Fars Forecast builder.

Scientific chain:
    raw FWI -> NoData fill -> bilinear -> master grid -> normalization
    fuel master raster -> percentile normalization
    DEM/slope -> master grid
    topography = 0.80*slope_norm + 0.20*aspect_norm
    risk = 100*(0.45*FWI_norm + 0.35*Fuel_norm + 0.20*Topography)

The scientific GeoTIFF is never smoothed or georeferenced differently for
the web. Web-only smoothing and four-corner web georeferencing are applied
only to the PNG visualization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.fill import fillnodata
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FWI = PROJECT_ROOT / "data/fwi/fwi_latest.tif"
DEFAULT_MASTER = PROJECT_ROOT / "fars_fire_fuel_hazard_60m.tif"
DEFAULT_DEM = PROJECT_ROOT / "dem_fars.tif"
DEFAULT_SLOPE = PROJECT_ROOT / "fars_slope_60m_light.tif"
DEFAULT_BOUNDARY = PROJECT_ROOT / "fars.geojson"
DEFAULT_CONFIG = PROJECT_ROOT / "config/model_config.json"
DEFAULT_OUTPUT_TIF = PROJECT_ROOT / "data/output/fire_risk_latest.tif"
DEFAULT_WEB_DIR = PROJECT_ROOT / "web/generated"

DEFAULT_FWI_FILL_DISTANCE = 20.0
DEFAULT_FWI_SMOOTHING = 1
DEFAULT_WEB_SMOOTHING_RADIUS = 18.0
ALPHA_VALUE = 235

# ---------------------------------------------------------------------------
# Web-only four-corner georeferencing control points.
#
# These are the exact bounding-box corners calculated from fars.geojson.
# The scientific GeoTIFF is NOT modified by these values.
# ---------------------------------------------------------------------------
FARS_WEB_WEST = 50.603183099720354
FARS_WEB_EAST = 55.58006519994905
FARS_WEB_SOUTH = 27.04561700022441
FARS_WEB_NORTH = 31.669598899969287


def parse_args():
    p = argparse.ArgumentParser(
        description="Build Smart Fars wildfire forecast"
    )

    p.add_argument(
        "--fwi",
        type=Path,
        default=DEFAULT_FWI
    )

    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG
    )

    p.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER
    )

    p.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM
    )

    p.add_argument(
        "--slope",
        type=Path,
        default=DEFAULT_SLOPE
    )

    p.add_argument(
        "--boundary",
        type=Path,
        default=DEFAULT_BOUNDARY
    )

    p.add_argument(
        "--output-tif",
        type=Path,
        default=DEFAULT_OUTPUT_TIF
    )

    p.add_argument(
        "--web-dir",
        type=Path,
        default=DEFAULT_WEB_DIR
    )

    return p.parse_args()


def ensure_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_boundary(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    geometries = []

    if data.get("type") == "FeatureCollection":

        for feature in data.get(
            "features",
            []
        ):

            if feature.get("geometry"):
                geometries.append(
                    shape(
                        feature["geometry"]
                    )
                )

    elif data.get("type") == "Feature":

        if data.get("geometry"):
            geometries.append(
                shape(
                    data["geometry"]
                )
            )

    else:

        geometries.append(
            shape(data)
        )

    if not geometries:
        raise ValueError(
            f"No geometry found in {path}"
        )

    geom = unary_union(
        geometries
    )

    if geom.is_empty:
        raise ValueError(
            "Fars boundary geometry is empty"
        )

    return geom, "EPSG:4326"


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


def load_master(path):
    with rasterio.open(path) as src:

        return {
            "data":
                src.read(1).astype(
                    "float32"
                ),

            "width":
                src.width,

            "height":
                src.height,

            "transform":
                src.transform,

            "crs":
                src.crs,

            "nodata":
                src.nodata,

            "profile":
                src.profile.copy(),
        }


def read_to_master_grid(
    source_path,
    master
):
    with rasterio.open(
        source_path
    ) as src:

        source = src.read(1).astype(
            "float32"
        )

        source_nodata = src.nodata

        invalid = (
            ~np.isfinite(source)
        )

        if source_nodata is not None:

            invalid |= np.isclose(
                source,
                float(
                    source_nodata
                )
            )

        source[invalid] = -9999.0

        destination = np.full(
            (
                master["height"],
                master["width"]
            ),
            -9999.0,
            dtype="float32",
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-9999.0,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            dst_nodata=-9999.0,
            resampling=Resampling.bilinear,
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


def fill_fwi_nodata(
    array,
    source_nodata
):
    fwi = np.array(
        array,
        dtype="float32",
        copy=True
    )

    valid = (
        np.isfinite(fwi)
        &
        (fwi >= 0.0)
    )

    if source_nodata is not None:

        valid &= ~np.isclose(
            fwi,
            float(
                source_nodata
            )
        )

    nodata_before = int(
        np.count_nonzero(
            ~valid
        )
    )

    if nodata_before == 0:

        return fwi, {
            "nodata_before":
                0,

            "filled_pixels":
                0,

            "nodata_after":
                0,

            "max_search_distance":
                DEFAULT_FWI_FILL_DISTANCE,

            "smoothing_iterations":
                DEFAULT_FWI_SMOOTHING,
        }

    work = np.array(
        fwi,
        dtype="float32",
        copy=True
    )

    work[~valid] = 0.0

    filled = fillnodata(
        image=work,
        mask=valid.astype(
            "uint8"
        ),
        max_search_distance=(
            DEFAULT_FWI_FILL_DISTANCE
        ),
        smoothing_iterations=(
            DEFAULT_FWI_SMOOTHING
        ),
    ).astype(
        "float32"
    )

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
        "nodata_before":
            nodata_before,

        "filled_pixels":
            filled_pixels,

        "nodata_after":
            nodata_after,

        "max_search_distance":
            DEFAULT_FWI_FILL_DISTANCE,

        "smoothing_iterations":
            DEFAULT_FWI_SMOOTHING,
    }


def prepare_fwi(
    path,
    master
):
    with rasterio.open(
        path
    ) as src:

        raw = src.read(1).astype(
            "float32"
        )

        nodata = src.nodata

        src_transform = src.transform

        src_crs = src.crs

    print("\nFWI source:")

    print(
        "  Size:",
        raw.shape
    )

    print(
        "  CRS:",
        src_crs
    )

    print(
        "  NoData:",
        nodata
    )

    filled, stats = fill_fwi_nodata(
        raw,
        nodata
    )

    print(
        "  NoData before:",
        stats["nodata_before"]
    )

    print(
        "  Filled:",
        stats["filled_pixels"]
    )

    print(
        "  NoData after:",
        stats["nodata_after"]
    )

    source = np.where(
        np.isfinite(filled),
        filled,
        -9999.0
    ).astype(
        "float32"
    )

    destination = np.full(
        (
            master["height"],
            master["width"]
        ),
        -9999.0,
        dtype="float32",
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=-9999.0,
        dst_transform=master["transform"],
        dst_crs=master["crs"],
        dst_nodata=-9999.0,
        resampling=Resampling.bilinear,
    )

    destination[
        ~np.isfinite(destination)
        |
        np.isclose(
            destination,
            -9999.0
        )
    ] = np.nan

    return (
        destination,
        stats
    )


def normalize_linear(
    array,
    minimum,
    maximum
):
    if maximum <= minimum:
        raise ValueError(
            "Invalid normalization range"
        )

    out = np.full(
        array.shape,
        np.nan,
        dtype="float32"
    )

    valid = np.isfinite(
        array
    )

    if np.any(valid):

        out[valid] = (
            (
                np.clip(
                    array[valid],
                    minimum,
                    maximum
                )
                -
                minimum
            )
            /
            (maximum - minimum)
        ).astype(
            "float32"
        )

    return out


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
            "No valid pixels for fuel normalization"
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
            "Invalid fuel percentile range"
        )

    out = np.full(
        array.shape,
        np.nan,
        dtype="float32"
    )

    out[valid] = (
        (
            np.clip(
                array[valid],
                low,
                high
            )
            -
            low
        )
        /
        (high - low)
    ).astype(
        "float32"
    )

    return (
        out,
        low,
        high
    )


def calculate_aspect_risk(
    dem,
    transform
):
    out = np.full(
        dem.shape,
        np.nan,
        dtype="float32"
    )

    valid = np.isfinite(
        dem
    )

    if not np.any(valid):
        return out

    dx = (
        abs(
            float(
                transform.a
            )
        )
        or
        1.0
    )

    dy = (
        abs(
            float(
                transform.e
            )
        )
        or
        1.0
    )

    safe_dem = np.array(
        dem,
        dtype="float64",
        copy=True
    )

    fill_value = float(
        np.nanmedian(
            safe_dem[valid]
        )
    )

    safe_dem[
        ~np.isfinite(safe_dem)
    ] = fill_value

    dz_dy, dz_dx = np.gradient(
        safe_dem,
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
        +
        360.0
    ) % 360.0

    good = (
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

    out[
        good & north
    ] = 0.3

    out[
        good & east_west
    ] = 0.6

    out[
        good & south
    ] = 1.0

    return out


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

    topo = (
        slope_weight
        *
        slope_norm
        +
        aspect_weight
        *
        aspect_norm
    )

    topo[
        ~np.isfinite(topo)
    ] = np.nan

    return topo


def compute_web_size(
    width,
    height,
    max_dimension
):
    scale = min(
        1.0,
        float(max_dimension)
        /
        float(
            max(width, height)
        )
    )

    return (
        max(
            1,
            int(
                round(
                    width * scale
                )
            )
        ),

        max(
            1,
            int(
                round(
                    height * scale
                )
            )
        ),
    )


def box_blur_2d(
    array,
    radius
):
    radius = min(
        64,
        int(
            round(radius)
        )
    )

    if radius <= 0:

        return np.array(
            array,
            dtype="float32",
            copy=True
        )

    src = np.array(
        array,
        dtype="float32",
        copy=True
    )

    padded = np.pad(
        src,
        (
            (
                radius,
                radius
            ),
            (
                radius,
                radius
            )
        ),
        mode="edge"
    )

    c = np.cumsum(
        np.cumsum(
            padded,
            axis=0,
            dtype="float32"
        ),
        axis=1,
        dtype="float32"
    )

    c = np.pad(
        c,
        (
            (
                1,
                0
            ),
            (
                1,
                0
            )
        ),
        mode="constant"
    )

    size = (
        2 * radius
        +
        1
    )

    result = (
        c[
            size:,
            size:
        ]
        -
        c[
            :-size,
            size:
        ]
        -
        c[
            size:,
            :-size
        ]
        +
        c[
            :-size,
            :-size
        ]
    ) / float(
        size * size
    )

    return result.astype(
        "float32"
    )


def smooth_web_risk(
    risk,
    valid_mask,
    radius
):
    if radius <= 0:

        return np.array(
            risk,
            dtype="float32",
            copy=True
        )

    valid = valid_mask.astype(
        "float32"
    )

    values = np.where(
        valid_mask,
        risk,
        0.0
    ).astype(
        "float32"
    )

    numerator = box_blur_2d(
        values,
        radius
    )

    denominator = box_blur_2d(
        valid,
        radius
    )

    out = np.full(
        risk.shape,
        np.nan,
        dtype="float32"
    )

    good = (
        denominator > 0.001
    )

    out[good] = (
        numerator[good]
        /
        denominator[good]
    )

    return np.clip(
        out,
        0.0,
        100.0
    )


def risk_to_rgb(
    risk
):
    stops = np.array(
        [
            0,
            20,
            40,
            60,
            80,
            100
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
            [136, 14, 79],
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

    rgb[
        valid,
        0
    ] = np.interp(
        values,
        stops,
        colors[:, 0]
    ).astype(
        "uint8"
    )

    rgb[
        valid,
        1
    ] = np.interp(
        values,
        stops,
        colors[:, 1]
    ).astype(
        "uint8"
    )

    rgb[
        valid,
        2
    ] = np.interp(
        values,
        stops,
        colors[:, 2]
    ).astype(
        "uint8"
    )

    return (
        rgb,
        valid
    )


def web_bounds(
    transform,
    width,
    height,
    crs
):
    pts = []

    samples = 200

    for i in range(
        samples + 1
    ):

        t = (
            i
            /
            float(samples)
        )

        pts.extend(
            [
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
                ),
            ]
        )

    xy = [
        transform * p
        for p in pts
    ]

    xs = np.array(
        [
            p[0]
            for p in xy
        ],
        dtype="float64"
    )

    ys = np.array(
        [
            p[1]
            for p in xy
        ],
        dtype="float64"
    )

    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = transformer.transform(
        xs,
        ys
    )

    lon = np.asarray(
        lon
    )

    lat = np.asarray(
        lat
    )

    good = (
        np.isfinite(lon)
        &
        np.isfinite(lat)
    )

    if not np.any(good):

        raise ValueError(
            "Could not calculate web bounds"
        )

    return [
        float(
            np.min(
                lon[good]
            )
        ),

        float(
            np.min(
                lat[good]
            )
        ),

        float(
            np.max(
                lon[good]
            )
        ),

        float(
            np.max(
                lat[good]
            )
        ),
    ]


def create_web_png(
    risk,
    master,
    boundary_geometry,
    boundary_crs,
    web_dir,
    max_dimension,
    smoothing_radius
):
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
    # Resize the model result for the web only.
    # --------------------------------------------------------------

    risk_safe = np.where(
        np.isfinite(risk),
        risk,
        0.0
    ).astype(
        "float32"
    )

    risk_image = Image.fromarray(
        risk_safe,
        mode="F"
    )

    risk_resized = risk_image.resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.BICUBIC
    )

    risk_web = np.array(
        risk_resized,
        dtype="float32",
        copy=True
    )

    valid_image = Image.fromarray(
        (
            np.isfinite(risk)
            *
            255
        ).astype(
            "uint8"
        ),
        mode="L"
    )

    valid_web = np.asarray(
        valid_image.resize(
            (
                target_width,
                target_height
            ),
            Image.Resampling.NEAREST
        ),
        dtype="uint8"
    ) > 0

    risk_web[
        ~valid_web
    ] = np.nan

    # --------------------------------------------------------------
    # WEB-ONLY smoothing.
    # Scientific GeoTIFF is not changed.
    # --------------------------------------------------------------

    risk_web = smooth_web_risk(
        risk_web,
        valid_web,
        smoothing_radius
    )

    rgb, finite = risk_to_rgb(
        risk_web
    )

    rgb_image = Image.fromarray(
        rgb,
        mode="RGB"
    )

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

    # ------------------------------------------------------------------
    # Four-point web georeferencing.
    #
    # SOURCE:
    #   NW / NE / SE / SW = actual corners of the scientific raster
    #
    # TARGET:
    #   NW / NE / SE / SW = corners of the Fars boundary bounding box
    #
    # ONLY the web PNG uses this transform.
    # The scientific GeoTIFF remains completely unchanged.
    # ------------------------------------------------------------------

    source_bounds = web_bounds(
        master["transform"],
        width,
        height,
        master["crs"]
    )

    target_bounds = [
        FARS_WEB_WEST,
        FARS_WEB_SOUTH,
        FARS_WEB_EAST,
        FARS_WEB_NORTH,
    ]

    source_west, source_south, source_east, source_north = (
        source_bounds
    )

    target_west, target_south, target_east, target_north = (
        target_bounds
    )

    # Web PNG is published in EPSG:4326.
    # The four TARGET control points define its geographic transform.

    target_transform = Affine(
        (
            target_east
            -
            target_west
        )
        /
        float(
            target_width
        ),

        0.0,

        target_west,

        0.0,

        -(
            target_north
            -
            target_south
        )
        /
        float(
            target_height
        ),

        target_north,
    )

    geometry_web = reproject_geometry(
        boundary_geometry,
        boundary_crs,
        "EPSG:4326"
    )

    province_mask = rasterize(
        [
            (
                mapping(
                    geometry_web
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

        dtype="uint8",
    ).astype(
        bool
    )

    control_points = {
        "source": {
            "NW": [
                source_west,
                source_north
            ],

            "NE": [
                source_east,
                source_north
            ],

            "SE": [
                source_east,
                source_south
            ],

            "SW": [
                source_west,
                source_south
            ],
        },

        "target": {
            "NW": [
                target_west,
                target_north
            ],

            "NE": [
                target_east,
                target_north
            ],

            "SE": [
                target_east,
                target_south
            ],

            "SW": [
                target_west,
                target_south
            ],
        },
    }

    alpha = np.where(
        province_mask
        &
        finite,
        ALPHA_VALUE,
        0
    ).astype(
        "uint8"
    )

    out = Image.fromarray(
        np.dstack(
            [
                rgb_final,
                alpha
            ]
        ),
        mode="RGBA"
    )

    png_path = (
        web_dir
        /
        "fire_risk_latest.png"
    )

    out.save(
        png_path,
        format="PNG",
        optimize=True
    )

    # --------------------------------------------------------------
    # Publish TARGET bounds, not original raster bounds.
    # --------------------------------------------------------------

    bounds = target_bounds

    meta = {
        "image_size": [
            target_width,
            target_height
        ],

        "master_size": [
            width,
            height
        ],

        "bounds":
            bounds,

        "control_points":
            control_points,

        "image_url":
            "generated/fire_risk_latest.png",

        "web": {
            "crs":
                "EPSG:4326",

            "georeferencing":
                "four-corner control-point mapping",

            "bounds":
                bounds,

            "control_points":
                control_points,

            "image_url":
                "generated/fire_risk_latest.png",

            "smoothing": {
                "enabled":
                    smoothing_radius > 0,

                "method":
                    "web-only weighted box smoothing",

                "radius":
                    float(
                        smoothing_radius
                    ),
            },
        },

        "master_crs": (
            master["crs"].to_string()
            if hasattr(
                master["crs"],
                "to_string"
            )
            else str(
                master["crs"]
            )
        ),

        "alpha":
            ALPHA_VALUE,
    }

    metadata_path = (
        web_dir
        /
        "fire_risk_latest.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2
        )

    return (
        png_path,
        metadata_path,
        meta
    )


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
            "No valid risk cells inside Fars"
        )

    values = risk[valid]

    return {
        "valid_cells":
            int(
                values.size
            ),

        "min":
            float(
                np.min(values)
            ),

        "max":
            float(
                np.max(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "std":
            float(
                np.std(values)
            ),

        "classes": {
            "very_low":
                int(
                    np.count_nonzero(
                        values < 20
                    )
                ),

            "low":
                int(
                    np.count_nonzero(
                        (
                            values >= 20
                        )
                        &
                        (
                            values < 40
                        )
                    )
                ),

            "moderate":
                int(
                    np.count_nonzero(
                        (
                            values >= 40
                        )
                        &
                        (
                            values < 60
                        )
                    )
                ),

            "high":
                int(
                    np.count_nonzero(
                        (
                            values >= 60
                        )
                        &
                        (
                            values < 80
                        )
                    )
                ),

            "critical":
                int(
                    np.count_nonzero(
                        values >= 80
                    )
                ),
        },
    }


def main():

    args = parse_args()

    print(
        "=" * 70
    )

    print(
        "SMART FARS FORECAST"
    )

    print(
        "=" * 70
    )

    for path, label in [
        (
            args.fwi,
            "FWI raster"
        ),
        (
            args.master,
            "Master Raster"
        ),
        (
            args.dem,
            "DEM"
        ),
        (
            args.slope,
            "Slope"
        ),
        (
            args.boundary,
            "Fars boundary"
        ),
        (
            args.config,
            "Model config"
        ),
    ]:

        ensure_file(
            path,
            label
        )

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

    topo_weight = float(
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
        topo_weight,
        1.0,
        atol=1e-6
    ):

        raise ValueError(
            "Model weights must sum to 1"
        )

    norm = config.get(
        "normalization",
        {}
    )

    fuel_p_low = float(
        norm.get(
            "fuel_percentile_low",
            1.0
        )
    )

    fuel_p_high = float(
        norm.get(
            "fuel_percentile_high",
            99.0
        )
    )

    fwi_min = float(
        norm.get(
            "fwi_min",
            0.0
        )
    )

    fwi_max = float(
        norm.get(
            "fwi_max",
            100.0
        )
    )

    slope_max = float(
        norm.get(
            "slope_max",
            45.0
        )
    )

    topo_cfg = config.get(
        "topography",
        {}
    )

    slope_weight = float(
        topo_cfg.get(
            "slope_weight",
            0.80
        )
    )

    aspect_weight = float(
        topo_cfg.get(
            "aspect_weight",
            0.20
        )
    )

    web_cfg = config.get(
        "web",
        {}
    )

    max_dimension = int(
        web_cfg.get(
            "max_dimension",
            3000
        )
    )

    web_smoothing_radius = float(
        web_cfg.get(
            "smoothing_radius",
            DEFAULT_WEB_SMOOTHING_RADIUS
        )
    )

    print()
    print(
        "Model weights:"
    )

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
        topo_weight
    )

    print(
        "Web smoothing radius:",
        web_smoothing_radius
    )

    print()
    print(
        "Loading Master Raster..."
    )

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

    print()
    print(
        "Loading Fars boundary..."
    )

    boundary, boundary_crs = (
        load_boundary(
            args.boundary
        )
    )

    boundary_master = (
        reproject_geometry(
            boundary,
            boundary_crs,
            master["crs"]
        )
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

        all_touched=False,
    )

    print(
        "  Province cells:",
        int(
            np.count_nonzero(
                province_mask
            )
        )
    )

    print()
    print(
        "Preparing Fuel..."
    )

    fuel = np.array(
        master["data"],
        dtype="float32",
        copy=True
    )

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
            fuel_p_low,
            fuel_p_high,
        )
    )

    print()
    print(
        "Preparing DEM..."
    )

    dem = read_to_master_grid(
        args.dem,
        master
    )

    dem[
        ~province_mask
    ] = np.nan

    print()
    print(
        "Preparing Slope..."
    )

    slope = read_to_master_grid(
        args.slope,
        master
    )

    slope[
        ~province_mask
    ] = np.nan

    print()
    print(
        "Calculating Topography..."
    )

    topography = build_topography(
        dem,
        slope,
        master["transform"],
        slope_max,
        slope_weight,
        aspect_weight,
    )

    topography[
        ~province_mask
    ] = np.nan

    print()
    print(
        "Preparing FWI..."
    )

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

    print()
    print(
        "Calculating wildfire risk..."
    )

    valid = (
        province_mask
        &
        np.isfinite(
            fwi_norm
        )
        &
        np.isfinite(
            fuel_norm
        )
        &
        np.isfinite(
            topography
        )
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

            topo_weight
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

    stats = calculate_statistics(
        risk,
        province_mask
    )

    print()
    print(
        "Forecast statistics:"
    )

    print(
        f"  Min    : "
        f"{stats['min']:.3f}"
    )

    print(
        f"  Max    : "
        f"{stats['max']:.3f}"
    )

    print(
        f"  Mean   : "
        f"{stats['mean']:.3f}"
    )

    print(
        f"  Median : "
        f"{stats['median']:.3f}"
    )

    args.output_tif.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    profile = (
        master["profile"].copy()
    )

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
        crs=master["crs"],
    )

    print()
    print(
        "Writing scientific GeoTIFF..."
    )

    with rasterio.open(
        args.output_tif,
        "w",
        **profile
    ) as dst:

        dst.write(
            np.where(
                np.isfinite(risk),
                risk,
                -9999.0
            ).astype(
                "float32"
            ),
            1,
        )

        dst.set_band_description(
            1,
            "Smart Fars Wildfire Risk (0-100)"
        )

    print()
    print(
        "Building Web GIS PNG..."
    )

    png_path, web_metadata_path, web_meta = (
        create_web_png(
            risk=risk,
            master=master,
            boundary_geometry=boundary,
            boundary_crs=boundary_crs,
            web_dir=args.web_dir,
            max_dimension=max_dimension,
            smoothing_radius=web_smoothing_radius,
        )
    )

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
                topo_weight,
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
                    0.3,
            },
        },

        "fwi_processing": {
            "raw_source":
                str(
                    args.fwi
                ),

            "fillnodata":
                fwi_fill_stats,

            "resampling_to_master":
                "bilinear",

            "normalization_min":
                fwi_min,

            "normalization_max":
                fwi_max,
        },

        "fuel_normalization": {
            "method":
                "percentile",

            "low_percentile":
                fuel_p_low,

            "high_percentile":
                fuel_p_high,

            "low_value":
                fuel_low,

            "high_value":
                fuel_high,
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
                ),
            ],
        },

        "statistics":
            stats,

        "web":
            web_meta,

        "outputs": {
            "raster":
                str(
                    args.output_tif
                ),

            "web_png":
                str(
                    png_path
                ),

            "web_metadata":
                str(
                    web_metadata_path
                ),
        },
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

    print()
    print(
        "=" * 70
    )

    print(
        "SUCCESS"
    )

    print(
        "=" * 70
    )

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

    print(
        "FWI: Raw -> fillnodata -> "
        "Bilinear -> Master -> normalization"
    )

    print(
        "Risk: scientific raster unchanged "
        "by web smoothing"
    )

    print(
        "Web smoothing radius:",
        web_smoothing_radius
    )

    print(
        "Web georeferencing:",
        "four-corner control-point mapping"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
