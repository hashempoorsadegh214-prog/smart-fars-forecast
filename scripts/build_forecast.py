# -*- coding: utf-8 -*-

"""Smart Fars Forecast builder.

Scientific processing:
    FWI
        -> NoData fill
        -> reprojection to master grid
        -> normalization

    Fuel
        -> percentile normalization

    DEM + Slope
        -> topography

    Risk:
        100 * (
            0.45 * FWI_norm
            + 0.35 * Fuel_norm
            + 0.20 * Topography
        )

IMPORTANT
---------
The scientific GeoTIFF keeps the original master raster grid.

For WEB output only:
    - the scientific risk raster is reprojected to EPSG:4326
    - a real geographic web grid is created
    - Fars boundary is rasterized on that web grid
    - values outside Fars are forced to NoData/transparent
    - smoothing is applied only for visualization
    - the Fars mask is applied again after smoothing

No artificial four-corner deformation is used.
"""


import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from PIL import Image

from pyproj import Transformer

from rasterio.enums import Resampling

from rasterio.features import geometry_mask

from rasterio.fill import fillnodata

from rasterio.transform import array_bounds

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_bounds,
)

from shapely.geometry import (
    mapping,
    shape,
)

from shapely.ops import (
    transform as shp_transform,
    unary_union,
)


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_FWI = (
    PROJECT_ROOT
    / "data/fwi/fwi_latest.tif"
)

DEFAULT_MASTER = (
    PROJECT_ROOT
    / "fars_fire_fuel_hazard_60m.tif"
)

DEFAULT_DEM = (
    PROJECT_ROOT
    / "dem_fars.tif"
)

DEFAULT_SLOPE = (
    PROJECT_ROOT
    / "fars_slope_60m_light.tif"
)

DEFAULT_BOUNDARY = (
    PROJECT_ROOT
    / "fars.geojson"
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/model_config.json"
)

DEFAULT_OUTPUT_TIF = (
    PROJECT_ROOT
    / "data/output/fire_risk_latest.tif"
)

DEFAULT_WEB_DIR = (
    PROJECT_ROOT
    / "web/generated"
)


# ============================================================
# PROCESSING SETTINGS
# ============================================================

DEFAULT_FWI_FILL_DISTANCE = 20.0

DEFAULT_FWI_SMOOTHING = 1

DEFAULT_WEB_SMOOTHING_RADIUS = 18.0

ALPHA_VALUE = 235


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build Smart Fars wildfire forecast."
        )
    )

    parser.add_argument(
        "--fwi",
        type=Path,
        default=DEFAULT_FWI,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
    )

    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
    )

    parser.add_argument(
        "--slope",
        type=Path,
        default=DEFAULT_SLOPE,
    )

    parser.add_argument(
        "--boundary",
        type=Path,
        default=DEFAULT_BOUNDARY,
    )

    parser.add_argument(
        "--output-tif",
        type=Path,
        default=DEFAULT_OUTPUT_TIF,
    )

    parser.add_argument(
        "--web-dir",
        type=Path,
        default=DEFAULT_WEB_DIR,
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_file(
    path,
    label,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


# ============================================================
# LOAD FARS BOUNDARY
# ============================================================

def load_boundary(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        data = json.load(handle)

    geometries = []

    if data.get(
        "type"
    ) == "FeatureCollection":

        for feature in data.get(
            "features",
            [],
        ):

            geometry = feature.get(
                "geometry"
            )

            if geometry:

                geometries.append(
                    shape(
                        geometry
                    )
                )

    elif data.get(
        "type"
    ) == "Feature":

        geometry = data.get(
            "geometry"
        )

        if geometry:

            geometries.append(
                shape(
                    geometry
                )
            )

    else:

        geometries.append(
            shape(
                data
            )
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

    return (
        geometry,
        "EPSG:4326",
    )


# ============================================================
# REPROJECT GEOMETRY
# ============================================================

def reproject_geometry(
    geometry,
    source_crs,
    target_crs,
):

    if (
        str(source_crs)
        ==
        str(target_crs)
    ):

        return geometry

    transformer = (
        Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
        )
    )

    return shp_transform(
        transformer.transform,
        geometry,
    )


# ============================================================
# LOAD MASTER
# ============================================================

def load_master(path):

    with rasterio.open(
        path
    ) as src:

        return {
            "data":
                src.read(
                    1
                ).astype(
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


# ============================================================
# ALIGN RASTER TO MASTER
# ============================================================

def read_to_master_grid(
    source_path,
    master,
):

    with rasterio.open(
        source_path
    ) as src:

        source = (
            src.read(
                1
            ).astype(
                "float32"
            )
        )

        source_nodata = src.nodata

        invalid = (
            ~np.isfinite(
                source
            )
        )

        if source_nodata is not None:

            invalid |= np.isclose(
                source,
                float(
                    source_nodata
                )
            )

        source[
            invalid
        ] = -9999.0

        destination = np.full(
            (
                master["height"],
                master["width"],
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

            dst_transform=master[
                "transform"
            ],
            dst_crs=master[
                "crs"
            ],
            dst_nodata=-9999.0,

            resampling=Resampling.bilinear,
        )

    destination[
        ~np.isfinite(
            destination
        )
        |
        np.isclose(
            destination,
            -9999.0,
        )
    ] = np.nan

    return destination


# ============================================================
# FWI NODATA FILL
# ============================================================

def fill_fwi_nodata(
    array,
    source_nodata,
):

    fwi = np.array(
        array,
        dtype="float32",
        copy=True,
    )

    valid = (
        np.isfinite(
            fwi
        )
        &
        (
            fwi >= 0.0
        )
    )

    if source_nodata is not None:

        valid &= ~np.isclose(
            fwi,
            float(
                source_nodata
            ),
        )

    nodata_before = int(
        np.count_nonzero(
            ~valid
        )
    )

    if nodata_before == 0:

        return (
            fwi,
            {
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
            },
        )

    work = np.array(
        fwi,
        dtype="float32",
        copy=True,
    )

    work[
        ~valid
    ] = 0.0

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
        ~np.isfinite(
            filled
        )
        |
        (
            filled < 0.0
        )
    ] = np.nan

    filled_pixels = int(
        np.count_nonzero(
            (~valid)
            &
            np.isfinite(
                filled
            )
        )
    )

    nodata_after = int(
        np.count_nonzero(
            ~np.isfinite(
                filled
            )
        )
    )

    return (
        filled,
        {
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
        },
    )


# ============================================================
# PREPARE FWI
# ============================================================

def prepare_fwi(
    path,
    master,
):

    with rasterio.open(
        path
    ) as src:

        raw = (
            src.read(
                1
            ).astype(
                "float32"
            )
        )

        nodata = src.nodata

        source_transform = (
            src.transform
        )

        source_crs = src.crs

    print()
    print(
        "FWI SOURCE"
    )
    print(
        "----------"
    )

    print(
        "Size:",
        raw.shape,
    )

    print(
        "CRS:",
        source_crs,
    )

    print(
        "NoData:",
        nodata,
    )

    filled, fill_stats = (
        fill_fwi_nodata(
            raw,
            nodata,
        )
    )

    print(
        "NoData before:",
        fill_stats[
            "nodata_before"
        ],
    )

    print(
        "Filled:",
        fill_stats[
            "filled_pixels"
        ],
    )

    print(
        "NoData after:",
        fill_stats[
            "nodata_after"
        ],
    )

    source = np.where(
        np.isfinite(
            filled
        ),
        filled,
        -9999.0,
    ).astype(
        "float32"
    )

    destination = np.full(
        (
            master["height"],
            master["width"],
        ),
        -9999.0,
        dtype="float32",
    )

    reproject(
        source=source,
        destination=destination,

        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=-9999.0,

        dst_transform=master[
            "transform"
        ],
        dst_crs=master[
            "crs"
        ],
        dst_nodata=-9999.0,

        resampling=Resampling.bilinear,
    )

    destination[
        ~np.isfinite(
            destination
        )
        |
        np.isclose(
            destination,
            -9999.0,
        )
    ] = np.nan

    return (
        destination,
        fill_stats,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_linear(
    array,
    minimum,
    maximum,
):

    if maximum <= minimum:

        raise ValueError(
            "Invalid normalization range."
        )

    output = np.full(
        array.shape,
        np.nan,
        dtype="float32",
    )

    valid = np.isfinite(
        array
    )

    if np.any(valid):

        output[
            valid
        ] = (
            (
                np.clip(
                    array[
                        valid
                    ],
                    minimum,
                    maximum,
                )
                -
                minimum
            )
            /
            (
                maximum
                -
                minimum
            )
        ).astype(
            "float32"
        )

    return output


def normalize_percentile(
    array,
    mask,
    low_percentile,
    high_percentile,
):

    valid = (
        np.isfinite(
            array
        )
        &
        mask
    )

    if not np.any(
        valid
    ):

        raise ValueError(
            "No valid pixels available "
            "for fuel normalization."
        )

    values = array[
        valid
    ]

    low = float(
        np.percentile(
            values,
            low_percentile,
        )
    )

    high = float(
        np.percentile(
            values,
            high_percentile,
        )
    )

    if high <= low:

        raise ValueError(
            "Invalid fuel percentile range."
        )

    output = np.full(
        array.shape,
        np.nan,
        dtype="float32",
    )

    output[
        valid
    ] = (
        (
            np.clip(
                array[
                    valid
                ],
                low,
                high,
            )
            -
            low
        )
        /
        (
            high
            -
            low
        )
    ).astype(
        "float32"
    )

    return (
        output,
        low,
        high,
    )


# ============================================================
# ASPECT
# ============================================================

def calculate_aspect_risk(
    dem,
    transform,
):

    output = np.full(
        dem.shape,
        np.nan,
        dtype="float32",
    )

    valid = np.isfinite(
        dem
    )

    if not np.any(
        valid
    ):

        return output

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
        copy=True,
    )

    fill_value = float(
        np.nanmedian(
            safe_dem[
                valid
            ]
        )
    )

    safe_dem[
        ~np.isfinite(
            safe_dem
        )
    ] = fill_value

    dz_dy, dz_dx = np.gradient(
        safe_dem,
        dy,
        dx,
    )

    aspect = (
        np.degrees(
            np.arctan2(
                dz_dx,
                -dz_dy,
            )
        )
        +
        360.0
    ) % 360.0

    good = (
        valid
        &
        np.isfinite(
            aspect
        )
    )

    north = (
        (
            aspect >= 315.0
        )
        |
        (
            aspect < 45.0
        )
    )

    east_west = (
        (
            (
                aspect >= 45.0
            )
            &
            (
                aspect < 135.0
            )
        )
        |
        (
            (
                aspect >= 225.0
            )
            &
            (
                aspect < 315.0
            )
        )
    )

    south = (
        (
            aspect >= 135.0
        )
        &
        (
            aspect < 225.0
        )
    )

    output[
        good & north
    ] = 0.3

    output[
        good & east_west
    ] = 0.6

    output[
        good & south
    ] = 1.0

    return output


# ============================================================
# TOPOGRAPHY
# ============================================================

def build_topography(
    dem,
    slope,
    transform,
    slope_max,
    slope_weight,
    aspect_weight,
):

    slope_norm = normalize_linear(
        slope,
        0.0,
        slope_max,
    )

    aspect_norm = (
        calculate_aspect_risk(
            dem,
            transform,
        )
    )

    topography = (
        slope_weight
        *
        slope_norm
        +
        aspect_weight
        *
        aspect_norm
    )

    topography[
        ~np.isfinite(
            topography
        )
    ] = np.nan

    return topography


# ============================================================
# WEB SIZE
# ============================================================

def compute_web_size(
    width,
    height,
    max_dimension,
):

    largest = max(
        width,
        height,
    )

    if largest <= 0:

        raise ValueError(
            "Invalid raster dimensions."
        )

    scale = min(
        1.0,
        float(
            max_dimension
        )
        /
        float(
            largest
        ),
    )

    return (
        max(
            1,
            int(
                round(
                    width
                    *
                    scale
                )
            )
        ),

        max(
            1,
            int(
                round(
                    height
                    *
                    scale
                )
            )
        ),
    )


# ============================================================
# WEB SMOOTHING
# ============================================================

def box_blur_2d(
    array,
    radius,
):

    radius = min(
        64,
        int(
            round(
                radius
            )
        ),
    )

    if radius <= 0:

        return np.array(
            array,
            dtype="float32",
            copy=True,
        )

    source = np.asarray(
        array,
        dtype="float32",
    )

    padded = np.pad(
        source,
        (
            (
                radius,
                radius,
            ),
            (
                radius,
                radius,
            ),
        ),
        mode="edge",
    )

    cumulative = np.cumsum(
        np.cumsum(
            padded,
            axis=0,
            dtype="float32",
        ),
        axis=1,
        dtype="float32",
    )

    cumulative = np.pad(
        cumulative,
        (
            (
                1,
                0,
            ),
            (
                1,
                0,
            ),
        ),
        mode="constant",
    )

    size = (
        2 * radius
        +
        1
    )

    result = (
        cumulative[
            size:,
            size:,
        ]

        -

        cumulative[
            :-size,
            size:,
        ]

        -

        cumulative[
            size:,
            :-size,
        ]

        +

        cumulative[
            :-size,
            :-size,
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
    radius,
):

    if radius <= 0:

        output = np.array(
            risk,
            dtype="float32",
            copy=True,
        )

        output[
            ~valid_mask
        ] = np.nan

        return output

    valid_float = (
        valid_mask.astype(
            "float32"
        )
    )

    values = np.where(
        valid_mask,
        risk,
        0.0,
    ).astype(
        "float32"
    )

    numerator = box_blur_2d(
        values,
        radius,
    )

    denominator = box_blur_2d(
        valid_float,
        radius,
    )

    output = np.full(
        risk.shape,
        np.nan,
        dtype="float32",
    )

    good = (
        denominator > 0.001
    )

    output[
        good
    ] = (
        numerator[
            good
        ]
        /
        denominator[
            good
        ]
    )

    output = np.clip(
        output,
        0.0,
        100.0,
    )

    # NEVER allow smoothing to create values
    # outside the valid Fars web mask.

    output[
        ~valid_mask
    ] = np.nan

    return output


# ============================================================
# COLOR
# ============================================================

def risk_to_rgb(
    risk,
):

    stops = np.array(
        [
            0.0,
            20.0,
            40.0,
            60.0,
            80.0,
            100.0,
        ],
        dtype="float32",
    )

    colors = np.array(
        [
            [
                255,
                245,
                157,
            ],

            [
                253,
                216,
                53,
            ],

            [
                251,
                140,
                0,
            ],

            [
                229,
                57,
                53,
            ],

            [
                229,
                57,
                53,
            ],

            [
                136,
                14,
                79,
            ],
        ],
        dtype="float32",
    )

    rgb = np.zeros(
        (
            risk.shape[0],
            risk.shape[1],
            3,
        ),
        dtype="uint8",
    )

    valid = np.isfinite(
        risk
    )

    if not np.any(
        valid
    ):

        return (
            rgb,
            valid,
        )

    values = np.clip(
        risk[
            valid
        ],
        0.0,
        100.0,
    )

    rgb[
        valid,
        0
    ] = np.interp(
        values,
        stops,
        colors[:, 0],
    ).astype(
        "uint8"
    )

    rgb[
        valid,
        1
    ] = np.interp(
        values,
        stops,
        colors[:, 1],
    ).astype(
        "uint8"
    )

    rgb[
        valid,
        2
    ] = np.interp(
        values,
        stops,
        colors[:, 2],
    ).astype(
        "uint8"
    )

    return (
        rgb,
        valid,
    )


# ============================================================
# WEB REPROJECTION
# ============================================================

def create_web_png(
    risk,
    master,
    boundary_geometry,
    boundary_crs,
    web_dir,
    max_dimension,
    smoothing_radius,
):

    web_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_width = int(
        master["width"]
    )

    source_height = int(
        master["height"]
    )

    source_transform = (
        master["transform"]
    )

    source_crs = (
        master["crs"]
    )

    # --------------------------------------------------------
    # TRUE SOURCE RASTER BOUNDS
    # --------------------------------------------------------

    source_left, source_bottom, source_right, source_top = (
        array_bounds(
            source_height,
            source_width,
            source_transform,
        )
    )

    print()
    print(
        "WEB REPROJECTION"
    )
    print(
        "----------------"
    )

    print(
        "Source CRS:",
        source_crs,
    )

    print(
        "Source size:",
        source_width,
        "x",
        source_height,
    )

    print(
        "Source bounds:",
        [
            float(source_left),
            float(source_bottom),
            float(source_right),
            float(source_top),
        ],
    )

    # --------------------------------------------------------
    # REAL GEOGRAPHIC BOUNDS OF THE MASTER RASTER
    #
    # Densify is important when source CRS is projected.
    # We do NOT simply transform the four corners.
    # --------------------------------------------------------

    geographic_bounds = (
        transform_bounds(
            source_crs,
            "EPSG:4326",

            source_left,
            source_bottom,
            source_right,
            source_top,

            densify_pts=101,
        )
    )

    geo_left = float(
        geographic_bounds[0]
    )

    geo_bottom = float(
        geographic_bounds[1]
    )

    geo_right = float(
        geographic_bounds[2]
    )

    geo_top = float(
        geographic_bounds[3]
    )

    print(
        "Geographic bounds:",
        [
            geo_left,
            geo_bottom,
            geo_right,
            geo_top,
        ],
    )

    # --------------------------------------------------------
    # DEFAULT WEB TRANSFORM
    #
    # This creates a genuine geographic raster grid.
    # --------------------------------------------------------

    (
        default_web_transform,
        default_web_width,
        default_web_height,
    ) = calculate_default_transform(
        source_crs,
        "EPSG:4326",

        source_width,
        source_height,

        left=source_left,
        bottom=source_bottom,
        right=source_right,
        top=source_top,
    )

    print(
        "Calculated web size:",
        default_web_width,
        "x",
        default_web_height,
    )

    # --------------------------------------------------------
    # LIMIT WEB SIZE
    # --------------------------------------------------------

    web_width, web_height = (
        compute_web_size(
            default_web_width,
            default_web_height,
            max_dimension,
        )
    )

    # --------------------------------------------------------
    # BUILD EXACT AXIS-ALIGNED EPSG:4326 GRID
    #
    # This is the important difference from the previous
    # versions. We are not deforming the image by four
    # control points. We actually resample the raster into
    # a geographic coordinate grid.
    # --------------------------------------------------------

    from affine import Affine

    web_transform = Affine(
        (
            geo_right
            -
            geo_left
        )
        /
        float(
            web_width
        ),

        0.0,

        geo_left,

        0.0,

        -(
            geo_top
            -
            geo_bottom
        )
        /
        float(
            web_height
        ),

        geo_top,
    )

    # --------------------------------------------------------
    # REPROJECT SCIENTIFIC RISK -> WEB EPSG:4326
    #
    # Scientific raster remains untouched.
    # --------------------------------------------------------

    web_risk = np.full(
        (
            web_height,
            web_width,
        ),
        np.nan,
        dtype="float32",
    )

    source_risk = np.where(
        np.isfinite(
            risk
        ),
        risk,
        -9999.0,
    ).astype(
        "float32"
    )

    reproject(
        source=source_risk,
        destination=web_risk,

        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=-9999.0,

        dst_transform=web_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,

        resampling=Resampling.bilinear,
    )

    web_risk[
        ~np.isfinite(
            web_risk
        )
    ] = np.nan

    # --------------------------------------------------------
    # REPROJECT FARS GEOMETRY TO EPSG:4326
    # --------------------------------------------------------

    boundary_web = (
        reproject_geometry(
            boundary_geometry,
            boundary_crs,
            "EPSG:4326",
        )
    )

    # --------------------------------------------------------
    # BUILD EXACT WEB FARS MASK
    #
    # This mask is on the SAME transform and SAME dimensions
    # as the web PNG.
    # --------------------------------------------------------

    web_fars_mask = geometry_mask(
        [
            mapping(
                boundary_web
            )
        ],

        out_shape=(
            web_height,
            web_width,
        ),

        transform=web_transform,

        invert=True,

        all_touched=False,
    )

    # --------------------------------------------------------
    # ABSOLUTE WEB MASK
    # --------------------------------------------------------

    web_risk[
        ~web_fars_mask
    ] = np.nan

    # --------------------------------------------------------
    # WEB-ONLY SMOOTHING
    # --------------------------------------------------------

    web_risk = smooth_web_risk(
        web_risk,
        web_fars_mask,
        smoothing_radius,
    )

    # --------------------------------------------------------
    # FINAL MASK AGAIN
    # --------------------------------------------------------

    web_risk[
        ~web_fars_mask
    ] = np.nan

    # --------------------------------------------------------
    # COLORS
    # --------------------------------------------------------

    rgb, finite = risk_to_rgb(
        web_risk
    )

    # --------------------------------------------------------
    # ALPHA
    #
    # Only cells simultaneously inside the Fars mask and
    # numerically valid are visible.
    # --------------------------------------------------------

    alpha = np.where(
        web_fars_mask
        &
        finite,

        ALPHA_VALUE,

        0,
    ).astype(
        "uint8"
    )

    rgba = np.dstack(
        [
            rgb,
            alpha,
        ]
    )

    image = Image.fromarray(
        rgba,
        mode="RGBA",
    )

    png_path = (
        web_dir
        /
        "fire_risk_latest.png"
    )

    image.save(
        png_path,
        format="PNG",
        optimize=True,
    )

    # --------------------------------------------------------
    # WEB BOUNDS
    #
    # EXACTLY the bounds represented by the web transform.
    # --------------------------------------------------------

    bounds = [
        geo_left,
        geo_bottom,
        geo_right,
        geo_top,
    ]

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    master_crs_string = (
        master["crs"].to_string()
        if hasattr(
            master["crs"],
            "to_string",
        )
        else str(
            master["crs"]
        )
    )

    metadata = {

        "image_size": [
            int(web_width),
            int(web_height),
        ],

        "master_size": [
            int(source_width),
            int(source_height),
        ],

        "bounds":
            bounds,

        "image_url":
            "generated/fire_risk_latest.png",

        "master_crs":
            master_crs_string,

        "alpha":
            ALPHA_VALUE,

        "web": {

            "crs":
                "EPSG:4326",

            "bounds":
                bounds,

            "image_url":
                "generated/fire_risk_latest.png",

            "georeferencing":
                "true raster reprojection",

            "georeferencing_method":
                "reproject master risk grid to EPSG:4326",

            "source_crs":
                master_crs_string,

            "target_crs":
                "EPSG:4326",

            "artificial_control_points":
                False,

            "four_corner_remapping":
                False,

            "boundary_mask":
                "Fars GeoJSON rasterized on web grid",

            "mask_reapplied_after_reprojection":
                True,

            "mask_reapplied_after_smoothing":
                True,

            "outside_fars":
                "transparent",

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
    }

    metadata_path = (
        web_dir
        /
        "fire_risk_latest.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "WEB OUTPUT"
    )
    print(
        "----------"
    )

    print(
        "Web CRS:",
        "EPSG:4326",
    )

    print(
        "Web size:",
        web_width,
        "x",
        web_height,
    )

    print(
        "Web bounds:",
        bounds,
    )

    print(
        "Fars mask applied:",
        True,
    )

    print(
        "Mask reapplied after smoothing:",
        True,
    )

    print(
        "Artificial four-corner remapping:",
        False,
    )

    return (
        png_path,
        metadata_path,
        metadata,
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    risk,
    province_mask,
):

    valid = (
        np.isfinite(
            risk
        )
        &
        province_mask
    )

    if not np.any(
        valid
    ):

        raise ValueError(
            "No valid risk cells inside Fars."
        )

    values = risk[
        valid
    ]

    return {

        "valid_cells":
            int(
                values.size
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values
                )
            ),

        "classes": {

            "very_low":
                int(
                    np.count_nonzero(
                        values < 20.0
                    )
                ),

            "low":
                int(
                    np.count_nonzero(
                        (
                            values >= 20.0
                        )
                        &
                        (
                            values < 40.0
                        )
                    )
                ),

            "moderate":
                int(
                    np.count_nonzero(
                        (
                            values >= 40.0
                        )
                        &
                        (
                            values < 60.0
                        )
                    )
                ),

            "high":
                int(
                    np.count_nonzero(
                        (
                            values >= 60.0
                        )
                        &
                        (
                            values < 80.0
                        )
                    )
                ),

            "critical":
                int(
                    np.count_nonzero(
                        values >= 80.0
                    )
                ),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print()
    print(
        "=" * 70
    )

    print(
        "SMART FARS FORECAST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    required_files = [

        (
            args.fwi,
            "FWI raster",
        ),

        (
            args.master,
            "Master raster",
        ),

        (
            args.dem,
            "DEM raster",
        ),

        (
            args.slope,
            "Slope raster",
        ),

        (
            args.boundary,
            "Fars boundary",
        ),

        (
            args.config,
            "Model configuration",
        ),
    ]

    for path, label in required_files:

        ensure_file(
            path,
            label,
        )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = load_json(
        args.config
    )

    weights = config.get(
        "weights",
        {},
    )

    fwi_weight = float(
        weights.get(
            "fwi",
            0.45,
        )
    )

    fuel_weight = float(
        weights.get(
            "fuel",
            0.35,
        )
    )

    topo_weight = float(
        weights.get(
            "topography",
            0.20,
        )
    )

    if not np.isclose(
        (
            fwi_weight
            +
            fuel_weight
            +
            topo_weight
        ),
        1.0,
        atol=1e-6,
    ):

        raise ValueError(
            "FWI + Fuel + Topography weights "
            "must sum to 1."
        )

    normalization = (
        config.get(
            "normalization",
            {},
        )
    )

    fuel_percentile_low = float(
        normalization.get(
            "fuel_percentile_low",
            1.0,
        )
    )

    fuel_percentile_high = float(
        normalization.get(
            "fuel_percentile_high",
            99.0,
        )
    )

    fwi_min = float(
        normalization.get(
            "fwi_min",
            0.0,
        )
    )

    fwi_max = float(
        normalization.get(
            "fwi_max",
            100.0,
        )
    )

    slope_max = float(
        normalization.get(
            "slope_max",
            45.0,
        )
    )

    topography_config = (
        config.get(
            "topography",
            {},
        )
    )

    slope_weight = float(
        topography_config.get(
            "slope_weight",
            0.80,
        )
    )

    aspect_weight = float(
        topography_config.get(
            "aspect_weight",
            0.20,
        )
    )

    web_config = (
        config.get(
            "web",
            {},
        )
    )

    max_dimension = int(
        web_config.get(
            "max_dimension",
            3000,
        )
    )

    web_smoothing_radius = float(
        web_config.get(
            "smoothing_radius",
            DEFAULT_WEB_SMOOTHING_RADIUS,
        )
    )

    print()
    print(
        "MODEL WEIGHTS"
    )

    print(
        "FWI:",
        fwi_weight,
    )

    print(
        "Fuel:",
        fuel_weight,
    )

    print(
        "Topography:",
        topo_weight,
    )

    print(
        "Web smoothing:",
        web_smoothing_radius,
    )

    # --------------------------------------------------------
    # MASTER
    # --------------------------------------------------------

    print()
    print(
        "LOADING MASTER RASTER"
    )

    print(
        "---------------------"
    )

    master = load_master(
        args.master
    )

    print(
        "Width:",
        master["width"],
    )

    print(
        "Height:",
        master["height"],
    )

    print(
        "CRS:",
        master["crs"],
    )

    print(
        "Transform:",
        master["transform"],
    )

    print(
        "Resolution:",
        abs(
            float(
                master[
                    "transform"
                ].a
            )
        ),
        "x",
        abs(
            float(
                master[
                    "transform"
                ].e
            )
        ),
    )

    # --------------------------------------------------------
    # FARS BOUNDARY
    # --------------------------------------------------------

    print()
    print(
        "BUILDING FARS MASK"
    )

    print(
        "------------------"
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
            master["crs"],
        )
    )

    # EXACT SAME MASTER GRID
    # as the scientific raster.

    province_mask = geometry_mask(
        [
            mapping(
                boundary_master
            )
        ],

        out_shape=(
            master["height"],
            master["width"],
        ),

        transform=master["transform"],

        invert=True,

        all_touched=False,
    )

    province_cells = int(
        np.count_nonzero(
            province_mask
        )
    )

    if province_cells == 0:

        raise ValueError(
            "Fars boundary does not overlap "
            "the master raster."
        )

    print(
        "Province cells:",
        province_cells,
    )

    # --------------------------------------------------------
    # FUEL
    # --------------------------------------------------------

    print()
    print(
        "PREPARING FUEL"
    )

    print(
        "--------------"
    )

    fuel = np.array(
        master["data"],
        dtype="float32",
        copy=True,
    )

    if master["nodata"] is not None:

        fuel[
            np.isclose(
                fuel,
                float(
                    master["nodata"]
                ),
            )
        ] = np.nan

    fuel[
        ~np.isfinite(
            fuel
        )
    ] = np.nan

    fuel_norm, fuel_low, fuel_high = (
        normalize_percentile(
            fuel,
            province_mask,
            fuel_percentile_low,
            fuel_percentile_high,
        )
    )

    fuel_norm[
        ~province_mask
    ] = np.nan

    print(
        "Fuel low:",
        fuel_low,
    )

    print(
        "Fuel high:",
        fuel_high,
    )

    # --------------------------------------------------------
    # DEM
    # --------------------------------------------------------

    print()
    print(
        "PREPARING DEM"
    )

    print(
        "-------------"
    )

    dem = read_to_master_grid(
        args.dem,
        master,
    )

    dem[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------
    # SLOPE
    # --------------------------------------------------------

    print()
    print(
        "PREPARING SLOPE"
    )

    print(
        "---------------"
    )

    slope = read_to_master_grid(
        args.slope,
        master,
    )

    slope[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------
    # TOPOGRAPHY
    # --------------------------------------------------------

    print()
    print(
        "CALCULATING TOPOGRAPHY"
    )

    print(
        "----------------------"
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

    # --------------------------------------------------------
    # FWI
    # --------------------------------------------------------

    print()
    print(
        "PREPARING FWI"
    )

    print(
        "-------------"
    )

    fwi, fwi_fill_stats = (
        prepare_fwi(
            args.fwi,
            master,
        )
    )

    fwi[
        ~province_mask
    ] = np.nan

    fwi_norm = normalize_linear(
        fwi,
        fwi_min,
        fwi_max,
    )

    fwi_norm[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    print()
    print(
        "CALCULATING WILDFIRE RISK"
    )

    print(
        "-------------------------"
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
            master["width"],
        ),
        np.nan,
        dtype="float32",
    )

    risk[
        valid
    ] = (
        100.0
        *
        (
            fwi_weight
            *
            fwi_norm[
                valid
            ]

            +

            fuel_weight
            *
            fuel_norm[
                valid
            ]

            +

            topo_weight
            *
            topography[
                valid
            ]
        )
    )

    risk = np.clip(
        risk,
        0.0,
        100.0,
    )

    # ABSOLUTE SCIENTIFIC MASK

    risk[
        ~province_mask
    ] = np.nan

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics = (
        calculate_statistics(
            risk,
            province_mask,
        )
    )

    print()
    print(
        "FORECAST STATISTICS"
    )

    print(
        "-------------------"
    )

    print(
        "Valid cells:",
        statistics[
            "valid_cells"
        ],
    )

    print(
        "Minimum:",
        f"{statistics['min']:.3f}",
    )

    print(
        "Maximum:",
        f"{statistics['max']:.3f}",
    )

    print(
        "Mean:",
        f"{statistics['mean']:.3f}",
    )

    print(
        "Median:",
        f"{statistics['median']:.3f}",
    )

    print(
        "Std:",
        f"{statistics['std']:.3f}",
    )

    # --------------------------------------------------------
    # SCIENTIFIC GEOTIFF
    # --------------------------------------------------------

    args.output_tif.parent.mkdir(
        parents=True,
        exist_ok=True,
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

        transform=master[
            "transform"
        ],

        crs=master[
            "crs"
        ],
    )

    print()
    print(
        "WRITING SCIENTIFIC GEOTIFF"
    )

    print(
        "--------------------------"
    )

    with rasterio.open(
        args.output_tif,
        "w",
        **profile,
    ) as dst:

        dst.write(
            np.where(
                np.isfinite(
                    risk
                ),
                risk,
                -9999.0,
            ).astype(
                "float32"
            ),
            1,
        )

        dst.set_band_description(
            1,
            "Smart Fars Wildfire Risk (0-100)",
        )

    # --------------------------------------------------------
    # WEB
    # --------------------------------------------------------

    print()
    print(
        "BUILDING WEB GIS"
    )

    print(
        "----------------"
    )

    (
        png_path,
        web_metadata_path,
        web_metadata,
    ) = create_web_png(

        risk=risk,

        master=master,

        boundary_geometry=boundary,

        boundary_crs=boundary_crs,

        web_dir=args.web_dir,

        max_dimension=max_dimension,

        smoothing_radius=(
            web_smoothing_radius
        ),
    )

    # --------------------------------------------------------
    # MASTER BOUNDS
    # --------------------------------------------------------

    master_left, master_bottom, master_right, master_top = (
        array_bounds(
            master["height"],
            master["width"],
            master["transform"],
        )
    )

    master_geographic_bounds = (
        transform_bounds(
            master["crs"],
            "EPSG:4326",

            master_left,
            master_bottom,
            master_right,
            master_top,

            densify_pts=101,
        )
    )

    master_crs_string = (
        master["crs"].to_string()
        if hasattr(
            master["crs"],
            "to_string",
        )
        else str(
            master["crs"]
        )
    )

    # --------------------------------------------------------
    # COMPLETE SCIENTIFIC METADATA
    # --------------------------------------------------------

    metadata = {

        "model":
            "Smart Fars Forecast",

        "risk_range":
            [
                0.0,
                100.0,
            ],

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
                fuel_percentile_low,

            "high_percentile":
                fuel_percentile_high,

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

            "crs":
                master_crs_string,

            "transform":
                list(
                    master["transform"]
                )[:6],

            "resolution": [

                abs(
                    float(
                        master[
                            "transform"
                        ].a
                    )
                ),

                abs(
                    float(
                        master[
                            "transform"
                        ].e
                    )
                ),
            ],

            "geographic_bounds": {

                "west":
                    float(
                        master_geographic_bounds[0]
                    ),

                "south":
                    float(
                        master_geographic_bounds[1]
                    ),

                "east":
                    float(
                        master_geographic_bounds[2]
                    ),

                "north":
                    float(
                        master_geographic_bounds[3]
                    ),
            },
        },

        "boundary": {

            "path":
                str(
                    args.boundary
                ),

            "crs":
                str(
                    boundary_crs
                ),

            "mask_on_master_grid":
                True,

            "pixels_inside_fars":
                province_cells,

            "outside_fars":
                "NoData",
        },

        "statistics":
            statistics,

        "web":
            web_metadata,

        "outputs": {

            "raster":
                str(
                    args.output_tif
                ),

            "raster_metadata":
                str(
                    args.output_tif.with_suffix(
                        ".json"
                    )
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

    # --------------------------------------------------------
    # WRITE SCIENTIFIC METADATA
    # --------------------------------------------------------

    metadata_path = (
        args.output_tif.with_suffix(
            ".json"
        )
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

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

    print()

    print(
        "Scientific raster:",
        args.output_tif,
    )

    print(
        "Scientific metadata:",
        metadata_path,
    )

    print(
        "Web PNG:",
        png_path,
    )

    print(
        "Web metadata:",
        web_metadata_path,
    )

    print()

    print(
        "Scientific master grid preserved:",
        True,
    )

    print(
        "Scientific raster modified by web reprojection:",
        False,
    )

    print(
        "Web raster reprojected to EPSG:4326:",
        True,
    )

    print(
        "Fars mask applied to web raster:",
        True,
    )

    print(
        "Fars mask reapplied after smoothing:",
        True,
    )

    print(
        "Artificial four-corner remapping:",
        False,
    )

    print(
        "Outside Fars:",
        "Transparent",
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )
