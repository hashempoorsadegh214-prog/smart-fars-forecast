
#!/usr/bin/env python3

"""
Smart Fars Forecast
Build continuous wildfire risk for Fars Province.

INPUTS
------
fars.geojson
dem_fars.tif
fars_slope_60m_light.tif
fars_fire_fuel_hazard_60m.tif
data/fwi/fwi_latest.tif
config/model_config.json

OUTPUTS
-------
data/output/fire_risk_latest.tif
data/output/fire_risk_latest.json

web/generated/fire_risk_latest.png
web/generated/fire_risk_latest.json

IMPORTANT
---------
The 60 m scientific raster is never smoothed or altered for Web display.

Only the Web PNG is interpolated for visualization.
The Fars boundary is applied again to the Web image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT / "fars_fire_fuel_hazard_60m.tif"
)

DEM_PATH = (
    ROOT / "dem_fars.tif"
)

SLOPE_PATH = (
    ROOT / "fars_slope_60m_light.tif"
)

BOUNDARY_PATH = (
    ROOT / "fars.geojson"
)


# ============================================================
# BOUNDARY
# ============================================================

def load_boundary(
    path: Path,
):
    """
    Load Fars boundary from GeoJSON.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    geojson_type = data.get("type")

    if geojson_type == "FeatureCollection":

        geometries = [
            shape(feature["geometry"])
            for feature in data.get(
                "features",
                [],
            )
            if feature.get("geometry")
        ]

        if not geometries:
            raise ValueError(
                "No geometries found in fars.geojson."
            )

        geometry = unary_union(
            geometries
        )

    elif geojson_type == "Feature":

        geometry = shape(
            data["geometry"]
        )

    else:

        geometry = shape(
            data
        )

    if geometry.is_empty:
        raise ValueError(
            "Fars boundary geometry is empty."
        )

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    return geometry


def transform_boundary(
    geometry,
    destination_crs,
):
    """
    Transform boundary from WGS84 to the Master Raster CRS.
    """

    source_crs = "EPSG:4326"

    if str(destination_crs) == source_crs:
        return geometry

    transformer = Transformer.from_crs(
        source_crs,
        destination_crs,
        always_xy=True,
    )

    return shapely_transform(
        transformer.transform,
        geometry,
    )


# ============================================================
# RASTER ALIGNMENT
# ============================================================

def read_to_master_grid(
    path: Path,
    master: rasterio.DatasetReader,
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject/resample one raster directly onto the Master Grid.
    """

    output = np.full(
        (
            master.height,
            master.width,
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(
        path
    ) as source:

        if source.crs is None:
            raise RuntimeError(
                f"{path.name} has no CRS."
            )

        reproject(
            source=rasterio.band(
                source,
                1,
            ),
            destination=output,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=master.transform,
            dst_crs=master.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    return output


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_by_percentile(
    values: np.ndarray,
    valid: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """
    Percentile normalization to 0-1.
    """

    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    selected = values[
        valid
    ]

    if selected.size == 0:
        raise RuntimeError(
            "No valid values available for normalization."
        )

    low = float(
        np.nanpercentile(
            selected,
            low_percentile,
        )
    )

    high = float(
        np.nanpercentile(
            selected,
            high_percentile,
        )
    )

    if (
        not math.isfinite(low)
        or not math.isfinite(high)
    ):
        raise RuntimeError(
            "Invalid percentile values."
        )

    if high <= low:

        low = float(
            np.nanmin(selected)
        )

        high = float(
            np.nanmax(selected)
        )

    if high <= low:

        result[valid] = 0.0

        return result

    clipped = np.clip(
        values,
        low,
        high,
    )

    result[valid] = (
        (
            clipped[valid]
            - low
        )
        /
        (
            high
            - low
        )
    ).astype(
        np.float32
    )

    return result


# ============================================================
# ASPECT
# ============================================================

def calculate_aspect_risk(
    dem: np.ndarray,
    x_resolution: float,
    y_resolution: float,
) -> np.ndarray:
    """
    Convert DEM aspect into the specified risk weighting.

    North      = 0.30
    East/West  = 0.60
    South      = 1.00
    """

    row_gradient, col_gradient = np.gradient(
        dem.astype(
            np.float32
        ),
        abs(y_resolution),
        abs(x_resolution),
    )

    dz_dx = col_gradient

    dz_dnorth = -row_gradient

    aspect = (
        np.degrees(
            np.arctan2(
                dz_dx,
                dz_dnorth,
            )
        )
        + 360.0
    ) % 360.0

    result = np.full(
        aspect.shape,
        np.nan,
        dtype=np.float32,
    )

    north = (
        (aspect >= 337.5)
        |
        (aspect < 22.5)
    )

    south = (
        (aspect >= 112.5)
        &
        (aspect < 247.5)
    )

    east_west = ~(
        north
        |
        south
    )

    result[north] = 0.30

    result[east_west] = 0.60

    result[south] = 1.00

    return result


# ============================================================
# WEB VISUALIZATION
# ============================================================

def colorize_risk(
    risk: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Convert continuous 0-100 risk values to RGB.

    This function creates a continuous gradient.
    """

    anchors = np.array(
        [
            [0.0, 46.0, 125.0, 50.0],
            [20.0, 156.0, 204.0, 101.0],
            [40.0, 253.0, 216.0, 53.0],
            [60.0, 245.0, 124.0, 0.0],
            [80.0, 183.0, 28.0, 28.0],
            [100.0, 136.0, 14.0, 14.0],
        ],
        dtype=np.float32,
    )

    values = np.clip(
        risk,
        0.0,
        100.0,
    )

    rgb = np.zeros(
        (
            risk.shape[0],
            risk.shape[1],
            3,
        ),
        dtype=np.float32,
    )

    for index in range(
        len(anchors) - 1
    ):

        low = anchors[
            index,
            0,
        ]

        high = anchors[
            index + 1,
            0,
        ]

        selected = (
            valid
            &
            (values >= low)
            &
            (values <= high)
        )

        if not np.any(
            selected
        ):
            continue

        ratio = (
            values[selected]
            - low
        ) / (
            high
            - low
        )

        rgb[selected, 0] = (
            anchors[index, 1]
            +
            ratio
            *
            (
                anchors[
                    index + 1,
                    1,
                ]
                -
                anchors[
                    index,
                    1,
                ]
            )
        )

        rgb[selected, 1] = (
            anchors[index, 2]
            +
            ratio
            *
            (
                anchors[
                    index + 1,
                    2,
                ]
                -
                anchors[
                    index,
                    2,
                ]
            )
        )

        rgb[selected, 2] = (
            anchors[index, 3]
            +
            ratio
            *
            (
                anchors[
                    index + 1,
                    3,
                ]
                -
                anchors[
                    index,
                    3,
                ]
            )
        )

    return np.clip(
        rgb,
        0,
        255,
    ).astype(
        np.uint8
    )


def create_web_gradient(
    risk: np.ndarray,
    province_mask: np.ndarray,
    output_path: Path,
    max_dimension: int,
) -> None:
    """
    Create a high-resolution smooth Web visualization.

    Scientific raster:
        untouched

    Web image:
        interpolated only for display

    Boundary:
        reapplied after interpolation
    """

    valid = (
        province_mask
        &
        np.isfinite(risk)
    )

    if not np.any(
        valid
    ):
        raise RuntimeError(
            "No valid pixels available for Web visualization."
        )

    rgb = colorize_risk(
        risk,
        valid,
    )

    # --------------------------------------------------------
    # Make invalid pixels explicitly transparent.
    # --------------------------------------------------------

    alpha = np.where(
        valid,
        235,
        0,
    ).astype(
        np.uint8
    )

    rgba = np.dstack(
        (
            rgb,
            alpha,
        )
    )

    source_image = Image.fromarray(
        rgba,
        mode="RGBA",
    )

    source_mask = Image.fromarray(
        (
            province_mask.astype(
                np.uint8
            )
            * 255
        ),
        mode="L",
    )

    # --------------------------------------------------------
    # Keep a higher Web resolution.
    # This is only visualization.
    # --------------------------------------------------------

    original_width = source_image.width
    original_height = source_image.height

    scale = min(
        1.0,
        max_dimension
        /
        max(
            original_width,
            original_height,
        ),
    )

    if scale < 1.0:

        target_size = (
            max(
                1,
                int(
                    original_width
                    * scale
                ),
            ),
            max(
                1,
                int(
                    original_height
                    * scale
                ),
            ),
        )

    else:

        target_size = (
            original_width,
            original_height,
        )

    # --------------------------------------------------------
    # Interpolate color separately.
    # --------------------------------------------------------

    web_rgb = source_image.convert(
        "RGB"
    ).resize(
        target_size,
        Image.Resampling.BICUBIC,
    )

    # --------------------------------------------------------
    # Interpolate boundary mask separately.
    # --------------------------------------------------------

    web_mask = source_mask.resize(
        target_size,
        Image.Resampling.BILINEAR,
    )

    mask_array = np.asarray(
        web_mask,
        dtype=np.uint8,
    )

    # Strict threshold.
    # Pixels below this value become fully transparent.
    web_alpha = np.where(
        mask_array >= 180,
        235,
        0,
    ).astype(
        np.uint8
    )

    web_rgb_array = np.asarray(
        web_rgb,
        dtype=np.uint8,
    )

    web_rgba = np.dstack(
        (
            web_rgb_array,
            web_alpha,
        )
    )

    final_image = Image.fromarray(
        web_rgba,
        mode="RGBA",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_image.save(
        output_path,
        format="PNG",
        optimize=True,
    )


# ============================================================
# FWI DATE
# ============================================================

def get_fwi_date(
    fwi_path: Path,
) -> str | None:

    metadata_path = (
        fwi_path.with_suffix(
            ".json"
        )
    )

    if not metadata_path.exists():
        return None

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    return (
        data.get(
            "target_date"
        )
        or
        data.get(
            "forecast_date"
        )
    )


# ============================================================
# WEB BOUNDS
# ============================================================

def get_web_bounds(
    master: rasterio.DatasetReader,
) -> list[float]:
    """
    Convert Master Raster bounds to WGS84 for Leaflet.
    """

    transformer = Transformer.from_crs(
        master.crs,
        "EPSG:4326",
        always_xy=True,
    )

    left, bottom = transformer.transform(
        master.bounds.left,
        master.bounds.bottom,
    )

    right, top = transformer.transform(
        master.bounds.right,
        master.bounds.top,
    )

    return [
        float(left),
        float(bottom),
        float(right),
        float(top),
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fwi",
        default="data/fwi/fwi_latest.tif",
    )

    parser.add_argument(
        "--config",
        default="config/model_config.json",
    )

    parser.add_argument(
        "--output-tif",
        default=(
            "data/output/"
            "fire_risk_latest.tif"
        ),
    )

    parser.add_argument(
        "--web-dir",
        default="web/generated",
    )

    args = parser.parse_args()

    fwi_path = (
        ROOT
        /
        args.fwi
    )

    config_path = (
        ROOT
        /
        args.config
    )

    output_path = (
        ROOT
        /
        args.output_tif
    )

    web_dir = (
        ROOT
        /
        args.web_dir
    )

    if not fwi_path.exists():
        raise FileNotFoundError(
            f"FWI file not found: {fwi_path}"
        )

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master Raster not found: {MASTER_PATH}"
        )

    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"DEM not found: {DEM_PATH}"
        )

    if not SLOPE_PATH.exists():
        raise FileNotFoundError(
            f"Slope raster not found: {SLOPE_PATH}"
        )

    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"Boundary not found: {BOUNDARY_PATH}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    # --------------------------------------------------------
    # WLC weights
    # --------------------------------------------------------

    weights = config[
        "weights"
    ]

    fwi_weight = float(
        weights["fwi"]
    )

    fuel_weight = float(
        weights["fuel"]
    )

    topography_weight = float(
        weights["topography"]
    )

    total_weight = (
        fwi_weight
        +
        fuel_weight
        +
        topography_weight
    )

    if abs(
        total_weight - 1.0
    ) > 1e-6:

        raise ValueError(
            "WLC weights must sum to 1.0."
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    normalization = config[
        "normalization"
    ]

    fuel_low = float(
        normalization[
            "fuel_percentile_low"
        ]
    )

    fuel_high = float(
        normalization[
            "fuel_percentile_high"
        ]
    )

    fwi_min = float(
        normalization[
            "fwi_min"
        ]
    )

    fwi_max = float(
        normalization[
            "fwi_max"
        ]
    )

    slope_max = float(
        normalization[
            "slope_max"
        ]
    )

    if fwi_max <= fwi_min:
        raise ValueError(
            "FWI max must be greater than FWI min."
        )

    if slope_max <= 0:
        raise ValueError(
            "Slope maximum must be greater than zero."
        )

    # --------------------------------------------------------
    # Master Raster
    # --------------------------------------------------------

    with rasterio.open(
        MASTER_PATH
    ) as master:

        if master.crs is None:
            raise RuntimeError(
                "Master Raster has no CRS."
            )

        master_shape = (
            master.height,
            master.width,
        )

        # ----------------------------------------------------
        # Fars boundary in Master CRS
        # ----------------------------------------------------

        fars_boundary = (
            load_boundary(
                BOUNDARY_PATH
            )
        )

        fars_boundary_master = (
            transform_boundary(
                fars_boundary,
                master.crs,
            )
        )

        # ----------------------------------------------------
        # STRICT MASTER GRID MASK
        # ----------------------------------------------------

        province_mask = geometry_mask(
            [fars_boundary_master],
            out_shape=master_shape,
            transform=master.transform,
            invert=True,
            all_touched=False,
        )

        # ----------------------------------------------------
        # Fuel = Master Raster
        # ----------------------------------------------------

        fuel = master.read(
            1
        ).astype(
            np.float32
        )

        if master.nodata is not None:

            fuel_valid = (
                fuel
                !=
                master.nodata
            )

        else:

            fuel_valid = (
                np.isfinite(
                    fuel
                )
            )

        fuel_valid &= np.isfinite(
            fuel
        )

        fuel_valid &= province_mask

        # ----------------------------------------------------
        # Fuel normalization
        # ----------------------------------------------------

        fuel_norm = (
            normalize_by_percentile(
                fuel,
                fuel_valid,
                fuel_low,
                fuel_high,
            )
        )

        # ----------------------------------------------------
        # DEM
        # ----------------------------------------------------

        dem = read_to_master_grid(
            DEM_PATH,
            master,
            Resampling.bilinear,
        )

        # ----------------------------------------------------
        # Slope
        # ----------------------------------------------------

        slope = read_to_master_grid(
            SLOPE_PATH,
            master,
            Resampling.bilinear,
        )

        # ----------------------------------------------------
        # FWI
        # ----------------------------------------------------

        fwi = read_to_master_grid(
            fwi_path,
            master,
            Resampling.bilinear,
        )

        # ----------------------------------------------------
        # Final valid mask
        # ----------------------------------------------------

        valid = (
            province_mask
            &
            np.isfinite(
                fuel_norm
            )
            &
            np.isfinite(
                dem
            )
            &
            np.isfinite(
                slope
            )
            &
            np.isfinite(
                fwi
            )
        )

        # ----------------------------------------------------
        # Slope normalization
        # ----------------------------------------------------

        slope_clipped = np.clip(
            slope,
            0.0,
            slope_max,
        )

        slope_norm = np.full(
            master_shape,
            np.nan,
            dtype=np.float32,
        )

        slope_norm[valid] = (
            slope_clipped[valid]
            /
            slope_max
        )

        # ----------------------------------------------------
        # Aspect
        # ----------------------------------------------------

        aspect_norm = (
            calculate_aspect_risk(
                dem,
                master.res[0],
                master.res[1],
            )
        )

        # ----------------------------------------------------
        # Topography
        # ----------------------------------------------------

        topography = np.full(
            master_shape,
            np.nan,
            dtype=np.float32,
        )

        topography_valid = (
            valid
            &
            np.isfinite(
                slope_norm
            )
            &
            np.isfinite(
                aspect_norm
            )
        )

        topography[
            topography_valid
        ] = (
            0.80
            *
            slope_norm[
                topography_valid
            ]
            +
            0.20
            *
            aspect_norm[
                topography_valid
            ]
        )

        # ----------------------------------------------------
        # FWI normalization
        # ----------------------------------------------------

        fwi_clipped = np.clip(
            fwi,
            fwi_min,
            fwi_max,
        )

        fwi_norm = np.full(
            master_shape,
            np.nan,
            dtype=np.float32,
        )

        fwi_norm[valid] = (
            (
                fwi_clipped[valid]
                -
                fwi_min
            )
            /
            (
                fwi_max
                -
                fwi_min
            )
        )

        # ----------------------------------------------------
        # WLC
        # ----------------------------------------------------

        risk = np.full(
            master_shape,
            np.nan,
            dtype=np.float32,
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

        risk[valid] = np.clip(
            risk[valid],
            0.0,
            100.0,
        )

        # ----------------------------------------------------
        # Scientific Output
        # ----------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        profile = master.profile.copy()

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        output_array = np.where(
            valid,
            risk,
            -9999.0,
        ).astype(
            np.float32
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as destination:

            destination.write(
                output_array,
                1,
            )

            destination.set_band_description(
                1,
                "Continuous wildfire risk 0-100",
            )

        valid_values = risk[
            valid
        ]

        if valid_values.size == 0:
            raise RuntimeError(
                "Final risk raster contains no valid pixels."
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "model": "Smart Fars Forecast",
            "fwi_source": (
                "Copernicus GWIS / ECMWF"
            ),
            "fwi_target_date": (
                get_fwi_date(
                    fwi_path
                )
            ),
            "master_raster": (
                MASTER_PATH.name
            ),
            "boundary": (
                BOUNDARY_PATH.name
            ),
            "grid": {
                "width": int(
                    master.width
                ),
                "height": int(
                    master.height
                ),
                "resolution_x": float(
                    master.res[0]
                ),
                "resolution_y": float(
                    master.res[1]
                ),
                "crs": str(
                    master.crs
                ),
                "bounds": [
                    float(
                        master.bounds.left
                    ),
                    float(
                        master.bounds.bottom
                    ),
                    float(
                        master.bounds.right
                    ),
                    float(
                        master.bounds.top
                    ),
                ],
            },
            "web": {
                "crs": "EPSG:4326",
                "bounds": get_web_bounds(
                    master
                ),
            },
            "weights": {
                "fwi": fwi_weight,
                "fuel": fuel_weight,
                "topography": (
                    topography_weight
                ),
            },
            "topography": {
                "slope_weight": 0.80,
                "aspect_weight": 0.20,
                "north": 0.30,
                "east_west": 0.60,
                "south": 1.00,
            },
            "statistics": {
                "min": float(
                    np.nanmin(
                        valid_values
                    )
                ),
                "max": float(
                    np.nanmax(
                        valid_values
                    )
                ),
                "mean": float(
                    np.nanmean(
                        valid_values
                    )
                ),
                "valid_pixels": int(
                    valid_values.size
                ),
            },
        }

    # --------------------------------------------------------
    # Scientific metadata
    # --------------------------------------------------------

    raster_metadata = (
        output_path.with_suffix(
            ".json"
        )
    )

    with raster_metadata.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Web outputs
    # --------------------------------------------------------

    web_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    web_png = (
        web_dir
        /
        "fire_risk_latest.png"
    )

    web_json = (
        web_dir
        /
        "fire_risk_latest.json"
    )

    web_config = config.get(
        "web",
        {},
    )

    max_dimension = int(
        web_config.get(
            "max_dimension",
            3000,
        )
    )

    create_web_gradient(
        risk=risk,
        province_mask=province_mask,
        output_path=web_png,
        max_dimension=max_dimension,
    )

    web_metadata = dict(
        metadata
    )

    web_metadata["image"] = {
        "file": (
            "fire_risk_latest.png"
        ),
        "display": (
            "continuous gradient"
        ),
        "interpolation": (
            "bicubic"
        ),
        "boundary_mask": True,
    }

    with web_json.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            web_metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "========================================"
    )

    print(
        "Smart Fars Forecast completed."
    )

    print(
        "Scientific raster:",
        output_path,
    )

    print(
        "Web image:",
        web_png,
    )

    print(
        "Valid pixels:",
        valid_values.size,
    )

    print(
        "Risk min:",
        float(
            np.nanmin(
                valid_values
            )
        ),
    )

    print(
        "Risk max:",
        float(
            np.nanmax(
                valid_values
            )
        ),
    )

    print(
        "Risk mean:",
        float(
            np.nanmean(
                valid_values
            )
        ),
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
