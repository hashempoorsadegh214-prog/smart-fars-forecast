#!/usr/bin/env python3

"""
Build continuous wildfire risk for Fars Province.

Inputs:
    fars.geojson
    dem_fars.tif
    fars_slope_60m_light.tif
    fars_fire_fuel_hazard_60m.tif
    data/fwi/fwi_latest.tif
    config/model_config.json

Outputs:
    data/output/fire_risk_latest.tif
    data/output/fire_risk_latest.json
    web/generated/fire_risk_latest.png
    web/generated/fire_risk_latest.json
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

MASTER_PATH = ROOT / "fars_fire_fuel_hazard_60m.tif"
DEM_PATH = ROOT / "dem_fars.tif"
SLOPE_PATH = ROOT / "fars_slope_60m_light.tif"
BOUNDARY_PATH = ROOT / "fars.geojson"


def load_boundary(path: Path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("type") == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in data.get("features", [])
            if feature.get("geometry")
        ]

        if not geometries:
            raise ValueError("No geometries found in boundary.")

        return unary_union(geometries)

    if data.get("type") == "Feature":
        return shape(data["geometry"])

    return shape(data)


def transform_boundary(geometry, destination_crs):
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


def read_to_master_grid(
    path: Path,
    master: rasterio.DatasetReader,
    resampling: Resampling,
) -> np.ndarray:
    output = np.full(
        (master.height, master.width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(path) as source:
        if source.crs is None:
            raise RuntimeError(
                f"{path.name} has no CRS."
            )

        reproject(
            source=rasterio.band(source, 1),
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


def normalize_by_percentile(
    values: np.ndarray,
    valid: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    selected = values[valid]

    if selected.size == 0:
        raise RuntimeError(
            "No valid values for normalization."
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
        low = float(np.nanmin(selected))
        high = float(np.nanmax(selected))

    if high <= low:
        result[valid] = 0.0
        return result

    clipped = np.clip(
        values,
        low,
        high,
    )

    result[valid] = (
        (clipped[valid] - low)
        / (high - low)
    ).astype(np.float32)

    return result


def calculate_aspect_risk(
    dem: np.ndarray,
    x_resolution: float,
    y_resolution: float,
) -> np.ndarray:
    row_gradient, col_gradient = np.gradient(
        dem.astype(np.float32),
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
        | (aspect < 22.5)
    )

    south = (
        (aspect >= 112.5)
        & (aspect < 247.5)
    )

    east_west = ~(north | south)

    result[north] = 0.30
    result[east_west] = 0.60
    result[south] = 1.00

    return result


def create_web_gradient(
    risk: np.ndarray,
    province_mask: np.ndarray,
    output_path: Path,
    max_dimension: int,
) -> None:
    """
    Create a smoother web preview.

    Important:
        This is ONLY for visualization.
        The original 60 m GeoTIFF values are not changed.

    The province mask is applied again after resizing so that
    colored pixels cannot leak outside the Fars boundary.
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
        dtype=np.uint8,
    )

    valid = (
        province_mask
        & np.isfinite(risk)
    )

    for index in range(
        len(anchors) - 1
    ):
        low_value = anchors[
            index,
            0,
        ]

        high_value = anchors[
            index + 1,
            0,
        ]

        selected = (
            valid
            & (values >= low_value)
            & (values <= high_value)
        )

        if not np.any(selected):
            continue

        ratio = (
            values[selected] - low_value
        ) / (
            high_value - low_value
        )

        r1 = anchors[index, 1]
        g1 = anchors[index, 2]
        b1 = anchors[index, 3]

        r2 = anchors[index + 1, 1]
        g2 = anchors[index + 1, 2]
        b2 = anchors[index + 1, 3]

        rgb[..., 0][selected] = (
            r1 + ratio * (r2 - r1)
        ).astype(np.uint8)

        rgb[..., 1][selected] = (
            g1 + ratio * (g2 - g1)
        ).astype(np.uint8)

        rgb[..., 2][selected] = (
            b1 + ratio * (b2 - b1)
        ).astype(np.uint8)

    source_image = Image.fromarray(
        rgb,
        mode="RGB",
    )

    mask_array = (
        province_mask.astype(np.uint8) * 255
    )

    source_mask = Image.fromarray(
        mask_array,
        mode="L",
    )

    scale = min(
        1.0,
        max_dimension / max(
            source_image.size
        ),
    )

    if scale < 1.0:
        target_size = (
            max(
                1,
                int(
                    source_image.width
                    * scale
                ),
            ),
            max(
                1,
                int(
                    source_image.height
                    * scale
                ),
            ),
        )
    else:
        target_size = source_image.size

    # Smooth only the WEB visualization.
    # This does not modify the 60 m scientific raster.
    web_rgb = source_image.resize(
        target_size,
        Image.Resampling.BICUBIC,
    )

    # Resize the province mask separately.
    # The mask is thresholded after resizing to prevent
    # color leakage outside the province.
    web_mask = source_mask.resize(
        target_size,
        Image.Resampling.BILINEAR,
    )

    alpha_array = np.asarray(
        web_mask,
        dtype=np.uint8,
    )

    alpha_array = np.where(
        alpha_array >= 128,
        235,
        0,
    ).astype(np.uint8)

    rgba = np.dstack(
        (
            np.asarray(
                web_rgb,
                dtype=np.uint8,
            ),
            alpha_array,
        )
    )

    final_image = Image.fromarray(
        rgba,
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


def get_fwi_date(
    fwi_path: Path,
) -> str | None:
    metadata_path = (
        fwi_path.with_suffix(".json")
    )

    if not metadata_path.exists():
        return None

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return (
        data.get("target_date")
        or data.get("forecast_date")
    )


def get_web_bounds(
    master: rasterio.DatasetReader,
) -> list[float]:
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
        default="data/output/fire_risk_latest.tif",
    )

    parser.add_argument(
        "--web-dir",
        default="web/generated",
    )

    args = parser.parse_args()

    fwi_path = ROOT / args.fwi
    config_path = ROOT / args.config
    output_path = ROOT / args.output_tif
    web_dir = ROOT / args.web_dir

    if not fwi_path.exists():
        raise FileNotFoundError(
            f"FWI file not found: {fwi_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    weights = config["weights"]

    fwi_weight = float(
        weights["fwi"]
    )

    fuel_weight = float(
        weights["fuel"]
    )

    topography_weight = float(
        weights["topography"]
    )

    if abs(
        (
            fwi_weight
            + fuel_weight
            + topography_weight
        )
        - 1.0
    ) > 1e-6:
        raise ValueError(
            "WLC weights must sum to 1.0."
        )

    fuel_low = float(
        config["normalization"][
            "fuel_percentile_low"
        ]
    )

    fuel_high = float(
        config["normalization"][
            "fuel_percentile_high"
        ]
    )

    fwi_min = float(
        config["normalization"][
            "fwi_min"
        ]
    )

    fwi_max = float(
        config["normalization"][
            "fwi_max"
        ]
    )

    slope_max = float(
        config["normalization"][
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

    with rasterio.open(
        MASTER_PATH
    ) as master:

        if master.crs is None:
            raise RuntimeError(
                "Master raster has no CRS."
            )

        master_shape = (
            master.height,
            master.width,
        )

        boundary = transform_boundary(
            load_boundary(
                BOUNDARY_PATH
            ),
            master.crs,
        )

        province_mask = geometry_mask(
            [boundary],
            out_shape=master_shape,
            transform=master.transform,
            invert=True,
        )

        fuel = master.read(
            1
        ).astype(np.float32)

        if master.nodata is not None:
            fuel_valid = (
                fuel != master.nodata
            )
        else:
            fuel_valid = (
                np.isfinite(fuel)
            )

        fuel_valid &= np.isfinite(
            fuel
        )

        fuel_valid &= province_mask

        fuel_norm = normalize_by_percentile(
            fuel,
            fuel_valid,
            fuel_low,
            fuel_high,
        )

        dem = read_to_master_grid(
            DEM_PATH,
            master,
            Resampling.bilinear,
        )

        slope = read_to_master_grid(
            SLOPE_PATH,
            master,
            Resampling.bilinear,
        )

        fwi = read_to_master_grid(
            fwi_path,
            master,
            Resampling.bilinear,
        )

        valid = (
            province_mask
            & np.isfinite(fuel_norm)
            & np.isfinite(dem)
            & np.isfinite(slope)
            & np.isfinite(fwi)
        )

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
            / slope_max
        )

        aspect_norm = calculate_aspect_risk(
            dem,
            master.res[0],
            master.res[1],
        )

        topography = np.full(
            master_shape,
            np.nan,
            dtype=np.float32,
        )

        topography_valid = (
            valid
            & np.isfinite(slope_norm)
            & np.isfinite(aspect_norm)
        )

        topography[
            topography_valid
        ] = (
            0.80
            * slope_norm[
                topography_valid
            ]
            +
            0.20
            * aspect_norm[
                topography_valid
            ]
        )

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
                - fwi_min
            )
            /
            (
                fwi_max
                - fwi_min
            )
        )

        risk = np.full(
            master_shape,
            np.nan,
            dtype=np.float32,
        )

        risk[valid] = (
            100.0
            * (
                fwi_weight
                * fwi_norm[valid]
                +
                fuel_weight
                * fuel_norm[valid]
                +
                topography_weight
                * topography[valid]
            )
        )

        risk[valid] = np.clip(
            risk[valid],
            0.0,
            100.0,
        )

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
        ).astype(np.float32)

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

        valid_values = risk[valid]

        metadata = {
            "model": "Smart Fars Forecast",
            "fwi_source": (
                "Copernicus GWIS / ECMWF"
            ),
            "fwi_target_date": (
                get_fwi_date(fwi_path)
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

    web_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    web_png = (
        web_dir
        / "fire_risk_latest.png"
    )

    web_json = (
        web_dir
        / "fire_risk_latest.json"
    )

    create_web_gradient(
        risk=risk,
        province_mask=province_mask,
        output_path=web_png,
        max_dimension=int(
            config["web"][
                "max_dimension"
            ]
        ),
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
        "Saved raster:",
        output_path,
    )

    print(
        "Saved web image:",
        web_png,
    )

    print(
        "Valid pixels:",
        valid_values.size,
    )


if __name__ == "__main__":
    main()
