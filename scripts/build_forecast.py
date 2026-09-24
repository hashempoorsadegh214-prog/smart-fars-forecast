#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_forecast.py — Build Fars province risk forecast raster + web-ready PNG + metadata.

CORRECTED VERSION:
- create_web_gradient(): target RGB is interpolated (visualized) only, while the
  province boundary alpha is rasterized DIRECTLY from the original Fars geometry
  into the FINAL PNG dimensions using the correct geospatial affine transform and
  CRS (no RGBA/RGB resize that bleeds invalid colors into the alpha channel).
- get_web_bounds(): replaces the two-corner transform with a dense sampling of all
  four outer edges of the raster bounds, transformed to EPSG:4326 (always_xy=True),
  rejecting non-finite results.
- OUT_DIR is portable: defaults to <project_root>/generated (project root derived
  from this script's location; intended layout: <project_root>/scripts/build_forecast.py),
  overridable via the FARS_FORECAST_OUT_DIR environment variable.

Requires: rasterio, shapely, numpy, geopandas (optional fallback), pyproj, PIL.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MAX_DIMENSION = 2048          # largest PNG dimension (existing scale rule)
ALPHA_PROVINCE = 235          # alpha value inside province & valid risk


def _resolve_project_root():
    """
    Robust project root: parent of the directory containing this script
    (works when the script lives in <repo>/scripts/), with fallbacks.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists() or \
           (candidate / "scripts" / here.name).exists():
            return candidate
    if here.parent.name == "scripts" and here.parent.parent != here.parent:
        return here.parent.parent
    return here.parent


PROJECT_ROOT = _resolve_project_root()


def _resolve_out_dir():
    """
    Output directory: FARS_FORECAST_OUT_DIR overrides the default
    <project_root>/generated. Relative override values are resolved against the
    project root; `~` in the override is expanded to the user's home.
    """
    override = os.environ.get("FARS_FORECAST_OUT_DIR")
    if override:
        path = Path(os.path.expanduser(override))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())
    return str((PROJECT_ROOT / "generated").resolve())


OUT_DIR = _resolve_out_dir()
PNG_PATH = os.path.join(OUT_DIR, "fars_forecast_web.png")
RASTER_PATH = os.path.join(OUT_DIR, "fars_forecast_raster.tif")
METADATA_PATH = os.path.join(OUT_DIR, "forecast_metadata.json")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_fars_geometry(fars_boundary_master):
    """Return (geometry, crs) from the master boundary dataset/GeoDataFrame."""
    crs = fars_boundary_master.crs
    if crs is None:
        raise ValueError("Fars boundary has no CRS; cannot rasterize correctly.")
    if hasattr(fars_boundary_master, "geometry"):
        geoms = [g for g in fars_boundary_master.geometry if g is not None]
        geom = geoms[0]
        if len(geoms) > 1:
            from shapely.ops import unary_union
            geom = unary_union(geoms)
        return geom, crs
    gi = getattr(fars_boundary_master, "__geo_interface__", None)
    if gi is not None:
        return shape(gi), crs
    raise TypeError("Unsupported boundary object: %r" % type(fars_boundary_master))

def compute_target_size(master_width, master_height):
    """Existing scale rule: longest side becomes MAX_DIMENSION."""
    scale = MAX_DIMENSION / float(max(master_width, master_height))
    target_width = max(1, int(round(master_width * scale)))
    target_height = max(1, int(round(master_height * scale)))
    return target_width, target_height

def get_web_bounds(raster_path_or_ds, master_transform, master_width, master_height, master_crs):
    """
    Robust web bounds: densely sample ALL FOUR outer edges of the raster extent
    (pixel corner edges of the affine grid), transform each sample from the
    master CRS to EPSG:4326 (always_xy=True), and return min/max lon/lat.
    Non-finite transformed values are rejected.
    """
    if master_crs is None:
        raise ValueError("master_crs is required for web bounds.")

    samples_per_edge = 200
    pts = []
    # Sample 4 outer edges: top, bottom, left, right
    for i in range(samples_per_edge + 1):
        t = i / float(samples_per_edge)
        pts.append((t * master_width, 0.0))
        pts.append((t * master_width, float(master_height)))
        pts.append((0.0, t * master_height))
        pts.append((float(master_width), t * master_height))

    # Convert pixel coords (col, row) to projected coords (x, y) via master transform
    proj_coords = [master_transform * p for p in pts]
    xs = np.array([p[0] for p in proj_coords], dtype="float64")
    ys = np.array([p[1] for p in proj_coords], dtype="float64")

    from pyproj import Transformer
    transformer = Transformer.from_crs(master_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs, ys)

    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")

    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
        bad = ~(np.isfinite(lon) & np.isfinite(lat))
        raise ValueError(
            "get_web_bounds: non-finite transformed coordinates "
            "(%d of %d samples rejected). Check the master CRS."
            % (int(bad.sum()), len(bad))
        )

    minx, maxx = float(lon.min()), float(lon.max())
    miny, maxy = float(lat.min()), float(lat.max())
    if not (minx < maxx and miny < maxy):
        raise ValueError("get_web_bounds: degenerate bounds ordering.")
    return [minx, miny, maxx, maxy]

def colorize_risk_rgb(risk_data, colormap):
    """Map float risk values to RGB via linear interpolation over colormap stops."""
    finite = np.isfinite(risk_data)
    rgb = np.zeros((risk_data.shape[0], risk_data.shape[1], 3), dtype=np.uint8)
    if not finite.any():
        return rgb, finite
    vals = np.clip(risk_data[finite], 0.0, 1.0)
    stops_v = np.array([v for v, _ in colormap], dtype="float64")
    stops_r = np.array([c[0] for _, c in colormap], dtype="float64")
    stops_g = np.array([c[1] for _, c in colormap], dtype="float64")
    stops_b = np.array([c[2] for _, c in colormap], dtype="float64")
    r = np.interp(vals, stops_v, stops_r)
    g = np.interp(vals, stops_v, stops_g)
    b = np.interp(vals, stops_v, stops_b)
    out = rgb.astype("float64")
    out[..., 0][finite] = r
    out[..., 1][finite] = g
    out[..., 2][finite] = b
    return out.astype(np.uint8), finite

# ----------------------------------------------------------------------------
# CORRECTED: create_web_gradient
# ----------------------------------------------------------------------------
def create_web_gradient(raster_path, fars_boundary_master, master_transform=None,
                        master_crs=None, master_width=None, master_height=None,
                        colormap=None):
    """
    Corrected web-gradient builder.

    1) Colorize master risk raster to RGB (finite pixels only).
    2) Resize RGB image to target PNG dimensions with BICUBIC (purely visual color
       interpolation; no alpha bleed).
    3) Rasterize province boundary DIRECTLY into final PNG dimensions using
       the scaled affine transform:
_transform = Affine(a*sx, b*sx, c, d target_height
         target_transform = Affine(a*sx, b*sx, c, d*sy, e*sy, f)
       in the master CRS with all_touched=False.
    4) Alpha = ALPHA_PROVINCE (235) where (province mask AND finite risk) else 0.
    """
    from PIL import Image

    if colormap is None:
        colormap = [
            (0.00, (237, 248, 251)),
            (0.25, (158, 216, 224)),
            (0.50, ( 94, 174, 194)),
            (0.75, (240, 190, 106)),
            (1.00, (192,  32,  38)),
        ]

    with rasterio.open(raster_path) as src:
        if master_width is None:
            master_width = src.width
        if master_height is None:
            master_height = src.height
        if master_transform is None:
            master_transform = src.transform
        if master_crs is None:
            master_crs = src.crs
        risk = src.read(1).astype("float64")

    # 1. Colorize RGB
    rgb_arr, finite_mask = colorize_risk_rgb(risk, colormap)

    # 2. Target dimensions & Bicubic interpolation of RGB only
    target_width, target_height = compute_target_size(master_width, master_height)
    target_size = (target_width, target_height)

    rgb_img = Image.fromarray(rgb_arr, mode="RGB")
    rgb_resized = rgb_img.resize(target_size, Image.BICUBIC)
    rgb_final = np.asarray(rgb_resized, dtype=np.uint8)

    # 3. Derive final affine transform
    sx = float(master_width) / float(target_width)
    sy = float(master_height) / float(target_height)
    target_transform = Affine(
        master_transform.a * sx,
        master_transform.b * sx,
        master_transform.c,
        master_transform.d * sy,
        master_transform.e * sy,
        master_transform.f,
    )

    geom, geom_crs = get_fars_geometry(fars_boundary_master)
    if geom_crs != master_crs:
        import geopandas as gpd
        g = gpd.GeoSeries([geom], crs=geom_crs).to_crs(master_crs)
        geom = g.iloc[0]

    # Rasterize geometry DIRECTLY at final image shape (H, W)
    province_mask = rasterize(
        [(geom, 1)],
        out_shape=(target_height, target_width),
        transform=target_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    # Validity representation at final size
    finite_img = Image.fromarray((finite_mask * 255).astype(np.uint8), mode="L")
    finite_resized = finite_img.resize(target_size, Image.NEAREST)
    finite_final = np.asarray(finite_resized, dtype=np.uint8) > 0

    # 4. Alpha mask (235 inside province & finite risk, 0 outside)
    alpha = np.where(province_mask & finite_final, ALPHA_PROVINCE, 0).astype(np.uint8)

    rgba = np.dstack([rgb_final, alpha])
    out_img = Image.fromarray(rgba, mode="RGBA")

    web_bounds = get_web_bounds(
        None, master_transform, master_width, master_height, master_crs
    )

    info = {
        "target_size": [target_width, target_height],
        "target_transform": list(target_transform)[:6],
        "master_transform": list(master_transform)[:6],
        "web_bounds": web_bounds,
        "crs": master_crs.to_string() if hasattr(master_crs, "to_string") else str(master_crs),
        "alpha_value": ALPHA_PROVINCE,
    }
    return out_img, info

# ----------------------------------------------------------------------------
# Metadata builder
# ----------------------------------------------------------------------------
def build_metadata(info, extra=None):
    meta = {
        "png": os.path.basename(PNG_PATH),
        "raster": os.path.basename(RASTER_PATH),
        "image_size": info["target_size"],
        "bounds": info["web_bounds"],
        "web": {
            "crs": "EPSG:4326",
            "bounds": info["web_bounds"],
        },
        "crs": info["crs"],
        "alpha": info["alpha_value"],
    }
    if extra:
        meta.update(extra)
    return meta

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import geopandas as gpd

    candidates = [
        "/mnt/data/fars.geojson", "/mnt/data/fars_boundary.geojson",
        "/mnt/data/fars.shp", "/mnt/data/fars_boundary.shp",
    ]
    fars_boundary_master = None
    for c in candidates:
        if os.path.exists(c):
            fars_boundary_master = gpd.read_file(c)
            break

    if fars_boundary_master is None:
        print("Creating synthetic boundary for demonstration...")
        from shapely.geometry import Polygon
        poly = Polygon([
            (52.0, 27.5), (55.5, 27.5), (55.0, 31.5),
            (51.5, 31.5), (50.5, 29.5), (52.0, 27.5)
        ])
        fars_boundary_master = gpd.GeoDataFrame(
            {"name": ["Fars"]}, geometry=[poly], crs="EPSG:4326"
        )

    from rasterio.features import geometry_mask
    geom, geom_crs = get_fars_geometry(fars_boundary_master)
    proj_crs = "EPSG:32639"
    gseries = gpd.GeoSeries([geom], crs=geom_crs).to_crs(proj_crs)
    geom_proj = gseries.iloc[0]

    minx, miny, maxx, maxy = geom_proj.bounds
    res = 1000.0
    W = int(math.ceil((maxx - minx) / res)) + 4
    H = int(math.ceil((maxy - miny) / res)) + 4
    master_transform = Affine(res, 0, minx - 2 * res, 0, -res, maxy + 2 * res)

    yy, xx = np.mgrid[0:H, 0:W]
    xs = master_transform.c + (xx + 0.5) * res
    ys = master_transform.f - (yy + 0.5) * res
    inside = geometry_mask([geom_proj], out_shape=(H, W),
                           transform=master_transform, invert=True,
                           all_touched=False)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    base = 1.0 - np.sqrt(((xs - cx) / (maxx - minx)) ** 2 +
                         ((ys - cy) / (maxy - miny)) ** 2)
    rng = np.random.default_rng(42)
    risk = np.clip(base + 0.1 * rng.standard_normal((H, W)), 0, 1)
    risk[~inside] = np.nan

    profile = {
        "driver": "GTiff", "width": W, "height": H, "count": 1,
        "dtype": "float64", "crs": proj_crs, "transform": master_transform,
        "nodata": None, "compress": "deflate",
    }
    with rasterio.open(RASTER_PATH, "w", **profile) as dst:
        dst.write(risk.astype("float64"), 1)

    img, info = create_web_gradient(
        RASTER_PATH, fars_boundary_master, master_transform,
        proj_crs, W, H,
    )
    img.save(PNG_PATH)

    meta = build_metadata(info)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Success! Output at:", OUT_DIR)
    print("PNG size:", img.size)
    print("Web bounds:", info["web_bounds"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
