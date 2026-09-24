
# Smart Fars Forecast

سامانه پیش‌بینی و پهنه‌بندی پیوسته خطر حریق در استان فارس.

## داده‌های پایه

- `fars.geojson` : مرز محاسباتی استان فارس
- `fars_fire_fuel_hazard_60m.tif` : لایه سوخت و Master Raster
- `fars_slope_60m_light.tif` : لایه شیب
- `dem_fars.tif` : مدل ارتفاع رقومی

## منبع FWI

FWI پیش‌بینی‌شده از:

Copernicus GWIS / ECMWF

دریافت می‌شود.

## منطق پردازش

```text
fars.geojson
        ↓
Master Raster = fars_fire_fuel_hazard_60m.tif
        ↓
DEM + Slope + Fuel + FWI
        ↓
Mask محدوده فارس
        ↓
هم‌ترازسازی Grid
        ↓
Outlier Control
        ↓
Normalization
        ↓
Aspect
        ↓
Topography
        ↓
WLC
        ↓
Fire Risk 0–100
        ↓
Web GIS
