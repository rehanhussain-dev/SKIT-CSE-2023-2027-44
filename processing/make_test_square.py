import json
import rasterio
from rasterio.warp import transform

RASTER = "processing/data/synthetic_grid.tif"
OUT = "processing/data/test_1ha.geojson"
SIDE_M = 100.0  # 100 m x 100 m = 1 hectare
SHIFT_PX = 100  # move the square this many pixels down from the top edge

with rasterio.open(RASTER) as src:
    pixel = src.res[0]
    # Start from a pixel-aligned corner so the square edges
    # fall exactly on pixel boundaries
    left = src.bounds.left + 80 * pixel
    top = src.bounds.top - SHIFT_PX * pixel
    crs = src.crs

right = left + SIDE_M
bottom = top - SIDE_M

# Square corners in UTM (metres), first point repeated to close the ring
xs = [left, right, right, left, left]
ys = [top, top, bottom, bottom, top]

# GeoJSON standard is WGS84 lon/lat, so convert from UTM
lons, lats = transform(crs, "EPSG:4326", xs, ys)
ring = [[lon, lat] for lon, lat in zip(lons, lats)]

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "test_1ha_square"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
    ],
}

with open(OUT, "w") as f:
    json.dump(geojson, f, indent=2)

print("Saved", OUT)
print("Square in UTM:", left, bottom, right, top)