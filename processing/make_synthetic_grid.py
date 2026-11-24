"""Generate a synthetic 2.5 m prediction grid (kg per pixel) for one field polygon.

Usage:
    python processing/make_synthetic_grid.py
    python processing/make_synthetic_grid.py --geojson my_polygon.json

The grid covers the bounding box of the polygon (plus a small margin), is stored
in UTM zone 43N (EPSG:32643, metres) and is filled with random yield values.
It is fake data, used only to test masking and aggregation before the real
model output exists.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import shape
import geopandas as gpd

PIXEL_SIZE = 2.5          # metres, same resolution as the super-resolved bands
UTM_CRS = "EPSG:32643"    # UTM zone 43N, covers Jaipur, Alwar, Bikaner, Jodhpur
MARGIN_PIXELS = 20        # extra pixels around the polygon so the edge is visible
NODATA = -9999.0

# Rehan's pilot plot (PILOT_RAJ_01), used when no GeoJSON file is given
DEFAULT_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [75.842, 28.128],
        [75.846, 28.128],
        [75.846, 28.124],
        [75.842, 28.124],
        [75.842, 28.128],
    ]],
}


def load_polygon(geojson_path):
    """Return a shapely polygon (lon/lat) from a GeoJSON file or the default."""
    if geojson_path is None:
        return shape(DEFAULT_POLYGON)

    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    # Accept a bare geometry, a Feature, or a FeatureCollection with one feature
    if data.get("type") == "FeatureCollection":
        data = data["features"][0]
    if data.get("type") == "Feature":
        data = data["geometry"]
    return shape(data)


def build_grid(polygon_lonlat, seed=42):
    """Create the random yield array and its georeferencing."""
    # Reproject to metres so the pixel size is really 2.5 m
    polygon_utm = gpd.GeoSeries([polygon_lonlat], crs="EPSG:4326").to_crs(UTM_CRS)
    min_x, min_y, max_x, max_y = polygon_utm.total_bounds

    margin = MARGIN_PIXELS * PIXEL_SIZE
    min_x, min_y = min_x - margin, min_y - margin
    max_x, max_y = max_x + margin, max_y + margin

    width = int(np.ceil((max_x - min_x) / PIXEL_SIZE))
    height = int(np.ceil((max_y - min_y) / PIXEL_SIZE))

    # Wheat is roughly 0.4 kg/m2, so a 6.25 m2 pixel holds about 2.5 kg.
    # Values are drawn around that number with some random spread.
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=2.5, scale=0.4, size=(height, width))
    values = np.clip(values, 0.5, 4.5).astype("float32")

    transform = from_origin(min_x, max_y, PIXEL_SIZE, PIXEL_SIZE)
    return values, transform


def main():
    parser = argparse.ArgumentParser(description="Make a synthetic yield GeoTIFF.")
    parser.add_argument("--geojson", help="GeoJSON file with the field polygon")
    parser.add_argument("--out", default="processing/data/synthetic_grid.tif")
    args = parser.parse_args()

    polygon = load_polygon(args.geojson)
    values, transform = build_grid(polygon)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=UTM_CRS,
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(values, 1)

    print(f"Saved: {out_path}")
    print(f"Grid size: {values.shape[1]} x {values.shape[0]} pixels")
    print(f"Pixel size: {PIXEL_SIZE} m, CRS: {UTM_CRS}")
    print(f"Yield range: {values.min():.2f} to {values.max():.2f} kg per pixel")


if __name__ == "__main__":
    main()