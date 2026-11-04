"""Keep only the pixels inside a field polygon and zero out everything else.

Usage:
    python processing/mask_field.py
    python processing/mask_field.py --geojson my_polygon.json

A pixel counts as inside the field when its centre lies inside the polygon
(the "centroid inside polygon" rule from the project spec).
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # save the picture to a file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask

# Reuse the polygon loader from the grid generator (same folder)
from make_synthetic_grid import load_polygon


def mask_field(values, transform, raster_crs, polygon_lonlat, nodata):
    """Return (masked_values, inside) for one field polygon.

    values:  2D array of kg per pixel
    inside:  True for pixels that belong to the field and hold real data
    masked:  the same array with every pixel outside the field set to 0
    """
    # The polygon is in degrees; the raster is in metres, so convert it first
    polygon_in_raster_crs = gpd.GeoSeries([polygon_lonlat], crs="EPSG:4326").to_crs(
        raster_crs
    )

    # all_touched=False means a pixel counts only if its centre is inside.
    # invert=True makes inside pixels True (the default is the opposite).
    inside = geometry_mask(
        list(polygon_in_raster_crs.geometry),
        out_shape=values.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    # Pixels without data must never count towards the yield
    has_data = (values != nodata) & ~np.isnan(values)
    inside = inside & has_data

    masked = np.where(inside, values, 0.0).astype("float32")
    return masked, inside


def save_before_after(values, inside, out_path):
    """Save a two-panel picture: the whole grid, then only the field pixels."""
    after = np.ma.masked_where(~inside, values)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(11, 5))

    image = ax_before.imshow(values, cmap="YlGn")
    ax_before.set_title("Before: whole grid")

    ax_after.imshow(after, cmap="YlGn", vmin=values.min(), vmax=values.max())
    ax_after.set_title("After: only pixels inside the field")

    for axis in (ax_before, ax_after):
        axis.set_xlabel("pixel column")
        axis.set_ylabel("pixel row")

    fig.colorbar(image, ax=[ax_before, ax_after], label="yield (kg per pixel)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Mask a yield grid with a polygon.")
    parser.add_argument("--raster", default="processing/data/synthetic_grid.tif")
    parser.add_argument("--geojson", help="GeoJSON file with the field polygon")
    parser.add_argument("--picture", default="processing/figures/mask_before_after.png")
    args = parser.parse_args()

    polygon = load_polygon(args.geojson)

    with rasterio.open(args.raster) as src:
        values = src.read(1)
        masked, inside = mask_field(values, src.transform, src.crs, polygon, src.nodata)
        pixel_area = abs(src.res[0] * src.res[1])  # square metres per pixel

    save_before_after(values, inside, Path(args.picture))

    inside_count = int(inside.sum())
    print(f"Pixels in grid:        {values.size}")
    print(f"Pixels inside field:   {inside_count}")
    print(f"Pixels zeroed outside: {values.size - inside_count}")
    print(f"Field area from pixels: {inside_count * pixel_area / 10000:.2f} ha")
    print(f"Picture saved: {args.picture}")


if __name__ == "__main__":
    main()