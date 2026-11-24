"""Sum the yield of all pixels inside one field and report it in metric tons.

Usage:
    python processing/aggregate.py
    python processing/aggregate.py --unit kg_per_m2
    python processing/aggregate.py --geojson my_polygon.json

The model's output unit is not final yet, so it is a single setting (--unit):
    kg_per_pixel : every pixel value is already kilograms in that pixel
    kg_per_m2    : every pixel value is kg per square metre, so it is
                   multiplied by the pixel area (6.25 m2 for a 2.5 m pixel)
"""

import argparse

import rasterio

from make_synthetic_grid import load_polygon
from mask_field import mask_field

DEFAULT_UNIT = "kg_per_pixel"
SUPPORTED_UNITS = ("kg_per_pixel", "kg_per_m2")


def aggregate(values, inside, pixel_area_m2, unit=DEFAULT_UNIT):
    """Return the field totals as a dictionary.

    values:         2D array from the model
    inside:         True for pixels that belong to the field
    pixel_area_m2:  area of one pixel in square metres
    unit:           how to read the pixel values (see module docstring)
    """
    if unit not in SUPPORTED_UNITS:
        raise ValueError(f"unit must be one of {SUPPORTED_UNITS}, got '{unit}'")

    field_values = values[inside]

    if unit == "kg_per_pixel":
        total_kg = float(field_values.sum())
    else:
        total_kg = float((field_values * pixel_area_m2).sum())

    pixel_count = int(inside.sum())
    area_hectares = pixel_count * pixel_area_m2 / 10000
    yield_tons = total_kg / 1000

    return {
        "unit": unit,
        "pixel_count": pixel_count,
        "area_hectares": round(area_hectares, 4),
        "yield_tons": round(yield_tons, 3),
        # Guard against an empty field (no pixel inside the polygon)
        "mean_yield_tons_per_hectare": (
            round(yield_tons / area_hectares, 3) if area_hectares > 0 else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Total yield of one field.")
    parser.add_argument("--raster", default="processing/data/synthetic_grid.tif")
    parser.add_argument("--geojson", help="GeoJSON file with the field polygon")
    parser.add_argument("--unit", default=DEFAULT_UNIT, choices=SUPPORTED_UNITS)
    args = parser.parse_args()

    polygon = load_polygon(args.geojson)

    with rasterio.open(args.raster) as src:
        values = src.read(1)
        _, inside = mask_field(values, src.transform, src.crs, polygon, src.nodata)
        pixel_area = abs(src.res[0] * src.res[1])

    result = aggregate(values, inside, pixel_area, args.unit)

    print(f"Unit used:             {result['unit']}")
    print(f"Pixels inside field:   {result['pixel_count']}")
    print(f"Field area:            {result['area_hectares']} ha")
    print(f"Total yield:           {result['yield_tons']} metric tons")
    print(f"Mean yield:            {result['mean_yield_tons_per_hectare']} t/ha")


if __name__ == "__main__":
    main()