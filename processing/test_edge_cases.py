import unittest

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, box

from aggregate import aggregate
from mask_field import mask_field

UTM = "EPSG:32643"
NODATA = -9999.0
PIXEL = 2.5
PIXEL_AREA = PIXEL * PIXEL
ORIGIN_X, ORIGIN_Y = 500000.0, 3100000.0
SIZE = 100  # 100 x 100 pixels = 250 m x 250 m, every pixel holds 1 kg


def make_grid():
    values = np.ones((SIZE, SIZE), dtype="float32")
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL)
    return values, transform


def to_lonlat(geometry_utm):
    """The pipeline expects lon/lat, so convert our metre-based shapes."""
    return gpd.GeoSeries([geometry_utm], crs=UTM).to_crs("EPSG:4326").iloc[0]


def run(polygon_utm, values=None):
    grid_values, transform = make_grid()
    if values is None:
        values = grid_values
    _, inside = mask_field(
        values, transform, UTM, to_lonlat(polygon_utm), NODATA
    )
    return aggregate(values, inside, PIXEL_AREA)


class EdgeCaseTests(unittest.TestCase):
    def test_one_hectare_uniform_square(self):
        # 100 m x 100 m, edges on pixel boundaries: 40 x 40 pixels
        result = run(box(500050, 3099800, 500150, 3099900))
        self.assertEqual(result["pixel_count"], 1600)
        self.assertEqual(result["area_hectares"], 1.0)
        self.assertEqual(result["yield_tons"], 1.6)

    def test_polygon_smaller_than_one_pixel(self):
        # 1 m x 1 m square that does not contain any pixel centre
        result = run(box(500050.1, 3099898.9, 500051.1, 3099899.9))
        self.assertEqual(result["pixel_count"], 0)
        self.assertEqual(result["yield_tons"], 0.0)
        self.assertEqual(result["mean_yield_tons_per_hectare"], 0.0)

    def test_polygon_crossing_raster_edge(self):
        # Left half sticks out of the raster: only 20 x 40 pixels remain
        result = run(box(499950, 3099800, 500050, 3099900))
        self.assertEqual(result["pixel_count"], 800)

    def test_nodata_pixels_are_not_counted(self):
        values, _ = make_grid()
        values[40, 20:60] = NODATA  # one full row of the square is empty
        result = run(box(500050, 3099800, 500150, 3099900), values)
        self.assertEqual(result["pixel_count"], 1560)

    def test_multipolygon(self):
        parts = MultiPolygon(
            [
                box(500050, 3099850, 500100, 3099900),
                box(500150, 3099850, 500200, 3099900),
            ]
        )
        result = run(parts)
        self.assertEqual(result["pixel_count"], 800)


if __name__ == "__main__":
    unittest.main()