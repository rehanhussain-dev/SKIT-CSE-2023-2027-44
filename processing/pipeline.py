"""One function that turns a field polygon into a result dictionary."""
from make_synthetic_grid import NODATA, PIXEL_SIZE, UTM_CRS, build_grid
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import shape
from sklearn.ensemble import RandomForestRegressor

from aggregate import DEFAULT_UNIT, aggregate
from mask_field import mask_field
from shap_mock import RNG_SEED, make_mock_table, stress_shares, stress_split


@lru_cache(maxsize=1)
def mock_stress_split():
    """Train the mock model once and reuse the result for every field."""
    table = make_mock_table()
    features = table.drop(columns="yield_t_ha")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=RNG_SEED
    )
    model.fit(features, table["yield_t_ha"])
    return stress_split(stress_shares(model, features))


def process_field(polygon_geojson, raster_path, unit=DEFAULT_UNIT):
    """Return yield, area and stress split for one GeoJSON polygon."""
    polygon_lonlat = shape(polygon_geojson)

    with rasterio.open(raster_path) as src:
        values = src.read(1).astype("float32")
        transform = src.transform
        raster_crs = src.crs
        nodata = src.nodata if src.nodata is not None else np.nan
        pixel_area_m2 = abs(src.res[0] * src.res[1])

    _, inside = mask_field(values, transform, raster_crs, polygon_lonlat, nodata)
    result = aggregate(values, inside, pixel_area_m2, unit)
    result["stress_breakdown"] = dict(mock_stress_split())
    return result

def process_polygon(polygon_geojson, unit=DEFAULT_UNIT):
    """Same result as process_field, but builds the mock grid in memory.

    This works for any polygon, so no saved raster file is needed.
    """
    polygon_lonlat = shape(polygon_geojson)
    values, transform = build_grid(polygon_lonlat)

    _, inside = mask_field(values, transform, UTM_CRS, polygon_lonlat, NODATA)
    result = aggregate(values, inside, PIXEL_SIZE * PIXEL_SIZE, unit)
    result["stress_breakdown"] = dict(mock_stress_split())
    return result


if __name__ == "__main__":
    data = json.loads(
        Path("processing/data/test_1ha.geojson").read_text(encoding="utf-8")
    )
    geometry = data["features"][0]["geometry"]
    print(process_field(geometry, "processing/data/synthetic_grid.tif"))
    print(process_polygon(geometry))