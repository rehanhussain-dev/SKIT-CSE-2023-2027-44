"""
reproject_s2_utm.py
-------------------
Reprojects Sentinel-2 patches from WGS84 (EPSG:4326) into UTM Zone 43N (EPSG:32643)
at 10.0m resolution and standardizes the canvas to 128x128 centered on the parcel.
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine

TARGET_CRS = "EPSG:32643"
TARGET_RES = 10.0
PATCH_SIZE = 128


def reproject_to_utm_128(tif_path: str):
    with rasterio.open(tif_path) as src:
        if str(src.crs) == TARGET_CRS:
            return False  # Already in UTM 43N

        nodata = src.nodata if src.nodata is not None else 0.0
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds, resolution=TARGET_RES
        )

        # Reproject into temporary UTM array
        reprojected = np.full((src.count, height, width), nodata, dtype=src.dtypes[0])
        for b in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=reprojected[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                src_nodata=nodata,
                dst_nodata=nodata,
            )

        reprojected = np.nan_to_num(reprojected, nan=nodata, posinf=nodata, neginf=nodata)

        # Locate active farm center in the reprojected UTM array to center-crop/pad to 128x128
        active_rows, active_cols = np.where((reprojected[0] != nodata) & (reprojected[0] != 0))
        if len(active_rows) > 0:
            center_r = int(np.mean(active_rows))
            center_c = int(np.mean(active_cols))
        else:
            center_r, center_c = height // 2, width // 2

        half = PATCH_SIZE // 2
        r_start = center_r - half
        c_start = center_c - half

        canvas = np.full((src.count, PATCH_SIZE, PATCH_SIZE), nodata, dtype=src.dtypes[0])

        # Compute overlap slice between reprojected raster and 128x128 target canvas
        src_r0 = max(0, r_start)
        src_r1 = min(height, r_start + PATCH_SIZE)
        src_c0 = max(0, c_start)
        src_c1 = min(width, c_start + PATCH_SIZE)

        dst_r0 = src_r0 - r_start
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = src_c0 - c_start
        dst_c1 = dst_c0 + (src_c1 - src_c0)

        canvas[:, dst_r0:dst_r1, dst_c0:dst_c1] = reprojected[:, src_r0:src_r1, src_c0:src_c1]

        # Compute exact new UTM origin for top-left of 128x128 canvas
        new_x_origin = transform.c + (c_start * transform.a)
        new_y_origin = transform.f + (r_start * transform.e)
        final_transform = Affine(transform.a, 0.0, new_x_origin, 0.0, transform.e, new_y_origin)

        profile = src.profile.copy()
        profile.update({
            "crs": TARGET_CRS,
            "transform": final_transform,
            "width": PATCH_SIZE,
            "height": PATCH_SIZE,
            "nodata": nodata,
        })

    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(canvas)
    return True


if __name__ == "__main__":
    s2_dir = "/Users/rehanhussain/Documents/my-srgan-data/patches/s2_128"
    files = sorted(glob.glob(os.path.join(s2_dir, "*.tif")))
    print(f"Reprojecting {len(files)} Sentinel-2 patches from EPSG:4326 to {TARGET_CRS} (10m)...")

    converted = sum(1 for f in files if reproject_to_utm_128(f))
    print(f"Done. Converted {converted} patches to {TARGET_CRS}.")