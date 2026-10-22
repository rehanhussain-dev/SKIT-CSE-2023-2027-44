"""
export_patches.py
-----------------
Tiles input multispectral Sentinel-2/Planet GeoTIFF rasters into
standardized 128x128 patches for the SRGAN super-resolution training pipeline.
Applies centered constant zero-padding for small parcels to prevent
kaleidoscopic mirror-reflection artifacts.
"""

import os
import glob
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine


PATCH_SIZE = 128
STRIDE = 64
MAX_NODATA_RATIO = 0.90  # Allow centered small parcels with background padding


def slice_raster_to_patches(input_tif, output_dir, patch_size=PATCH_SIZE, stride=STRIDE):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_tif))[0]

    with rasterio.open(input_tif) as src:
        width = src.width
        height = src.height
        nodata = src.nodata if src.nodata is not None else 0
        data = src.read()

        # Center-pad small parcels directly to patch_size x patch_size
        if width < patch_size or height < patch_size:
            pad_top = (patch_size - height) // 2
            pad_bottom = patch_size - height - pad_top
            pad_left = (patch_size - width) // 2
            pad_right = patch_size - width - pad_left

            # Symmetrical constant zero padding
            padded = np.pad(
                data,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=nodata,
            )

            # Adjust geospatial geotransform origin for top-left padding shift
            orig_transform = src.transform
            new_x_origin = orig_transform.c - (pad_left * orig_transform.a)
            new_y_origin = orig_transform.f - (pad_top * orig_transform.e)

            new_transform = Affine(
                orig_transform.a,
                orig_transform.b,
                new_x_origin,
                orig_transform.d,
                orig_transform.e,
                new_y_origin,
            )

            profile = src.profile.copy()
            profile.update({
                "width": patch_size,
                "height": patch_size,
                "transform": new_transform,
                "nodata": nodata,
            })

            out_file = os.path.join(output_dir, f"{base_name}_padded_patch_0.tif")
            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(padded)

            print(f"  Center-padded -> {out_file}")
            return 1

        # Sliding window tiling for rasters larger than patch_size
        total_saved = 0
        patch_idx = 0

        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):
                window = Window(left, top, patch_size, patch_size)
                patch_data = src.read(window=window)

                zero_mask = (patch_data == nodata) | (patch_data == 0)
                zero_ratio = np.sum(zero_mask) / patch_data.size
                if zero_ratio > MAX_NODATA_RATIO:
                    continue

                transform = rasterio.windows.transform(window, src.transform)
                profile = src.profile.copy()
                profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": transform,
                    "nodata": nodata,
                })

                out_filename = os.path.join(output_dir, f"{base_name}_p{patch_idx:04d}.tif")
                with rasterio.open(out_filename, "w", **profile) as dst:
                    dst.write(patch_data)

                patch_idx += 1
                total_saved += 1

        print(f"[{base_name}] Processed {width}x{height} raster -> {total_saved} patches.")
        return total_saved


def batch_process_directory(input_dir, output_dir):
    tif_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.tif"), recursive=True))
    if not tif_files:
        print(f"No GeoTIFFs found in {input_dir}.")
        return

    print(f"Found {len(tif_files)} rasters in {input_dir}. Extracting {PATCH_SIZE}x{PATCH_SIZE} patches...")
    grand_total = 0
    for tif in tif_files:
        grand_total += slice_raster_to_patches(tif, output_dir)

    print(f"Done. Saved {grand_total} patches to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile GeoTIFFs into center-padded 128x128 patches.")
    parser.add_argument("--input_dir", required=True, help="Input directory of GeoTIFFs")
    parser.add_argument("--output_dir", required=True, help="Output directory for 128x128 patches")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Sliding window stride")
    args = parser.parse_args()

    batch_process_directory(args.input_dir, args.output_dir)