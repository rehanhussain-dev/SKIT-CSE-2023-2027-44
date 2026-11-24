"""
sanitize_patches.py
-------------------
Scrubs NaN/Inf values from S2 index bands (replacing with 0.0)
and identifies/removes empty nodata patches.
"""

import os
import glob
import numpy as np
import rasterio

def sanitize_directory(patch_dir: str, is_planet: bool = False):
    files = sorted(glob.glob(os.path.join(patch_dir, "*.tif")))
    print(f"\nSanitizing {len(files)} patches in: {patch_dir}")
    
    cleaned_count = 0
    empty_removed = 0

    for fpath in files:
        with rasterio.open(fpath) as src:
            profile = src.profile.copy()
            data = src.read()

        # Check if entire raster is zero/nodata
        valid_mask = data != (src.nodata if src.nodata is not None else 0.0)
        if not np.any(valid_mask):
            print(f"  [REMOVED EMPTY] {os.path.basename(fpath)}")
            os.remove(fpath)
            empty_removed += 1
            continue

        # Replace NaN and Inf with 0.0
        nan_inf_mask = np.isnan(data) | np.isinf(data)
        if np.any(nan_inf_mask):
            data[nan_inf_mask] = 0.0
            with rasterio.open(fpath, "w", **profile) as dst:
                dst.write(data)
            cleaned_count += 1

    print(f"  Sanitized {cleaned_count} files with Inf/NaN scrubbed to 0.0")
    if empty_removed > 0:
        print(f"  Removed {empty_removed} empty patches")

if __name__ == "__main__":
    s2_dir = "/Users/rehanhussain/Documents/my-srgan-data/patches/s2_128"
    planet_dir = "/Users/rehanhussain/Documents/my-srgan-data/patches/planet_128"

    sanitize_directory(s2_dir, is_planet=False)
    sanitize_directory(planet_dir, is_planet=True)