"""
run_srgan_inference.py

Purpose
-------
Runs the pretrained OpenSR-SRGAN "RGB-NIR" model (10m -> 2.5m, 4x) over a
Sentinel-2 GeoTIFF exported from Google Earth Engine (GEE). Designed to run
in Google Colab and be committed to git as part of the Sprint 2
(Spatial Super-Resolution) pipeline.

Project: Farm Yield Prediction using Very High Spatial Resolution Data
Sprint : Spatial Super-Resolution
Model  : ESAOpenSR / opensr_srgan (pretrained RGB-NIR, 4-band, 4x)

Usage (Colab)
-------------
1. Upload your GEE-exported GeoTIFF (or mount Google Drive) so its path
   is accessible, e.g. "/content/drive/MyDrive/farm_patches/field_01.tif".
2. Set INPUT_TIF below (or pass it as a CLI arg when run as a script).
3. Run all cells / run the script.
4. Output georeferenced SR GeoTIFF is written next to the input,
   suffixed "_SR.tif".

Assumptions / things to verify before trusting the output
-----------------------------------------------------------
- Your GeoTIFF has exactly 4 bands, in the order Red, Green, Blue, NIR
  (Sentinel-2 bands B4, B3, B2, B8). If your GEE export used a different
  band order, fix it in your GEE export script or reorder bands below.
- Pixel values are surface reflectance scaled to roughly 0-1
  (Sentinel-2 SR products are natively 0-10000; this script rescales
  automatically -- see `SCALE_FACTOR` below, but you MUST confirm your
  own export's native range using the inspection step first).
"""

import os
import sys
import numpy as np
import rasterio

# --------------------------------------------------------------------------
# 0. CONFIG -- edit these for your run
# --------------------------------------------------------------------------

# Path to your GEE-exported GeoTIFF (4-band Red-Green-Blue-NIR, 10m).
INPUT_TIF = "/content/drive/MyDrive/farm_patches/field_01.tif"

# Where to write the super-resolved output. Defaults to INPUT_TIF with
# an "_SR" suffix, sitting alongside the input.
OUTPUT_TIF = None  # leave None to auto-generate

# If your GEE export is raw Sentinel-2 digital numbers (0-10000), this
# rescales to ~0-1 reflectance before feeding the model. Set to 1.0 if
# your export is already scaled to 0-1 (check with the inspection step
# below before running the full job).
SCALE_FACTOR = 10000.0

# LR-space patch size, upscale factor, and tile overlap. 128 matches the
# patch size specified in your Form-1 GEE export pipeline. Do not change
# `FACTOR` -- it must match the pretrained model (RGB-NIR = 4x, 10m->2.5m).
WINDOW_SIZE = (128, 128)
FACTOR = 4
OVERLAP = 12
ELIMINATE_BORDER_PX = 2

DEVICE = "cuda"  # Colab GPU runtime required (Runtime > Change runtime type > GPU)


# --------------------------------------------------------------------------
# 1. Install dependencies (Colab-safe: skips if already installed)
# --------------------------------------------------------------------------
def install_dependencies():
    os.system(f"{sys.executable} -m pip install -q opensr_srgan opensr-utils rasterio")


# --------------------------------------------------------------------------
# 2. Inspect the GeoTIFF before running inference
# --------------------------------------------------------------------------
def inspect_geotiff(path):
    """
    Prints band count, dtype, value range, and CRS so you can confirm the
    file matches what the model expects (4 bands, R-G-B-NIR order,
    reflectance-like value range) BEFORE burning GPU time on a bad input.
    """
    with rasterio.open(path) as src:
        print(f"File: {path}")
        print(f"  Band count : {src.count}")
        print(f"  Dtype      : {src.dtypes[0]}")
        print(f"  CRS        : {src.crs}")
        print(f"  Size       : {src.width} x {src.height}")
        data = src.read()
        print(f"  Value range: min={data.min()}, max={data.max()}, mean={data.mean():.2f}")

        if src.count != 4:
            print(f"  WARNING: expected 4 bands (R,G,B,NIR), found {src.count}. "
                  f"Re-export from GEE with the correct band selection/order.")
        if data.max() > 20:
            print("  NOTE: values look like raw digital numbers (>20), "
                  "SCALE_FACTOR=10000 is probably correct.")
        else:
            print("  NOTE: values already look like 0-1 reflectance. "
                  "Set SCALE_FACTOR = 1.0 before running inference.")
    return


# --------------------------------------------------------------------------
# 3. Run inference using opensr-utils (handles tiling, blending, georeferencing)
# --------------------------------------------------------------------------
def run_inference(input_tif, output_tif):
    import torch
    from opensr_srgan import load_inference_model
    import opensr_utils

    device = DEVICE if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. This will be slow. "
              "In Colab: Runtime > Change runtime type > GPU.")

    print("Loading pretrained RGB-NIR SRGAN model...")
    model = load_inference_model("RGB-NIR").to(device)

    print(f"Running SR over {input_tif} ...")
    opensr_utils.large_file_processing(
        root=input_tif,
        model=model,
        window_size=WINDOW_SIZE,
        factor=FACTOR,
        overlap=OVERLAP,
        eliminate_border_px=ELIMINATE_BORDER_PX,
        device=device,
    )
    print(f"Done. Super-resolved output should be written alongside the input "
          f"(check opensr-utils console output above for the exact path).")


# --------------------------------------------------------------------------
# 4. Fallback: manual patch-based inference (if you need raw tensor control,
#    e.g. custom normalization, band reordering, or opensr-utils doesn't
#    support your file layout yet)
# --------------------------------------------------------------------------
def run_inference_manual(input_tif, output_tif):
    import torch
    from opensr_srgan import load_inference_model

    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = load_inference_model("RGB-NIR").to(device).eval()

    with rasterio.open(input_tif) as src:
        profile = src.profile
        arr = src.read().astype(np.float32) / SCALE_FACTOR  # (bands, H, W)
        arr = np.clip(arr, 0, 1)

    # Reorder bands here if your GEE export is not already R,G,B,NIR, e.g.:
    # arr = arr[[2, 1, 0, 3], :, :]  # example: swap from B,G,R,NIR to R,G,B,NIR

    lr = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 4, H, W)

    with torch.inference_mode():
        sr = model.predict_step(lr)

    sr_np = sr.squeeze(0).cpu().numpy()

    out_profile = profile.copy()
    out_profile.update(
        height=sr_np.shape[1],
        width=sr_np.shape[2],
        count=sr_np.shape[0],
        dtype="float32",
        transform=profile["transform"] * profile["transform"].scale(
            profile["width"] / sr_np.shape[2],
            profile["height"] / sr_np.shape[1],
        ),
    )

    with rasterio.open(output_tif, "w", **out_profile) as dst:
        dst.write(sr_np)

    print(f"Wrote manual SR output to {output_tif}")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        INPUT_TIF = sys.argv[1]

    if OUTPUT_TIF is None:
        base, ext = os.path.splitext(INPUT_TIF)
        OUTPUT_TIF = f"{base}_SR{ext}"

    install_dependencies()

    print("=" * 60)
    print("STEP 1: Inspecting input GeoTIFF")
    print("=" * 60)
    inspect_geotiff(INPUT_TIF)

    print("\n" + "=" * 60)
    print("STEP 2: Running SRGAN inference")
    print("=" * 60)
    try:
        run_inference(INPUT_TIF, OUTPUT_TIF)
    except Exception as e:
        print(f"opensr-utils path failed ({e}); falling back to manual inference.")
        run_inference_manual(INPUT_TIF, OUTPUT_TIF)
