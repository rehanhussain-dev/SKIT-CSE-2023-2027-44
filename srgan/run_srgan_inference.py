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

import argparse
import os
import numpy as np
import rasterio

# --------------------------------------------------------------------------
# 0. CONFIG -- edit these for your run
# --------------------------------------------------------------------------

# Path to your GEE-exported GeoTIFF. The GEE pipeline exports eight bands;
# the model receives Red, Green, Blue, NIR from that file.
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
MODEL_BAND_INDICES = (2, 1, 0, 3)  # B4, B3, B2, B8 in the GEE export order.


# --------------------------------------------------------------------------
# 1. Install dependencies (Colab-safe: skips if already installed)
# --------------------------------------------------------------------------
def install_dependencies():
    """Install optional runtime dependencies when explicitly requested."""
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", os.path.join(os.path.dirname(__file__), "requirements-srgan.txt")]
    )


# --------------------------------------------------------------------------
# 2. Inspect the GeoTIFF before running inference
# --------------------------------------------------------------------------
def inspect_geotiff(path):
    """
    Prints band count, dtype, value range, CRS, and the bands selected for the
    model BEFORE burning GPU time on a bad input.
    """
    with rasterio.open(path) as src:
        print(f"File: {path}")
        print(f"  Band count : {src.count}")
        print(f"  Dtype      : {src.dtypes[0]}")
        print(f"  CRS        : {src.crs}")
        print(f"  Size       : {src.width} x {src.height}")
        data = src.read()
        print(
            f"  Value range: min={data.min()}, max={data.max()}, mean={data.mean():.2f}"
        )

        if src.count != 4:
            print(
                f"  WARNING: expected 4 bands (R,G,B,NIR), found {src.count}. "
                f"Re-export from GEE with the correct band selection/order."
            )
        if src.count < 4:
            raise ValueError(f"Expected at least 4 bands, found {src.count}")
        if max(MODEL_BAND_INDICES) >= src.count:
            raise ValueError("The input does not contain the required RGB-NIR bands")
        print(f"  Model bands: {MODEL_BAND_INDICES} (R,G,B,NIR)")
        if data.max() > 20:
            print(
                "  NOTE: values look like raw digital numbers (>20), "
                "SCALE_FACTOR=10000 is probably correct."
            )
        else:
            print(
                "  NOTE: values already look like 0-1 reflectance. "
                "Set SCALE_FACTOR = 1.0 before running inference."
            )
    return


def load_model(device):
    """Load the pretrained four-band RGB-NIR model."""
    import torch
    from opensr_srgan import load_inference_model

    return load_inference_model("RGB-NIR").to(device).eval()


def read_model_input(src):
    """Read and normalize the GEE band layout into R-G-B-NIR tensors."""
    arr = src.read(MODEL_BAND_INDICES).astype(np.float32)
    if np.nanmax(arr) > 20:
        arr /= SCALE_FACTOR
    return np.clip(np.nan_to_num(arr, nan=0.0), 0, 1)


# --------------------------------------------------------------------------
# 3. Run inference using opensr-utils (handles tiling, blending, georeferencing)
# --------------------------------------------------------------------------
def run_inference(input_tif, output_tif):
    import torch

    device = DEVICE if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no GPU detected. This will be slow. "
            "In Colab: Runtime > Change runtime type > GPU."
        )

    print("Loading pretrained RGB-NIR SRGAN model...")
    model = load_model(device)

    with rasterio.open(input_tif) as src:
        source_profile = src.profile.copy()
        source_transform = src.transform
        source_height, source_width = src.height, src.width
        arr = read_model_input(src)

    lr = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.inference_mode():
        sr = model.predict_step(lr)
    sr_np = sr.squeeze(0).detach().cpu().numpy()

    output_profile = source_profile.copy()
    output_profile.update(
        count=sr_np.shape[0],
        dtype="float32",
        height=sr_np.shape[1],
        width=sr_np.shape[2],
        transform=source_transform * source_transform.scale(
            source_width / sr_np.shape[2], source_height / sr_np.shape[1]
        ),
    )
    print(
        f"Done. Super-resolved output should be written alongside the input "
        f"(check opensr-utils console output above for the exact path)."
    )
    with rasterio.open(output_tif, "w", **output_profile) as dst:
        dst.write(np.clip(sr_np, 0, 1).astype(np.float32))
    print(f"Wrote super-resolved GeoTIFF to {output_tif}")


# --------------------------------------------------------------------------
# 4. Fallback: manual patch-based inference (if you need raw tensor control,
#    e.g. custom normalization, band reordering, or opensr-utils doesn't
#    support your file layout yet)
# --------------------------------------------------------------------------
def run_inference_manual(input_tif, output_tif):
    import torch

    device = DEVICE if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    with rasterio.open(input_tif) as src:
        profile = src.profile
        arr = read_model_input(src)

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
        transform=profile["transform"]
        * profile["transform"].scale(
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
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tif", nargs="?", default=INPUT_TIF)
    parser.add_argument("--output", dest="output_tif")
    parser.add_argument(
        "--install", action="store_true", help="Install requirements-srgan.txt before inference"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_TIF = args.input_tif
    OUTPUT_TIF = args.output_tif
    if OUTPUT_TIF is None:
        base, ext = os.path.splitext(INPUT_TIF)
        OUTPUT_TIF = f"{base}_SR{ext}"

    if args.install:
        install_dependencies()

    print("=" * 60)
    print("STEP 1: Inspecting input GeoTIFF")
    print("=" * 60)
    inspect_geotiff(INPUT_TIF)

    print("\n" + "=" * 60)
    print("STEP 2: Running SRGAN inference")
    print("=" * 60)
    run_inference(INPUT_TIF, OUTPUT_TIF)
