"""Batch-upscale a Sentinel-2 patch time series from 10 m to 2.5 m.

Runs the pretrained RGB-NIR SRGAN over every GeoTIFF in a directory and
writes one georeferenced four-band (B4, B3, B2, B8) ``_SR.tif`` per input.
The outputs are what ``validate_sr_consistency.py`` checks before the
upscaled bands are handed to the temporal LSTM branch.

    py -3 srgan/upscale_timeseries.py srgan/low/LR --output-dir srgan/outputs/sr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_srgan_inference import (  # noqa: E402
    FACTOR,
    MODEL_BAND_INDICES,
    load_model,
    read_model_input,
    validate_inference_source,
)

SR_BAND_NAMES = ("B4", "B3", "B2", "B8")


def upscale_mask(valid, factor=FACTOR):
    """Repeat a (H, W) validity mask onto the (H*factor, W*factor) grid."""
    return np.repeat(np.repeat(valid, factor, axis=0), factor, axis=1)


def sr_profile(profile, transform, sr_shape, source_shape):
    """Build the output profile: same footprint, pixel size divided by the factor."""
    height, width = sr_shape
    source_height, source_width = source_shape
    out = profile.copy()
    out.update(
        driver="GTiff",
        count=len(SR_BAND_NAMES),
        dtype="float32",
        nodata=float("nan"),
        height=height,
        width=width,
        transform=transform * transform.scale(source_width / width, source_height / height),
    )
    return out


def upscale_file(model, input_path, output_path, device="cpu"):
    """Super-resolve one GeoTIFF. Returns the fraction of valid input pixels."""
    import rasterio
    import torch

    with rasterio.open(input_path) as src:
        validate_inference_source(src)
        profile = src.profile.copy()
        transform = src.transform
        source_shape = (src.height, src.width)
        valid = np.isfinite(src.read(MODEL_BAND_INDICES)).all(axis=0)
        arr = read_model_input(src)

    with torch.inference_mode():
        sr = model.predict_step(torch.from_numpy(arr).unsqueeze(0).to(device))
    sr_np = np.clip(sr.squeeze(0).detach().cpu().numpy(), 0, 1).astype(np.float32)

    expected = (source_shape[0] * FACTOR, source_shape[1] * FACTOR)
    if sr_np.shape[1:] != expected:
        raise ValueError(f"model returned {sr_np.shape[1:]}, expected {expected}")

    # Pixels that were invalid in the input carry no information; keep them nodata.
    sr_np[:, ~upscale_mask(valid)] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **sr_profile(profile, transform, expected, source_shape)) as dst:
        dst.write(sr_np)
        dst.descriptions = SR_BAND_NAMES
    return float(valid.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "sr")
    parser.add_argument("--min-valid-fraction", type=float, default=0.5,
                        help="Skip patches with fewer valid RGB-NIR pixels than this fraction")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import rasterio
    import torch

    paths = sorted(args.input_dir.glob("*.tif")) + sorted(args.input_dir.glob("*.tiff"))
    if not paths:
        raise SystemExit(f"No GeoTIFF patches found in {args.input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pretrained RGB-NIR SRGAN on {device}...")
    model = load_model(device)

    written = skipped = 0
    for path in paths:
        output_path = args.output_dir / f"{path.stem}_SR.tif"
        if output_path.exists() and not args.overwrite:
            written += 1
            continue
        try:
            with rasterio.open(path) as src:
                valid_fraction = float(np.isfinite(src.read(MODEL_BAND_INDICES)).all(axis=0).mean())
            if valid_fraction < args.min_valid_fraction:
                raise ValueError(f"only {valid_fraction:.0%} valid pixels")
            upscale_file(model, path, output_path, device)
        except ValueError as error:
            skipped += 1
            print(f"SKIP {path.name}: {error}")
            continue
        written += 1
    print(f"Upscaled {written} of {len(paths)} patches into {args.output_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
