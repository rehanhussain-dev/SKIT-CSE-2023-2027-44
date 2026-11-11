"""Upscale Sentinel-2 patches with the SRResNet trained on Sentinel-2 / PlanetScope pairs.

The checkpoint comes from ``kaggle_srgan_script.ipynb``. That model was trained on
Sentinel-2 bands resampled to a 12 m UTM grid (Blue, Green, Red, NIR) against 3 m
PlanetScope, so inference has to feed it the same thing: every input patch is
first reprojected to a 12 m UTM grid, then upscaled 4x to 3 m.

    py -3 srgan/run_srresnet_inference.py srgan/low/LR

For each input ``name.tif`` this writes, under ``--output-dir``:

    sr/name_SR.tif   3 m result, bands B4, B3, B2, B8 (same order as the OpenSR outputs)
    lr12/name.tif    the 12 m model input, bands B2, B3, B4, B8

The ``lr12`` files are the low-resolution side for ``validate_sr_consistency.py``:

    py -3 srgan/validate_sr_consistency.py --lr-dir srgan/outputs/srresnet/lr12 `
        --sr-dir srgan/outputs/srresnet/sr --hr-dir srgan/high/high `
        --output-dir srgan/outputs/srresnet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

FACTOR = 4
CHANNELS = 4
LR_PIXEL_M = 12.0
SCALE_FACTOR = 10000.0
GEE_BANDS = (1, 2, 3, 4)  # B2, B3, B4, B8 = Blue, Green, Red, NIR (model order)
SR_BAND_ORDER = (2, 1, 0, 3)  # model order -> B4, B3, B2, B8
SR_BAND_NAMES = ("B4", "B3", "B2", "B8")
LR_BAND_NAMES = ("B2", "B3", "B4", "B8")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "srresnet_best.pt"


def build_model(features=64, blocks=16):
    """Same architecture as ``SRResNet`` in the Kaggle notebook (state-dict compatible)."""
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels),
            )

        def forward(self, x):
            return x + self.body(x)

    class SRResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Conv2d(CHANNELS, features, 9, padding=4), nn.PReLU(features))
            self.residual = nn.Sequential(
                *[ResidualBlock(features) for _ in range(blocks)],
                nn.Conv2d(features, features, 3, padding=1), nn.BatchNorm2d(features),
            )
            self.upsample = nn.Sequential(
                nn.Conv2d(features, features * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(features),
                nn.Conv2d(features, features * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(features),
            )
            self.tail = nn.Sequential(
                nn.Conv2d(features, features, 3, padding=1), nn.PReLU(features),
                nn.Conv2d(features, CHANNELS, 9, padding=4), nn.Sigmoid(),
            )

        def forward(self, x):
            shallow = self.head(x)
            return self.tail(self.upsample(self.residual(shallow) + shallow))

    return SRResNet()


def load_srresnet(checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
    """Load the trained generator in eval mode."""
    import torch

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Download srresnet_best.pt from the Kaggle "
            "notebook's Output tab into srgan/checkpoints/."
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state = payload.get("model_state_dict", payload)
    bands = tuple(payload.get("bands", ("blue", "green", "red", "nir")))
    if bands != ("blue", "green", "red", "nir") or payload.get("factor", FACTOR) != FACTOR:
        raise ValueError(f"Unsupported checkpoint layout: bands={bands}, factor={payload.get('factor')}")
    if payload.get("arch") == "residual_sr_v1":
        # Checkpoint from train_residual_sr.py: bicubic plus a learned, input-consistent detail layer.
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_residual_sr import ResidualSR

        model = ResidualSR(payload.get("features", 48), payload.get("blocks", 8))
    else:
        model = build_model()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def utm_epsg(longitude, latitude):
    """EPSG code of the UTM zone containing a WGS84 point."""
    zone = min(int((longitude + 180) // 6) + 1, 60)
    return (32600 if latitude >= 0 else 32700) + zone


def snapped_grid(bounds, pixel=LR_PIXEL_M):
    """Largest grid of ``pixel``-sized cells inside ``bounds`` with its origin on a multiple of ``pixel``.

    Returns (left, top, width, height). Snapping keeps every date of a time series,
    and the 3 m outputs, on one shared pixel lattice.
    """
    left, bottom, right, top = bounds
    grid_left = math.ceil(left / pixel) * pixel
    grid_top = math.floor(top / pixel) * pixel
    width = int(math.floor((right - grid_left) / pixel))
    height = int(math.floor((grid_top - bottom) / pixel))
    if width < 1 or height < 1:
        raise ValueError(f"input is smaller than one {pixel:g} m pixel")
    return grid_left, grid_top, width, height


def normalize(values):
    """Reflectance or 0..10000 digital numbers -> 0..1, non-finite kept as NaN."""
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if finite.any() and float(values[finite].max()) > 20:
        values = values / SCALE_FACTOR
    values[~np.isfinite(values)] = np.nan
    return np.clip(values, 0, 1)


def read_lr_on_12m_grid(path):
    """Reproject the Blue, Green, Red, NIR bands of a GEE export to a 12 m UTM grid.

    Returns (array (4, h, w) with NaN nodata, transform, crs).
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("input has no CRS")
        if src.count < max(GEE_BANDS):
            raise ValueError(f"input needs at least {max(GEE_BANDS)} bands, found {src.count}")
        source = normalize(src.read(GEE_BANDS))
        if not np.isfinite(source).any():
            raise ValueError("no finite pixels in the Blue, Green, Red, NIR bands")
        if src.crs.is_projected:
            crs, bounds = src.crs, tuple(src.bounds)
        else:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            crs = CRS.from_epsg(utm_epsg((west + east) / 2, (south + north) / 2))
            bounds = transform_bounds(src.crs, crs, *src.bounds)
        left, top, width, height = snapped_grid(bounds)
        transform = from_origin(left, top, LR_PIXEL_M, LR_PIXEL_M)
        lr = np.full((CHANNELS, height, width), np.nan, dtype=np.float32)
        reproject(
            source=source, destination=lr,
            src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
            dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return lr, transform, crs


def super_resolve(model, lr, device="cpu"):
    """Run the model on a (4, h, w) Blue, Green, Red, NIR array with NaN nodata.

    Returns (4, 4h, 4w) in B4, B3, B2, B8 order; pixels under invalid input are NaN.
    """
    import torch

    valid = np.isfinite(lr).all(axis=0)
    with torch.inference_mode():
        sr = model(torch.from_numpy(np.nan_to_num(lr, nan=0.0)).unsqueeze(0).to(device))
    sr = np.clip(sr.squeeze(0).cpu().numpy().astype(np.float32), 0, 1)[list(SR_BAND_ORDER)]
    sr[:, ~np.repeat(np.repeat(valid, FACTOR, axis=0), FACTOR, axis=1)] = np.nan
    return sr


def write_geotiff(path, array, transform, crs, band_names):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", count=array.shape[0], height=array.shape[1], width=array.shape[2],
        dtype="float32", nodata=float("nan"), crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(array)
        dst.descriptions = band_names


def upscale_file(model, input_path, output_dir, device="cpu", min_valid_fraction=0.5):
    """Upscale one patch. Returns the SR path."""
    lr, transform, crs = read_lr_on_12m_grid(input_path)
    valid_fraction = float(np.isfinite(lr).all(axis=0).mean())
    if valid_fraction < min_valid_fraction:
        raise ValueError(f"only {valid_fraction:.0%} valid pixels")
    sr = super_resolve(model, lr, device)
    sr_path = output_dir / "sr" / f"{input_path.stem}_SR.tif"
    write_geotiff(sr_path, sr, transform * transform.scale(1 / FACTOR, 1 / FACTOR), crs, SR_BAND_NAMES)
    write_geotiff(output_dir / "lr12" / f"{input_path.stem}.tif", lr, transform, crs, LR_BAND_NAMES)
    return sr_path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A GeoTIFF or a directory of GeoTIFFs (GEE 8-band export)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=here / "outputs" / "srresnet")
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    args = parser.parse_args()

    import torch

    if args.input.is_dir():
        paths = sorted(args.input.glob("*.tif")) + sorted(args.input.glob("*.tiff"))
    else:
        paths = [args.input]
    if not paths or not paths[0].exists():
        raise SystemExit(f"No GeoTIFF found at {args.input}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_srresnet(args.checkpoint, device)
    print(f"Loaded {args.checkpoint.name} on {device}")

    written = 0
    for path in paths:
        try:
            upscale_file(model, path, args.output_dir, device, args.min_valid_fraction)
        except ValueError as error:
            print(f"SKIP {path.name}: {error}")
            continue
        written += 1
    print(f"Upscaled {written} of {len(paths)} patches into {args.output_dir} ({len(paths) - written} skipped)")
    raise SystemExit(0 if written else 1)


if __name__ == "__main__":
    main()
