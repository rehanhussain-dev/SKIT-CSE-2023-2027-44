"""Train a consistency-constrained 4x super-resolution model on Sentinel-2 / PlanetScope pairs.

This replaces the SRResNet from ``kaggle_srgan_script.ipynb``, whose checkpoint
fails ``validate_sr_consistency.py``. Three design changes address the measured
failures:

1. The network only predicts a correction on top of bicubic upsampling and its
   last layer starts at zero, so training starts from the bicubic baseline
   instead of from noise. There is no BatchNorm.
2. The output is projected so that averaging it back to the input grid returns
   the input exactly. The model can add detail inside a Sentinel-2 pixel but
   cannot change the reflectance of that pixel, so the NDVI time series the
   LSTM reads cannot drift or flatten.
3. PlanetScope targets are first brought to Sentinel-2 radiometry (a per-date,
   per-band linear fit at 12 m), and the checkpoint is chosen by NDVI agreement
   with PlanetScope on a held-out season, not by L1.

    py -3 srgan/train_residual_sr.py --lr-dir srgan/low/LR --hr-dir srgan/high/high

Runs on CPU in a few minutes; the patches are tiny. Writes
``srgan/checkpoints/residual_sr_best.pt``, which ``run_srresnet_inference.py``
loads with ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FACTOR = 4
CHANNELS = 4
SCALE_FACTOR = 10000.0
BAND_NAMES = ("blue", "green", "red", "nir")
LR_BANDS = (1, 2, 3, 4)  # GEE export: B2, B3, B4, B8
HR_BANDS_FALLBACK = (2, 4, 6, 8)  # PlanetScope SuperDove: blue, green, red, nir
RED, NIR = 2, 3  # positions in BAND_NAMES
ARCH = "residual_sr_v1"
MIN_VALID_FRACTION = 0.5
DATE_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, padding_mode="reflect"), nn.PReLU(features),
            nn.Conv2d(features, features, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x):
        return x + 0.2 * self.body(x)


class ResidualSR(nn.Module):
    """Bicubic upsampling plus a learned detail layer, consistent with the input by construction."""

    def __init__(self, features=48, blocks=8):
        super().__init__()
        self.features, self.blocks = features, blocks
        self.head = nn.Conv2d(CHANNELS, features, 3, padding=1, padding_mode="reflect")
        self.body = nn.Sequential(*[Block(features) for _ in range(blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * 4, 3, padding=1, padding_mode="reflect"), nn.PixelShuffle(2),
            nn.PReLU(features),
            nn.Conv2d(features, features * 4, 3, padding=1, padding_mode="reflect"), nn.PixelShuffle(2),
            nn.PReLU(features),
        )
        self.tail = nn.Conv2d(features, CHANNELS, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=FACTOR, mode="bicubic", align_corners=False)
        shallow = self.head(x)
        out = base + self.tail(self.upsample(shallow + self.body(shallow)))
        # Projection: whatever the network adds, each input pixel keeps its value on average.
        return out + F.interpolate(x - F.avg_pool2d(out, FACTOR), scale_factor=FACTOR, mode="nearest")


def save_checkpoint(path, model, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": ARCH, "model_state_dict": model.state_dict(), "features": model.features,
                "blocks": model.blocks, "channels": CHANNELS, "factor": FACTOR, "bands": BAND_NAMES, **extra}, path)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def season_of(name):
    year, month, _ = (int(part) for part in DATE_PATTERN.search(name).groups())
    start = year if month >= 7 else year - 1
    return f"{start}-{start + 1}"


def to_reflectance(raw):
    values = raw.astype(np.float32)
    finite = np.isfinite(values)
    if np.issubdtype(raw.dtype, np.integer) or (finite.any() and float(values[finite].max()) > 20):
        values = values / SCALE_FACTOR
    values[~np.isfinite(values)] = np.nan
    return values


def masked_block_mean(hr, valid):
    """Mean of each FACTOR x FACTOR block over valid pixels; NaN where a block has none."""
    channels, height, width = hr.shape
    shape = (height // FACTOR, FACTOR, width // FACTOR, FACTOR)
    weights = valid.reshape(shape).sum(axis=(1, 3)).astype(np.float32)
    sums = np.where(valid, hr, 0.0).reshape(channels, *shape).sum(axis=(2, 4))
    return np.where(weights > 0, sums / np.maximum(weights, 1), np.nan)


def harmonise(hr, lr, valid):
    """Per-band linear fit that maps PlanetScope reflectance onto Sentinel-2 at the 12 m grid."""
    coarse = masked_block_mean(hr, valid)
    out = hr.copy()
    for band in range(hr.shape[0]):
        keep = np.isfinite(coarse[band]) & np.isfinite(lr[band])
        x, y = coarse[band][keep], lr[band][keep]
        if keep.sum() < 10 or x.std() < 1e-6:
            gain, offset = 1.0, float(np.mean(y) - np.mean(x)) if keep.any() else 0.0
        else:
            gain, offset = np.polyfit(x, y, 1)
            if not 0.3 < gain < 3.0:  # an unstable fit on a flat scene: fall back to an offset
                gain, offset = 1.0, float(y.mean() - x.mean())
        out[band] = gain * hr[band] + offset
    return out


def prepare_pair(lr_path, hr_path):
    """Co-register one pair. Returns LR (4,h,w), harmonised HR (4,4h,4w), raw HR, mask (4h,4w)."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    with rasterio.open(hr_path) as src:
        names = [(d or "").lower() for d in src.descriptions]
        bands = tuple(names.index(n) + 1 for n in BAND_NAMES) if all(n in names for n in BAND_NAMES) \
            else HR_BANDS_FALLBACK
        raw = src.read(bands)
        crs, t = src.crs, src.transform
        nodata = src.nodata if src.nodata is not None else 0
    height, width = (raw.shape[1] // FACTOR) * FACTOR, (raw.shape[2] // FACTOR) * FACTOR
    raw = raw[:, :height, :width]
    valid = np.all(raw != nodata, axis=0)
    hr = to_reflectance(raw)

    lr = np.full((CHANNELS, height // FACTOR, width // FACTOR), np.nan, dtype=np.float32)
    with rasterio.open(lr_path) as src:
        source = to_reflectance(src.read(LR_BANDS))
        if not np.isfinite(source).any():
            raise ValueError("LR contains no finite pixels")
        reproject(source=source, destination=lr, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=Affine(t.a * FACTOR, t.b, t.c, t.d, t.e * FACTOR, t.f),
                  dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    lr = np.clip(lr, 0, 1)
    lr_valid = np.isfinite(lr).all(axis=0)
    valid &= np.isfinite(hr).all(axis=0) & np.repeat(np.repeat(lr_valid, FACTOR, 0), FACTOR, 1)
    if valid.mean() < MIN_VALID_FRACTION:
        raise ValueError(f"only {valid.mean():.0%} valid pixels")
    target = harmonise(hr, lr, valid)
    return (np.nan_to_num(lr, nan=0.0), np.nan_to_num(target, nan=0.0).astype(np.float32),
            np.nan_to_num(hr, nan=0.0), valid)


def load_samples(lr_dir, hr_dir):
    lr_files = {p.stem: p for p in Path(lr_dir).rglob("*.tif")}
    hr_files = {p.stem: p for p in Path(hr_dir).rglob("*.tif")}
    samples = []
    for name in sorted(set(lr_files) & set(hr_files)):
        try:
            samples.append((name, *prepare_pair(lr_files[name], hr_files[name])))
        except ValueError as error:
            print(f"  skipped {name}: {error}")
    return samples


def random_crop(sample, size, rng):
    """Random LR-aligned crop with flips and a transpose, so position in the patch carries no signal."""
    _, lr, target, _, valid = sample
    top = rng.randrange(lr.shape[1] - size + 1)
    left = rng.randrange(lr.shape[2] - size + 1)
    big = slice(top * FACTOR, (top + size) * FACTOR), slice(left * FACTOR, (left + size) * FACTOR)
    lr = lr[:, top:top + size, left:left + size]
    target, valid = target[:, big[0], big[1]], valid[big[0], big[1]]
    if rng.random() < 0.5:
        lr, target, valid = lr[:, :, ::-1], target[:, :, ::-1], valid[:, ::-1]
    if rng.random() < 0.5:
        lr, target, valid = lr[:, ::-1], target[:, ::-1], valid[::-1]
    if rng.random() < 0.5:
        lr, target, valid = lr.transpose(0, 2, 1), target.transpose(0, 2, 1), valid.T
    return lr.copy(), target.copy(), valid.astype(np.float32)[None].copy()


# --------------------------------------------------------------------------
# Loss and evaluation
# --------------------------------------------------------------------------
def ndvi_t(x):
    return (x[:, NIR] - x[:, RED]) / (x[:, NIR] + x[:, RED]).clamp_min(1e-3)


def loss_fn(prediction, target, mask, ndvi_weight=0.5, gradient_weight=0.5):
    bands = (torch.abs(prediction - target) * mask).sum() / (mask.sum() * CHANNELS).clamp_min(1)
    m = mask[:, 0]
    index = (torch.abs(ndvi_t(prediction) - ndvi_t(target)) * m).sum() / m.sum().clamp_min(1)
    # Edges: match horizontal and vertical NDVI differences where both pixels are valid.
    p, t = ndvi_t(prediction), ndvi_t(target)
    mx, my = m[:, :, 1:] * m[:, :, :-1], m[:, 1:] * m[:, :-1]
    gx = (torch.abs((p[:, :, 1:] - p[:, :, :-1]) - (t[:, :, 1:] - t[:, :, :-1])) * mx).sum() / mx.sum().clamp_min(1)
    gy = (torch.abs((p[:, 1:] - p[:, :-1]) - (t[:, 1:] - t[:, :-1])) * my).sum() / my.sum().clamp_min(1)
    return bands + ndvi_weight * index + gradient_weight * (gx + gy)


def ndvi_np(x):
    total = x[NIR] + x[RED]
    return np.where(total > 1e-6, (x[NIR] - x[RED]) / np.maximum(total, 1e-6), np.nan)


def pearson(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or a[keep].std() == 0 or b[keep].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


@torch.no_grad()
def evaluate(model, samples):
    """NDVI correlation with the untouched PlanetScope image, model versus bicubic, per date."""
    model.eval()
    rows = []
    for name, lr, _, hr_raw, valid in samples:
        x = torch.from_numpy(lr)[None]
        sr = model(x)[0].clamp(0, 1).numpy()
        bicubic = F.interpolate(x, scale_factor=FACTOR, mode="bicubic", align_corners=False)[0].clamp(0, 1).numpy()
        reference = ndvi_np(hr_raw)
        rows.append((name, pearson(ndvi_np(sr)[valid], reference[valid]),
                     pearson(ndvi_np(bicubic)[valid], reference[valid])))
    return rows


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr-dir", type=Path, default=here / "low" / "LR")
    parser.add_argument("--hr-dir", type=Path, default=here / "high" / "high")
    parser.add_argument("--output", type=Path, default=here / "checkpoints" / "residual_sr_best.pt")
    parser.add_argument("--validation-season", default="2024-2025",
                        help="Whole season kept out of training and used to choose the checkpoint")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=8, help="Training crop size in LR pixels")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    samples = load_samples(args.lr_dir, args.hr_dir)
    train = [s for s in samples if season_of(s[0]) != args.validation_season]
    held_out = [s for s in samples if season_of(s[0]) == args.validation_season]
    if not train or not held_out:
        raise SystemExit(f"Need pairs both inside and outside season {args.validation_season}")
    print(f"usable pairs: {len(samples)} | train: {len(train)} | held-out season {args.validation_season}: {len(held_out)}")

    model = ResidualSR()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            batch = [random_crop(rng.choice(train), args.crop, rng) for _ in range(args.batch_size)]
            lr, target, mask = (torch.from_numpy(np.stack(items)) for items in zip(*batch))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(lr), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        rows = evaluate(model, held_out)
        r_sr = float(np.nanmedian([row[1] for row in rows]))
        r_bicubic = float(np.nanmedian([row[2] for row in rows]))
        improved = r_sr > best
        if improved:
            best = r_sr
            save_checkpoint(args.output, model, epoch=epoch, validation_season=args.validation_season,
                            validation_ndvi_r=r_sr, validation_ndvi_r_bicubic=r_bicubic)
        if improved or epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss={total / args.steps_per_epoch:.5f} held-out NDVI r: "
                  f"model {r_sr:.4f} bicubic {r_bicubic:.4f}" + ("  * best" if improved else ""), flush=True)
    print(f"best held-out NDVI r {best:.4f} vs bicubic {r_bicubic:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
