"""Validate super-resolved patches before they feed the temporal LSTM branch.

For every low-resolution (LR, 10 m) patch with a matching ``_SR.tif`` this
script checks:

1. Georeferencing  - same CRS and footprint, pixel size exactly LR / 4.
2. Radiometry      - SR averaged back to 10 m must reproduce the LR bands.
3. Sharpening      - NDVI edge strength of SR versus plain bicubic upsampling.
4. Reference match - where a PlanetScope (3 m) patch exists for the same date,
                     NDVI agreement of SR and of bicubic with that reference,
                     over the whole overlap and at field boundaries only.
5. LSTM inputs     - the per-date NDVI mean series computed from SR must track
                     the LR series the LSTM is trained on, season by season.

    py -3 srgan/validate_sr_consistency.py --lr-dir srgan/low/LR \
        --sr-dir srgan/outputs/sr --hr-dir srgan/high/high

Writes ``validation_per_date.csv``, ``validation_report.json`` and (when
matplotlib is installed) ``validation_summary.png`` to ``--output-dir``.
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_srgan_inference import FACTOR, MODEL_BAND_INDICES, read_model_input  # noqa: E402

RED, NIR = 0, 3  # positions in the model band order B4, B3, B2, B8
DATE_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})")


# --------------------------------------------------------------------------
# Array-level metrics (no GeoTIFF dependency, covered by the unit tests)
# --------------------------------------------------------------------------
def block_mean(arr, factor=FACTOR):
    """Average (C, H*f, W*f) down to (C, H, W), ignoring NaN."""
    channels, height, width = arr.shape
    blocks = arr.reshape(channels, height // factor, factor, width // factor, factor)
    return np.nanmean(blocks, axis=(2, 4))


def block_std(image, factor=FACTOR):
    """Mean standard deviation inside each factor x factor block of a 2-D image."""
    height, width = image.shape
    blocks = image.reshape(height // factor, factor, width // factor, factor)
    return float(np.nanmean(np.nanstd(blocks, axis=(1, 3))))


def ndvi(bands):
    red, nir = bands[RED], bands[NIR]
    total = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (nir - red) / total, np.nan)


def pearson(a, b):
    """Pearson correlation over pixels that are finite in both arrays."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or a[keep].std() == 0 or b[keep].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def gradient_magnitude(image):
    """Per-pixel gradient magnitude; NaN wherever a neighbour is NaN."""
    rows, cols = np.gradient(image)
    return np.hypot(rows, cols)


def edge_strength(image, mask=None):
    """Steepness of the strongest edges: 95th percentile of gradient magnitude.

    A mean would not do: for a transition of fixed contrast the mean gradient is
    the same whether the edge spans one pixel or ten.
    """
    magnitude = gradient_magnitude(image)
    if mask is not None:
        magnitude = magnitude[mask]
    return float(np.nanpercentile(magnitude, 95)) if np.isfinite(magnitude).any() else float("nan")


def radiometric_consistency(lr, sr, factor=FACTOR):
    """Compare SR averaged back to the LR grid with the LR bands themselves."""
    degraded = block_mean(sr, factor)
    difference = degraded - lr
    return {
        "rmse": float(np.sqrt(np.nanmean(difference**2))),
        "bias": float(np.nanmean(difference)),
        "r": pearson(degraded, lr),
    }


def bicubic_upsample(arr, factor=FACTOR):
    """Bicubic baseline: what you get from 10 m data with no learned model."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
    up = F.interpolate(tensor, scale_factor=factor, mode="bicubic", align_corners=False)
    return up.squeeze(0).clamp(0, 1).numpy()


def season_of(date):
    """Rabi season label used by the LSTM pipeline, e.g. 2022-11-05 -> 2022-2023."""
    year, month = int(date[:4]), int(date[5:7])
    start = year if month >= 7 else year - 1
    return f"{start}-{start + 1}"


def lstm_series_consistency(rows):
    """Per-season agreement between the SR and LR NDVI-mean time series."""
    seasons = {}
    for row in rows:
        seasons.setdefault(row["season"], []).append(row)
    result = {}
    for season, items in sorted(seasons.items()):
        items.sort(key=lambda item: item["date"])
        lr = np.array([item["ndvi_mean_lr"] for item in items])
        sr = np.array([item["ndvi_mean_sr"] for item in items])
        result[season] = {
            "observations": len(items),
            "r": pearson(lr, sr),
            "max_abs_diff": float(np.max(np.abs(sr - lr))),
            "mean_diff": float(np.mean(sr - lr)),
        }
    return result


# --------------------------------------------------------------------------
# GeoTIFF-level checks
# --------------------------------------------------------------------------
def georeference_issues(lr_src, sr_src, factor=FACTOR):
    """Return a list of problems with how the SR grid sits on the LR grid."""
    issues = []
    if lr_src.crs != sr_src.crs:
        issues.append(f"CRS differs: {lr_src.crs} vs {sr_src.crs}")
    if (sr_src.height, sr_src.width) != (lr_src.height * factor, lr_src.width * factor):
        issues.append(f"size is {sr_src.width}x{sr_src.height}, expected {factor}x the input")
    tolerance = abs(lr_src.res[0]) * 1e-6
    if any(abs(a - b) > tolerance for a, b in zip(lr_src.bounds, sr_src.bounds)):
        issues.append("footprint (bounds) differs from the input")
    if any(abs(l / s - factor) > 1e-6 for l, s in zip(lr_src.res, sr_src.res)):
        issues.append(f"pixel size ratio is not {factor}")
    return issues


def read_reference_ndvi(hr_path, sr_src):
    """Resample the PlanetScope red/NIR bands onto the SR grid and return NDVI."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(hr_path) as hr:
        names = [name.lower() if name else "" for name in hr.descriptions]
        if "red" not in names or "nir" not in names:
            return None
        out = np.full((2, sr_src.height, sr_src.width), np.nan, dtype=np.float32)
        for index, name in enumerate(("red", "nir")):
            reproject(
                source=rasterio.band(hr, names.index(name) + 1),
                destination=out[index],
                src_nodata=hr.nodata if hr.nodata is not None else 0,
                dst_transform=sr_src.transform,
                dst_crs=sr_src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    red, nir = out
    total = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(total) & (total > 0), (nir - red) / total, np.nan)


def validate_pair(lr_path, sr_path, hr_path=None):
    """Compute all per-date metrics for one LR/SR pair."""
    import rasterio

    match = DATE_PATTERN.search(lr_path.stem)
    date = "-".join(match.groups()) if match else lr_path.stem

    with rasterio.open(lr_path) as lr_src, rasterio.open(sr_path) as sr_src:
        issues = georeference_issues(lr_src, sr_src)
        valid = np.isfinite(lr_src.read(MODEL_BAND_INDICES)).all(axis=0)
        lr = read_model_input(lr_src)
        lr[:, ~valid] = np.nan
        sr = sr_src.read().astype(np.float32)
        reference = read_reference_ndvi(hr_path, sr_src) if hr_path and not issues else None

    row = {"date": date, "season": season_of(date) if match else "", "file": lr_path.name,
           "valid_fraction": float(valid.mean()), "georef_issues": "; ".join(issues)}
    if issues:
        return row

    bicubic = bicubic_upsample(np.nan_to_num(lr, nan=0.0))
    bicubic[:, ~np.isfinite(sr[0])] = np.nan
    ndvi_lr, ndvi_sr, ndvi_bicubic = ndvi(lr), ndvi(sr), ndvi(bicubic)

    row.update({f"downsampled_{key}": value for key, value in radiometric_consistency(lr, sr).items()})
    row.update(
        ndvi_mean_lr=float(np.nanmean(ndvi_lr)),
        ndvi_mean_sr=float(np.nanmean(ndvi_sr)),
        ndvi_std_lr=float(np.nanstd(ndvi_lr)),
        ndvi_std_sr=float(np.nanstd(ndvi_sr)),
        edge_bicubic=edge_strength(ndvi_bicubic),
        edge_sr=edge_strength(ndvi_sr),
        subpixel_std_bicubic=block_std(ndvi_bicubic),
        subpixel_std_sr=block_std(ndvi_sr),
    )
    row["ndvi_mean_diff"] = row["ndvi_mean_sr"] - row["ndvi_mean_lr"]
    row["edge_ratio"] = row["edge_sr"] / row["edge_bicubic"] if row["edge_bicubic"] else float("nan")

    if reference is not None and np.isfinite(reference).mean() > 0.05:
        overlap = np.isfinite(reference) & np.isfinite(ndvi_sr)
        reference_edges = gradient_magnitude(reference)
        finite_edges = reference_edges[overlap & np.isfinite(reference_edges)]
        if finite_edges.size:
            boundary = overlap & (reference_edges >= np.percentile(finite_edges, 75))
            row.update(
                ref_r_sr=pearson(ndvi_sr[overlap], reference[overlap]),
                ref_r_bicubic=pearson(ndvi_bicubic[overlap], reference[overlap]),
                ref_boundary_r_sr=pearson(ndvi_sr[boundary], reference[boundary]),
                ref_boundary_r_bicubic=pearson(ndvi_bicubic[boundary], reference[boundary]),
                ref_edge=edge_strength(reference, overlap),
                ref_edge_sr=edge_strength(ndvi_sr, overlap),
                ref_edge_bicubic=edge_strength(ndvi_bicubic, overlap),
            )
    return row


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def column(rows, key):
    values = np.array([row[key] for row in rows if key in row and row[key] == row[key]], dtype=np.float64)
    return values


def summarise(rows, thresholds):
    """Aggregate per-date rows into a report with pass/fail checks."""
    checked = [row for row in rows if not row["georef_issues"]]
    bad_georef = [row["file"] for row in rows if row["georef_issues"]]
    seasons = lstm_series_consistency(checked)
    # A correlation needs a few points to mean anything.
    season_r = [s["r"] for s in seasons.values() if s["observations"] >= 5 and s["r"] == s["r"]]
    paired = [row for row in checked if "ref_r_sr" in row]

    rmse, ndvi_diff, edge_ratio = (column(checked, key) for key in ("downsampled_rmse", "ndvi_mean_diff", "edge_ratio"))
    report = {
        "patches_validated": len(rows),
        "thresholds": thresholds,
        "georeferencing": {"patches_with_issues": bad_georef},
        "radiometric_consistency": {
            "median_rmse": float(np.median(rmse)),
            "max_rmse": float(rmse.max()),
            "median_bias": float(np.median(column(checked, "downsampled_bias"))),
            "min_r": float(column(checked, "downsampled_r").min()),
        },
        "sharpening": {
            "median_edge_ratio_sr_vs_bicubic": float(np.median(edge_ratio)),
            "fraction_dates_sharper_than_bicubic": float((edge_ratio > 1).mean()),
            "median_subpixel_ndvi_std_sr": float(np.median(column(checked, "subpixel_std_sr"))),
            "median_subpixel_ndvi_std_bicubic": float(np.median(column(checked, "subpixel_std_bicubic"))),
        },
        "lstm_inputs": {
            "max_abs_ndvi_mean_diff": float(np.abs(ndvi_diff).max()),
            "mean_ndvi_mean_diff": float(ndvi_diff.mean()),
            "min_season_r": float(min(season_r)) if season_r else float("nan"),
            "seasons": seasons,
            "note": "The SR model outputs B4, B3, B2, B8 only. NDVI can be derived at 2.5 m; "
                    "NDRE (needs B5) and SI (needs B11) cannot, so those LSTM features still "
                    "come from the native-resolution bands.",
        },
    }
    checks = {
        "georeferencing": not bad_georef,
        "radiometric_consistency": bool(rmse.max() <= thresholds["max_rmse"]),
        "sharper_than_bicubic": bool(np.median(edge_ratio) > 1),
        "lstm_ndvi_series": bool(
            np.abs(ndvi_diff).max() <= thresholds["max_ndvi_mean_diff"]
            and season_r and min(season_r) >= thresholds["min_season_r"]
        ),
    }
    if paired:
        keys = ("ref_r_sr", "ref_r_bicubic", "ref_boundary_r_sr", "ref_boundary_r_bicubic",
                "ref_edge", "ref_edge_sr", "ref_edge_bicubic")
        medians = {key: float(np.median(column(paired, key))) for key in keys}
        report["planet_reference"] = {
            "paired_dates": len(paired),
            "median_ndvi_r_sr": medians["ref_r_sr"],
            "median_ndvi_r_bicubic": medians["ref_r_bicubic"],
            "median_boundary_ndvi_r_sr": medians["ref_boundary_r_sr"],
            "median_boundary_ndvi_r_bicubic": medians["ref_boundary_r_bicubic"],
            "median_edge_strength_reference": medians["ref_edge"],
            "median_edge_strength_sr": medians["ref_edge_sr"],
            "median_edge_strength_bicubic": medians["ref_edge_bicubic"],
            "fraction_dates_sr_beats_bicubic": float(
                (column(paired, "ref_r_sr") > column(paired, "ref_r_bicubic")).mean()
            ),
        }
        checks["matches_planet_at_least_as_well_as_bicubic"] = bool(
            medians["ref_r_sr"] >= medians["ref_r_bicubic"]
            and medians["ref_boundary_r_sr"] >= medians["ref_boundary_r_bicubic"]
        )
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def write_csv(rows, path):
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = sorted((row for row in rows if "ndvi_mean_sr" in row), key=lambda row: row["date"])
    index = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(index, [row["ndvi_mean_lr"] for row in rows], label="Sentinel-2 10 m", linewidth=2)
    axes[0].plot(index, [row["ndvi_mean_sr"] for row in rows], "--", label="SRGAN 2.5 m", linewidth=1.5)
    axes[0].set_ylabel("Patch mean NDVI")
    axes[0].set_title("NDVI series fed to the LSTM: native vs super-resolved")
    axes[0].legend()
    axes[1].plot(index, [row["downsampled_rmse"] for row in rows], color="tab:red")
    axes[1].set_ylabel("RMSE (reflectance)")
    axes[1].set_title("SR averaged back to 10 m vs original bands")
    axes[2].plot(index, [row["edge_ratio"] for row in rows], color="tab:green")
    axes[2].axhline(1.0, color="grey", linewidth=1)
    axes[2].set_ylabel("Edge strength ratio")
    axes[2].set_title("NDVI edge strength, SRGAN / bicubic (above 1 = sharper)")
    step = max(1, len(rows) // 12)
    axes[2].set_xticks(index[::step])
    axes[2].set_xticklabels([row["date"] for row in rows][::step], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr-dir", type=Path, default=here / "low" / "LR")
    parser.add_argument("--sr-dir", type=Path, default=here / "outputs" / "sr")
    parser.add_argument("--hr-dir", type=Path, default=None, help="Optional PlanetScope reference patches")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--max-rmse", type=float, default=0.01,
                        help="Largest allowed RMSE (0-1 reflectance) of SR averaged back to 10 m")
    parser.add_argument("--max-ndvi-mean-diff", type=float, default=0.02)
    parser.add_argument("--min-season-r", type=float, default=0.99)
    args = parser.parse_args()

    # Fully masked blocks legitimately average to NaN; numpy warns about each one.
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice|Degrees of freedom")

    rows = []
    for sr_path in sorted(args.sr_dir.glob("*_SR.tif")):
        lr_path = args.lr_dir / f"{sr_path.stem[:-3]}.tif"
        if not lr_path.exists():
            print(f"SKIP {sr_path.name}: no matching LR patch")
            continue
        hr_path = args.hr_dir / lr_path.name if args.hr_dir else None
        rows.append(validate_pair(lr_path, sr_path, hr_path if hr_path and hr_path.exists() else None))
    if not rows:
        raise SystemExit(f"No LR/SR pairs found in {args.lr_dir} and {args.sr_dir}")

    thresholds = {"max_rmse": args.max_rmse, "max_ndvi_mean_diff": args.max_ndvi_mean_diff,
                  "min_season_r": args.min_season_r}
    report = summarise(rows, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "validation_per_date.csv")
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_figure(rows, args.output_dir / "validation_summary.png")

    print(json.dumps({key: value for key, value in report.items() if key != "lstm_inputs"}, indent=2))
    print("lstm_inputs:", json.dumps({k: v for k, v in report["lstm_inputs"].items() if k != "seasons"}, indent=2))
    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
