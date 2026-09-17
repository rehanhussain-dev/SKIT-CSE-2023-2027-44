"""Review GeoTIFF patches before SRGAN inference."""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def review_patch(path, expected_size=128, expected_bands=8):
    """Return validation issues and summary metadata for one GeoTIFF."""
    issues = []
    with rasterio.open(path) as src:
        data = src.read(masked=True)
        if src.width != expected_size or src.height != expected_size:
            issues.append(f"shape is {src.width}x{src.height}, expected {expected_size}x{expected_size}")
        if src.count != expected_bands:
            issues.append(f"band count is {src.count}, expected {expected_bands}")
        if src.crs is None:
            issues.append("CRS is missing")
        if not src.transform:
            issues.append("affine transform is missing")
        if np.ma.count_masked(data) == data.size:
            issues.append("all pixels are masked")
        values = data.compressed()
        if values.size and (not np.isfinite(values).all()):
            issues.append("contains non-finite values")
        return {
            "path": str(path),
            "shape": f"{src.width}x{src.height}",
            "bands": src.count,
            "crs": str(src.crs),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "issues": issues,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bands", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.directory.glob("*.tif")) + sorted(args.directory.glob("*.tiff"))
    if not paths:
        raise SystemExit(f"No GeoTIFF patches found in {args.directory}")

    failed = 0
    for path in paths:
        result = review_patch(path, args.size, args.bands)
        status = "PASS" if not result["issues"] else "FAIL"
        print(f"{status} {result['path']}: {result['shape']}, {result['bands']} bands, {result['crs']}")
        for issue in result["issues"]:
            print(f"  - {issue}")
        failed += bool(result["issues"])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()