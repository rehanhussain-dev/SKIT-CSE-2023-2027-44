"""
qa_audit.py
-----------
User Story 3 - Week 1: Automated File Integrity & Structural Audit.
Scans Sentinel-2 and PlanetScope 128x128 GeoTIFF training patches to verify
GDAL stream readability, exact dimensions, channel counts, and spatial metadata.
"""

import os
import glob
import json
import argparse
from pathlib import Path
import rasterio

EXPECTED_WIDTH = 128
EXPECTED_HEIGHT = 128
EXPECTED_BANDS = 8


def audit_single_patch(filepath: str):
    """Audits a single raster patch for readability, shape, band count, and metadata."""
    issues = []
    metadata = {}

    try:
        with rasterio.open(filepath) as src:
            # 1. Structural dimensions check
            w, h, c = src.width, src.height, src.count
            dtype = src.dtypes[0]
            crs = str(src.crs) if src.crs else None
            transform = src.transform

            metadata = {
                "file": os.path.basename(filepath),
                "shape": (c, h, w),
                "dtype": dtype,
                "crs": crs,
                "nodata": src.nodata,
            }

            if w != EXPECTED_WIDTH or h != EXPECTED_HEIGHT:
                issues.append(f"Invalid dimensions: {w}x{h} (expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")

            if c != EXPECTED_BANDS:
                issues.append(f"Invalid band count: {c} (expected {EXPECTED_BANDS})")

            if crs is None:
                issues.append("Missing CRS metadata")

            if transform.is_identity:
                issues.append("Unset/Identity affine transform")

            # 2. Complete raster I/O read test
            try:
                data = src.read()
                if data.size == 0:
                    issues.append("Empty raster stream")
            except Exception as read_err:
                issues.append(f"Decompression/Read error: {str(read_err)}")

    except Exception as open_err:
        issues.append(f"File open failure: {str(open_err)}")

    return len(issues) == 0, issues, metadata


def run_week1_audit(s2_dir: str, planet_dir: str, report_output: str):
    print("=" * 65)
    print("USER STORY 3 - WEEK 1: PATCH INTEGRITY & STRUCTURAL AUDIT")
    print("=" * 65)

    audit_summary = {
        "sentinel2": {"total": 0, "passed": 0, "failed": 0, "errors": []},
        "planet": {"total": 0, "passed": 0, "failed": 0, "errors": []},
    }

    targets = [
        ("sentinel2", s2_dir, "Sentinel-2 (LR)"),
        ("planet", planet_dir, "PlanetScope (HR)"),
    ]

    for key, directory, label in targets:
        files = sorted(glob.glob(os.path.join(directory, "*.tif")))
        audit_summary[key]["total"] = len(files)
        print(f"\nScanning {label} in: {directory}")
        print(f"Total patches detected: {len(files)}")

        if not files:
            print("  [WARNING] No patches found in directory.")
            continue

        for fpath in files:
            is_valid, errors, meta = audit_single_patch(fpath)
            if is_valid:
                audit_summary[key]["passed"] += 1
            else:
                audit_summary[key]["failed"] += 1
                audit_summary[key]["errors"].append({
                    "file": os.path.basename(fpath),
                    "issues": errors,
                    "metadata": meta,
                })

        pass_rate = (audit_summary[key]["passed"] / len(files)) * 100 if files else 0
        print(f"  Passed: {audit_summary[key]['passed']} / {len(files)} ({pass_rate:.1f}%)")
        if audit_summary[key]["failed"] > 0:
            print(f"  [ALERT] {audit_summary[key]['failed']} patches failed structural inspection!")

    # Save audit report
    os.makedirs(os.path.dirname(os.path.abspath(report_output)), exist_ok=True)
    with open(report_output, "w") as jf:
        json.dump(audit_summary, jf, indent=2)

    print("\n" + "=" * 65)
    print(f"Audit report saved to: {report_output}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 1 structural and integrity audit")
    parser.add_argument(
        "--s2_dir",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "patches" / "s2_128"),
        help="Path to Sentinel-2 patches",
    )
    parser.add_argument(
        "--planet_dir",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "patches" / "planet_128"),
        help="Path to PlanetScope patches",
    )
    parser.add_argument(
        "--report_output",
        type=str,
        default="data/week1_integrity_report.json",
        help="Path for saving JSON audit summary",
    )
    args = parser.parse_args()

    run_week1_audit(args.s2_dir, args.planet_dir, args.report_output)