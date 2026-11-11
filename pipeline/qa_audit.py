"""
qa_audit.py
-----------
User Story 3 - Weeks 1 & 2: Structural and Radiometric Patch Quality Audit.
Inspects file integrity, exact dimensions, coordinate systems, surface reflectance
ranges, vegetation index bounds, and nodata/NaN boundary behavior.
"""

import os
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import rasterio

EXPECTED_WIDTH = 128
EXPECTED_HEIGHT = 128
EXPECTED_BANDS = 8


def audit_single_patch(filepath: str, is_planet: bool = False):
    """Audits structural integrity, band values, indices, and boundary metrics."""
    issues = []
    metadata = {}
    stats = {}

    try:
        with rasterio.open(filepath) as src:
            w, h, c = src.width, src.height, src.count
            dtype = str(src.dtypes[0])
            crs = str(src.crs) if src.crs else None
            transform = src.transform
            nodata_val = src.nodata if src.nodata is not None else 0.0

            metadata = {
                "file": os.path.basename(filepath),
                "shape": (c, h, w),
                "dtype": dtype,
                "crs": crs,
                "nodata": nodata_val,
            }

            # --- Week 1: Structural Checks ---
            if w != EXPECTED_WIDTH or h != EXPECTED_HEIGHT:
                issues.append(f"Invalid dimensions: {w}x{h} (expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")
            if c != EXPECTED_BANDS:
                issues.append(f"Invalid band count: {c} (expected {EXPECTED_BANDS})")
            if crs is None:
                issues.append("Missing CRS metadata")
            if transform.is_identity:
                issues.append("Identity/unset affine transform")

            # Read full array
            data = src.read()
            if data.size == 0:
                issues.append("Empty raster array")
                return False, issues, metadata, stats

            # --- Week 2: Radiometric & Spectral Checks ---
            # 1. Check for NaN or Inf leakage
            nan_count = int(np.isnan(data).sum())
            inf_count = int(np.isinf(data).sum())
            if nan_count > 0:
                issues.append(f"Contains {nan_count} NaN values")
            if inf_count > 0:
                issues.append(f"Contains {inf_count} Inf values")

            # Valid mask: exclude zero-padded boundary
            valid_mask = data != nodata_val
            if not np.any(valid_mask):
                issues.append("Patch contains 100% nodata/padding")
                return False, issues, metadata, stats

            # Optical Surface Reflectance checks: Bands 1-5 (S2) or Bands 1-8 (Planet SR)
            # S2 stack: [B2, B3, B4, B8, B11, NDVI, NDRE, SI]
            optical_bands = slice(0, 5) if not is_planet else slice(0, 8)
            optical_data = data[optical_bands]
            optical_valid = valid_mask[optical_bands]

            if np.any(optical_valid):
                opt_min = float(np.min(optical_data[optical_valid]))
                opt_max = float(np.max(optical_data[optical_valid]))

                stats["optical_min"] = opt_min
                stats["optical_max"] = opt_max

                # Scale check: If Planet is stored as 16-bit Surface Reflectance (scale factor 10,000)
                if is_planet and opt_max > 1.5:
                    if opt_min < 0 or opt_max > 12000:
                        issues.append(f"Planet raw digital number out of bounds: [{opt_min}, {opt_max}]")
                else:
                    if opt_min < -0.05:
                        issues.append(f"Negative surface reflectance: {opt_min:.4f}")
                    if opt_max > 1.20:
                        issues.append(f"Severe cloud saturation / flare: {opt_max:.4f}")

            # S2 Index checks: Band 6 (NDVI), Band 7 (NDRE), Band 8 (SI)
            if not is_planet and c >= 8:
                ndvi_vals = data[5][valid_mask[5]]
                ndre_vals = data[6][valid_mask[6]]
                si_vals = data[7][valid_mask[7]]

                if len(ndvi_vals) > 0:
                    ndvi_min, ndvi_max = float(np.min(ndvi_vals)), float(np.max(ndvi_vals))
                    stats["ndvi_range"] = [ndvi_min, ndvi_max]
                    if ndvi_min < -1.0 or ndvi_max > 1.0:
                        issues.append(f"NDVI out of theoretical bounds [-1, 1]: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

                if len(ndre_vals) > 0:
                    ndre_min, ndre_max = float(np.min(ndre_vals)), float(np.max(ndre_vals))
                    stats["ndre_range"] = [ndre_min, ndre_max]
                    if ndre_min < -1.0 or ndre_max > 1.0:
                        issues.append(f"NDRE out of theoretical bounds [-1, 1]: [{ndre_min:.3f}, {ndre_max:.3f}]")

                if len(si_vals) > 0:
                    si_min, si_max = float(np.min(si_vals)), float(np.max(si_vals))
                    stats["si_range"] = [si_min, si_max]
                    if si_min < 0.0:
                        issues.append(f"Salinity Index (SI) negative values: {si_min:.3f}")

    except Exception as e:
        issues.append(f"Read failure: {str(e)}")

    is_valid = len(issues) == 0
    return is_valid, issues, metadata, stats


def run_week2_audit(s2_dir: str, planet_dir: str, report_output: str):
    print("=" * 68)
    print("USER STORY 3 - WEEK 2: RADIOMETRIC & SPECTRAL RANGE VALIDATION")
    print("=" * 68)

    audit_summary = {
        "sentinel2": {"total": 0, "passed": 0, "flagged": 0, "details": []},
        "planet": {"total": 0, "passed": 0, "flagged": 0, "details": []},
    }

    targets = [
        ("sentinel2", s2_dir, "Sentinel-2 (LR)", False),
        ("planet", planet_dir, "PlanetScope (HR)", True),
    ]

    for key, directory, label, is_planet in targets:
        files = sorted(glob.glob(os.path.join(directory, "*.tif")))
        audit_summary[key]["total"] = len(files)
        print(f"\nAnalyzing {label} radiometric distributions in:\n  {directory}")
        print(f"Patches detected: {len(files)}")

        if not files:
            print("  [WARNING] No patches found.")
            continue

        for fpath in files:
            is_valid, issues, meta, stats = audit_single_patch(fpath, is_planet=is_planet)
            if is_valid:
                audit_summary[key]["passed"] += 1
            else:
                audit_summary[key]["flagged"] += 1
                audit_summary[key]["details"].append({
                    "file": os.path.basename(fpath),
                    "issues": issues,
                    "stats": stats,
                })

        pass_rate = (audit_summary[key]["passed"] / len(files)) * 100 if files else 0
        print(f"  Radiometric Pass Rate: {audit_summary[key]['passed']} / {len(files)} ({pass_rate:.1f}%)")
        if audit_summary[key]["flagged"] > 0:
            print(f"  [FLAGGED] {audit_summary[key]['flagged']} patches with anomalies.")
            for item in audit_summary[key]["details"][:3]:
                print(f"    - {item['file']}: {', '.join(item['issues'])}")
            if len(audit_summary[key]["details"]) > 3:
                print(f"    ... and {len(audit_summary[key]['details']) - 3} more.")

    os.makedirs(os.path.dirname(os.path.abspath(report_output)), exist_ok=True)
    with open(report_output, "w") as jf:
        json.dump(audit_summary, jf, indent=2)

    print("\n" + "=" * 68)
    print(f"Radiometric audit report written to: {report_output}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 2 radiometric and spectral range audit")
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
        default="data/week2_radiometric_report.json",
        help="Output path for JSON summary",
    )
    args = parser.parse_args()

    run_week2_audit(args.s2_dir, args.planet_dir, args.report_output)