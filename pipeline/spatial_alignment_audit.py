"""
spatial_alignment_audit.py
--------------------------
User Story 3 - Week 3: Spatial Alignment, CRS Validation & Temporal Pairing.
Verifies UTM Zone 43N (EPSG:32643) consistency, checks geographic centroid alignment
of active non-zero farm pixels between S2 (10m) and PlanetScope (3m) patches,
and pairs HR targets with nearest LR observations within a max day tolerance.
"""

import os
import re
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

EXPECTED_CRS = "EPSG:32643"
MAX_CENTROID_OFFSET_M = 15.0  # Max acceptable geographic drift in meters (1.5 S2 pixels)
MAX_TEMPORAL_GAP_DAYS = 3     # Max days between S2 LR and Planet HR capture


def extract_date(filename: str):
    match = re.search(r"(\d{8})", os.path.basename(filename))
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    return None


def inspect_spatial_footprint(tif_path: str):
    """Extracts CRS, resolution, and geographic centroid of non-zero farm pixels."""
    with rasterio.open(tif_path) as src:
        crs = str(src.crs) if src.crs else None
        res_x, res_y = src.res
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else 0.0
        data = src.read(1)  # Check Band 1 mask

        # Locate active farm pixels (non-padded interior)
        active_rows, active_cols = np.where((data != nodata) & (data != 0))
        if len(active_rows) == 0:
            return None

        mean_row = float(np.mean(active_rows))
        mean_col = float(np.mean(active_cols))

        # Convert pixel center to projected UTM coordinates (Easting, Northing)
        easting, northing = rasterio.transform.xy(transform, mean_row, mean_col)

        return {
            "file": os.path.basename(tif_path),
            "path": tif_path,
            "date": extract_date(tif_path),
            "crs": crs,
            "res_m": round(float(res_x), 2),
            "active_pixels": int(len(active_rows)),
            "easting": float(easting),
            "northing": float(northing),
        }


def run_week3_audit(s2_dir: str, planet_dir: str, report_out: str, pairs_csv_out: str):
    print("=" * 72)
    print("USER STORY 3 - WEEK 3: SPATIAL ALIGNMENT & TEMPORAL PAIRING AUDIT")
    print("=" * 72)

    s2_files = sorted(glob.glob(os.path.join(s2_dir, "*.tif")))
    planet_files = sorted(glob.glob(os.path.join(planet_dir, "*.tif")))

    s2_records = [r for f in s2_files if (r := inspect_spatial_footprint(f)) is not None]
    planet_records = [r for f in planet_files if (r := inspect_spatial_footprint(f)) is not None]

    # 1. CRS & Resolution Check
    s2_crs_issues = [r["file"] for r in s2_records if r["crs"] != EXPECTED_CRS]
    planet_crs_issues = [r["file"] for r in planet_records if r["crs"] != EXPECTED_CRS]

    print(f"\n[1] CRS & Projection Audit ({EXPECTED_CRS}):")
    print(f"  Sentinel-2 ({len(s2_records)} patches): {len(s2_records) - len(s2_crs_issues)}/{len(s2_records)} valid CRS")
    print(f"  PlanetScope ({len(planet_records)} patches): {len(planet_records) - len(planet_crs_issues)}/{len(planet_records)} valid CRS")

    # 2. Geographic Centroid Alignment Across Sensors
    s2_mean_e = np.mean([r["easting"] for r in s2_records])
    s2_mean_n = np.mean([r["northing"] for r in s2_records])
    pl_mean_e = np.mean([r["easting"] for r in planet_records])
    pl_mean_n = np.mean([r["northing"] for r in planet_records])

    global_offset_m = float(np.hypot(s2_mean_e - pl_mean_e, s2_mean_n - pl_mean_n))

    print("\n[2] Active Farm Centroid Registration:")
    print(f"  S2 Mean Farm Centroid:     ({s2_mean_e:.2f} E, {s2_mean_n:.2f} N)")
    print(f"  Planet Mean Farm Centroid: ({pl_mean_e:.2f} E, {pl_mean_n:.2f} N)")
    print(f"  Inter-Sensor Offset:       {global_offset_m:.2f} meters (Threshold: <= {MAX_CENTROID_OFFSET_M}m)")

    # 3. Temporal Nearest-Neighbor Pairing (HR -> LR)
    paired_rows = []
    unmatched_hr = []

    for pl in planet_records:
        if pl["date"] is None:
            continue

        # Find closest S2 patch by date
        best_s2 = min(s2_records, key=lambda s: abs((s["date"] - pl["date"]).days))
        day_diff = abs((best_s2["date"] - pl["date"]).days)
        spatial_offset_m = float(np.hypot(best_s2["easting"] - pl["easting"], best_s2["northing"] - pl["northing"]))

        if day_diff <= MAX_TEMPORAL_GAP_DAYS and spatial_offset_m <= MAX_CENTROID_OFFSET_M:
            yr = pl["date"].year
            rabi_season = f"{yr}-{yr + 1}" if pl["date"].month >= 10 else f"{yr - 1}-{yr}"

            paired_rows.append({
                "season": rabi_season,
                "planet_date": pl["date"].strftime("%Y-%m-%d"),
                "s2_date": best_s2["date"].strftime("%Y-%m-%d"),
                "delta_days": day_diff,
                "centroid_offset_m": round(spatial_offset_m, 2),
                "lr_s2_path": best_s2["path"],
                "hr_planet_path": pl["path"],
            })
        else:
            unmatched_hr.append({
                "planet_file": pl["file"],
                "planet_date": pl["date"].strftime("%Y-%m-%d"),
                "nearest_s2_date": best_s2["date"].strftime("%Y-%m-%d"),
                "delta_days": day_diff,
                "centroid_offset_m": round(spatial_offset_m, 2),
            })

    df_pairs = pd.DataFrame(paired_rows)
    os.makedirs(os.path.dirname(os.path.abspath(pairs_csv_out)), exist_ok=True)
    df_pairs.to_csv(pairs_csv_out, index=False)

    print(f"\n[3] LR-HR Temporal & Spatial Pairing (<= {MAX_TEMPORAL_GAP_DAYS} days, <= {MAX_CENTROID_OFFSET_M}m):")
    print(f"  Matched SRGAN Pairs: {len(df_pairs)} / {len(planet_records)} HR patches")
    print(f"  Unmatched HR Scenes: {len(unmatched_hr)} (exceeded temporal or spatial window)")
    if not df_pairs.empty:
        print(f"  Mean Temporal Gap:   {df_pairs['delta_days'].mean():.2f} days")
        print(f"  Mean Spatial Offset: {df_pairs['centroid_offset_m'].mean():.2f} meters")

    summary = {
        "expected_crs": EXPECTED_CRS,
        "s2_crs_issues": s2_crs_issues,
        "planet_crs_issues": planet_crs_issues,
        "global_centroid_offset_m": round(global_offset_m, 3),
        "total_hr_candidates": len(planet_records),
        "matched_pairs_count": len(df_pairs),
        "unmatched_hr_count": len(unmatched_hr),
        "unmatched_details": unmatched_hr,
    }

    os.makedirs(os.path.dirname(os.path.abspath(report_out)), exist_ok=True)
    with open(report_out, "w") as jf:
        json.dump(summary, jf, indent=2)

    print("\n" + "=" * 72)
    print(f"Spatial audit saved to:  {report_out}")
    print(f"Candidate pairs CSV to:  {pairs_csv_out}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Week 3 spatial alignment and pairing audit")
    parser.add_argument(
        "--s2_dir",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "patches" / "s2_128"),
    )
    parser.add_argument(
        "--planet_dir",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "patches" / "planet_128"),
    )
    parser.add_argument("--report_out", type=str, default="data/week3_spatial_report.json")
    parser.add_argument(
        "--pairs_csv_out",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "srgan_candidate_pairs.csv"),
    )
    args = parser.parse_args()

    run_week3_audit(args.s2_dir, args.planet_dir, args.report_out, args.pairs_csv_out)