"""
extract_timeseries_csv.py
-------------------------
Extracts field-level mean, median, min, max, and std for NDVI, NDRE, and SI 
from the multi-temporal Sentinel-2 GeoTIFF composites. 
Outputs a structured CSV file formatted for Pratik's LSTM-WOFOST growth trajectory network.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# Sentinel-2 8-band stack order defined in gee_ingestion.py:
# 1: B2 (Blue), 2: B3 (Green), 3: B4 (Red), 4: B8 (NIR),
# 5: B11 (SWIR1), 6: NDVI, 7: NDRE, 8: SI
BAND_INDEXES = {
    "NDVI": 6,
    "NDRE": 7,
    "SI": 8,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 4,
    "B11": 5
}


def extract_date_from_filename(filename: str):
    """Extracts YYYYMMDD date from standard filename (e.g. farmA_20251105.tif)."""
    match = re.search(r"(\d{8})", os.path.basename(filename))
    if match:
        raw_date = match.group(1)
        return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    return None


def extract_field_metrics(tif_path: str):
    """Computes spatial statistics over non-masked pixels within the parcel."""
    date_str = extract_date_from_filename(tif_path)
    if not date_str:
        return None

    with rasterio.open(tif_path) as src:
        # Read the indices: NDVI, NDRE, SI
        ndvi_band = src.read(BAND_INDEXES["NDVI"])
        ndre_band = src.read(BAND_INDEXES["NDRE"])
        si_band = src.read(BAND_INDEXES["SI"])
        nodata = src.nodata if src.nodata is not None else 0

        # Valid mask: ignore nodata, zeros, and non-finite values
        valid_mask = (
            (ndvi_band != nodata)
            & np.isfinite(ndvi_band)
            & (ndvi_band != 0)
            & np.isfinite(ndre_band)
            & np.isfinite(si_band)
        )

        valid_count = int(np.sum(valid_mask))
        if valid_count == 0:
            return None

        ndvi_vals = ndvi_band[valid_mask]
        ndre_vals = ndre_band[valid_mask]
        si_vals = si_band[valid_mask]

        # Calculate seasonal metadata
        obs_date = pd.to_datetime(date_str)
        year = obs_date.year
        month = obs_date.month

        # In Rajasthan, Rabi cycle runs from Oct (year) to Apr (year+1)
        if month >= 10:
            rabi_season = f"{year}-{year + 1}"
        else:
            rabi_season = f"{year - 1}-{year}"

        row = {
            "date": date_str,
            "season": rabi_season,
            "doy": obs_date.dayofyear,
            "valid_pixels": valid_count,
            # NDVI metrics
            "ndvi_mean": float(np.mean(ndvi_vals)),
            "ndvi_median": float(np.median(ndvi_vals)),
            "ndvi_std": float(np.std(ndvi_vals)),
            "ndvi_min": float(np.min(ndvi_vals)),
            "ndvi_max": float(np.max(ndvi_vals)),
            # NDRE metrics (Chlorophyll / N-status)
            "ndre_mean": float(np.mean(ndre_vals)),
            "ndre_median": float(np.median(ndre_vals)),
            "ndre_std": float(np.std(ndre_vals)),
            # Salinity / Soil Moisture Index (SI)
            "si_mean": float(np.mean(si_vals)),
            "si_median": float(np.median(si_vals)),
            "si_std": float(np.std(si_vals)),
            "source_file": os.path.basename(tif_path),
        }
        return row


def build_timeseries_dataset(s2_dir: str, output_csv: str):
    """Processes all GeoTIFF files and compiles a sorted time-series table."""
    tif_files = sorted(glob.glob(os.path.join(s2_dir, "*.tif")))
    print(f"Found {len(tif_files)} Sentinel-2 GeoTIFF files in {s2_dir}")

    records = []
    for filepath in tif_files:
        metrics = extract_field_metrics(filepath)
        if metrics:
            records.append(metrics)

    df = pd.DataFrame(records)
    if df.empty:
        print("No valid raster data extracted.")
        return

    # Sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSuccessfully extracted {len(df)} temporal observation steps!")
    print(f"Saved CSV to: {output_csv}")
    print("\nSummary of extracted seasons:")
    print(df["season"].value_counts().sort_index())
    print("\nSample Preview:")
    print(df[["date", "season", "ndvi_mean", "ndre_mean", "si_mean", "valid_pixels"]].head(8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract field-level index time series to CSV")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "s2"),
        help="Directory containing S2 GeoTIFF files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(Path.home() / "Documents" / "my-srgan-data" / "s2_vegetation_indices_timeseries.csv"),
        help="Target output CSV file path",
    )
    args = parser.parse_args()

    build_timeseries_dataset(args.input_dir, args.output_csv)