"""
IMD weather-data acquisition for VHR-YieldNet.

This module will be used to download/store IMD gridded
weather datasets for the WOFOST simulation.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOWNLOAD_DIR = PROJECT_ROOT / "wofost" / "weather" / "download"
RAW_DIR = PROJECT_ROOT / "wofost" / "weather" / "raw"

FIELD_LAT = 28.04996394302817
FIELD_LON = 75.12877653515253

SOWING_DATE = "2025-11-15"


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("VHR-YieldNet IMD weather acquisition")
    print()
    print(f"Field latitude : {FIELD_LAT}")
    print(f"Field longitude: {FIELD_LON}")
    print(f"Sowing date    : {SOWING_DATE}")
    print()
    print(f"Download folder: {DOWNLOAD_DIR}")
    print(f"Raw-data folder: {RAW_DIR}")


if __name__ == "__main__":
    main()