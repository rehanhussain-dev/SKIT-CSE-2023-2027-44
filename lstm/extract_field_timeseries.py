
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_ZIP = (
    PROJECT_ROOT
    / "vhr_yieldnet_weekly-20261103T161614Z-1-001.zip"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "s2_field_vegetation_indices_timeseries.csv"
)

MIN_VALID_COVERAGE = 0.80

# Original TARGET_ROI from the team's gee_ingestion.py.
# Coordinates are longitude, latitude (EPSG:4326).
FIELD_POLYGON = np.array([
    [75.12763058308391, 28.05011137020101],
    [75.12778615120678, 28.04940595785404],
    [75.12887512806682, 28.049486441375816],
    [75.12991046074657, 28.049798906242394],
    [75.1296100533369, 28.05064160999248],
    [75.12763058308391, 28.05011137020101],
])

# Zero-based indices from the original 8-band export.
BAND_INDEXES = {
    "ndvi": 5,
    "ndre": 6,
    "si": 7,
}


def create_field_mask(shape, transform):
    """Identify pixel centres inside the field polygon."""

    rows, cols = np.indices(shape)

    x = (
        transform[0, 0] * (cols + 0.5)
        + transform[0, 1] * (rows + 0.5)
        + transform[0, 3]
    )

    y = (
        transform[1, 0] * (cols + 0.5)
        + transform[1, 1] * (rows + 0.5)
        + transform[1, 3]
    )

    inside = np.zeros(shape, dtype=bool)

    # Ray-casting point-in-polygon algorithm.
    for v, w in zip(
        FIELD_POLYGON[:-1],
        FIELD_POLYGON[1:]
    ):
        if v[1] == w[1]:
            continue

        crossing = (
            ((y > v[1]) != (y > w[1]))
            & (
                x < (
                    (w[0] - v[0])
                    * (y - v[1])
                    / (w[1] - v[1])
                    + v[0]
                )
            )
        )

        inside ^= crossing

    return inside


def extract_observation(archive, filename):
    """Extract vegetation indices inside the field."""

    match = re.search(
        r"farmA_(\d{8})\.tif$",
        filename,
        re.IGNORECASE,
    )

    if match is None:
        raise ValueError(f"Unexpected filename: {filename}")

    date = pd.to_datetime(
        match.group(1),
        format="%Y%m%d",
    )

    with tifffile.TiffFile(
        io.BytesIO(archive.read(filename))
    ) as tif:

        data = tif.asarray()
        metadata = tif.geotiff_metadata

        if data.ndim != 3 or data.shape[2] != 8:
            raise ValueError(
                f"Unexpected raster shape: {data.shape}"
            )

        if metadata is None:
            raise ValueError(
                f"Missing GeoTIFF metadata: {filename}"
            )

        if metadata.get("GeographicTypeGeoKey") != 4326:
            raise ValueError(
                f"Unexpected CRS: {filename}"
            )

        if metadata.get("GTRasterTypeGeoKey") != 1:
            raise ValueError(
                f"Unsupported raster pixel type: {filename}"
            )

        if "ModelTransformation" not in metadata:
            raise ValueError(
                f"Missing spatial transform: {filename}"
            )

        transform = np.asarray(
            metadata["ModelTransformation"],
            dtype=float,
        )

        nodata_tag = tif.pages[0].tags.get(42113)

        if nodata_tag is None:
            raise ValueError(
                f"Missing NoData metadata: {filename}"
            )

        if float(nodata_tag.value) != -np.inf:
            raise ValueError(
                f"Unexpected NoData value: {filename}"
            )

    field_mask = create_field_mask(
        data.shape[:2],
        transform,
    )

    total_field_pixels = int(field_mask.sum())

    if total_field_pixels == 0:
        raise ValueError(
            f"No field pixels found: {filename}"
        )

    # Require valid values for all three indices.
    valid_mask = (
        field_mask
        & np.isfinite(data[:, :, 5:8]).all(axis=2)
    )

    valid_pixels = int(valid_mask.sum())

    coverage = valid_pixels / total_field_pixels

    season_year = (
        date.year if date.month >= 10
        else date.year - 1
    )

    row = {
        "field_id": "farmA",
        "date": date.strftime("%Y-%m-%d"),
        "season": f"{season_year}-{season_year + 1}",
        "doy": date.dayofyear,
        "total_raster_pixels": int(field_mask.size),
        "total_field_pixels": total_field_pixels,
        "valid_pixels": valid_pixels,
        "valid_coverage": coverage,
        "quality_pass": coverage >= MIN_VALID_COVERAGE,
        "source_file": Path(filename).name,
        "source_type": "Sentinel-2 nominal 10m export",
        "crop_identity": "unverified",
    }

    for name, index in BAND_INDEXES.items():

        values = data[:, :, index][valid_mask]

        if len(values) > 0:
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_std"] = float(values.std())
            row[f"{name}_min"] = float(values.min())
            row[f"{name}_max"] = float(values.max())

        else:
            for stat in [
                "mean", "median", "std", "min", "max"
            ]:
                row[f"{name}_{stat}"] = np.nan

    return row


def main():

    if not INPUT_ZIP.exists():
        raise FileNotFoundError(
            f"Satellite archive not found: {INPUT_ZIP}"
        )

    records = []

    with zipfile.ZipFile(INPUT_ZIP) as archive:

        files = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".tif")
        )

        if not files:
            raise ValueError("No GeoTIFF files found.")

        print("GeoTIFF files found:", len(files))

        for filename in files:
            records.append(
                extract_observation(archive, filename)
            )

    df = pd.DataFrame(records)

    df = df.sort_values("date").reset_index(drop=True)

    if df.duplicated(["field_id", "date"]).any():
        raise ValueError("Duplicate field observations found.")

    accepted = df[df["quality_pass"]]
    rejected = df[~df["quality_pass"]]

    if accepted.empty:
        raise ValueError("No observations passed quality checks.")

    features = ["ndvi_mean", "ndre_mean", "si_mean"]

    if not np.isfinite(
        accepted[features].to_numpy()
    ).all():
        raise ValueError("Invalid accepted vegetation indices.")

    # Preserve every observation for quality auditing.
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFIELD EXTRACTION: PASS")
    print("Total observations:", len(df))
    print("Accepted observations:", len(accepted))
    print("Rejected observations:", len(rejected))
    print("Minimum valid coverage:", MIN_VALID_COVERAGE)

    print(
        "Field pixel counts:",
        sorted(df["total_field_pixels"].unique())
    )

    print("\nACCEPTED OBSERVATIONS PER SEASON:")
    print(
        accepted.groupby("season").size().to_string()
    )

    print("\nREJECTED OBSERVATIONS:")

    print(
        rejected[
            ["date", "valid_pixels", "valid_coverage"]
        ].to_string(index=False)
    )

    print("\nACCEPTED VEGETATION INDEX STATISTICS:")

    print(
        accepted[features].describe().to_string()
    )

    print("\nSaved CSV:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
