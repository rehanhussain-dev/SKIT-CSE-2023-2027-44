import argparse
from datetime import datetime, timedelta
import ee


def init_gee(project_id="vhr-yieldnet"):
    try:
        ee.Initialize(project=project_id)
    except Exception as err:
        print(f"Standard init failed ({err}). Attempting auth...")
        ee.Authenticate()
        ee.Initialize(project=project_id)


def apply_cloud_score_plus(image):
    """
    Masks clouds/shadows using Cloud Score+ V1.
    Scales reflective optical bands to surface reflectance [0, 1].
    """
    qa_mask = image.select("cs_cdf").gte(0.60)
    optical_bands = image.select(["B2", "B3", "B4", "B5", "B8", "B11"])

    scaled_optical = optical_bands.toFloat().divide(10000.0)
    return scaled_optical.updateMask(qa_mask)


def compute_indices(image):
    """Computes agricultural indices required for biophysical fusion."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat()
    ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE").toFloat()

    red = image.select("B4")
    swir1 = image.select("B11")
    si = red.multiply(swir1).sqrt().rename("SI").toFloat()

    return image.addBands([ndvi, ndre, si])


def generate_weekly_intervals(start_str, end_str):
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    intervals = []

    current_dt = start_dt
    while current_dt < end_dt:
        next_dt = min(current_dt + timedelta(days=7), end_dt)

        # Includes 4-digit year, month abbreviation, and 2-digit day (e.g., 2025_Oct_15)
        file_tag = current_dt.strftime("%Y_%b_%d")
        intervals.append(
            (current_dt.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d"), file_tag)
        )
        current_dt = next_dt

    return intervals


def export_weekly_composites(
    roi_coords, start_date, end_date, drive_folder="vhr_yieldnet_weekly"
):
    aoi = ee.Geometry.Polygon(roi_coords)
    buffered_roi = aoi.buffer(60).bounds()

    intervals = generate_weekly_intervals(start_date, end_date)
    selected_bands = ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDRE", "SI"]

    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    print(
        f"Processing {len(intervals)} weekly windows into Drive folder '{drive_folder}'..."
    )

    for w_start, w_end, file_tag in intervals:
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(buffered_roi)
            .filterDate(w_start, w_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .filter(ee.Filter.lt("MEAN_SOLAR_ZENITH_ANGLE", 70))
            .linkCollection(
                csPlus.filterBounds(buffered_roi).filterDate(w_start, w_end), ["cs_cdf"]
            )
            .map(apply_cloud_score_plus)
            .map(compute_indices)
        )

        try:
            count = s2_collection.size().getInfo()
        except Exception as e:
            print(f"Query error on {file_tag}: {e}")
            continue

        if count == 0:
            print(f"Skipping {file_tag} ({w_start} -> {w_end}): Zero valid scenes.")
            continue

        composite = (
            s2_collection.select(selected_bands).median().clip(buffered_roi).toFloat()
        )

        filename = f"composite_{file_tag}"

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=filename,
            folder=drive_folder,
            fileNamePrefix=filename,
            scale=10,
            region=buffered_roi,
            crs="EPSG:4326",
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        print(f"Submitted: {filename}.tif [{w_start} to {w_end}] ({count} scenes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Pipeline with Cloud Score+"
    )
    parser.add_argument("--project", type=str, default="vhr-yieldnet")
    parser.add_argument("--start", type=str, default="2025-10-15")
    parser.add_argument("--end", type=str, default="2026-03-31")
    parser.add_argument("--folder", type=str, default="vhr_yieldnet_weekly")
    args = parser.parse_args()

    TARGET_ROI = [
        [75.12763058308391, 28.05011137020101],
        [75.12778615120678, 28.04940595785404],
        [75.12887512806682, 28.049486441375816],
        [75.12991046074657, 28.049798906242394],
        [75.1296100533369, 28.05064160999248],
        [75.12763058308391, 28.05011137020101],
    ]

    init_gee(args.project)
    export_weekly_composites(TARGET_ROI, args.start, args.end, drive_folder=args.folder)
