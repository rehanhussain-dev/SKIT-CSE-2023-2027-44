import argparse
from datetime import datetime, timedelta
import ee


def init_gee(project_id='vhr-yieldnet'):
  """Initialize Earth Engine with a specified Google Cloud Project."""
  try:
    ee.Initialize(project=project_id)
  except Exception as err:
    print(f'Standard initialization failed ({err}). Attempting auth flow...')
    ee.Authenticate()
    ee.Initialize(project=project_id)


def mask_s2_clouds(image):
  """Filters cirrus and opaque clouds using the Sentinel-2 QA60 bitmask."""
  qa = image.select('QA60')
  cloud_bit = 1 << 10
  cirrus_bit = 1 << 11
  mask = (
      qa.bitwiseAnd(cloud_bit)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit).eq(0))
  )
  return image.updateMask(mask).divide(10000).toFloat()


def compute_indices(image):
  """Calculates NDVI, NDRE, and SWIR Salinity Index (SI)."""
  ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').toFloat()
  ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE').toFloat()

  red = image.select('B4')
  swir1 = image.select('B11')
  si = red.multiply(swir1).sqrt().rename('SI').toFloat()

  return image.addBands([ndvi, ndre, si])


def generate_weekly_intervals(start_str, end_str):
  """Generates consecutive 7-day intervals for temporal time-series."""
  start_dt = datetime.strptime(start_str, '%Y-%m-%d')
  end_dt = datetime.strptime(end_str, '%Y-%m-%d')
  intervals = []

  current_dt = start_dt
  while current_dt < end_dt:
    next_dt = min(current_dt + timedelta(days=7), end_dt)
    file_tag = current_dt.strftime('%b_%d')
    intervals.append(
        (current_dt.strftime('%Y-%m-%d'), next_dt.strftime('%Y-%m-%d'), file_tag)
    )
    current_dt = next_dt
  return intervals


def export_weekly_composites(
    roi_coords, start_date, end_date, drive_folder='vhr_yieldnet_weekly'
):
  """Executes asynchronous GEE export tasks across all date windows."""
  roi = ee.Geometry.Polygon([roi_coords])
  intervals = generate_weekly_intervals(start_date, end_date)
  selected_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'NDVI', 'NDRE', 'SI'][cite: 2]

  print(
      f"Queuing {len(intervals)} weekly export tasks to Google Drive folder"
      f" '{drive_folder}'..."
  )

  for w_start, w_end, file_tag in intervals:
    s2_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(w_start, w_end)
        .filter(ee.Filter.lt('MEAN_SOLAR_ZENITH_ANGLE', 70))
        .map(mask_s2_clouds)
        .map(compute_indices)
    )

    try:
      count = s2_collection.size().getInfo()
    except Exception as e:
      print(f'Error checking scene count for {file_tag}: {e}. Skipping...')
      continue

    if count == 0:
      print(
          f'Skipping {file_tag} ({w_start} to {w_end}): No valid cloud-free'
          ' scenes.'
      )
      continue

    composite = (
        s2_collection.select(selected_bands).median().clip(roi).toFloat()
    )

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f'{file_tag}',
        folder=drive_folder,
        scale=10,
        region=roi,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True},
    )
    task.start()
    print(
        f'Export task started: {file_tag}.tif [{w_start} to {w_end}] (Found'
        f' {count} scenes)'
    )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Sentinel-2 Weekly Ingestion Pipeline'
  )
  parser.add_argument(
      '--project', type=str, default='vhr-yieldnet', help='GCP Project ID'
  )
  parser.add_argument(
      '--start',
      type=str,
      default='2025-10-15',
      help='Start Date (YYYY-MM-DD)',
  )
  parser.add_argument(
      '--end', type=str, default='2026-03-31', help='End Date (YYYY-MM-DD)'
  )
  parser.add_argument(
      '--folder',
      type=str,
      default='vhr_yieldnet_weekly',
      help='Google Drive target folder',
  )
  args = parser.parse_args()

  # Target Farm Parcel Coordinates (Shekhawati / Jhunjhunu-Sikar Agricultural Belt)
  TARGET_ROI = [
      [75.12990660891519, 28.049796953402886],
      [75.12963302359567, 28.050644391440418],
      [75.12854941115366, 28.050393474728637],
      [75.12885518298135, 28.04948922285366],
      [75.12990660891519, 28.049796953402886],
  ]

  init_gee(args.project)
  export_weekly_composites(
      TARGET_ROI, args.start, args.end, drive_folder=args.folder
  )