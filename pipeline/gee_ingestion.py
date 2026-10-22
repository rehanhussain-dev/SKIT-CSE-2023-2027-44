import argparse
from datetime import datetime, timedelta
from pathlib import Path
import requests
import ee

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

LOCAL_S2_DIR = Path.home() / "Documents" / "my-srgan-data" / "s2"
LOCAL_S2_DIR.mkdir(parents=True, exist_ok=True)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


# ── Google Drive Helpers ──────────────────────────────────────────────────────

def get_drive_service():
    cwd = Path.cwd()
    token_file = cwd / "token.json"
    credentials_file = cwd / "credentials.json"

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        refreshed = False
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                refreshed = True
            except RefreshError:
                print("Stored refresh token is invalid or expired. Re-authenticating...")
                if token_file.exists():
                    token_file.unlink()
                creds = None

        if not refreshed:
            if not credentials_file.exists():
                raise FileNotFoundError(
                    f"Missing 'credentials.json' in {cwd}. Place your Google Cloud credentials.json here."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_or_create_drive_folder(service, folder_name):
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder.get("id")


def file_exists_in_drive(service, file_name, folder_id):
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    return len(results.get("files", [])) > 0


def upload_to_drive(service, file_path: Path, target_drive_name: str, folder_id):
    if file_exists_in_drive(service, target_drive_name, folder_id):
        print(f"  [Drive] Skipping: '{target_drive_name}' already exists.")
        return

    file_metadata = {"name": target_drive_name, "parents": [folder_id]}
    media = MediaFileUpload(str(file_path), mimetype="image/tiff", chunksize=5 * 1024 * 1024, resumable=True)
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print(f"  [Drive] Uploaded: {target_drive_name} (ID: {uploaded_file.get('id')})")


# ── Earth Engine Logic ────────────────────────────────────────────────────────

def init_gee(project_id="vhr-yieldnet"):
    try:
        ee.Initialize(project=project_id)
    except Exception as err:
        print(f"Standard init failed ({err}). Authenticating...")
        ee.Authenticate()
        ee.Initialize(project=project_id)


def mask_cloud_score_plus(image):
    qa_mask = image.select("cs_cdf").gte(0.60)
    optical_bands = image.select(["B2", "B3", "B4", "B5", "B8", "B11"]).toFloat().divide(10000.0)
    return optical_bands.updateMask(qa_mask)


def mask_qa60(image):
    """Fallback mask for seasons prior to 2019."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    optical_bands = image.select(["B2", "B3", "B4", "B5", "B8", "B11"]).toFloat().divide(10000.0)
    return optical_bands.updateMask(mask)


def compute_indices(image):
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
        file_tag = current_dt.strftime("%Y%m%d")
        intervals.append(
            (current_dt.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d"), file_tag)
        )
        current_dt = next_dt

    return intervals


def process_rabi_seasons(roi_coords, season_years, drive_service, drive_folder_id):
    aoi = ee.Geometry.Polygon(roi_coords)
    buffered_roi = aoi.buffer(60).bounds()
    selected_bands = ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDRE", "SI"]

    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    for start_year in season_years:
        season_start = f"{start_year}-10-15"
        season_end = f"{start_year + 1}-03-31"
        print(f"\n{'='*55}\nProcessing Rabi Season: {start_year}-{start_year + 1}\n{'='*55}")

        intervals = generate_weekly_intervals(season_start, season_end)

        for w_start, w_end, file_tag in intervals:
            filename = f"farmA_{file_tag}.tif"
            local_path = LOCAL_S2_DIR / filename

            # Use Cloud Score+ if 2019 or later; fallback to QA60 for pre-2019
            use_cs_plus = int(file_tag[:4]) >= 2019

            if use_cs_plus:
                s2_col = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(buffered_roi)
                    .filterDate(w_start, w_end)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
                    .linkCollection(csPlus.filterBounds(buffered_roi).filterDate(w_start, w_end), ["cs_cdf"])
                    .map(mask_cloud_score_plus)
                    .map(compute_indices)
                )
            else:
                # Pre-2019 fallback to available L2A or Level-1C TOA
                s2_col = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(buffered_roi)
                    .filterDate(w_start, w_end)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
                    .map(mask_qa60)
                    .map(compute_indices)
                )

            try:
                count = s2_col.size().getInfo()
            except Exception as e:
                print(f"Query error on {file_tag}: {e}")
                continue

            if count == 0:
                print(f"Skipping {file_tag} ({w_start} -> {w_end}): Zero valid scenes.")
                continue

            composite = s2_col.select(selected_bands).median().clip(buffered_roi).toFloat()

            if local_path.exists():
                print(f"  [Local] File already exists: {local_path.name}")
            else:
                print(f"  [GEE] {filename} [{w_start} to {w_end}] ({count} scenes)...")
                try:
                    url = composite.getDownloadURL({
                        "name": f"farmA_{file_tag}",
                        "bands": selected_bands,
                        "region": buffered_roi,
                        "scale": 10,
                        "crs": "EPSG:4326",
                        "format": "GEO_TIFF",
                    })
                    resp = requests.get(url, stream=True)
                    resp.raise_for_status()

                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    print(f"  Saved: {local_path.name}")
                except Exception as dl_err:
                    print(f"  Download failed for {filename}: {dl_err}")
                    continue

            upload_to_drive(drive_service, local_path, filename, drive_folder_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10-Season Sentinel-2 Pipeline")
    parser.add_argument("--project", type=str, default="vhr-yieldnet")
    parser.add_argument("--start_year", type=int, default=2016, help="First Rabi season starting year (default: 2016)")
    parser.add_argument("--end_year", type=int, default=2025, help="Last Rabi season starting year (default: 2025)")
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
    drive_service = get_drive_service()
    drive_folder_id = get_or_create_drive_folder(drive_service, args.folder)

    season_list = list(range(args.start_year, args.end_year + 1))
    process_rabi_seasons(TARGET_ROI, season_list, drive_service, drive_folder_id)