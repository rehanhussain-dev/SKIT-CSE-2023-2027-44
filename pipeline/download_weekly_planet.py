"""
download_planet_weekly.py
-------------------------
Fetches weekly 3m PlanetScope scenes matching the Sentinel-2 Rabi season
intervals, clips them via the Planet Orders API, and uploads the surface
reflectance GeoTIFFs directly to a Google Drive folder named 'planet_weekly'.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Google Drive API Client Libraries
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()
API_KEY = os.getenv("PLANET_API_KEY")
if not API_KEY:
    raise SystemExit("Error: Set PLANET_API_KEY in your .env file.")

AUTH = (API_KEY, "")
DATA_URL = "https://api.planet.com/data/v1/quick-search"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"

# ── Target Farm Parcel Coordinates (Expanded AOI) ─────────────────────────────
COORDS = [
    [75.12763058308391, 28.05011137020101],
    [75.12778615120678, 28.04940595785404],
    [75.12887512806682, 28.049486441375816],
    [75.12991046074657, 28.049798906242394],
    [75.1296100533369, 28.05064160999248],
    [75.12763058308391, 28.05011137020101]
]

GEOMETRY = {"type": "Polygon", "coordinates": [COORDS]}

START_DATE = "2025-10-15"
END_DATE = "2026-03-31"
MAX_CLOUD_COVER = 0.20
ITEM_TYPE = "PSScene"
PRODUCT_BUNDLE = "analytic_8b_sr_udm2"

# Staging directory before uploading to Google Drive
LOCAL_TEMP_DIR = "temp_planet_downloads"
DRIVE_TARGET_FOLDER = "planet_weekly"
SCOPES = ['https://www.googleapis.com/auth/drive.file']

os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)


# ── Google Drive Helpers ──────────────────────────────────────────────────────

def get_drive_service():
    """Authenticates and returns the Google Drive v3 service."""
    creds = None
    token_file = 'token.json'
    credentials_file = 'credentials.json'

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    "Missing 'credentials.json' for Google Drive API. "
                    "Download it from your Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def get_or_create_drive_folder(service, folder_name):
    """Finds or creates a specific folder in the user's Google Drive root."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])

    if files:
        return files[0]['id']

    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')


def upload_to_drive(service, file_path, folder_id):
    """Uploads a single file to the specified Google Drive folder."""
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype='image/tiff', resumable=True)
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"  Uploaded to Google Drive: {file_name} (ID: {uploaded_file.get('id')})")


# ── Planet API Logic ──────────────────────────────────────────────────────────

def generate_weekly_intervals(start_str, end_str):
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    intervals = []
    
    current_dt = start_dt
    while current_dt < end_dt:
        next_dt = min(current_dt + timedelta(days=7), end_dt)
        tag = current_dt.strftime("%b_%d")
        intervals.append((
            current_dt.strftime("%Y-%m-%dT00:00:00Z"),
            next_dt.strftime("%Y-%m-%dT23:59:59Z"),
            tag
        ))
        current_dt = next_dt
    return intervals


def find_clearest_scene_in_window(w_start, w_end):
    search_payload = {
        "item_types": [ITEM_TYPE],
        "filter": {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": GEOMETRY},
                {"type": "DateRangeFilter", "field_name": "acquired", "config": {"gte": w_start, "lte": w_end}},
                {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": MAX_CLOUD_COVER}}
            ]
        }
    }
    resp = requests.post(DATA_URL, auth=AUTH, json=search_payload)
    if resp.status_code != 200:
        return None
    features = resp.json().get("features", [])
    if not features:
        return None
    features.sort(key=lambda f: f["properties"]["cloud_cover"])
    return features[0]["id"]


def submit_clipped_batch_order(item_ids):
    order_payload = {
        "name": "vhr_yieldnet_weekly_paired",
        "products": [
            {
                "item_ids": item_ids,
                "item_type": ITEM_TYPE,
                "product_bundle": PRODUCT_BUNDLE
            }
        ],
        "tools": [{"clip": {"aoi": GEOMETRY}}]
    }
    resp = requests.post(ORDERS_URL, auth=AUTH, json=order_payload)
    if resp.status_code != 202:
        raise RuntimeError(f"Order submission failed [{resp.status_code}]: {resp.text}")
    return resp.json()["id"]


def poll_and_transfer_order(order_id, drive_service, drive_folder_id, poll_interval=15):
    status_url = f"{ORDERS_URL}/{order_id}"
    print("Waiting for Planet engine to clip and package scenes...")

    while True:
        resp = requests.get(status_url, auth=AUTH)
        data = resp.json()
        state = data["state"]
        print(f"  Order State: {state}")

        if state == "success":
            results = data["_links"]["results"]
            print(f"Downloading and transferring {len(results)} outputs to Google Drive...")
            for item in results:
                file_name = item["name"].split("/")[-1]
                if file_name.endswith(".tif") and "AnalyticMS_SR" in file_name:
                    download_url = item["location"]
                    local_path = os.path.join(LOCAL_TEMP_DIR, file_name)
                    
                    # Stream file locally
                    print(f"  Fetching: {file_name}")
                    with requests.get(download_url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    
                    # Upload directly to Drive folder
                    upload_to_drive(drive_service, local_path, drive_folder_id)
                    
                    # Clean up local staging copy
                    if os.path.exists(local_path):
                        os.remove(local_path)

            print(f"\nAll files successfully transferred to Google Drive folder '{DRIVE_TARGET_FOLDER}'!")
            break

        elif state in ["failed", "cancelled"]:
            raise SystemExit(f"Planet order failed: {state}. Details: {data.get('error')}")

        time.sleep(poll_interval)


def main():
    print(f"Initializing Google Drive service for target folder '{DRIVE_TARGET_FOLDER}'...")
    drive_service = get_drive_service()
    drive_folder_id = get_or_create_drive_folder(drive_service, DRIVE_TARGET_FOLDER)
    print(f"Target Google Drive Folder ID: {drive_folder_id}")

    intervals = generate_weekly_intervals(START_DATE, END_DATE)
    print(f"Scanning Planet catalog across {len(intervals)} weekly windows...")

    matched_item_ids = []
    for w_start, w_end, tag in intervals:
        scene_id = find_clearest_scene_in_window(w_start, w_end)
        if scene_id:
            print(f"  [{tag}] Selected: {scene_id}")
            matched_item_ids.append(scene_id)
        else:
            print(f"  [{tag}] No clear scene found.")

    if not matched_item_ids:
        print("No scenes found matching criteria.")
        return

    matched_item_ids = list(dict.fromkeys(matched_item_ids))
    print(f"\nSubmitting batch clip order for {len(matched_item_ids)} weekly scenes...")
    order_id = submit_clipped_batch_order(matched_item_ids)
    poll_and_transfer_order(order_id, drive_service, drive_folder_id)


if __name__ == "__main__":
    main()