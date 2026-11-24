"""
download_planet_weekly.py
-------------------------
Fetches weekly 3m PlanetScope scenes across 10 Rabi seasons,
detects 8-band vs 4-band asset compatibility dynamically, clips via Planet
Orders API, saves locally to ~/Documents/my-srgan-data/planet/farmA_YYYYMMDD.tif,
and mirrors to Google Drive.
"""

import argparse
import os
import re
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

socket.setdefaulttimeout(120.0)

# Directory paths anchored to ~/Documents/my-srgan-data
BASE_DIR = Path.home() / "Documents" / "my-srgan-data"
PLANET_DIR = BASE_DIR / "planet"
PLANET_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file
env_path = Path.cwd() / ".env"
if not env_path.exists():
    env_path = BASE_DIR / ".env"
if not env_path.exists():
    env_path = Path.home() / "my-srgan-data" / ".env"

load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("PLANET_API_KEY")
if not API_KEY:
    raise SystemExit(f"Error: PLANET_API_KEY not found in .env (checked {env_path})")

AUTH = (API_KEY, "")
DATA_URL = "https://api.planet.com/data/v1/quick-search"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"

FARM_PREFIX = "farmA"
COORDS = [
    [75.12763058308391, 28.05011137020101],
    [75.12778615120678, 28.04940595785404],
    [75.12887512806682, 28.049486441375816],
    [75.12991046074657, 28.049798906242394],
    [75.1296100533369, 28.05064160999248],
    [75.12763058308391, 28.05011137020101],
]

GEOMETRY = {"type": "Polygon", "coordinates": [COORDS]}
MAX_CLOUD_COVER = 0.20
ITEM_TYPE = "PSScene"

DRIVE_TARGET_FOLDER = "planet_weekly"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


# ── Google Drive Helpers ──────────────────────────────────────────────────────

def get_drive_service():
    cwd = Path.cwd()
    token_file = cwd / "token.json" if (cwd / "token.json").exists() else BASE_DIR / "token.json"
    credentials_file = (
        cwd / "credentials.json"
        if (cwd / "credentials.json").exists()
        else BASE_DIR / "credentials.json"
    )

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
                    f"Missing 'credentials.json' in {cwd} or {BASE_DIR}."
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


def upload_to_drive(service, file_path: Path, target_drive_name: str, folder_id: str, max_retries: int = 5):
    if file_exists_in_drive(service, target_drive_name, folder_id):
        print(f"  [Drive] Skipping: '{target_drive_name}' already exists.")
        return

    file_metadata = {"name": target_drive_name, "parents": [folder_id]}
    media = MediaFileUpload(str(file_path), mimetype="image/tiff", chunksize=1024 * 1024, resumable=True)
    request = service.files().create(body=file_metadata, media_body=media, fields="id")

    response = None
    retry_count = 0
    while response is None:
        try:
            status, response = request.next_chunk(num_retries=3)
            if status:
                print(f"    [Drive Upload] {int(status.progress() * 100)}% transferred...")
        except (TimeoutError, socket.timeout, HttpError, ConnectionResetError) as err:
            retry_count += 1
            if retry_count > max_retries:
                print(f"  [Drive Error] Upload failed for {target_drive_name} ({err}). Local copy safe.")
                return
            wait_time = 2**retry_count
            print(f"    [Timeout] Retrying in {wait_time}s ({retry_count}/{max_retries})...")
            time.sleep(wait_time)

    print(f"  [Drive] Uploaded as: {target_drive_name} (ID: {response.get('id')})")


# ── Planet API Logic ──────────────────────────────────────────────────────────

def generate_weekly_intervals(start_str, end_str):
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    intervals = []

    current_dt = start_dt
    while current_dt < end_dt:
        next_dt = min(current_dt + timedelta(days=7), end_dt)
        tag = current_dt.strftime("%Y%m%d")
        intervals.append(
            (
                current_dt.strftime("%Y-%m-%dT00:00:00Z"),
                next_dt.strftime("%Y-%m-%dT23:59:59Z"),
                tag,
            )
        )
        current_dt = next_dt
    return intervals


def find_clearest_scene_with_bundle(w_start, w_end):
    """
    Finds the clearest scene and selects 'analytic_8b_sr_udm2' if the satellite
    supports 8-band, otherwise falls back to 'analytic_sr_udm2' (4-band).
    """
    search_payload = {
        "item_types": [ITEM_TYPE],
        "filter": {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": GEOMETRY},
                {"type": "DateRangeFilter", "field_name": "acquired", "config": {"gte": w_start, "lte": w_end}},
                {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": MAX_CLOUD_COVER}},
            ],
        },
    }
    resp = requests.post(DATA_URL, auth=AUTH, json=search_payload)
    if resp.status_code != 200:
        return None, None

    features = resp.json().get("features", [])
    if not features:
        return None, None

    features.sort(key=lambda f: f["properties"]["cloud_cover"])
    best = features[0]
    scene_id = best["id"]
    assets = best.get("assets", [])

    # Check if 8-band surface reflectance is supported
    if "ortho_analytic_8b_sr" in assets or best.get("properties", {}).get("instrument") == "PSB.SD":
        bundle = "analytic_8b_sr_udm2"
    else:
        bundle = "analytic_sr_udm2"

    return scene_id, bundle


def submit_orders_by_bundle(matched_pairs, season_tag):
    """Submits separate batch orders for 8-band and 4-band scenes."""
    grouped = {}
    for sid, bundle in matched_pairs:
        grouped.setdefault(bundle, []).append(sid)

    order_ids = []
    for bundle, item_ids in grouped.items():
        unique_ids = list(dict.fromkeys(item_ids))
        order_payload = {
            "name": f"rabi_{season_tag}_{bundle}_{int(time.time())}",
            "products": [
                {
                    "item_ids": unique_ids,
                    "item_type": ITEM_TYPE,
                    "product_bundle": bundle,
                }
            ],
            "tools": [{"clip": {"aoi": GEOMETRY}}],
        }
        resp = requests.post(ORDERS_URL, auth=AUTH, json=order_payload)
        if resp.status_code == 202:
            oid = resp.json()["id"]
            print(f"  [Order Submitted] Bundle '{bundle}' ({len(unique_ids)} scenes) -> ID: {oid}")
            order_ids.append(oid)
        else:
            print(f"  [Order Error] {bundle} [{resp.status_code}]: {resp.text}")

    return order_ids


def extract_acquisition_date(filename: str) -> str:
    match = re.search(r"(\d{8})_\d{6}", filename)
    if match:
        return match.group(1)
    fallback = re.search(r"(202\d{5})", filename)
    if fallback:
        return fallback.group(1)
    return datetime.utcnow().strftime("%Y%m%d")


def poll_and_transfer_order(order_id, drive_service, drive_folder_id, poll_interval=20):
    status_url = f"{ORDERS_URL}/{order_id}"
    print(f"Polling order {order_id}...")

    while True:
        resp = requests.get(status_url, auth=AUTH)
        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue

        data = resp.json()
        state = data.get("state")
        print(f"  Order State: {state}")

        if state == "success":
            results = data.get("_links", {}).get("results", [])
            print(f"Found {len(results)} outputs. Transferring GeoTIFFs...")

            for item in results:
                raw_filename = item["name"].split("/")[-1]

                # Match both 8-band and 4-band BOA surface reflectance GeoTIFFs
                if raw_filename.endswith(".tif") and "AnalyticMS_SR" in raw_filename:
                    acq_date = extract_acquisition_date(raw_filename)
                    formatted_name = f"{FARM_PREFIX}_{acq_date}.tif"
                    local_target_path = PLANET_DIR / formatted_name

                    if local_target_path.exists():
                        print(f"  [Local] File already exists: {formatted_name}")
                    else:
                        download_url = item["location"]
                        print(f"  Downloading: {raw_filename} -> {formatted_name}")
                        with requests.get(download_url, stream=True) as r:
                            r.raise_for_status()
                            with open(local_target_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        f.write(chunk)
                        print(f"  Saved locally: {local_target_path.name}")

                    upload_to_drive(drive_service, local_target_path, formatted_name, drive_folder_id)
            break

        elif state in ["failed", "cancelled"]:
            print(f"  Planet order {order_id} failed: {data.get('error')}")
            break

        time.sleep(poll_interval)


def run_pipeline(start_year, end_year):
    print(f"Initializing Google Drive service for target folder '{DRIVE_TARGET_FOLDER}'...")
    drive_service = get_drive_service()
    drive_folder_id = get_or_create_drive_folder(drive_service, DRIVE_TARGET_FOLDER)
    print(f"Target Google Drive Folder ID: {drive_folder_id}")

    for year in range(start_year, end_year + 1):
        season_start = f"{year}-10-15"
        season_end = f"{year + 1}-03-31"
        season_tag = f"{year}_{year + 1}"
        print(f"\n{'='*55}\nScanning Planet Catalog for Season: {season_tag}\n{'='*55}")

        intervals = generate_weekly_intervals(season_start, season_end)
        matched_pairs = []

        for w_start, w_end, tag in intervals:
            expected_local_file = PLANET_DIR / f"{FARM_PREFIX}_{tag}.tif"
            
            # Check local directory first to save API quota and orders
            if expected_local_file.exists():
                print(f"  [{tag}] Local copy exists ({expected_local_file.name}). Mirroring to Drive if missing...")
                upload_to_drive(drive_service, expected_local_file, expected_local_file.name, drive_folder_id)
                continue

            scene_id, bundle = find_clearest_scene_with_bundle(w_start, w_end)
            if scene_id:
                print(f"  [{tag}] Selected: {scene_id} (Bundle: {bundle})")
                matched_pairs.append((scene_id, bundle))
            else:
                print(f"  [{tag}] No clear scene found.")

        if matched_pairs:
            print(f"\nSubmitting batch orders for {len(matched_pairs)} new weekly scenes...")
            order_ids = submit_orders_by_bundle(matched_pairs, season_tag)
            for oid in order_ids:
                poll_and_transfer_order(oid, drive_service, drive_folder_id)
        else:
            print(f"No new scenes required for season {season_tag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Season Adaptive Planet Downloader")
    parser.add_argument("--start_year", type=int, default=2016, help="First Rabi season starting year (default: 2016)")
    parser.add_argument("--end_year", type=int, default=2025, help="Last Rabi season starting year (default: 2025)")
    args = parser.parse_args()

    run_pipeline(args.start_year, args.end_year)