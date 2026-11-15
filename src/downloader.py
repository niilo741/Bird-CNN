import os
import time
import csv
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ['API_KEY']
BASE_URL = "https://xeno-canto.org/api/3/recordings"

species = [
    "Common Cuckoo",
    "Common Blackbird",
    "Great Tit",
    "Common Chaffinch",
    "Yellowhammer",
    "Willow Warbler",
    "Great Spotted Woodpecker",
    "Hooded Crow",
    "Common Wood Pigeon",
    "Goldcrest",
]
countries = ["Finland", "Sweden", "Norway", "Estonia"]

max_per_species = 50
output_dir = Path("data/raw")
metadata_path = Path("data/metadata.csv")

output_dir.mkdir(parents=True, exist_ok=True)
metadata = []

for label, bird in enumerate(species):
    folder_name = bird.lower().replace(" ", "_")
    bird_dir = output_dir / folder_name
    bird_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    seen_ids = set()

    for country in countries:
        if downloaded_count >= max_per_species:
            break

        query = f'en:"={bird}" q:A cnt:{country} len:"5-60"'
        params = {
            "key": API_KEY,
            "query": query,
            "page": 1,
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for rec in data.get("recordings", []):
            if downloaded_count >= max_per_species:
                break

            xc_id = rec["id"]
            if xc_id in seen_ids or not rec.get("file"):
                continue
            seen_ids.add(xc_id)

            file_url = rec["file"]
            if file_url.startswith("//"):
                file_url = "https:" + file_url

            ext = Path(rec.get("file-name", ".mp3")).suffix or ".mp3"
            dest_path = bird_dir / f"XC{xc_id}{ext}"

            if not dest_path.exists():
                audio_response = requests.get(file_url, timeout=60)
                audio_response.raise_for_status()
                dest_path.write_bytes(audio_response.content)
                time.sleep(1.1)

            metadata.append({
                "xc_id": xc_id,
                "species": bird,
                "label": label,
                "country": country,
                "filepath": str(dest_path),
                "length": rec.get("length", ""),
            })
            downloaded_count += 1

        time.sleep(1.1)

    print(f"{bird}: total {downloaded_count} recordings ready.")

if metadata:
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    print(f"\nSaved metadata for {len(metadata)} files to {metadata_path}")
