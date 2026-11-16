import os
import time
import csv
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.labels import load_classes, save_classes, get_label

load_dotenv()
API_KEY = os.environ["API_KEY"]
BASE_URL = "https://xeno-canto.org/api/3/recordings"

# Add new species anywhere in this list; their labels come from classes.json, not list order.
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
countries = ["Finland", "Sweden", "Norway", "Estonia", "Denmark", "Latvia", "Lithuania", "Germany", "Poland"]

max_per_species = 150  # target TOTAL per species, including recordings already downloaded
# Higher targets for species the model confuses most; raise or add entries as needed.
species_targets = {"Common Blackbird": 300, "Great Spotted Woodpecker": 300}
output_dir = Path("data/raw")
metadata_path = Path("data/metadata.csv")
FIELDNAMES = ["xc_id", "species", "label", "country", "filepath", "length"]


def load_existing():
    """IDs already in metadata.csv and how many recordings each species has."""
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        return set(df["xc_id"].astype(str)), Counter(df["species"])
    return set(), Counter()


def append_rows(rows):
    new_file = not metadata_path.exists()
    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def iter_recordings(query):
    """Yield recordings from every result page, not just page 1."""
    page, num_pages = 1, 1
    while page <= num_pages:
        response = requests.get(
            BASE_URL, params={"key": API_KEY, "query": query, "page": page}, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        num_pages = int(data.get("numPages", 1))
        yield from data.get("recordings", [])
        page += 1
        time.sleep(1.1)


output_dir.mkdir(parents=True, exist_ok=True)
classes = load_classes()
known_ids, counts = load_existing()

for bird in species:
    label = get_label(bird, classes)
    save_classes(classes)

    bird_dir = output_dir / bird.lower().replace(" ", "_")
    bird_dir.mkdir(parents=True, exist_ok=True)

    have = counts[bird]
    target = species_targets.get(bird, max_per_species)
    new_rows = []

    for country in countries:
        if have >= target:
            break

        query = f'en:"={bird}" q:">C" cnt:{country} len:"5-60"'
        for rec in iter_recordings(query):
            if have >= target:
                break

            xc_id = str(rec["id"])
            if xc_id in known_ids or not rec.get("file"):
                continue
            known_ids.add(xc_id)

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

            new_rows.append({
                "xc_id": xc_id,
                "species": bird,
                "label": label,
                "country": country,
                "filepath": str(dest_path),
                "length": rec.get("length", ""),
            })
            have += 1

    if new_rows:
        append_rows(new_rows)  # saved per species, so a crash doesn't lose earlier work
    print(f"{bird}: +{len(new_rows)} new, {have} total.")
