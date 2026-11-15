import json
from pathlib import Path

import pandas as pd

CLASSES_PATH = Path("data/classes.json")


def load_classes():
    """Return {species_name: label}. Labels are permanent and only ever appended."""
    if CLASSES_PATH.exists():
        return json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return {}


def save_classes(classes):
    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSES_PATH.write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")


def get_label(species, classes):
    """Look up a species' label, assigning the next free number if it's new."""
    if species not in classes:
        classes[species] = len(classes)
    return classes[species]


def num_classes():
    return len(load_classes())


if __name__ == "__main__":
    # One-off bootstrap: build classes.json from the labels already in metadata.csv,
    # so existing data keeps exactly the same label numbers.
    if CLASSES_PATH.exists():
        print(f"{CLASSES_PATH} already exists, not overwriting.")
    else:
        df = pd.read_csv("data/metadata.csv")
        pairs = df[["species", "label"]].drop_duplicates().sort_values("label")
        classes = {row.species: int(row.label) for row in pairs.itertuples()}
        assert sorted(classes.values()) == list(range(len(classes))), "labels must be 0..N-1"
        save_classes(classes)
        print(f"Wrote {len(classes)} classes to {CLASSES_PATH}")
