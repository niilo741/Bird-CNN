import csv
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

input_metadata = Path("data/metadata.csv")
output_dir = Path("data/processed")
output_metadata = Path("data/processed_metadata.csv")
processed_ids_path = Path("data/processed_ids.txt")  # recordings already handled
FIELDNAMES = ["chunk_path", "xc_id", "species", "label"]

sample_rate = 22050
chunk_seconds = 5
chunk_samples = sample_rate * chunk_seconds
hop_samples = chunk_samples // 2
min_tail_seconds = 2
min_rms = 0.003

n_mels = 128
n_fft = 2048
hop_length = 512
fmax = 10000


def slice_audio(y):
    chunks = []
    min_tail_samples = int(min_tail_seconds * sample_rate)
    for start in range(0, len(y), hop_samples):
        chunk = y[start : start + chunk_samples]
        if len(chunk) == chunk_samples:
            chunks.append(chunk)
        elif len(chunk) >= min_tail_samples:
            chunks.append(np.pad(chunk, (0, chunk_samples - len(chunk))))
            break
        else:
            break
    return chunks


def log_mel_tensor(chunk):
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel[np.newaxis, :, :].astype(np.float32)


def load_done_ids():
    if processed_ids_path.exists():
        return set(processed_ids_path.read_text().split())
    if output_metadata.exists():
        # First run after this change: treat everything already in processed_metadata.csv as done.
        ids = set(pd.read_csv(output_metadata)["xc_id"].astype(str))
        processed_ids_path.write_text("".join(f"{i}\n" for i in sorted(ids)))
        return ids
    return set()


def append_chunks(rows):
    new_file = not output_metadata.exists()
    with open(output_metadata, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def mark_done(xc_id):
    with open(processed_ids_path, "a") as f:
        f.write(f"{xc_id}\n")


df = pd.read_csv(input_metadata)
done = load_done_ids()
todo = df[~df["xc_id"].astype(str).isin(done)]
print(f"{len(todo)} new recordings to process ({len(df) - len(todo)} already done).")

output_dir.mkdir(parents=True, exist_ok=True)
total_chunks = 0

for i, (_, row) in enumerate(todo.iterrows(), start=1):
    filepath = Path(row["filepath"])
    xc_id = str(row["xc_id"])

    species_out_dir = output_dir / row["species"].lower().replace(" ", "_")
    species_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        y, _ = librosa.load(filepath, sr=sample_rate, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception as e:
        print(f"[!] skipping {filepath.name}: {e}")
        continue  # not marked done, so it is retried next run

    rows = []
    chunks = slice_audio(y)
    if chunks:
        rms_values = [np.sqrt(np.mean(c ** 2)) for c in chunks]
        max_rms = max(rms_values)
        for chunk_idx, (chunk, rms) in enumerate(zip(chunks, rms_values)):
            if rms < min_rms or rms < 0.2 * max_rms:
                continue
            out_path = species_out_dir / f"XC{xc_id}_chunk{chunk_idx}.npy"
            np.save(out_path, log_mel_tensor(chunk))
            rows.append({
                "chunk_path": str(out_path),
                "xc_id": xc_id,
                "species": row["species"],
                "label": int(row["label"]),
            })

    if rows:
        append_chunks(rows)
    mark_done(xc_id)  # also marks recordings with 0 usable chunks, so they aren't redone
    total_chunks += len(rows)

    if i % 50 == 0 or i == len(todo):
        print(f"  Processed {i}/{len(todo)} new files...")

print(f"\nDone! Added {total_chunks} spectrogram chunks to {output_metadata}")
