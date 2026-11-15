import librosa
import csv
import numpy as np
from pathlib import Path
import pandas as pd

input_metadata = Path("data/metadata.csv")
output_dir = Path("data/processed")
output_metadata = Path("data/processed_metadata.csv")

sample_rate = 22050
chunk_seconds = 5
chunk_samples = sample_rate * chunk_seconds
hop_samples = chunk_samples // 2
min_tail_seconds = 2
min_rms = 0.003

n_mels = 128
n_fft = 2048
hop_lenght = 512
fmax = 10000

# Defining a function to split audio into 5-second chunks
def slice_audio(y):
    chunks = []
    min_tail_samples = int(min_tail_seconds*sample_rate)

    for start in range(0, len(y), hop_samples):
        chunk = y[start : start + chunk_samples]

        if len(chunk) == chunk_samples:
            chunks.append(chunk)
        elif len(chunk) >= min_tail_samples:
            padded = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(padded)
            break
        else:
            break
    return chunks

# Defining function to convert a 5-second waveform into a normalized (1,128,216) float32 array
def log_mel_tensor(chunk):
   
    mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_fft=n_fft, hop_length=hop_lenght, n_mels=n_mels, fmax=fmax)
    
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    return log_mel[np.newaxis, :, :].astype(np.float32)

# Loading metadata
df = pd.read_csv(input_metadata)
output_dir.mkdir(parents=True, exist_ok=True)
processed_rows = []

print(f'Processing {len(df)} recordings...')

for idx, row in df.iterrows():
    filepath = Path(row['filepath'])
    species = row['species']
    label = row['label']
    xc_id = row['xc_id']

    folder_name = species.lower().replace(" ", "_")
    species_out_dir = output_dir / folder_name
    species_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        y, _ = librosa.load(filepath, sr=sample_rate, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception as e:
        print(f'[!] skipping {filepath.name}: {e}')

    chunks = slice_audio(y)
    if not chunks:
        continue

    # Calculate loudness (RMS) for each chunk to filter out silence
    rms_values = [np.sqrt(np.mean(c ** 2)) for c in chunks]
    max_rms = max(rms_values)

    # 3 & 4. Filter silent chunks and save spectrograms
    saved_count = 0
    for chunk_idx, (chunk, rms) in enumerate(zip(chunks, rms_values)):
        if rms < min_rms or rms < 0.2 * max_rms:
            continue

        spec_tensor = log_mel_tensor(chunk)
        out_path = species_out_dir / f"XC{xc_id}_chunk{chunk_idx}.npy"
        np.save(out_path, spec_tensor)

        processed_rows.append({
            "chunk_path": str(out_path),
            "xc_id": xc_id,
            "species": species,
            "label": label})
        saved_count += 1

    if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
        print(f"  Processed {idx + 1}/{len(df)} files...")

    # Save index for PyTorch Dataset and train/val splitting
    if processed_rows:
        with open(output_metadata, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=processed_rows[0].keys())
            writer.writeheader()
            writer.writerows(processed_rows)
        print(f"\nDone! Saved {len(processed_rows)} spectrogram chunks to {output_metadata}")