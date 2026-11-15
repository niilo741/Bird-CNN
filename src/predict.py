import argparse

import librosa
import numpy as np
import torch

from src.checkpoint import load_checkpoint
from src.model import BirdCNN, ModelConfig
from src.preprocess import log_mel_tensor, sample_rate, slice_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, BirdCNN, ModelConfig)
    model.eval()
    names = sorted(ckpt["classes"], key=ckpt["classes"].get)

    # Same steps as preprocess.py, so the model sees what it was trained on.
    y, _ = librosa.load(args.audio, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    chunks = slice_audio(y)
    if not chunks:
        raise SystemExit("Audio is shorter than 2 seconds after trimming silence.")

    x = torch.from_numpy(np.stack([log_mel_tensor(c) for c in chunks]))
    with torch.no_grad():
        probs = model(x).softmax(1).mean(0)
    # shortcut: always names a known species, even for silence or an unknown bird;
    # add a background/noise class when using this on field recordings.
    for p, i in zip(*probs.topk(min(3, len(names)))):
        print(f"{p:6.1%}  {names[i]}")
