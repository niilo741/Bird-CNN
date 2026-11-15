import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.labels import load_classes


def data_version(csv_path=Path("data/processed_metadata.csv")):
    """Short fingerprint of exactly which chunks existed when the model was trained."""
    paths = sorted(pd.read_csv(csv_path)["chunk_path"].unique())
    return hashlib.md5("\n".join(paths).encode()).hexdigest()[:10]


def save_checkpoint(path, model, cfg, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "classes": load_classes(),
        "data_version": data_version(),
        "metrics": metrics or {},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, path)


def load_checkpoint(path, model_cls, model_cfg_cls):
    ckpt = torch.load(path, map_location="cpu")
    model_cfg = model_cfg_cls(**ckpt["config"]["model"])
    model = model_cls(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt
