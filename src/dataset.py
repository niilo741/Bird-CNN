from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.labels import num_classes
from src.splits import split_for


class BirdDataset(Dataset):
    
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _apply_spec_augment(self, spec):
        spec = spec.copy()
        _, n_mels, n_steps = spec.shape
        f = np.random.randint(0, 16)
        if f > 0:
            f0 = np.random.randint(0, n_mels - f)
            spec[:, f0 : f0 + f, :] = 0.0
        t = np.random.randint(0, 24)
        if t > 0:
            t0 = np.random.randint(0, n_steps - t)
            spec[:, :, t0 : t0 + t] = 0.0
        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = np.load(row["chunk_path"])
        if self.augment:
            spec = self._apply_spec_augment(spec)
        return torch.from_numpy(spec).float(), torch.tensor(row["label"], dtype=torch.long)


def get_dataloaders(csv_path=Path("data/processed_metadata.csv"), batch_size=32):
    df = pd.read_csv(csv_path).drop_duplicates("chunk_path")
    df["split"] = df["xc_id"].map(split_for)

    loaders = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name].reset_index(drop=True)
        is_train = name == "train"
        loaders[name] = DataLoader(
            BirdDataset(part, augment=is_train),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=0,
        )

    # Class weights for CrossEntropyLoss(weight=...): 1/sqrt(count), sized by classes.json
    # so the vector length always matches the model's output layer.
    counts = np.bincount(df.loc[df["split"] == "train", "label"], minlength=num_classes())
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w[counts > 0].mean()
    class_weights = torch.tensor(w, dtype=torch.float32)

    return loaders["train"], loaders["val"], loaders["test"], class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()
    for name, dl in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"{name}: {len(dl.dataset)} chunks, {dl.dataset.df['xc_id'].nunique()} recordings")
    x, y = next(iter(train_loader))
    print(f"X shape: {x.shape} | Y shape: {y.shape}")
    print("Class weights:", class_weights.numpy().round(2))
