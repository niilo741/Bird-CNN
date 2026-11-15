import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader


class BirdDataset(Dataset):

    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _apply_spec_augment(self, spec):

        spec = spec.copy()
        _, n_mels, n_steps = spec.shape

        # Frequency mask (horizontal strip)
        f = np.random.randint(0, 16)
        if f > 0:
            f0 = np.random.randint(0, n_mels - f)
            spec[:, f0 : f0 + f, :] = 0.0

        # Time mask (vertical strip)
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
        x = torch.from_numpy(spec).float()
        y = torch.tensor(row['label'], dtype=torch.long)

        return x, y


def get_dataloaders(csv_path=Path("data/processed_metadata.csv"), batch_size=32):
    df = pd.read_csv(csv_path)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    train_idx, test_idx = next(splitter.split(df, groups=df["xc_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_dataset = BirdDataset(train_df, augment=True)
    test_dataset = BirdDataset(test_df, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"X shape: {x.shape} | Y shape: {y.shape}")