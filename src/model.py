from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_classes: int
    channels: tuple = (32, 64, 128, 256)
    dropout: float = 0.3


class BirdCNN(nn.Module):
    """Input: (batch, 1, n_mels, time) log-mel chunks. Output: (batch, num_classes) logits."""

    def __init__(self, cfg):
        super().__init__()
        layers, c_in = [], 1
        for c in cfg.channels:
            layers += [
                nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(c_in, cfg.num_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=2)  # average over frequency -> (batch, channels, time)
        x = x.amax(dim=2) + x.mean(dim=2)  # max + mean over time, so any clip length works
        return self.fc(self.dropout(x))


def expand_head(model, num_classes):
    """Grow the output layer for newly added species, keeping the weights of existing ones."""
    old = model.fc
    if num_classes == old.out_features:
        return
    assert num_classes > old.out_features, "labels are append-only; classes cannot be removed"
    model.fc = nn.Linear(old.in_features, num_classes)
    with torch.no_grad():
        model.fc.weight[: old.out_features] = old.weight
        model.fc.bias[: old.out_features] = old.bias


if __name__ == "__main__":
    model = BirdCNN(ModelConfig(num_classes=10)).eval()
    x = torch.randn(2, 1, 128, 216)
    before = model(x)
    assert before.shape == (2, 10)
    expand_head(model, 12)
    after = model(x)
    assert after.shape == (2, 12) and torch.allclose(before, after[:, :10], atol=1e-6)
    print(f"OK, {sum(p.numel() for p in model.parameters()):,} parameters")
