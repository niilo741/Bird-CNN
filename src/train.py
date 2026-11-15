import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn

from src.checkpoint import load_checkpoint, save_checkpoint
from src.dataset import get_dataloaders
from src.labels import load_classes, num_classes
from src.model import BirdCNN, ModelConfig, expand_head


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 32
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.4
    seed: int = 0


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)


@torch.no_grad()
def predict(model, loader, device):
    """Logits and labels for a whole loader, in dataset order."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels).numpy()


def macro_f1(y_true, y_pred):
    # Only classes that occur in this split: a species with no val recordings can't be scored.
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="checkpoint to fine-tune from (e.g. after adding species)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        model, ckpt = load_checkpoint(args.resume, BirdCNN, ModelConfig)
        expand_head(model, num_classes())
        model_cfg = ModelConfig(**{**ckpt["config"]["model"], "num_classes": num_classes()})
        cfg = Config(model_cfg, TrainConfig(epochs=15, lr=3e-4))
        out_path = "checkpoints/finetuned.pt"
    else:
        cfg = Config(ModelConfig(num_classes=num_classes()))
        model = BirdCNN(cfg.model)
        out_path = "checkpoints/best.pt"
    tc = cfg.train
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    model.to(device)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=tc.batch_size)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=tc.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tc.lr, epochs=tc.epochs, steps_per_epoch=len(train_loader)
    )

    best_f1 = -1.0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Mixup: blend each chunk with another one and take the same blend of their losses.
            lam = np.random.beta(tc.mixup_alpha, tc.mixup_alpha)
            perm = torch.randperm(len(x), device=device)
            out = model(lam * x + (1 - lam) * x[perm])
            loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[perm])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(x)

        logits, y_val = predict(model, val_loader, device)
        pred = logits.argmax(1).numpy()
        val_f1, val_acc = macro_f1(y_val, pred), (pred == y_val).mean()
        saved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(out_path, model, cfg, {"val_macro_f1": float(val_f1), "epoch": epoch})
            saved = "  (saved)"
        print(f"epoch {epoch:2d}  train loss {total_loss / len(train_loader.dataset):.3f}  "
              f"val acc {val_acc:.3f}  val macro-F1 {val_f1:.3f}{saved}")

    # Test report for the best checkpoint, not the last epoch.
    model, _ = load_checkpoint(out_path, BirdCNN, ModelConfig)
    logits, y_test = predict(model.to(device), test_loader, device)
    pred = logits.argmax(1).numpy()
    names = sorted(load_classes(), key=load_classes().get)
    print(f"\nTest ({out_path})")
    print(classification_report(y_test, pred, labels=range(len(names)), target_names=names, zero_division=0))
    print(f"chunk accuracy     {(pred == y_test).mean():.3f}")
    print(f"chunk macro-F1     {macro_f1(y_test, pred):.3f}")

    # Per recording: average the chunk logits. Test loader is unshuffled, so rows line up with df.
    df = test_loader.dataset.df.assign(row=range(len(y_test)))
    rec_correct = [
        logits[g["row"].tolist()].mean(0).argmax().item() == g["label"].iloc[0] for _, g in df.groupby("xc_id")
    ]
    print(f"recording accuracy {np.mean(rec_correct):.3f}  ({len(rec_correct)} recordings)")


if __name__ == "__main__":
    main()
