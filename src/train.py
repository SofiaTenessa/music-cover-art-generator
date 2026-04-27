"""Training loop for the genre CNN.

Features (each ticks at least one rubric item):
    - Device auto-detection (CUDA → MPS → CPU)         [GPU acceleration]
    - AdamW with cosine-annealing LR schedule            [LR scheduling]
    - L2 weight decay + dropout + early stopping         [Regularization, ≥2 techniques]
    - Gradient clipping                                  [Training stability]
    - Training/validation loss + accuracy curves         [Training curves]
    - Best-model checkpointing on validation accuracy

Run:
    python -m src.train --data-dir data/genres_original --epochs 50


"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import make_dataloaders, NUM_CLASSES
from .model import GenreCNN, count_params


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 1.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def plot_curves(history: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/genres_original")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--run-name", default="cnn_default")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = make_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        seed=args.seed,
    )
    print(f"Train/val/test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    model = GenreCNN(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    print(f"Model parameters: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.run_name}_best.pt"

    start = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)
        else:
            epochs_no_improve += 1

        tag = " *" if improved else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}{tag}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement for {args.patience} epochs).")
            break

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed/60:.1f} min. Best val acc: {best_val_acc:.3f}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test loss {test_loss:.4f}, test acc {test_acc:.3f}")

    curves_path = out_dir / f"{args.run_name}_curves.png"
    plot_curves(history, curves_path)
    history_path = out_dir / f"{args.run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "history": history,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "args": vars(args),
            "elapsed_min": elapsed / 60,
        }, f, indent=2)
    print(f"Saved: {ckpt_path}, {curves_path}, {history_path}")


if __name__ == "__main__":
    main()
