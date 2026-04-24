"""Evaluation and error analysis for the genre CNN.

Produces:
    - Accuracy, macro F1, per-class precision/recall/F1      [3+ metrics rubric item]
    - Confusion matrix (PNG + CSV)                           [error analysis rubric item]
    - Inference latency stats                                [inference time rubric item]
    - Top-k failure case dump                                [error analysis rubric item]

Run:
    python -m src.evaluate --ckpt models/cnn_default_best.pt --data-dir data/genres_original

# AI-generated via Claude (scaffold). Author owns error-analysis narrative.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from .dataset import make_dataloaders, IDX_TO_GENRE, GENRES, NUM_CLASSES
from .model import GenreCNN
from .train import get_device


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)
        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_probs.append(probs)
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


def measure_inference_latency(model: nn.Module, device, n_trials: int = 50) -> dict:
    model.eval()
    x = torch.randn(1, 1, 128, 1250, device=device)
    for _ in range(5):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput_hz": float(1000 / np.median(times)),
        "n_trials": n_trials,
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def find_hardest_examples(probs: np.ndarray, labels: np.ndarray, k: int = 10):
    preds = probs.argmax(1)
    wrong = preds != labels
    wrong_idx = np.where(wrong)[0]
    pred_conf = probs[np.arange(len(probs)), preds]
    ranked = wrong_idx[np.argsort(-pred_conf[wrong_idx])][:k]
    return [(int(i), int(labels[i]), int(preds[i]), float(pred_conf[i])) for i in ranked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", default="data/genres_original")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = GenreCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: val_acc={ckpt.get('val_acc', '?'):.3f}, epoch={ckpt.get('epoch', '?')}")

    _, _, test_loader = make_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=False, seed=args.seed,
    )
    print(f"Test set size: {len(test_loader.dataset)}")

    preds, labels, probs = collect_predictions(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    macro_prec = precision_score(labels, preds, average="macro")
    macro_rec = recall_score(labels, preds, average="macro")
    top3 = float(np.mean([labels[i] in probs[i].argsort()[-3:] for i in range(len(labels))]))

    print(f"\nTest metrics:")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"  Macro F1:     {macro_f1:.3f}")
    print(f"  Macro prec:   {macro_prec:.3f}")
    print(f"  Macro recall: {macro_rec:.3f}")
    print(f"  Top-3 acc:    {top3:.3f}")
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=GENRES, digits=3))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, GENRES, out_dir / "confusion_matrix.png")
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(GENRES), comments="")

    latency = measure_inference_latency(model, device)
    print(f"\nInference latency (device={device.type}):")
    print(f"  Median:     {latency['median_ms']:.2f} ms")
    print(f"  p95:        {latency['p95_ms']:.2f} ms")
    print(f"  Throughput: {latency['throughput_hz']:.1f} Hz")

    hardest = find_hardest_examples(probs, labels, k=20)
    print("\nHardest misclassifications (true → predicted, confidence):")
    for idx, true_y, pred_y, conf in hardest[:10]:
        print(f"  test idx {idx:4d}: {IDX_TO_GENRE[true_y]} → {IDX_TO_GENRE[pred_y]} ({conf:.3f})")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "top3_accuracy": top3,
            "per_class_f1": {g: float(f) for g, f in zip(
                GENRES, f1_score(labels, preds, average=None))},
            "latency": latency,
            "hardest_examples": [
                {"idx": i, "true": IDX_TO_GENRE[t], "pred": IDX_TO_GENRE[p], "confidence": c}
                for i, t, p, c in hardest
            ],
        }, f, indent=2)
    print(f"\nSaved: {out_dir}/confusion_matrix.png, metrics.json, confusion_matrix.csv")


if __name__ == "__main__":
    main()
