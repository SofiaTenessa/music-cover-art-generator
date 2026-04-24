"""Custom CNN architecture for music genre classification from log-mel spectrograms.

A small VGG-style 2-D CNN operating on (1, 128, ~1250) input spectrograms.
Design choices to discuss in the technical walkthrough:

    - Four conv blocks with increasing channel counts (32 → 64 → 128 → 256).
    - BatchNorm after every conv for stable training.
    - MaxPool after every block to downsample in both freq and time.
    - Dropout in the classifier head (regularization rubric item).
    - Global average pooling to reduce dependence on input length.

# AI-generated via Claude (scaffold). Author tuned depth, channel counts, dropout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class GenreCNN(nn.Module):
    """CNN classifier for mel-spectrograms.

    Input:  (batch, 1, n_mels=128, n_frames~1250)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-D penultimate-layer embedding (pre-classifier)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.gap(x).flatten(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GenreCNN(num_classes=10)
    x = torch.randn(2, 1, 128, 1250)
    y = model(x)
    print(f"Input: {tuple(x.shape)}  → Output: {tuple(y.shape)}")
    print(f"Trainable params: {count_params(model):,}")
