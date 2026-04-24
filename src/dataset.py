"""PyTorch Dataset, DataLoader, and train/val/test split for GTZAN.

Also implements SpecAugment — a data-augmentation technique for spectrograms
that masks random time and frequency bands during training.

Reference: Park et al., SpecAugment (Interspeech 2019).

# AI-generated via Claude (scaffold). SpecAugment + split logic reviewed by author.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .preprocessing import AudioConfig, DEFAULT_CONFIG, preprocess_file


GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}
NUM_CLASSES = len(GENRES)


class SpecAugment:
    """Random time- and frequency-masking on log-mel spectrograms."""

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        prob: float = 0.8,
    ):
        self.F = freq_mask_param
        self.T = time_mask_param
        self.mF = n_freq_masks
        self.mT = n_time_masks
        self.prob = prob

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return spec
        spec = spec.copy()
        n_mels, n_frames = spec.shape
        for _ in range(self.mF):
            f = random.randint(0, self.F)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[f0:f0 + f, :] = spec.mean()
        for _ in range(self.mT):
            t = random.randint(0, self.T)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, t0:t0 + t] = spec.mean()
        return spec


class GTZANDataset(Dataset):
    """GTZAN music genre dataset."""

    def __init__(
        self,
        data_dir: str | Path,
        config: AudioConfig = DEFAULT_CONFIG,
        transform: Callable | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []
        for genre in GENRES:
            genre_dir = self.data_dir / genre
            if not genre_dir.exists():
                continue
            for wav in sorted(genre_dir.glob("*.wav")):
                self.samples.append((wav, GENRE_TO_IDX[genre]))

        if not self.samples:
            raise RuntimeError(
                f"No audio files found under {self.data_dir}. "
                "Did you download the GTZAN dataset? See SETUP.md."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            spec = preprocess_file(path, self.config)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform is not None:
            spec = self.transform(spec)
        tensor = torch.from_numpy(spec).unsqueeze(0).float()
        return tensor, label


def make_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = True,
    config: AudioConfig = DEFAULT_CONFIG,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders with 70/15/15 split."""
    train_aug = SpecAugment() if augment else None
    full_train = GTZANDataset(data_dir, config=config, transform=train_aug)
    full_eval = GTZANDataset(data_dir, config=config, transform=None)

    n = len(full_train)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test], generator=generator)

    train_set = torch.utils.data.Subset(full_train, list(train_idx))
    val_set = torch.utils.data.Subset(full_eval, list(val_idx))
    test_set = torch.utils.data.Subset(full_eval, list(test_idx))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/genres_original"
    train_loader, val_loader, test_loader = make_dataloaders(data_dir, batch_size=4, num_workers=0)
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, labels: {y.tolist()}")
    print(f"Train / val / test sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")
