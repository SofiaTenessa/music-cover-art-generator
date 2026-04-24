"""Basic sanity tests — these don't require the dataset or weights to run.

Run:
    python -m pytest tests/
"""
from __future__ import annotations

import numpy as np
import torch

from src.model import GenreCNN, count_params
from src.prompt_builder import GENRE_STYLES, build_prompt
from src.preprocessing import (
    AudioConfig, audio_to_log_mel, normalize_spectrogram, extract_mood_features,
)


def test_cnn_forward_shape():
    model = GenreCNN(num_classes=10)
    x = torch.randn(2, 1, 128, 1250)
    y = model(x)
    assert y.shape == (2, 10)


def test_cnn_embedding_shape():
    model = GenreCNN(num_classes=10)
    x = torch.randn(2, 1, 128, 1250)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 256)


def test_cnn_param_count_is_reasonable():
    model = GenreCNN(num_classes=10)
    n = count_params(model)
    assert 500_000 < n < 10_000_000, f"Unexpected param count: {n}"


def test_prompt_builder_all_genres():
    mood = {"tempo_bpm": 120, "energy": 0.1, "brightness": 2500}
    for genre in GENRE_STYLES:
        p = build_prompt(genre, mood)
        assert len(p.positive) > 30
        assert len(p.negative) > 10


def test_log_mel_shape():
    cfg = AudioConfig()
    audio = np.random.randn(cfg.n_samples).astype(np.float32) * 0.1
    spec = audio_to_log_mel(audio, cfg)
    assert spec.shape[0] == cfg.n_mels
    assert spec.ndim == 2


def test_normalize_zero_mean_unit_var():
    x = np.random.randn(10, 20).astype(np.float32) * 3 + 5
    y = normalize_spectrogram(x)
    assert abs(y.mean()) < 1e-4
    assert abs(y.std() - 1.0) < 1e-3


def test_mood_features_keys():
    cfg = AudioConfig()
    audio = np.random.randn(cfg.n_samples).astype(np.float32) * 0.1
    mood = extract_mood_features(audio, cfg)
    for key in ("tempo_bpm", "energy", "brightness", "roughness", "key_estimate"):
        assert key in mood
