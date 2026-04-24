"""Audio → mel-spectrogram preprocessing.

Converts raw audio waveforms into normalized log-mel-spectrograms suitable for
CNN input. Also extracts auxiliary features (MFCCs, spectral statistics) used
by the baseline model and for mood-feature extraction.

# AI-generated via Claude (scaffold). Parameters and design reviewed by author.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


# ---- Config ---------------------------------------------------------------

@dataclass
class AudioConfig:
    """Preprocessing hyperparameters. Shared between training and inference."""
    sample_rate: int = 22050        # GTZAN is 22050 Hz natively
    duration_sec: float = 29.0      # GTZAN clips are 30s; trim slightly to avoid short-clip edge cases
    n_mels: int = 128               # mel bins
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 8000

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


DEFAULT_CONFIG = AudioConfig()


# ---- Core preprocessing ---------------------------------------------------

def load_audio(path: str | Path, config: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Load an audio file, resample to `config.sample_rate`, pad/truncate to fixed length.

    Returns a 1-D float32 numpy array of length `config.n_samples`.
    """
    audio, _ = librosa.load(str(path), sr=config.sample_rate, mono=True)
    target = config.n_samples
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), mode="constant")
    else:
        audio = audio[:target]
    return audio.astype(np.float32)


def audio_to_log_mel(audio: np.ndarray, config: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Convert a 1-D audio array to a log-mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Per-spectrogram zero-mean, unit-variance normalization."""
    mean = spec.mean()
    std = spec.std() + 1e-8
    return (spec - mean) / std


def preprocess_file(path: str | Path, config: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """End-to-end: audio file path → normalized log-mel spectrogram ready for CNN."""
    audio = load_audio(path, config)
    spec = audio_to_log_mel(audio, config)
    return normalize_spectrogram(spec)


# ---- Mood / auxiliary features --------------------------------------------

def extract_mood_features(audio: np.ndarray, config: AudioConfig = DEFAULT_CONFIG) -> dict:
    """Extract interpretable mood-related audio features."""
    sr = config.sample_rate

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_val = float(np.asarray(tempo).flatten()[0])

    rms = librosa.feature.rms(y=audio)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    key_idx = int(np.argmax(chroma.mean(axis=1)))
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    return {
        "tempo_bpm": tempo_val,
        "energy": float(rms.mean()),
        "brightness": float(spectral_centroid.mean()),
        "roughness": float(zero_crossing_rate.mean()),
        "key_estimate": key_names[key_idx],
    }


def extract_mfcc(audio: np.ndarray, config: AudioConfig = DEFAULT_CONFIG, n_mfcc: int = 20) -> np.ndarray:
    """Classical MFCC features used by the baseline logistic-regression model."""
    mfccs = librosa.feature.mfcc(
        y=audio, sr=config.sample_rate, n_mfcc=n_mfcc,
        n_fft=config.n_fft, hop_length=config.hop_length,
    )
    mean = mfccs.mean(axis=1)
    std = mfccs.std(axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing <audio_file>")
        sys.exit(1)
    spec = preprocess_file(sys.argv[1])
    print(f"Spectrogram shape: {spec.shape}, dtype: {spec.dtype}")
    print(f"Min: {spec.min():.3f}, max: {spec.max():.3f}, mean: {spec.mean():.3f}")
