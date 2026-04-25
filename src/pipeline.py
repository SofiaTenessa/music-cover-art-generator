"""End-to-end pipeline: audio → genre + mood → text prompt → cover art.

Stages:
    1. Load + preprocess audio
    2. Extract log-mel spectrogram + mood features
    3. Run CNN for genre prediction
    4. Build text prompt from genre + mood
    5. Generate cover art with Stable Diffusion

CLI:
    python -m src.pipeline --audio path/to/song.wav --output cover.png

# AI-generated via Claude (scaffold). Author owns stage integration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import IDX_TO_GENRE, NUM_CLASSES
from .model import GenreCNN
from .preprocessing import (
    AudioConfig, DEFAULT_CONFIG, audio_to_log_mel, extract_mood_features,
    load_audio, normalize_spectrogram,
)
from .prompt_builder import Prompt, build_prompt, refine_prompt
from .train import get_device


@dataclass
class PipelineResult:
    genre: str
    genre_probs: dict
    mood_features: dict
    prompt: Prompt
    image: Image.Image

    def save(self, out_path: str | Path, save_metadata: bool = True):
        out_path = Path(out_path)
        self.image.save(out_path)
        if save_metadata:
            meta_path = out_path.with_suffix(".txt")
            with open(meta_path, "w") as f:
                f.write(f"Predicted genre: {self.genre}\n\n")
                f.write("Top-3 genre probabilities:\n")
                top3 = sorted(self.genre_probs.items(), key=lambda kv: -kv[1])[:3]
                for g, p in top3:
                    f.write(f"  {g}: {p:.3f}\n")
                f.write("\nAudio features:\n")
                for k, v in self.mood_features.items():
                    f.write(f"  {k}: {v}\n")
                f.write(f"\nPositive prompt:\n  {self.prompt.positive}\n")
                f.write(f"\nNegative prompt:\n  {self.prompt.negative}\n")


class CoverArtPipeline:
    def __init__(
        self,
        cnn_checkpoint: str | Path,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | None = None,
        audio_config: AudioConfig = DEFAULT_CONFIG,
    ):
        self.device = device or get_device()
        self.audio_config = audio_config

        self.cnn = GenreCNN(num_classes=NUM_CLASSES).to(self.device)
        ckpt = torch.load(cnn_checkpoint, map_location=self.device)
        self.cnn.load_state_dict(ckpt["model_state_dict"])
        self.cnn.eval()

        self._diffusion_model_id = diffusion_model_id
        self._sd_pipe = None

    def _ensure_sd(self):
        if self._sd_pipe is not None:
            return
        from diffusers import StableDiffusionPipeline

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self._diffusion_model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)
        if self.device.type == "mps":
            pipe.enable_attention_slicing()
        self._sd_pipe = pipe

    @torch.no_grad()
    def classify_audio(self, audio_path: str | Path) -> tuple[str, dict, dict]:
        audio = load_audio(audio_path, self.audio_config)
        mood = extract_mood_features(audio, self.audio_config)
        spec = normalize_spectrogram(audio_to_log_mel(audio, self.audio_config))
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        logits = self.cnn(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        genre_idx = int(probs.argmax())
        genre_probs = {IDX_TO_GENRE[i]: float(probs[i]) for i in range(len(probs))}
        return IDX_TO_GENRE[genre_idx], genre_probs, mood

    def generate_image(
        self,
        prompt: Prompt,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        self._ensure_sd()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self._sd_pipe(
            prompt=prompt.positive,
            negative_prompt=prompt.negative,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

    def run(
        self,
        audio_path: str | Path,
        lyrics: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> PipelineResult:
        """
        Run the full pipeline: analyze audio → build prompt (optionally with lyrics) → generate cover art.
        """
        genre, genre_probs, mood = self.classify_audio(audio_path)
        prompt = build_prompt(genre, mood_features=mood, lyrics=lyrics)
        image = self.generate_image(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return PipelineResult(
            genre=genre,
            genre_probs=genre_probs,
            mood_features=mood,
            prompt=prompt,
            image=image,
        )

    def generate_from_prompt_text(
        self,
        prompt_text: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Generate an image from a raw prompt string (useful for refinement).

        Args:
            prompt_text: The positive prompt
            negative_prompt: The negative prompt (to avoid)
            num_inference_steps: Number of diffusion steps (higher = slower but higher quality)
            guidance_scale: How strongly to follow the prompt (higher = stronger)
            seed: Optional random seed for reproducibility

        Returns:
            PIL Image object
        """
        self._ensure_sd()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self._sd_pipe(
            prompt=prompt_text,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="outputs/cover.png")
    parser.add_argument("--ckpt", default="models/cnn_default_best.pt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lyrics", default=None, help="Optional lyrics to influence cover art style")
    args = parser.parse_args()

    pipeline = CoverArtPipeline(
        cnn_checkpoint=args.ckpt,
        diffusion_model_id=args.model_id,
    )
    result = pipeline.run(
        args.audio,
        lyrics=args.lyrics,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Predicted genre: {result.genre}")
    print(f"Prompt: {result.prompt.positive}")
    print(f"Saved cover art: {args.output}")


if __name__ == "__main__":
    main()
