"""
LoRA integration for Chapel Covers.

Extends the CoverArtPipeline to support loading and using trained LoRA weights
during image generation.

Usage:
    from src.lora_integration import CoverArtPipelineWithLoRA

    pipeline = CoverArtPipelineWithLoRA(
        cnn_checkpoint="models/cnn_default_best.pt",
        diffusion_model_id="runwayml/stable-diffusion-v1-5"
    )

    # Load trained LoRA weights
    pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

    # Generate with Duke style
    image = pipeline.generate_from_prompt_text(
        "dark moody album cover",
        use_lora=True,
        lora_scale=0.7  # Blend LoRA: 0.0 = original, 1.0 = full LoRA
    )
"""
from __future__ import annotations

from pathlib import Path
import torch
from PIL import Image

from .pipeline import CoverArtPipeline


class CoverArtPipelineWithLoRA(CoverArtPipeline):
    """Extended pipeline with LoRA support."""

    def __init__(
        self,
        cnn_checkpoint: str | Path,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | None = None,
        audio_config=None,
    ):
        super().__init__(
            cnn_checkpoint=cnn_checkpoint,
            diffusion_model_id=diffusion_model_id,
            device=device,
            audio_config=audio_config or __import__("src.preprocessing", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG,
        )
        self._lora_weights_path = None
        self._lora_scale = 0.7  # Default LoRA blending strength

    def load_lora_weights(
        self,
        lora_path: str | Path,
        lora_scale: float = 0.7,
    ) -> None:
        """
        Load trained LoRA weights.

        Args:
            lora_path: Path to saved LoRA weights (from lora_train.py)
            lora_scale: Blending strength [0, 1]
                - 0.0: Use original model
                - 0.5: 50% LoRA influence
                - 1.0: Full LoRA influence
        """
        self._ensure_sd()

        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        print(f"📂 Loading LoRA weights from: {lora_path}")

        try:
            # Load LoRA weights into the UNet (supports both .bin and .safetensors)
            from peft import PeftModel
            self._sd_pipe.unet = PeftModel.from_pretrained(
                self._sd_pipe.unet,
                str(lora_path),
            )
            self._lora_weights_path = lora_path
            self._lora_scale = lora_scale

            print(f"✓ LoRA loaded with scale {lora_scale}")
        except Exception as e:
            print(f"⚠️  Error loading LoRA: {e}")
            raise

    def unload_lora_weights(self) -> None:
        """Unload LoRA weights and return to original model."""
        if self._sd_pipe is None:
            return

        # Reload original model state
        self._sd_pipe.unet.set_attn_processor(None)
        self._lora_weights_path = None
        print("✓ LoRA weights unloaded")

    def generate_from_prompt_text(
        self,
        prompt_text: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        use_lora: bool = False,
        lora_scale: float | None = None,
    ) -> Image.Image:
        """
        Generate image with optional LoRA enhancement.

        Args:
            prompt_text: Positive prompt
            negative_prompt: Negative prompt
            num_inference_steps: Diffusion steps
            guidance_scale: Guidance scale
            seed: Random seed
            use_lora: Whether to use loaded LoRA weights
            lora_scale: Override LoRA blending strength (if not set, uses loaded scale)

        Returns:
            Generated PIL Image
        """
        self._ensure_sd()

        # Set LoRA scale if provided
        if use_lora and lora_scale is not None:
            self._lora_scale = lora_scale

        # Use LoRA scale in prompt if enabled
        if use_lora and self._lora_weights_path is not None:
            prompt_with_lora = f"{prompt_text} <lora:chapel_covers:{self._lora_scale}>"
        else:
            prompt_with_lora = prompt_text

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self._sd_pipe(
            prompt=prompt_with_lora if use_lora else prompt_text,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    def generate_image_batch(
        self,
        prompt_texts: list[str],
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        use_lora: bool = False,
        lora_scale: float | None = None,
    ) -> list[Image.Image]:
        """
        Generate multiple images in batch (useful for A/B testing LoRA).

        Args:
            prompt_texts: List of prompts
            negative_prompt: Negative prompt (shared)
            num_inference_steps: Diffusion steps
            guidance_scale: Guidance scale
            use_lora: Use LoRA
            lora_scale: LoRA blending strength

        Returns:
            List of PIL Images
        """
        images = []
        for prompt_text in prompt_texts:
            img = self.generate_from_prompt_text(
                prompt_text=prompt_text,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_lora=use_lora,
                lora_scale=lora_scale,
            )
            images.append(img)
        return images
