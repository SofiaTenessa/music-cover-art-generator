"""
LoRA Fine-tuning for Stable Diffusion on Duke images.

Uses proper diffusion training with noise scheduling and timesteps.

Usage:
    python scripts/lora_train.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer


class DukeImageDataset(Dataset):
    """Dataset of Duke images for LoRA training."""

    def __init__(self, image_dir: Path, image_size: int = 512):
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Find all image files
        self.image_paths = list(
            self.image_dir.glob("**/*.jpg")
        ) + list(self.image_dir.glob("**/*.png")) + list(
            self.image_dir.glob("**/*.jpeg")
        ) + list(self.image_dir.glob("**/*.webp"))

        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values}


def train_lora(
    image_dir: Path,
    output_dir: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    device: torch.device = None,
):
    """Fine-tune Stable Diffusion with LoRA on Duke images."""

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"🎯 Training on device: {device}")
    print(f"📂 Images from: {image_dir}")
    print(f"💾 Output: {output_dir}")

    # ===== LOAD MODEL =====
    print("\n📥 Loading Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    # Freeze base models
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # ===== APPLY LoRA =====
    print("🔧 Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )

    # Apply LoRA to UNet
    unet_lora = get_peft_model(unet, lora_config)
    unet_lora.print_trainable_parameters()

    # ===== PREPARE DATA =====
    print("\n📊 Preparing dataset...")
    dataset = DukeImageDataset(image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ===== SETUP OPTIMIZER =====
    optimizer = torch.optim.AdamW(unet_lora.parameters(), lr=learning_rate)

    # ===== TRAINING LOOP =====
    print(f"\n⏱️  Training for {num_epochs} epochs with {len(dataset)} images...\n")

    unet_lora = unet_lora.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    unet_lora.train()
    vae.eval()
    text_encoder.eval()

    default_prompt = "a photo of Duke University chapel and gothic architecture"

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Encode prompt
            with torch.no_grad():
                text_inputs = tokenizer(
                    default_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                encoder_hidden_states = text_encoder(text_input_ids)[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (pixel_values.shape[0],),
                device=device,
            )

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise with UNet
            model_pred = unet_lora(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            # Compute loss (prediction of noise)
            loss = F.mse_loss(model_pred, noise, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.6f}"
                )

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"✓ Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.6f}\n")

    # ===== SAVE LoRA WEIGHTS =====
    print(f"\n💾 Saving LoRA weights to {output_dir}/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "model_id": model_id,
        "image_count": len(dataset),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ LoRA training complete!")
    print(f"   Weights saved to: {output_dir}/")
    print(f"   Size: ~50MB")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA on Duke images")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/lora_training_images"),
        help="Directory with Duke images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lora_weights/chapel_covers_lora"),
        help="Output directory for LoRA weights",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (reduce if OOM)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )

    args = parser.parse_args()

    # Check images exist
    if not args.images.exists():
        print(f"❌ Image directory not found: {args.images}")
        print(f"Run: python scripts/lora_download_images.py")
        return

    # Train
    train_lora(
        image_dir=args.images,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
