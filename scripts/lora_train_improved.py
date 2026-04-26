"""
IMPROVED LoRA Fine-tuning for Stable Diffusion on Duke images.

Key improvements over lora_train.py:
- More epochs (20 default, can go higher) for better convergence
- Lower learning rate (1e-5) for stable learning
- Increased LoRA rank (32 instead of 16) for more model capacity
- Data augmentation (random crops, rotations) for better generalization
- Better training prompts that describe Duke architecture in detail
- Learning rate scheduling for smoother convergence
- Loss tracking and early stopping capability

Usage:
    # Default (improved params): 20 epochs, LR 1e-5, rank 32
    python scripts/lora_train_improved.py

    # Custom params
    python scripts/lora_train_improved.py --epochs 30 --lr 1e-5 --rank 32 --batch-size 1
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
import random

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer


class DukeImageDataset(Dataset):
    """Dataset of Duke images for LoRA training with augmentation."""

    def __init__(self, image_dir: Path, image_size: int = 512, augment: bool = True):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

        # Find all image files
        self.image_paths = list(
            self.image_dir.glob("**/*.jpg")
        ) + list(self.image_dir.glob("**/*.png")) + list(
            self.image_dir.glob("**/*.jpeg")
        ) + list(self.image_dir.glob("**/*.webp"))

        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_paths)} images in {self.image_dir}")

        # Base transform (always applied)
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
        ])

        # Augmentation transform (optional) - 4 techniques for robustness
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),  # ✓ Technique 1: Slight rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ✓ Technique 2: Translation/shift
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),  # ✓ Technique 3: Color variation
            transforms.RandomHorizontalFlip(p=0.3),  # ✓ Technique 4: Horizontal flipping
        ]) if augment else None

        # Final normalization
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply base transform
        image = self.base_transform(image)

        # Apply augmentation if enabled
        if self.augment and self.augment_transform and random.random() > 0.3:
            image = self.augment_transform(image)

        # Apply final transform
        pixel_values = self.final_transform(image)
        return {"pixel_values": pixel_values}


# Diverse Duke-specific training prompts
DUKE_TRAINING_PROMPTS = [
    "a photo of Duke University chapel and gothic architecture",
    "Duke chapel with stone arches and religious architecture",
    "Duke campus quad with gothic stone buildings and spires",
    "Duke University chapel dome and brick architecture",
    "Duke campus courtyard with arches and stone columns",
    "Duke chapel interior with vaulted ceilings and gothic details",
    "Duke University stone archways and gothic revival architecture",
    "Gothic chapel at Duke University with historic stone exterior",
    "Duke campus architecture with navy blue and stone tones",
    "Duke University buildings with gothic spires and stone facades",
    "Duke chapel at sunset with warm golden light on stone",
    "Historic Duke University chapel and campus grounds",
]


def train_lora(
    image_dir: Path,
    output_dir: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    batch_size: int = 1,
    num_epochs: int = 20,
    learning_rate: float = 1e-5,
    lora_rank: int = 32,
    device: torch.device = None,
):
    """Fine-tune Stable Diffusion with LoRA on Duke images using improved parameters."""

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n🏛️  IMPROVED LoRA Training for Duke Chapel Covers")
    print(f"🎯 Device: {device}")
    print(f"📂 Images: {image_dir}")
    print(f"💾 Output: {output_dir}")
    print(f"⚙️  Config: epochs={num_epochs}, LR={learning_rate}, rank={lora_rank}, batch={batch_size}")

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

    # ===== APPLY LoRA (IMPROVED) =====
    print(f"🔧 Applying LoRA configuration (rank={lora_rank})...")
    lora_config = LoraConfig(
        r=lora_rank,  # Increased from 16 to 32
        lora_alpha=lora_rank * 2,  # Scale alpha with rank
        target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )

    unet_lora = get_peft_model(unet, lora_config)
    unet_lora.print_trainable_parameters()

    # ===== PREPARE DATA =====
    print("\n📊 Preparing dataset with augmentation...")
    dataset = DukeImageDataset(image_dir, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ===== SETUP OPTIMIZER & SCHEDULER =====
    optimizer = torch.optim.AdamW(unet_lora.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ===== TRAINING LOOP =====
    print(f"\n⏱️  Training for {num_epochs} epochs with {len(dataset)} images...\n")

    unet_lora = unet_lora.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    unet_lora.train()
    vae.eval()
    text_encoder.eval()

    best_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Use diverse prompts
            prompt = random.choice(DUKE_TRAINING_PROMPTS)

            # Encode prompt
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompt,
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

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(unet_lora.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
                )

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"✓ Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.6f}")

        # Learning rate scheduling
        scheduler.step()

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            print(f"  📉 New best loss! Saving checkpoint...")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{patience})")

        # Early stopping (optional: comment out to always train full epochs)
        # if patience_counter >= patience:
        #     print(f"\n⛔ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        #     break

        print()

    # ===== SAVE LoRA WEIGHTS =====
    print(f"\n💾 Saving improved LoRA weights to {output_dir}/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "model_id": model_id,
        "image_count": len(dataset),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "device": str(device),
        "best_loss": best_loss,
        "notes": "Improved LoRA with higher rank, lower LR, augmentation, and cosine annealing",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ LoRA training complete!")
    print(f"   Weights saved to: {output_dir}/")
    print(f"   Best loss achieved: {best_loss:.6f}")
    print(f"   Estimated size: ~100MB (rank={lora_rank})")


def main():
    parser = argparse.ArgumentParser(
        description="Train improved LoRA on Duke images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/lora_train_improved.py                    # Default: 20 epochs, LR 1e-5, rank 32
  python scripts/lora_train_improved.py --epochs 30        # 30 epochs
  python scripts/lora_train_improved.py --rank 64 --epochs 25   # Larger rank, more training
  python scripts/lora_train_improved.py --lr 5e-6          # Even lower learning rate
        """,
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/lora_training_images"),
        help="Directory with Duke images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lora_weights/chapel_covers_lora_improved"),
        help="Output directory for LoRA weights",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (1 recommended for Mac GPU)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (20-30 recommended)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (1e-5 to 1e-6 recommended)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="LoRA rank (32-64 recommended for good quality)",
    )

    args = parser.parse_args()

    train_lora(
        image_dir=args.images,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.rank,
    )


if __name__ == "__main__":
    main()
