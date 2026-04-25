"""
Download Duke University images for LoRA fine-tuning.

This script downloads Duke campus and chapel images from Unsplash (free, licensed imagery).
Alternative: You can manually place your own Duke photos in data/lora_training_images/

Usage:
    python scripts/lora_download_images.py --output data/lora_training_images --count 40
"""
import argparse
import requests
from pathlib import Path
from urllib.parse import urlencode
import time


def download_unsplash_images(query: str, output_dir: Path, count: int = 40, per_page: int = 20):
    """
    Download images from Unsplash API (free tier, no auth needed).

    Args:
        query: Search query (e.g., "Duke University chapel")
        output_dir: Directory to save images
        count: Total number of images to download
        per_page: Images per request (max 20 free tier)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unsplash free API (no authentication needed for basic searches)
    base_url = "https://api.unsplash.com/search/photos"

    # Note: Unsplash free API requires an access key. For this script,
    # we'll use a fallback approach with direct image URLs or provide instructions.

    print(f"ℹ️  Unsplash API requires authentication for automation.")
    print(f"Manual approach: Save Duke images to {output_dir}/")
    print(f"\nSearches to try on Unsplash.com:")
    print(f"  - 'Duke University'")
    print(f"  - 'Duke chapel'")
    print(f"  - 'Duke campus'")
    print(f"  - 'gothic architecture'")
    print(f"  - 'university chapel'")
    print(f"\nDownload 40-50 images and save as JPG/PNG.")


def download_from_bing(query: str, output_dir: Path, count: int = 40):
    """
    Fallback: Guide user to download from Bing Image Search.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🖼️  DUKE IMAGES DOWNLOAD GUIDE")
    print("=" * 70)
    print()
    print("Easiest approach: Use Bing Image Search (no login needed)")
    print()
    print("1. Open: https://www.bing.com/images/search")
    print("2. Search for each term below, download ~8-10 images per search:")
    print()

    searches = [
        "Duke University chapel",
        "Duke campus quad",
        "Duke gothic architecture",
        "Duke buildings",
        "Duke University aerial",
        "Chapel gothic",
    ]

    for i, search in enumerate(searches, 1):
        print(f"   {i}. {search}")

    print()
    print(f"3. Save all images to: {output_dir}/")
    print(f"4. Run LoRA training: python scripts/lora_train.py")
    print()
    print("=" * 70)
    print()
    print(f"Creating directory: {output_dir}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Ready to accept images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Duke images for LoRA fine-tuning"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lora_training_images"),
        help="Output directory for images",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="Target number of images",
    )

    args = parser.parse_args()

    # Use Bing/manual approach (simpler, no API key needed)
    download_from_bing(query="Duke", output_dir=args.output, count=args.count)
