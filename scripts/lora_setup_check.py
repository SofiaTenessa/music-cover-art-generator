#!/usr/bin/env python3
"""
LoRA setup validation script.

Checks all dependencies and setup before training.

Usage:
    python scripts/lora_setup_check.py
"""
from pathlib import Path
import sys
import importlib

def check_import(module_name: str, package_name: str = None) -> bool:
    """Try importing a module."""
    if package_name is None:
        package_name = module_name
    try:
        importlib.import_module(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} — pip install {package_name} --break-system-packages")
        return False

def main():
    print("\n" + "=" * 70)
    print("🏛️  Chapel Covers LoRA Setup Checker")
    print("=" * 70 + "\n")

    all_good = True

    # Check Python version
    print("📌 Python Version")
    print(f"  {sys.version}")
    if sys.version_info < (3, 8):
        print("  ⚠️  Python 3.8+ required")
        all_good = False
    else:
        print("  ✓ OK")
    print()

    # Check core dependencies
    print("📦 Core Dependencies")
    core_deps = [
        ("torch", "torch"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
    ]
    for module, name in core_deps:
        if not check_import(module, name):
            all_good = False
    print()

    # Check diffusers
    print("🎨 Diffusers (Stable Diffusion)")
    if not check_import("diffusers"):
        print("    Run: pip install diffusers --break-system-packages")
        all_good = False
    print()

    # Check LoRA libraries
    print("⚙️  LoRA Libraries")
    lora_deps = [
        ("peft", "peft"),
        ("accelerate", "accelerate"),
    ]
    for module, name in lora_deps:
        if not check_import(module, name):
            all_good = False
    print()

    # Check Flask (optional, for server)
    print("🌐 Flask (Optional, for REST API)")
    check_import("flask", "Flask")
    print()

    # Check file structure
    print("📁 File Structure")
    project_root = Path(__file__).resolve().parent.parent
    required_dirs = [
        "src",
        "app",
        "scripts",
    ]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ — missing")
            all_good = False
    print()

    # Check models
    print("🤖 Models")
    cnn_path = project_root / "models" / "cnn_default_best.pt"
    if cnn_path.exists():
        print(f"  ✓ CNN checkpoint found: {cnn_path.name}")
    else:
        print(f"  ✗ CNN checkpoint missing: models/cnn_default_best.pt")
        print(f"    Train with: python -m src.train")
        all_good = False
    print()

    # Check/create data directories
    print("📂 Data Directories")
    data_dirs = [
        ("data/lora_training_images", "Duke training images"),
        ("lora_weights", "LoRA weights output"),
    ]
    for dir_name, desc in data_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ ({desc})")
            # Check if images exist
            if "training_images" in dir_name:
                images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
                if images:
                    print(f"    → {len(images)} images found")
                else:
                    print(f"    → No images yet (download ~40-50)")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {dir_name}/ ({desc})")
    print()

    # Summary
    print("=" * 70)
    if all_good:
        print("✅ Setup looks good! Ready to train LoRA.")
        print("\nNext steps:")
        print("  1. Place Duke images in: data/lora_training_images/")
        print("  2. Run: python scripts/lora_train.py")
        print("  3. Test: python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora")
    else:
        print("⚠️  Setup incomplete. Fix issues above, then retry.")
        print("\nInstall missing packages:")
        print("  pip install peft accelerate diffusers --break-system-packages")
    print("=" * 70 + "\n")

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
