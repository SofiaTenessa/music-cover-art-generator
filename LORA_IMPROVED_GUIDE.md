# LoRA Training Improvements Guide

## Problem Statement

The original LoRA training (10 epochs, rank 16, LR 5e-5) produced models that could somewhat learn Duke aesthetics, but:
- Generated repetitive basilica images regardless of training image diversity
- Didn't learn fine details of Duke architecture effectively
- Loss didn't converge smoothly

The original refinement system tried to force compliance through repetition and caps, but hit fundamental Stable Diffusion 1.5 limitations (e.g., "add rain" doesn't reliably add rain).

## Improvements Made

### 1. **Increased LoRA Rank (16 → 32)**
- **Why:** Higher rank = more parameters = more model capacity
- **Trade-off:** Larger weights (~100MB vs 50MB), but much richer Duke aesthetic learning
- **Impact:** Can now represent more nuanced architectural details

### 2. **Lower Learning Rate (5e-5 → 1e-5)**
- **Why:** Stable Diffusion is large and sensitive; smaller steps prevent catastrophic forgetting
- **Trade-off:** Takes slightly longer to train
- **Impact:** Smoother convergence, less risk of overfitting or distorting base model

### 3. **More Training Epochs (10 → 20-30)**
- **Why:** Need more iterations to fully learn a visual style from only 41 images
- **Trade-off:** ~3x training time
- **Impact:** Model can cycle through all images multiple times, learning robust features

### 4. **Data Augmentation**
New `DukeImageDataset` includes:
- Random rotations (±15°)
- Random translations (±10% shift)
- Color jitter (brightness, contrast, saturation variations)

**Why:** Augmentation prevents overfitting, helps model learn style at different angles/lighting
**Impact:** More generalizable Duke aesthetics across diverse compositions

### 5. **Diverse Training Prompts**
Replaced single prompt with 12 Duke-specific prompts:
```python
"Duke chapel with stone arches and religious architecture"
"Duke campus quad with gothic stone buildings and spires"
"Gothic chapel at Duke University with historic stone exterior"
# ... 9 more variations
```

**Why:** Each prompt teaches the model different aspects of Duke
**Impact:** More robust style learning, less overfitting to single architectural feature

### 6. **Learning Rate Scheduling**
Added cosine annealing (smooth LR decay):
- Starts at `lr` (1e-5)
- Gradually decreases to ~0 by final epoch
- **Impact:** Better fine-tuning at later epochs, prevents overshooting

### 7. **Gradient Clipping**
Added `torch.nn.utils.clip_grad_norm(max_norm=1.0)`:
- **Why:** Prevents exploding gradients on Mac GPU
- **Impact:** More stable training, fewer convergence issues

## How to Use

### Quick Start (Improved)
```bash
# Using improved defaults: 20 epochs, rank 32, LR 1e-5
python scripts/lora_train_improved.py

# Output: lora_weights/chapel_covers_lora_improved/
```

### Custom Configuration
```bash
# Very thorough training (recommended for best results)
python scripts/lora_train_improved.py \
  --epochs 30 \
  --rank 32 \
  --lr 1e-5

# Faster training (less time, slightly lower quality)
python scripts/lora_train_improved.py \
  --epochs 15 \
  --rank 16 \
  --lr 1e-5

# Maximum capacity (best quality, requires more VRAM)
python scripts/lora_train_improved.py \
  --epochs 25 \
  --rank 64 \
  --lr 1e-5
```

### Update Flask Server to Use Improved Weights
```bash
# After training completes, update the server:
python app/flask_server_lora.py \
  --lora-path lora_weights/chapel_covers_lora_improved
```

## Hyperparameter Recommendations

| Goal | Config |
|------|--------|
| **Fast iteration** | `--epochs 15 --rank 16 --lr 1e-5` |
| **Balanced** (recommended) | `--epochs 20 --rank 32 --lr 1e-5` |
| **High quality** | `--epochs 30 --rank 32 --lr 1e-5` |
| **Maximum detail** | `--epochs 25 --rank 64 --lr 1e-5` |

## Expected Results

### Training Loss Curve
With improved params, you should see:
```
Epoch 1: Loss ~0.15-0.17
Epoch 5: Loss ~0.12-0.13
Epoch 10: Loss ~0.10-0.11
Epoch 15: Loss ~0.09-0.10
Epoch 20: Loss ~0.08-0.09
```

(Loss may fluctuate due to augmentation, but trend should be downward)

### Generated Images
**With improved LoRA:**
- More detailed gothic architecture (not just basilica)
- Better preservation of image content (less oversaturation of Duke style)
- More natural lighting and composition
- Duke chapel visible in diverse contexts (not repetitive)

## What Didn't Get Fixed

### Rain Feature (Removed)
❌ Stable Diffusion v1.5 fundamentally struggles with specific object addition requests like "add rain"
- Even with aggressive prompting and high weights, rain often doesn't render
- Other precipitation effects equally unreliable
- **Solution:** Remove from refinement system, focus on style/mood refinements that work

### Effective Refinements (What Actually Works)
✅ These refinements reliably work with Stable Diffusion 1.5:
- **Darkness/Brightness:** "darker", "brighter", "more light"
- **Colors:** "add blue", "add gold", "monochrome", "more colorful"
- **Mood/Atmosphere:** "dramatic", "peaceful", "moody", "energetic"
- **Detail:** "sharper", "softer", "more detail", "minimal"
- **Lighting:** "sunset", "sunrise", "night", "dusk"
- **Weather:** "foggy", "misty", "overcast", "stormy" (atmospheric effects, not objects)

❌ Avoid these (unreliable with Stable Diffusion 1.5):
- Specific objects: "add rain", "add birds", "add people"
- Complex scenes: "add a crowd", "add specific buildings"
- Too many details: "add rain, wind, and lightning"

## Training Time Estimates

On Mac with MPS (M1/M2/M3):
| Config | Time |
|--------|------|
| `--epochs 15 --rank 16` | ~20-30 min |
| `--epochs 20 --rank 32` | ~50-70 min |
| `--epochs 30 --rank 32` | ~80-120 min |
| `--epochs 25 --rank 64` | ~100-150 min |

On NVIDIA GPU:
- ~3-5x faster than MPS estimates above


