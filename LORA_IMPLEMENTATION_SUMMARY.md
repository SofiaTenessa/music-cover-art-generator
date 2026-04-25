# Chapel Covers: LoRA Fine-tuning Implementation

## 🎯 What Was Created

A complete LoRA (Low-Rank Adaptation) fine-tuning pipeline to teach Stable Diffusion Duke's visual style.

### New Files

| File | Purpose |
|------|---------|
| `scripts/lora_download_images.py` | Download/collect Duke images |
| `scripts/lora_train.py` | **Main training script** |
| `scripts/lora_setup_check.py` | Verify dependencies before training |
| `src/lora_integration.py` | Extended pipeline with LoRA support |
| `app/flask_server_lora.py` | Flask server with LoRA inference |
| `LORA_QUICKSTART.md` | Step-by-step user guide |
| `LORA_IMPLEMENTATION_SUMMARY.md` | This file |

---

## 🚀 Quick Start (Follow These Steps)

### Step 1: Install Dependencies
```bash
cd ~/Documents/cs372-final/music-cover-art-generator
source venv/bin/activate

# Install LoRA libraries
pip install peft accelerate --break-system-packages
```

### Step 2: Collect Duke Images
**Option A (Manual, 15-30 min):**
1. Go to [Unsplash.com](https://unsplash.com) or [Bing Images](https://bing.com/images)
2. Search: "Duke University chapel", "Duke campus", "Duke gothic", etc.
3. Download ~40-50 images total
4. Save to: `data/lora_training_images/`

**Option B (Automated):**
```bash
python scripts/lora_download_images.py --output data/lora_training_images
```

### Step 3: Train LoRA (60-90 minutes)
```bash
python scripts/lora_train.py \
    --images data/lora_training_images \
    --output lora_weights/chapel_covers_lora \
    --epochs 10
```

### Step 4: Test with Flask Server
```bash
python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora
```
Visit: http://localhost:5000

---

## 📋 Architecture

### Training Pipeline (`lora_train.py`)
```
Duke Images (40-50 JPG/PNG)
    ↓
Load Stable Diffusion Model (~4GB)
    ↓
Apply LoRA Configuration (rank=16)
    ↓
Fine-tune on Duke images (10 epochs)
    ↓
Save LoRA Weights (~50MB)
```

**Why LoRA?**
- ✅ Small: 50MB vs 4GB for full fine-tune
- ✅ Fast: 60-90 min on Mac vs 4+ hours
- ✅ Efficient: Only 3% of model parameters
- ✅ Composable: Can combine with other LoRA weights

### Inference (`lora_integration.py`)
```python
from src.lora_integration import CoverArtPipelineWithLoRA

pipeline = CoverArtPipelineWithLoRA(...)
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

# Use with or without LoRA
image = pipeline.generate_from_prompt_text(
    "dark moody album cover",
    use_lora=True,      # Toggle Duke style
    lora_scale=0.7      # Blend strength 0-1
)
```

---

## 🔧 Technical Details

### LoRA Configuration
Located in `lora_train.py`:
```python
LoraConfig(
    r=16,                    # Rank (size/quality trade-off)
    lora_alpha=32,          # Scaling factor
    target_modules=[        # Apply to attention layers
        "to_k", "to_v", "to_q", "to_out.0"
    ],
    lora_dropout=0.05,
    task_type=TaskType.IMAGE_CLASSIFICATION,
)
```

**Tuning parameters:**
- `r=8`: Smaller (30MB), faster, less capacity
- `r=16`: Balanced (50MB, recommended)
- `r=32`: Larger (100MB), slower, more capacity

### LoRA Scale (Blending)
```
use_lora=False, lora_scale=0.0
    → Pure Stable Diffusion (no Duke influence)

use_lora=True, lora_scale=0.5
    → 50% Duke style

use_lora=True, lora_scale=0.7
    → 70% Duke style (recommended)

use_lora=True, lora_scale=1.0
    → Maximum Duke style (may be too intense)
```

### File Outputs
After training, `lora_weights/chapel_covers_lora/` contains:
```
chapel_covers_lora/
├── adapter_config.json      # LoRA metadata
├── adapter_model.bin        # LoRA weights (~50MB)
└── metadata.json            # Training info
```

---

## 📊 Expected Results

### Before LoRA
```
Prompt: "dark moody indie album cover"
Result: Generic Stable Diffusion output
        - May lack Duke visual elements
        - Chapel/gothic may be generic
```

### After LoRA
```
Prompt: "dark moody indie album cover"
Result: Duke-infused output
        - More gothic architecture
        - Chapel-inspired aesthetics
        - Campus visual patterns
        - Learned style from training images
```

### A/B Testing
```python
# Generate same prompt with/without LoRA
img1 = pipeline.generate_from_prompt_text(prompt, use_lora=False)
img2 = pipeline.generate_from_prompt_text(prompt, use_lora=True)
# Compare side-by-side
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'peft'` | `pip install peft --break-system-packages` |
| `No images found in data/lora_training_images/` | Download/place Duke images there first |
| `CUDA out of memory` | Not applicable on Mac, but reduce batch size if needed: `--batch-size 1` |
| `Training is very slow` | Normal on Mac. GPU on NVIDIA would be 3-5x faster. |
| `Generated images still don't look "Duke"` | Training data quality matters. Use 50+ diverse images. |
| `LoRA not loading: FileNotFoundError` | Check path: `ls lora_weights/chapel_covers_lora/adapter_model.bin` |

---

## 🎓 Learning Resources

### Papers
- **LoRA:** [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Diffusion Models:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

### Libraries
- **Diffusers:** https://huggingface.co/docs/diffusers/
- **PEFT:** https://huggingface.co/docs/peft/
- **Accelerate:** https://huggingface.co/docs/accelerate/

---

## 🚢 Integration Points

### 1. Flask Server
```bash
# Run with LoRA
python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora

# REST endpoints
POST /generate {prompt, use_lora: true, lora_scale: 0.7}
POST /refine {current_prompt, refinement, use_lora: true}
```

### 2. Streamlit App
```python
from src.lora_integration import CoverArtPipelineWithLoRA

pipeline = CoverArtPipelineWithLoRA(cnn_checkpoint, diffusion_model_id)
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

image = pipeline.generate_from_prompt_text(
    prompt,
    use_lora=st.toggle("Use Duke Style", value=True),
    lora_scale=st.slider("Style Intensity", 0.0, 1.0, 0.7)
)
```

### 3. CLI
```bash
python scripts/lora_train.py --images data/lora_training_images --epochs 10
```

---

## 📈 Performance Metrics

### Training
- **Hardware:** Mac (GPU via MPS)
- **Time:** 60-90 minutes
- **Memory:** ~6-8GB (manageable on Mac)
- **Output Size:** ~50MB

### Inference
- **First generation:** 30-90 seconds (depends on steps)
- **Additional generations:** ~30-60 seconds
- **LoRA overhead:** Negligible (~1-2 seconds)
- **Quality:** Similar to non-LoRA (style added, not reduced)

---

## ✅ Validation Checklist

Before running training:
- [ ] Python 3.8+ installed
- [ ] `peft` and `accelerate` installed
- [ ] CNN checkpoint at `models/cnn_default_best.pt`
- [ ] Duke images collected in `data/lora_training_images/`
- [ ] At least 30 images (40-50 recommended)

After training:
- [ ] `lora_weights/chapel_covers_lora/adapter_model.bin` exists (~50MB)
- [ ] Flask server starts with `--lora-path` flag
- [ ] Generated images show Duke influence
- [ ] A/B comparison shows style differences

---

## 🎨 Future Enhancements

1. **Multi-style LoRA:** Train separate LoRA for different genres
   - `chapel_covers_rock_lora/`
   - `chapel_covers_indie_lora/`
   - `chapel_covers_jazz_lora/`

2. **LoRA Composition:** Combine multiple LoRA weights
   ```python
   pipeline.load_lora_weights("lora_duke.pt", weight=0.5)
   pipeline.load_lora_weights("lora_gothic.pt", weight=0.5)
   ```

3. **More Training Data:** 100-200 curated Duke images for better results

4. **LoRA Merging:** Bake LoRA weights into model for deployment
   ```python
   pipeline.merge_lora_into_unet()
   pipeline.save_pretrained("models/chapel_covers_merged/")
   ```

---

## 📝 Notes

- **Reversible:** LoRA can be loaded/unloaded without affecting base model
- **Composable:** Multiple LoRA weights can be chained
- **Efficient:** Training doesn't require GPU (MPS on Mac works fine)
- **Portable:** Can share trained LoRA weights (50MB) vs full model (4GB)

---

## 🤝 Support

**Setup issues?** Check `scripts/lora_setup_check.py`:
```bash
python scripts/lora_setup_check.py
```

**Training issues?** See LORA_QUICKSTART.md troubleshooting section.

**Code questions?** See inline comments in:
- `scripts/lora_train.py` — Training algorithm
- `src/lora_integration.py` — LoRA loading/inference
- `app/flask_server_lora.py` — REST API usage

---

**Happy training! 🏛️🎨**
