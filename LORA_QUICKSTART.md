# LoRA Fine-tuning Quick Start

**Goal:** Learn Duke's visual style and inject it into generated album covers.

**Timeline:** 2-4 hours total
- Image collection: 15-30 min
- LoRA training: 60-90 min
- Integration & testing: 30-60 min

---

## Step 1: Collect Duke Images (15-30 min)

You need ~40-50 Duke/chapel photos. Two approaches:

### Option A: Manual Download (Easiest)
1. Go to [Unsplash.com](https://unsplash.com) or [Bing Image Search](https://bing.com/images)
2. Search for these terms, download 8-10 images each:
   - "Duke University chapel"
   - "Duke campus quad"
   - "Duke gothic architecture"
   - "Duke buildings"
   - "Duke aerial"
   - "Chapel gothic"
3. Save all images to: `data/lora_training_images/`
4. Rename them: `duke_001.jpg`, `duke_002.jpg`, etc.

### Option B: Automated Download (Requires setup)
```bash
cd /Users/sofiapereztenessa/Documents/cs372-final/music-cover-art-generator
python scripts/lora_download_images.py --output data/lora_training_images
```

**Verify images loaded:**
```bash
ls data/lora_training_images/ | wc -l  # Should show ~40-50
```

---

## Step 2: Install LoRA Dependencies (5 min)

```bash
cd /Users/sofiapereztenessa/Documents/cs372-final/music-cover-art-generator

# Activate venv (if not already)
source venv/bin/activate

# Install LoRA libraries
pip install peft accelerate --break-system-packages
```

**Verify:**
```bash
python -c "import peft; print('✓ peft installed')"
```

---

## Step 3: Train LoRA (60-90 min)

```bash
cd /Users/sofiapereztenessa/Documents/cs372-final/music-cover-art-generator
python scripts/lora_train.py \
    --images data/lora_training_images \
    --output lora_weights/chapel_covers_lora \
    --epochs 10 \
    --batch-size 2
```

**What's happening:**
- Loading Stable Diffusion (~2-3 GB on first run)
- Extracting Duke visual features from images
- Fine-tuning ~3% of model parameters (LoRA)
- Saving ~50MB weights

**Expected output:**
```
📥 Loading Stable Diffusion model...
🔧 Applying LoRA configuration...
📊 Preparing dataset...
Found 43 images in data/lora_training_images

⏱️  Training for 10 epochs with 43 images...
Epoch 1/10 | Batch 1/22 | Loss: 0.1234
Epoch 1/10 | Batch 5/22 | Loss: 0.0987
...
✅ LoRA training complete!
   Weights saved to: lora_weights/chapel_covers_lora/
   Size: ~50MB
```

**Troubleshooting:**
- **OOM (Out of Memory):** Reduce batch size: `--batch-size 1`
- **Slow on Mac:** Use float32 (we do by default). This is normal.
- **Missing diffusers:** `pip install diffusers --break-system-packages`

---

## Step 4: Test LoRA with Flask Server (15 min)

### Option A: Flask REST API
```bash
cd /Users/sofiapereztenessa/Documents/cs372-final/music-cover-art-generator
python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora
```

Then open http://localhost:5000

- Upload a song
- Generate cover with **"Use Duke Style"** toggle

### Option B: Streamlit App
```bash
cd /Users/sofiapereztenessa/Documents/cs372-final/music-cover-art-generator
streamlit run app/streamlit_app.py
```

Then manually use LoRA in code:
```python
from src.lora_integration import CoverArtPipelineWithLoRA

pipeline = CoverArtPipelineWithLoRA(
    cnn_checkpoint="models/cnn_default_best.pt"
)
pipeline.load_lora_weights("lora_weights/chapel_covers_lora", lora_scale=0.7)

image = pipeline.generate_from_prompt_text(
    "dark moody album cover",
    use_lora=True
)
image.show()
```

---

## Step 5: Integrate into Your App

### In Flask server:
```python
from src.lora_integration import CoverArtPipelineWithLoRA

pipeline = CoverArtPipelineWithLoRA(cnn_checkpoint, diffusion_model_id)
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

# Generate with Duke style
image = pipeline.generate_from_prompt_text(
    "indie album cover",
    use_lora=True,
    lora_scale=0.7  # Blend strength: 0-1
)
```

### LoRA Scale Tuning:
- **0.0:** Original Stable Diffusion (no Duke style)
- **0.5:** Subtle Duke influence
- **0.7:** Strong Duke style (recommended)
- **1.0:** Maximum Duke style (may be too intense)

---

## File Structure

```
music-cover-art-generator/
├── scripts/
│   ├── lora_download_images.py    # Download Duke images
│   └── lora_train.py               # Train LoRA weights
├── src/
│   ├── lora_integration.py         # LoRA pipeline wrapper
│   └── pipeline.py                 # Original pipeline
├── app/
│   ├── flask_server_lora.py        # Flask with LoRA support
│   └── streamlit_app.py            # Streamlit with manual LoRA
├── data/
│   └── lora_training_images/       # ← Place Duke images here
├── lora_weights/
│   └── chapel_covers_lora/         # ← Trained LoRA weights (auto-created)
└── models/
    └── cnn_default_best.pt         # CNN checkpoint (pre-existing)
```

---

## Common Workflows

### A/B Test: With vs Without LoRA
```python
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

prompt = "dark moody indie album cover"

# Original
img_original = pipeline.generate_from_prompt_text(prompt, use_lora=False)

# With LoRA
img_with_lora = pipeline.generate_from_prompt_text(prompt, use_lora=True, lora_scale=0.7)

# Display side-by-side
import matplotlib.pyplot as plt
plt.subplot(121); plt.imshow(img_original); plt.title("Original")
plt.subplot(122); plt.imshow(img_with_lora); plt.title("With Duke LoRA")
plt.show()
```

### Generate Batch with Different Scales
```python
prompts = [
    "dark moody album cover",
    "colorful indie artwork",
    "gothic chapel vibes"
]

for scale in [0.0, 0.5, 0.7, 1.0]:
    images = pipeline.generate_image_batch(
        prompts,
        use_lora=True,
        lora_scale=scale
    )
    # Save images
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **LoRA not loading** | Check path: `ls -la lora_weights/chapel_covers_lora/` |
| **"No images found"** | Ensure `data/lora_training_images/` has JPG/PNG files |
| **Memory error during training** | Reduce batch size: `--batch-size 1` |
| **Training is very slow** | Normal on Mac. GPU training is ~3x faster. |
| **Generated images don't look "Duke"** | Add more diverse Duke images, retrain, or increase epochs |

---

## Next Steps

1. **Improve training data:** Collect 100+ diverse Duke images for better results
2. **Experiment with LoRA rank:** Try `r=8` (smaller) or `r=32` (larger) in `lora_train.py`
3. **Fine-tune prompts:** Combine LoRA with genre-specific prompts for best results
4. **Deploy:** Use `flask_server_lora.py` as production endpoint

---

## References

- **LoRA Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Diffusers Docs:** https://huggingface.co/docs/diffusers/
- **PEFT Library:** https://huggingface.co/docs/peft/

---

**Questions?** Check the code comments in:
- `scripts/lora_train.py` — Training logic
- `src/lora_integration.py` — LoRA loading/inference
- `app/flask_server_lora.py` — REST API integration
