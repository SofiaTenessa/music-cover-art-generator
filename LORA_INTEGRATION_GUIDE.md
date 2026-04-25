# LoRA Integration Guide for Chapel Covers

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Uploads Audio                      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   ┌────▼──────────┐    ┌────────▼────────┐
   │  CNN Genre    │    │  Audio Features │
   │  Classifier   │    │  (Mood, Tempo)  │
   └────┬──────────┘    └────────┬────────┘
        │                        │
        └────────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Build Text Prompt    │
         │  (Genre + Mood)       │
         └───────────┬───────────┘
                     │
      ┌──────────────┴──────────────┐
      │                             │
  ┌───▼──────────────┐    ┌────────▼─────────┐
  │ Stable Diffusion │    │  Duke LoRA      │
  │  (Base Model)    │    │  Weights (~50MB)│
  └───┬──────────────┘    └────────┬─────────┘
      │                           │
      └───────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │  Combine via    │
         │  LoRA Scale     │
         │  (0.0 - 1.0)    │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Generate Image │
         │  with Duke      │
         │  Visual Style   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Display Cover  │
         │  Art to User    │
         └─────────────────┘
```

---

## Implementation Details

### 1. Flask Server with LoRA

**File:** `app/flask_server_lora.py`

```python
from src.lora_integration import CoverArtPipelineWithLoRA

# Initialize with LoRA support
pipeline = CoverArtPipelineWithLoRA(
    cnn_checkpoint="models/cnn_default_best.pt",
    diffusion_model_id="runwayml/stable-diffusion-v1-5"
)

# Load trained Duke style weights
pipeline.load_lora_weights("lora_weights/chapel_covers_lora", lora_scale=0.7)

# Generate with Duke style
image = pipeline.generate_from_prompt_text(
    prompt_text="dark moody indie album cover",
    negative_prompt="blurry, low quality",
    use_lora=True,      # Enable Duke style
    lora_scale=0.7      # Strength: 0-1
)
```

**REST API:**
```bash
# Start server
python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora

# GET /status
curl http://localhost:5000/status
# → {"lora_loaded": true, "lora_path": "lora_weights/chapel_covers_lora"}

# POST /generate
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "dark moody indie cover",
    "use_lora": true,
    "lora_scale": 0.7
  }'
```

---

### 2. Extending Streamlit App

**File:** `app/streamlit_app.py`

Add this after step 3 (Generate):

```python
if st.session_state.generated_image is not None:
    st.markdown('<h3>🎭 LoRA Controls</h3>', unsafe_allow_html=True)
    
    # Add LoRA toggles
    use_lora = st.checkbox(
        "Enhance with Duke style (LoRA)",
        value=True,
        help="Apply learned Duke visual style"
    )
    
    if use_lora:
        lora_scale = st.slider(
            "Style Intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0.0 = original, 1.0 = maximum Duke style"
        )
    else:
        lora_scale = 0.0
    
    if st.button("Regenerate with LoRA Settings"):
        with st.spinner("Regenerating with Duke style..."):
            new_image = st.session_state.pipeline.generate_from_prompt_text(
                prompt_text=st.session_state.current_prompt.positive,
                negative_prompt=NEGATIVE_PROMPT,
                use_lora=use_lora,
                lora_scale=lora_scale
            )
            st.session_state.generated_image = new_image
            st.rerun()
```

---

### 3. CLI Usage

**Generate single image with LoRA:**

```python
# Create script: generate_with_lora.py
from src.lora_integration import CoverArtPipelineWithLoRA

pipeline = CoverArtPipelineWithLoRA(
    cnn_checkpoint="models/cnn_default_best.pt"
)
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

prompts = [
    ("dark moody", 0.5),
    ("colorful indie", 0.7),
    ("gothic chapel", 1.0),
]

for prompt, scale in prompts:
    image = pipeline.generate_from_prompt_text(
        prompt,
        use_lora=True,
        lora_scale=scale
    )
    image.save(f"cover_{scale}.png")
```

Run:
```bash
python generate_with_lora.py
```

---

## Training Workflow

### Full Training Pipeline

```bash
# 1. Check setup
python scripts/lora_setup_check.py

# 2. Download/place images
python scripts/lora_download_images.py

# 3. Train LoRA
python scripts/lora_train.py \
    --images data/lora_training_images \
    --output lora_weights/chapel_covers_lora \
    --epochs 10 \
    --batch-size 2

# 4. Test with Flask
python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora

# 5. Test with Streamlit
streamlit run app/streamlit_app.py
```

**Expected timeline:**
- Setup check: 1 min
- Image download: 20 min
- Training: 60-90 min
- Testing: 10 min
- **Total: 2-3 hours**

---

## A/B Testing Guide

### Compare LoRA scales

```python
from src.lora_integration import CoverArtPipelineWithLoRA
from PIL import Image
import matplotlib.pyplot as plt

pipeline = CoverArtPipelineWithLoRA("models/cnn_default_best.pt")
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

prompt = "dark moody indie album cover"
scales = [0.0, 0.3, 0.5, 0.7, 1.0]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, scale in zip(axes, scales):
    image = pipeline.generate_from_prompt_text(
        prompt,
        use_lora=(scale > 0),
        lora_scale=scale
    )
    ax.imshow(image)
    ax.set_title(f"Scale: {scale}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("lora_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Compare different Duke images

```python
# Train LoRA on different image sets
for image_set in ["chapel_only", "campus_only", "all_duke"]:
    print(f"Training on {image_set}...")
    
    import subprocess
    subprocess.run([
        "python", "scripts/lora_train.py",
        "--images", f"data/{image_set}",
        "--output", f"lora_weights/{image_set}_lora",
        "--epochs", "10"
    ])

# Compare outputs
for scale in [0.5, 0.7, 1.0]:
    for model in ["chapel_only", "campus_only", "all_duke"]:
        pipeline.load_lora_weights(f"lora_weights/{model}_lora")
        image = pipeline.generate_from_prompt_text(prompt, use_lora=True, lora_scale=scale)
        image.save(f"compare_{model}_{scale}.png")
```

---

## Production Deployment

### Option 1: Flask with LoRA

```bash
# Start production server
python app/flask_server_lora.py \
    --lora-path lora_weights/chapel_covers_lora \
    --host 0.0.0.0 \
    --port 8000
```

### Option 2: Merge LoRA into Model (One-time)

```python
# Merge LoRA weights into base model
from src.lora_integration import CoverArtPipelineWithLoRA

pipeline = CoverArtPipelineWithLoRA("models/cnn_default_best.pt")
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

# Merge LoRA into UNet
pipeline._ensure_sd()
pipeline._sd_pipe.unet = pipeline._sd_pipe.unet.merge_and_unload()

# Save merged model
pipeline._sd_pipe.save_pretrained("models/chapel_covers_duke_merged")

# Now use merged model (no LoRA loading needed)
# Slightly faster inference, same results
```

---

## Parameter Tuning

### LoRA Rank (in `lora_train.py`)

```python
LoraConfig(r=8)     # Small: 20MB, less capacity
LoraConfig(r=16)    # Medium: 50MB, balanced (default)
LoraConfig(r=32)    # Large: 100MB, more capacity
```

**When to use:**
- `r=8`: Quick experiments, limited training data
- `r=16`: Most use cases (recommended)
- `r=32`: Large datasets, complex styles

### LoRA Scale (in inference)

```python
# Subtle
pipeline.generate_from_prompt_text(prompt, use_lora=True, lora_scale=0.3)

# Balanced
pipeline.generate_from_prompt_text(prompt, use_lora=True, lora_scale=0.7)

# Strong
pipeline.generate_from_prompt_text(prompt, use_lora=True, lora_scale=1.0)
```

### Training Epochs

```bash
# Quick test
python scripts/lora_train.py --epochs 3

# Standard
python scripts/lora_train.py --epochs 10

# Extended
python scripts/lora_train.py --epochs 20
```

---

## Monitoring & Debugging

### During Training

```bash
# Monitor loss in real-time
python scripts/lora_train.py --epochs 10 2>&1 | grep "Loss:"
```

### After Training

```python
# Check LoRA metadata
import json
with open("lora_weights/chapel_covers_lora/metadata.json") as f:
    meta = json.load(f)
    print(f"Trained on {meta['image_count']} images")
    print(f"Epochs: {meta['epochs']}")
```

### Quality Assessment

```python
# Visual inspection
pipeline.load_lora_weights("lora_weights/chapel_covers_lora")

# Test different genres
for genre in ["rock", "indie", "jazz", "pop"]:
    prompt = f"{genre} album cover, duke style"
    image = pipeline.generate_from_prompt_text(prompt, use_lora=True)
    image.save(f"test_{genre}.png")

# Manual review: Do images show Duke style?
```

---

## Troubleshooting Integration

| Issue | Solution |
|-------|----------|
| Flask won't start with LoRA | Check path: `ls lora_weights/chapel_covers_lora/adapter_model.bin` |
| Images don't look different with/without LoRA | Training data quality matters; retrain with 50+ diverse images |
| Generation is slow | Normal (~30-90s). LoRA overhead is negligible. |
| LoRA scale doesn't seem to work | Verify `use_lora=True` is set in generation call |
| Out of memory during generation | Reduce image size in prompt or use smaller LoRA rank |

---

## Next Steps

1. **Collect images** → `data/lora_training_images/`
2. **Install dependencies** → `pip install peft accelerate`
3. **Train LoRA** → `python scripts/lora_train.py`
4. **Test Flask** → `python app/flask_server_lora.py --lora-path ...`
5. **Iterate** → Adjust LoRA rank, training epochs, scale

**Questions?** See LORA_QUICKSTART.md or LORA_IMPLEMENTATION_SUMMARY.md
