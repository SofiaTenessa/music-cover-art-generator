# Setup Instructions

## Requirements

- Python 3.10 or newer
- ~5 GB free disk space (GTZAN dataset + model weights)
- Recommended: Apple Silicon Mac (M1/M2/M3/M4), NVIDIA GPU, or Google Colab
- Stable Diffusion needs at least 8 GB RAM (16 GB+ recommended)

## Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd music-cover-art-generator
```

## Step 2: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

## Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

On Apple Silicon (M1/M2/M3/M4), PyTorch will automatically use the MPS backend — no extra setup needed.

On NVIDIA GPUs, install the CUDA-enabled PyTorch build:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 4: Download the GTZAN dataset

GTZAN is a classic music genre classification dataset: 1000 30-second audio clips across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).

Download from Kaggle:
```bash
# Option A: Kaggle CLI (requires kaggle account + API key)
pip install kaggle
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d data/
```

Or download manually from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification and unzip into `data/`.

The expected structure after extraction:
```
data/
  genres_original/
    blues/
      blues.00000.wav
      ...
    classical/
    country/
    ...
```

## Step 5: Train the genre classifier

**Option A — Train locally (Apple Silicon or NVIDIA GPU):**
```bash
python -m src.train --data-dir data/genres_original --epochs 50 --batch-size 32
```
Training should take 10-20 minutes on M4 Pro and produce `models/cnn_best.pt`.

**Option B — Train on Colab (free T4 GPU):**
Open `notebooks/train_cnn_colab.ipynb` in Google Colab, mount your Drive, upload the dataset, and run all cells. Download the resulting `cnn_best.pt` and place it in `models/`.

## Step 6: Generate a cover art

```bash
python -m src.pipeline --audio path/to/song.wav --output outputs/cover.png
```

First run will download Stable Diffusion weights (~4 GB) — this takes a few minutes but only happens once.

## Step 7: Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Open your browser to http://localhost:8501.

## Troubleshooting

**"MPS backend not available"** — Make sure you're on PyTorch 2.0+ and macOS 12.3+. Verify with:
```python
import torch
print(torch.backends.mps.is_available())  # should print True
```

**"Out of memory" when running Stable Diffusion** — Try `--generator sdxl-turbo` in the pipeline to use a smaller, faster model, or run generation on Colab.

**Librosa fails to load .wav files** — Install FFmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: https://ffmpeg.org/download.html

**Kaggle download fails** — Make sure you've accepted the dataset's terms on the Kaggle website and placed your API token at `~/.kaggle/kaggle.json`.
