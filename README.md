# Music Genre → Cover Art Generator

A multi-stage machine learning pipeline that takes an audio clip as input, predicts its musical genre and mood, and generates an AI album cover that visually reflects the music.

Final project for CS 372: Introduction to Applied Machine Learning, Spring 2026.

## What it Does

This project turns any 30-second audio clip into a custom album cover. The system works in three stages: (1) a custom-trained convolutional neural network analyzes a mel-spectrogram of the input audio to predict its genre and extract mood features, (2) a prompt-construction module translates these predictions into a descriptive text prompt, and (3) a pretrained Stable Diffusion model generates an album cover from that prompt. The goal is to explore cross-modal generation — translating information from the audio modality into the visual modality — in a way that produces creative, genre-faithful artwork for music that doesn't have existing cover art (independent artists, personal recordings, AI-generated music).

## Quick Start

```bash
# 1. Clone and set up environment (see SETUP.md for details)
git clone <your-repo-url>
cd music-cover-art-generator
pip install -r requirements.txt

# 2. Train the genre classifier (or download pretrained weights)
#    See notebooks/train_cnn_colab.ipynb for GPU training on Colab
python -m src.train

# 3. Run the pipeline end-to-end on an audio file
python -m src.pipeline --audio path/to/song.wav --output cover.png

# 4. Or launch the Streamlit web app
streamlit run app/streamlit_app.py
```

## Video Links

- **Demo video (non-technical):** [link-to-demo-video]
- **Technical walkthrough:** [link-to-walkthrough-video]

## Evaluation

*Placeholder — fill in with your actual results.*

| Model | Test Accuracy | F1 (macro) | Inference time |
|-------|---------------|------------|----------------|
| Random baseline | 10.0% | 0.10 | — |
| Logistic regression on MFCCs | TBD | TBD | TBD |
| Custom CNN (ours) | TBD | TBD | TBD |
| Custom CNN + SpecAugment | TBD | TBD | TBD |

Qualitative evaluation of generated cover art: see `docs/qualitative_results.md`.

## Project Structure

```
music-cover-art-generator/
├── src/                  # Core ML code
│   ├── preprocessing.py  # Audio → mel-spectrogram
│   ├── dataset.py        # PyTorch Dataset & augmentation
│   ├── model.py          # Custom CNN architecture
│   ├── baselines.py      # Baseline models for comparison
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Metrics, confusion matrix, error analysis
│   ├── prompt_builder.py # Genre/mood → text prompt
│   └── pipeline.py       # End-to-end: audio → cover art
├── app/
│   └── streamlit_app.py  # Web UI
├── notebooks/
│   └── train_cnn_colab.ipynb  # GPU training on Colab
├── data/                 # GTZAN dataset (not committed)
├── models/               # Saved model weights (not committed)
├── outputs/              # Generated cover art examples
├── tests/                # Basic sanity tests
├── docs/                 # Design notes, qualitative results
├── SETUP.md              # Installation instructions
├── ATTRIBUTION.md        # AI tool usage and sources
└── requirements.txt      # Python dependencies
```

## Individual Contributions

This is a solo project. All work was completed by Sofia Perez Tenessa, with AI coding assistance documented in `ATTRIBUTION.md`.

## License

Educational use only. Dataset (GTZAN) has its own licensing terms — see `data/README.md`.
