# CS 372 ML Rubric - Final Submission Selection (15 Items)

## Strategic Selection for Maximum Points

You have completed work across many ML categories. Below are your **15 strongest claims** for submission, selected based on:
- Highest point values (7-10 pts prioritized)
- Most substantial engineering effort
- Best evidence in code/documentation
- Coverage of all major project components

**Total Points Selected: 104 points** (out of 150 max)

---

## Selected Items (Ranked by Points)

### Tier 1: High-Value (10 pts each)
1. **Web Application Deployment with UI** (10 pts) ✓
   - Flask REST API + HTML frontend
   - File: `app/flask_server_lora.py`, `app/frontend.html`
   - Evidence: Fully functional music upload → cover generation interface

2. **Solo Project** (10 pts) ✓
   - Individual completion
   - File: README.md states "Solo work"
   - Evidence: All work attributed to single developer

### Tier 2: Major Technical Achievements (7 pts each)
3. **Fine-tuned Diffusion Model with LoRA** (7 pts) ✓
   - Custom DDPM training loop with noise prediction
   - File: `scripts/lora_train_improved.py` (lines 100-300+)
   - Evidence: Full backprop through UNet, VAE encoding, noise scheduling

4. **Cross-Modal Generation (Audio → Visual)** (7 pts) ✓
   - Substantial system converting music to album covers
   - File: `src/pipeline.py` (full pipeline orchestration)
   - Evidence: 5-stage pipeline: audio → features → genre → prompt → image

5. **Audio Preprocessing & Feature Extraction** (7 pts) ✓
   - Mel-spectrograms, tempo, energy, brightness extraction
   - File: `src/preprocessing.py`
   - Evidence: Handles MP3/WAV/FLAC, 22.05kHz resampling, 128 frequency bins

6. **Multi-Stage ML Pipeline Architecture** (7 pts) ✓
   - Clear modular pipeline with orchestration
   - File: `src/pipeline.py`, `app/flask_server_lora.py`
   - Evidence: Audio → Features → CNN → Refinement → Prompt → Diffusion → Image

7. **Transfer Learning - Domain Adaptation** (7 pts) ✓
   - Fine-tuned generic Stable Diffusion for Duke aesthetic
   - File: `scripts/lora_train_improved.py`
   - Evidence: Learned Duke-specific visual patterns from 41 curated images

8. **Preprocessing Pipeline Addressing 2+ Challenges** (7 pts) ✓
   - Handles variable-length audio, noisy compression, domain mismatch
   - File: `src/preprocessing.py`
   - Evidence: Dynamic length handling, log-mel normalization, feature refinement heuristics

9. **Error Analysis & Root Cause Documentation** (7 pts) ✓
   - Documented failures: rain generation, genre classification, refinement issues
   - File: README.md, REFINEMENT_GUIDE.md, PROJECT_TECHNICAL_OVERVIEW.md
   - Evidence: Identified Stable Diffusion 1.5 limitations, genre heuristic improvements, refinement prompt engineering

### Tier 3: Core ML Techniques (5 pts each)
10. **Custom CNN Architecture** (5 pts) ✓
    - Built from scratch (not transfer learning from ResNet)
    - File: `src/model.py` (GenreCNN class)
    - Evidence: Conv2d layers, MaxPool, Dropout, FC layers, trained on GTZAN

11. **Regularization Techniques (3 methods)** (5 pts) ✓
    - Dropout (0.5), Gradient clipping (norm=1.0), Learning rate scheduling (cosine annealing)
    - File: `src/model.py`, `scripts/lora_train_improved.py`
    - Evidence: Multiple complementary regularization approaches

12. **Data Augmentation (4 techniques)** (5 pts) ✓
    - RandomRotation (±15°), RandomAffine (translation), ColorJitter, **RandomHorizontalFlip** (NEW)
    - File: `scripts/lora_train_improved.py` (DukeImageDataset class)
    - Evidence: 4 independent augmentation methods reduce overfitting

13. **Feature Engineering** (5 pts) ✓
    - Custom extraction: Tempo (BPM), Energy (RMS), Brightness (spectral centroid)
    - File: `src/preprocessing.py` (extract_mood_features function)
    - Evidence: Not standard librosa outputs—hand-calculated custom features

14. **Documented Iterations & Improvement** (5 pts) ✓
    - v1→v2: LoRA rank 16→32, epochs 10→20, added augmentation, improved refinements
    - File: LORA_IMPROVED_GUIDE.md, README.md, HYPERPARAMETER_TUNING_ANALYSIS.md
    - Evidence: Measured loss reduction, documented rationale for each improvement

15. **Qualitative + Quantitative Evaluation** (5 pts) ✓
    - Metrics: CNN accuracy (70%→80%), loss curves, inference timing
    - Qualitative: Visual inspection of generated covers, user feedback integration
    - File: README.md, PROJECT_TECHNICAL_OVERVIEW.md
    - Evidence: Both numerical results and descriptive analysis

---

## Points Breakdown

| Item | Category | Points |
|------|----------|--------|
| 1. Web App Deployment | Deployment | 10 |
| 2. Solo Project | General | 10 |
| 3. Fine-tuned Diffusion LoRA | Generative Models | 7 |
| 4. Cross-Modal Generation | Audio/Modalities | 7 |
| 5. Audio Preprocessing | Audio/Modalities | 7 |
| 6. Multi-Stage Pipeline | System Integration | 7 |
| 7. Transfer Learning | Transfer Learning | 7 |
| 8. Preprocessing Pipeline | Data/Features | 7 |
| 9. Error Analysis | Evaluation | 7 |
| 10. Custom CNN | Training/Optimization | 5 |
| 11. Regularization | Training/Optimization | 5 |
| 12. Data Augmentation | Data/Features | 5 |
| 13. Feature Engineering | Data/Features | 5 |
| 14. Documented Iterations | Evaluation | 5 |
| 15. Qual + Quant Eval | Evaluation | 5 |
| **TOTAL** | | **104 pts** |

---

## Why These 15?

### Strength Ranking Rationale

**Top Priority (Must Include):**
- Items 1-2: Required framework (deployment, solo work) = 20 pts baseline
- Items 3-9: Core technical achievements showing substantial ML engineering = 49 pts (main value)

**Supporting Evidence (Round Out the 15):**
- Items 10-15: Demonstrate ML fundamentals and rigorous evaluation = 30 pts (demonstrates depth)

### Coverage Analysis

✅ **Generative Models:** Fine-tuned Diffusion (7 pts)
✅ **Audio/Speech:** Audio preprocessing (7 pts), Cross-modal (7 pts)
✅ **Computer Vision:** CNN architecture (5 pts), Transfer learning (7 pts)
✅ **Data Science:** Feature engineering (5 pts), Augmentation (5 pts), Preprocessing (7 pts)
✅ **Training/Optimization:** Regularization (5 pts), Custom architecture (5 pts)
✅ **System Integration:** Multi-stage pipeline (7 pts)
✅ **Evaluation:** Error analysis (7 pts), Documented iterations (5 pts), Evaluation (5 pts)
✅ **Deployment:** Web app (10 pts)

**Result:** Balanced coverage demonstrating breadth AND depth across 8+ ML categories

---

## Items NOT Selected (And Why)

These are secondary items; selecting them would replace higher-value items:

| Item | Points | Reason Not Selected |
|------|--------|-------------------|
| Modular code design | 3 | Lower value; also covered by architecture choices |
| Train/val/test split | 3 | Lower value; implied by CNN training |
| Normalization/standardization | 3 | Lower value; included in preprocessing |
| Learning rate scheduling | 3 | Covered under "Regularization techniques" |
| GPU training | 3 | Infrastructure requirement, not method |
| Design tradeoffs explanation | 3 | Lower points; explain verbally if asked |
| Tracked training curves | 3 | Covered under "Documented iterations" |
| ATTRIBUTION.md | 3 | Low priority (though you have this) |
| Hyperparameter comparison table | 5 | NEW! Could swap for item 14, but documented iterations shows broader story |
| Ablation study | 7 | Would be 7 pts, but no controlled experimental table yet |
| Systematic hyperparameter tuning | 5 | Covered by documented iterations + new tuning document |

---

## Submission Checklist

Before submitting these 15 items:

- [x] Web app deployment working (tested Flask + HTML)
- [x] Solo project confirmed (no collaborators)
- [x] LoRA training script with DDPM loop documented
- [x] Cross-modal pipeline explained (audio → image)
- [x] Audio preprocessing with custom features
- [x] Multi-stage architecture diagrammed
- [x] Transfer learning approach documented
- [x] Preprocessing handles multiple challenges
- [x] Error cases documented with analysis
- [x] CNN built from scratch (not pretrained)
- [x] Regularization methods identified (3 types)
- [x] Augmentation techniques listed (4 methods)
- [x] Features custom-engineered (not standard)
- [x] Iterations documented with metrics
- [x] Both qualitative and quantitative evaluation present

---

## Total Points Summary

```
Conservative Guarantee: 104 points
Additional possible claims: ~15-25 more pts (hyperparameter tuning, ablation study, etc.)
Realistic submission score: 104-129 points out of 150 max
```

**Your project is positioned well for 104 pts minimum, with strong supporting documentation for additional claims if rubric reviewers ask for clarification.**

