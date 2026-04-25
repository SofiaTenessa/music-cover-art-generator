# Improving Genre Classification

## Current Problem

The CNN achieves ~70% accuracy on GTZAN dataset, which means:
- Classical → Predicted as Pop (wrong)
- Rock → Blues 47%, Rock 21% (mostly wrong)
- Misclassifications cascade through the entire pipeline

## Root Causes

1. **GTZAN dataset is small** (~1000 samples, 10 genres, 30 sec each)
2. **CNN architecture is basic** (simple 2D CNN on mel-spectrograms)
3. **No transfer learning** from larger pre-trained models
4. **Genre boundaries are fuzzy** (Rock/Metal/Blues overlap significantly)

## Quick Fixes (No Retraining)

### Option 1: Use Pre-trained Model (Easiest, ~30 mins)

Replace the CNN with a pre-trained music classifier from Hugging Face:

```python
# In src/pipeline.py, replace classify_audio():

from transformers import pipeline as hf_pipeline

classifier = hf_pipeline("audio-classification", 
                         model="facebook/wav2vec2-xlarge-960h-self-supervised")
```

This uses Meta's wav2vec2 model (~1B parameters) trained on 960+ hours of audio.

**Pros:** Better accuracy, faster implementation  
**Cons:** Slightly slower inference, requires audio preprocessing

### Option 2: Retrain CNN (Better long-term, ~2-4 hours)

You need:
1. **Better training data** (100-200 labeled songs per genre)
2. **Better augmentation** (pitch shift, time stretch, noise)
3. **Better architecture** (ResNet instead of simple CNN)

Steps:
```bash
# 1. Collect/label 50+ songs per genre
mkdir -p data/genre_training_data/{rock,pop,classical,blues,jazz}/

# 2. Retrain CNN
python -m src.train \
  --data-dir data/genre_training_data \
  --epochs 50 \
  --batch-size 16

# 3. Replace model
cp models/new_best.pt models/cnn_default_best.pt
```

### Option 3: Ensemble Approach (Balanced, ~1 hour)

Combine multiple classifiers:
1. Your CNN (fast, ~70% accuracy)
2. Librosa beat/chroma analysis (detects rock/metal via timbre)
3. Tempo-based heuristics (EDM is 120-140 BPM)

```python
def classify_audio_ensemble(audio_path):
    # Get CNN prediction
    cnn_genre, cnn_probs = cnn_classifier(audio_path)
    
    # Get audio features
    features = extract_advanced_features(audio_path)
    
    # Adjust CNN prediction based on features
    if features["tempo"] > 160 and "rock" not in cnn_genre:
        # Likely metal/punk, not pop
        adjust_probs(cnn_probs, boost_metal=True)
    
    if features["spectral_centroid"] > 5000:
        # Bright high-frequency content = rock/pop/metal
        reduce_probs(cnn_probs, ["classical", "jazz"])
    
    return combined_prediction(cnn_probs, feature_probs)
```

---

## Recommended Path

For **Chapel Covers final submission**:

1. **Quick win:** Add genre-specific prompt fallbacks
   ```python
   # In prompt_builder.py
   FALLBACK_MAPPINGS = {
       "Pop": "Rock",  # If CNN says Pop but looks energetic → Rock
       "Classical": "Jazz",  # If CNN says Classical but tempo > 100 → Jazz
   }
   ```

2. **Medium effort:** Use Librosa to detect genre signals
   ```python
   def refine_genre_with_features(cnn_genre, audio_features):
       energy = audio_features["energy"]
       tempo = audio_features["tempo_bpm"]
       spectral = audio_features["spectral_centroid"]
       
       if energy > 0.2 and tempo < 90:
           return "Blues"  # Slow + energetic = Blues
       if spectral > 6000 and tempo > 100:
           return "Rock"  # Bright + fast = Rock
       
       return cnn_genre  # Default to CNN
   ```

3. **Long term:** Retrain CNN or use pre-trained transformer

---

## Testing Accuracy

After making changes, test with:

```bash
python -m src.evaluate \
  --model models/cnn_default_best.pt \
  --test-dir data/genre_test_songs/
```

Expected accuracy by approach:
- Current CNN: ~70%
- With feature refinement: ~75-80%
- With pre-trained model: ~85-90%
- With retrained ensemble: ~90%+

---

## Files to Modify

1. `src/pipeline.py` — `classify_audio()` method
2. `src/prompt_builder.py` — Add genre confidence filtering
3. `src/preprocessing.py` — Extract more audio features

Would you like me to implement Option 1 (pre-trained model) or Option 3 (ensemble)?
