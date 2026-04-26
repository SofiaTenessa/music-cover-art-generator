# Example: Pop Music with Sad Lyrics

## Scenario
User uploads:
- **Song:** "Upbeat pop track with cheerful production"
- **Genre detected:** Pop (from CNN)
- **Lyrics:** "I'm alone, tears falling, goodbye my love, pain inside..."

---

## What Happens in Each Stage

### Stage 1: Audio Analysis

```
Audio file → Librosa processing

Features extracted:
- tempo_bpm: 110 (moderate-fast, typical for pop)
- energy: 0.16 (moderate-high, pop is usually energetic)
- brightness: 3200 (HIGH, pop has lots of high frequencies)

Why high brightness?
Pop uses:
  ✓ Bright synthesizers
  ✓ High-pitched vocals
  ✓ Crisp drums
  ✓ Upbeat production
→ Lots of high-frequency content = high brightness score
```

### Stage 2: Genre Classification

```
CNN processes mel-spectrogram

Output probabilities:
- pop: 0.45 ← HIGHEST (cheerful, upbeat characteristics)
- disco: 0.18
- pop: 0.12
- rock: 0.10
- ...other genres: < 0.10

Prediction: "Pop" ✓
```

### Stage 3: Genre Refinement

```python
# Check if features suggest a different genre
if cnn_genre == "pop" and energy > 0.12 and tempo < 90:
    # Could be classical
    return "classical"

# In this case:
# cnn_genre = "pop"
# energy = 0.16 (> 0.12) ✓
# tempo = 110 (NOT < 90) ✗
# → Keep "pop" (no override)

Final genre: "Pop"
```

### Stage 4: Build the Prompt (THE KEY PART!)

```python
# Pop genre style (from GENRE_STYLES)
style = "BRIGHT and COLORFUL, VIBRANT Duke blue and gold tones, "
        "SATURATED colors, clean modern aesthetic, cheerful "
        "energetic vibe, HIGHLY SATURATED color grading, "
        "polished professional photography, colorful composition"

# Audio-derived mood (from describe_mood)
mood_from_audio = describe_mood({
    "tempo_bpm": 110,   # steady, mid-tempo
    "energy": 0.16,     # intense
    "brightness": 3200  # bright, sharp
})
# → Returns: "steady, mid-tempo, intense, bright, sharp"

# Lyrics-derived mood (from lyrics_to_mood)
lyrics = "I'm alone, tears falling, goodbye my love, pain inside..."
lyrics_mood = lyrics_to_mood(lyrics)

# Keyword counting:
# "melancholic" keywords: ["sad", "lonely", "tears", "cry", "gone", "miss", "goodbye"]
# Found: "tears" (1), "goodbye" (1), "alone" + "pain" not in melancholic list
# 
# "dark" keywords: ["dark", "pain", "suffer", "death", "alone", "lost"]
# Found: "pain" (1), "alone" (1) = 2 matches
#
# Winner: "dark" (2 > 2 if only melancholic)
# Actually: "melancholic" = 2 (tears, goodbye)
#          "dark" = 2 (pain, alone)
# Tie! Return highest in dict order: "dark"
# → Returns: "dark"

# Combine everything:
prompt_parts = [
    DUKE_AESTHETIC_BASE,
    "vibrant colorful campus quad with bright light",  # Pop subject
    "BRIGHT and COLORFUL, VIBRANT Duke blue and gold tones, "
    "SATURATED colors, clean modern aesthetic, cheerful energetic vibe",  # Pop style
    "steady, mid-tempo, intense, bright, sharp",  # Audio mood
    "dark emotional tone",  # Lyrics mood ← CONFLICT!
    QUALITY_TAGS
]

final_prompt = (
    "Duke university campus aesthetic, stone architecture with arches, "
    "chapel visible in background, campus quad atmosphere, indie album cover, "
    "vibrant colorful campus quad with bright light, "
    "BRIGHT and COLORFUL, VIBRANT Duke blue and gold tones, "
    "SATURATED colors, clean modern aesthetic, cheerful energetic vibe, "
    "HIGHLY SATURATED color grading, polished professional photography, "
    "colorful composition, "
    "steady, mid-tempo, intense, bright, sharp, "
    "dark emotional tone, "
    "album cover, square format, centered composition, highly detailed, "
    "cinematic lighting, 4k, professional photography, trending on artstation, "
    "artistic aesthetic, no text"
)
```

### Stage 5: Image Generation (Stable Diffusion)

```
Stable Diffusion processes the prompt:

Sees TWO CONFLICTING INSTRUCTIONS:
1. "BRIGHT and COLORFUL, VIBRANT, SATURATED colors, 
    cheerful energetic vibe, HIGHLY SATURATED"
2. "dark emotional tone"

How SD resolves conflicts:
- Token weights matter (CAPS = higher weight)
- BRIGHT appears 3x, DARK appears 1x
- → Colorful dominates, but adds dark undertones
- Result: COLORFUL BUT MOODY

Generation process (30 steps):
Step 1-10:   Pure random noise
Step 11-20:  Large-scale composition 
             → Bright, vibrant color palette emerges
Step 21-30:  Fine details
             → Adds darker shadows, moody lighting
             
Final image characteristics:
✓ Vibrant, saturated colors (BRIGHT, COLORFUL wins)
✓ Cheerful composition (Pop style)
✓ BUT with dark shadows and moody mood (dark emotional tone added)
✓ Overall: Cheerful pop aesthetic with sad/introspective undertones
```

---

## 📊 The Result: "Sad Pop" Aesthetic

### Visual Output:

```
┌─────────────────────────────────────┐
│   VIBRANT, COLORFUL CAMPUS SCENE     │
│                                      │
│  ✓ Bright Duke blue and gold tones  │
│  ✓ Saturated, vivid colors          │
│  ✓ Clean, modern aesthetic          │
│  ✓ Energetic composition            │
│                                      │
│  BUT ALSO:                           │
│  ✓ Dark shadows in corners          │
│  ✓ Moody lighting underneath        │
│  ✓ Emotional, introspective feeling  │
│  ✓ Contrast between bright & dark   │
│                                      │
│  Result: Happy on the surface,      │
│          sad underneath             │
│          (Classic sad pop vibe!)    │
└─────────────────────────────────────┘
```

---

## 🎨 Real-World Comparison

**What you'd get is similar to:**

- **Olivia Rodrigo's "drivers license"** album art:
  - Bright colors, clean design
  - But melancholic, introspective mood
  
- **Billie Eilish's "When We All Fall Asleep" cover:**
  - Colorful, artistic
  - But dark, moody undertones
  
- **Taylor Swift's "Lover"** (pop) vs **"folklore"** (sad):
  - Same artist, different emotional vibes
  - Your system would produce that contrast

---

## 🎯 The Key Insight: Prompt Layering

The system works because **it layers instructions**:

```
Layer 1 (Genre): "bright, colorful, cheerful"
Layer 2 (Audio):  "intense, sharp" 
Layer 3 (Lyrics): "dark emotional tone"

Result = Blend of all three
      = "Sad pop" (oxymoron that actually makes sense!)
```

**Without lyrics:**
```
Only Layers 1 & 2:
= Pure bright pop cover
```

**With sad lyrics:**
```
All three layers:
= Bright pop + dark emotion
= Nuanced, realistic cover!
```

---

## 📈 Prompt Composition Breakdown

```
Word frequency in final prompt:

BRIGHT/COLORFUL/VIBRANT/SATURATED terms: 12+ mentions
  → Heavy weight toward colorful aesthetic

dark/shadow/moody terms: 2-3 mentions
  → But weighted heavily by "dark emotional tone"
  
Result: ~80% colorful, ~20% dark
      = Colorful-dominant with dark undertones
```

---

## 🔄 What If You Clicked "Make it Darker"?

If user then clicked the "make it darker" refinement button:

```
Original prompt: "...BRIGHT and COLORFUL...dark emotional tone..."

Refine button appends: "EXTREMELY DARK, BLACK dominant, DARK GREY 
                       dominant, BLACK and grey color palette, 
                       minimal light..."

New prompt: "...BRIGHT and COLORFUL...dark emotional tone...
             EXTREMELY DARK, BLACK dominant..."

Conflict increases:
- BRIGHT (original) vs EXTREMELY DARK (refinement)
- Refinement usually wins (stronger terms)
- Result: Dark-dominant with colorful undertones
- = "Sad pop" becomes more genuinely SAD
```

---

## 💭 The Philosophy

This is actually **smart design**:

Real music often has **contradictions:**
- Happy beats, sad lyrics (most emotional pop songs)
- Dark production, upbeat energy (emo, alt-pop)
- Energetic tempo, melancholic lyrics (indie rock)

**Without lyrics input:** System only sees the energy/speed
**With lyrics input:** System captures the emotional truth

Example songs this describes:
- "Someone Like You" (Adele) - pop melody + heartbreak
- "Creep" (Radiohead) - indie rock + deep insecurity
- "The Middle" (Jimmy Eat World) - upbeat + depression
- "Bitter Sweet Symphony" (The Verve) - bright strings + despair

---

## 🎬 Visual Summary

```
Pop Song + Happy Audio Features + Sad Lyrics
                    ↓
┌─────────────────────────────────────────┐
│     PROMPT COMBINATION:                  │
│  "BRIGHT, COLORFUL, VIBRANT, SATURATED  │
│   cheerful energetic vibe... dark       │
│   emotional tone, sad mood"             │
└─────────────────────────────────────────┘
                    ↓
      Stable Diffusion interprets:
      
      "Make something COLORFUL and CHEERFUL
       but with a DARK, SAD underlying emotion"
                    ↓
        Album Cover Output:
        
        ✓ Vibrant, saturated colors
        ✓ Colorful Duke campus
        ✓ BUT with moody shadows
        ✓ Dark emotional undertones
        ✓ Introspective feeling
        ✓ Melancholic vibe
        
        = "Sad Pop" aesthetic!
```

---

## 🧪 How to Test This

```bash
python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora
```

Try these combinations:

1. **Upbeat Pop + Happy Lyrics:**
   - "Love you forever, dancing in the sun!"
   - Result: Bright, cheerful, vivid colors ✓

2. **Upbeat Pop + Sad Lyrics:**
   - "Alone, tears falling, goodbye my love..."
   - Result: Bright with dark mood (this scenario!) ✓

3. **Rock + Sad Lyrics:**
   - "Pain, suffering, lost forever..."
   - Result: Dark, gritty, moody ✓

4. **No Lyrics:**
   - Pop detected → Pure bright, colorful result ✓

You'll see how **lyrics dramatically shift the visual mood** while preserving the genre aesthetic!

---

## Summary

**For Pop with Sad Lyrics:**

| Component | Input | Output |
|-----------|-------|--------|
| Genre | Pop | BRIGHT, COLORFUL, VIBRANT |
| Audio | Energetic, sharp | Intense, bright |
| Lyrics | Sad, alone, tears | Dark emotional tone |
| **Result** | **Blend of all** | **Sad pop vibe!** |

The system creates a **visually cheerful but emotionally melancholic cover** - exactly what real sad pop songs need! 🎵💔

