# Album Cover Refinement Guide
Quick reference for what refinements actually work with Chapel Covers.

> **Important:** Stable Diffusion 1.5 is better at modifying existing images than adding completely new objects. These refinements adjust mood, style, and atmosphere—not adding things from scratch.

## Refinements That Work ✅

### Lighting & Darkness
```
"darker"          → Much darker image, heavy shadows, low-key lighting
"make it darker"  → Same as above
"gloomier"        → Dark, moody, sad lighting, overcast mood
"brighter"        → Bright, well-lit, sunny daylight
"brighten"        → Same as brighter
```

### Time of Day
```
"sunset"          → Golden/orange tones, warm evening light
"sunrise"         → Golden dawn light, morning sun
"night"           → Dark, nocturnal, stars/moonlight
"dusk"            → Twilight hour, golden hour transition
"dawn"            → Early morning, first light
"daytime"         → Full bright daylight
"midday"          → Bright sun overhead
```

### Colors & Saturation
```
"monochrome"      → Black and white, no color
"black and white" → Same as monochrome
"grayscale"       → Desaturated, gray tones
"more colorful"   → Vibrant, saturated, bold colors
"vibrant"         → Saturated, vivid, chromatic intensity
"colorful"        → Bold colors, rich palette
"add blue"        → Blue tones, blue atmosphere, Duke blue
"add gold"        → Golden tones, warm gold, amber
"add purple"      → Purple/violet tones
"add red"         → Red/crimson tones
```

### Mood & Atmosphere
```
"dramatic"        → High contrast, bold, powerful composition
"peaceful"        → Calm, serene, tranquil
"calm"            → Relaxed, gentle, tranquil
"moody"           → Dark mood, atmospheric, brooding
"energetic"       → Vibrant, dynamic, lively, bold
"intense"         → Powerful, striking, dramatic
"bold"            → Strong, striking, confident
```

### Detail & Sharpness
```
"sharper"         → Sharp, crisp, clear details, high definition
"sharp"           → Same as sharper
"softer"          → Soft focus, blurry, gentle, delicate
"soft"            → Same as softer
"more detail"     → Texture, fine details, intricate architecture
"detailed"        → Fine details, visible texture, crisp
"less detail"     → Soft focus, minimalist
"minimal"         → Minimal, simple, clean, uncluttered
```

### Contrast & Texture
```
"more contrast"   → High contrast, bold shadows, bright highlights
"high contrast"   → Same as above
"less contrast"   → Low contrast, soft transitions, gentle
"add texture"     → Textured, rough, visible grain
"film grain"      → Grainy, analog film, vintage aesthetic
"grainy"          → Grain, film, vintage look
```

### Weather & Atmosphere
```
"foggy"           → Fog, mist, hazy, atmospheric
"fog"             → Same as foggy
"misty"           → Mist, foggy, hazy
"add clouds"      → Cloudy, cloud formations, cloud cover
"clouds"          → Same as add clouds
"overcast"        → Overcast, gray sky, cloud cover
"stormy"          → Dark storm clouds, dramatic, threatening weather
"snow"            → Snowy, snowfall, white snow, winter
```

---

## Refinements That DON'T Work ❌

### Specific Objects (SD 1.5 Limitation)
```
"add rain"        → REMOVED - doesn't reliably render precipitation
"rainy"           → REMOVED
"rain"            → REMOVED
"add people"      → Won't work - can't add humans reliably
"add birds"       → Won't work - can't add animals
"add cars"        → Won't work - can't add vehicles
"add flowers"     → Won't work - can't add plants
"add buildings"   → Won't work - can't add new structures
```

**Why?** Stable Diffusion v1.5 struggles with explicit object addition. It's designed to generate from scratch or modify existing images, not reliably add specific new elements.

---

## Usage Examples

### Good Refinement Flow
```
1. Generate cover:  "Rock song, energetic, fast tempo"
2. First refine:    "darker"
   → Image gets darker, moodier
   
3. Second refine:   "more contrast"
   → Added shadow depth, striking composition
   
4. Third refine:    "add blue"
   → Duke blue tones emphasized
```

### Bad Refinement (Won't Help)
```
1. Generate cover
2. Refine with:     "add rain"
   → Image likely unchanged or rain barely visible
   → Frustrating!
```

### Better Alternative
```
1. Generate cover
2. Refine with:     "darker and stormy"
   → Gets darker, ominous mood
   → Creates rainy/stormy *atmosphere* even without literal rain
   
3. Or refine with:  "overcast"
   → Gray, cloudy sky tone
   → Suggests rain without requiring explicit precipitation
```

---

## Pro Tips

### Effective Combinations
- **Moody rock:** darker + high contrast + dramatic
- **Dreamy pop:** softer + brighter + more colorful
- **Melancholic jazz:** darker + blue + peaceful
- **Intense metal:** high contrast + dramatic + more contrast
- **Warm indie:** sunset + warmer golds + textured

### What Actually Changes the Image
- **Lighting:** darker, brighter, sunset, sunrise, night
- **Color:** monochrome, vibrant, add blue/gold/purple/red
- **Mood:** dramatic, peaceful, energetic, moody, calm
- **Detail:** sharper, softer, textured, grainy, minimal
- **Atmosphere:** foggy, stormy, overcast, bright

### What Likely Won't Work
- Asking for new objects
- Asking for people or animals
- Asking for specific scenes ("add a beach")
- Combining too many requests at once

---

## Refinement Strategy

**Best approach:**
1. Generate initial cover
2. Use 1-2 refinements max per iteration
3. Stick to style/mood refinements (listed above ✅)
4. Avoid object-based refinements (listed above ❌)

**Why?** 
- Stable Diffusion 1.5 works best with simple, style-based instructions
- Multiple refinements can contradict each other
- Object addition is unreliable due to model limitations

---

## Report Issues
If a refinement doesn't seem to work as expected:
1. Try a simpler version: "darker" instead of "make it much darker"
2. Try the opposite: if "darker" doesn't work, try "brighter" to verify the system works
3. Check that you're not asking for objects (rain, people, animals, etc.)
4. Restart Flask server and try again

---

## Technical Note

The refinement system in `src/prompt_builder.py` has 100+ mappings covering all the ✅ refinements above. Each mapping uses aggressive language and repetition to maximize the chances Stable Diffusion prioritizes it. Even so, style/mood refinements work much better than object additions.

See `LORA_IMPROVED_GUIDE.md` for technical details on why certain refinements are more reliable.
