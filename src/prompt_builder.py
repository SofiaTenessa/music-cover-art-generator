"""Build a text prompt for Stable Diffusion from CNN predictions + audio features.

This is the glue between the audio-classification and image-generation stages.
For Chapel Covers, prompts are Duke-inspired with gothic architecture, navy/royal blue tones,
and campus aesthetics. The quality of the final cover art depends heavily on the prompt quality.

Design principles:
    - Genre sets the overall visual language.
    - Mood features (tempo, energy, brightness) add expressive variation.
    - Lyrics are analyzed for emotional mood to further customize the aesthetic.
    - The prompt is structured as (subject) + (Duke-inspired style) + (mood) + (quality tags).
    - A negative prompt pushes Stable Diffusion away from common failure modes.

# AI-generated via Claude (scaffold). Author owns the prompt design / tuning.
"""
from __future__ import annotations

from dataclasses import dataclass


# ===== DUKE-INSPIRED AESTHETIC =====
# Every prompt should evoke the Duke campus: stone architecture, chapel in background,
# campus quad atmosphere, with balanced Duke colors and warm lighting.
# Modern indie album cover aesthetic, not overly dark/gothic.
# NO official Duke logos or copyrighted marks—only architectural and color inspiration.

DUKE_AESTHETIC_BASE = (
    "Duke university campus aesthetic, stone architecture with arches, chapel visible in background, "
    "campus quad atmosphere, indie album cover, modern photography style, professional composition, "
    "artistic lighting, film grain texture, square album cover format, "
    "Duke Navy Blue (#012169) and Duke Royal Blue (#00539B) tones, white accents"
)

GENRE_STYLES: dict[str, dict] = {
    "blues": {
        "style": "melancholic but warm, rich analog photograph, golden-hour light, deep blues with amber warmth, introspective mood",
        "subjects": ["stone archway with warm window light", "campus courtyard at dusk", "architectural detail with soft shadows"],
    },
    "classical": {
        "style": "elegant and refined, symmetrical stone architecture, warm candlelit tones, sophisticated composition, professional lighting",
        "subjects": ["ornate stone entrance framed beautifully", "arches creating balanced geometric composition", "warm light through historic corridor"],
    },
    "country": {
        "style": "warm and inviting, golden-hour Americana aesthetic, rich earth tones, bright blue accents, natural and approachable",
        "subjects": ["sun-lit campus quad with warm glow", "stone archway in golden light", "tree-lined university grounds at golden hour"],
    },
    "disco": {
        "style": "VIBRANT and COLORFUL, celebratory energy, SATURATED rich color grading with BRIGHT blues and golds, VIVID colors, energetic but sophisticated, glossy editorial feel, HIGHLY SATURATED composition",
        "subjects": ["COLORFUL vibrant light play on stone architecture", "VIBRANT colorful campus courtyard scene", "illuminated arches with COLORFUL vibrant highlights"],
    },
    "hiphop": {
        "style": "bold and confident, high-energy color palette, urban campus aesthetic, street-art inspired but professional, rich contrast",
        "subjects": ["dynamic campus archway with bold lighting", "expressive architectural detail", "energetic courtyard composition"],
    },
    "jazz": {
        "style": "sophisticated and cool, balanced blue and gold tones, geometric composition, smooth refined aesthetic, cool jazz vibes",
        "subjects": ["architectural arches creating clean lines", "campus detail with sophisticated lighting", "stone patterns with cool-toned light"],
    },
    "metal": {
        "style": "intense and dramatic, high contrast with bold color accents, moody but not desaturated, powerful composition, energetic",
        "subjects": ["dramatic stone tower with bold lighting", "architectural detail with striking shadows", "intense campus scene with strong tones"],
    },
    "pop": {
        "style": "BRIGHT and COLORFUL, VIBRANT Duke blue and gold tones, SATURATED colors, clean modern aesthetic, cheerful energetic vibe, HIGHLY SATURATED color grading, polished professional photography, colorful composition",
        "subjects": ["vibrant colorful campus quad with bright light", "bright colorful geometric architecture", "COLORFUL stone detail with vibrant bold lighting"],
    },
    "reggae": {
        "style": "warm and uplifting, tropical color palette meeting Duke campus, sunset golds and greens, joyful organic aesthetic",
        "subjects": ["sun-lit campus with warm vegetation", "golden-lit stone with natural elements", "bright campus courtyard with organic feel"],
    },
    "rock": {
        "style": "energetic and raw, powerful color palette with blues and golds, film grain texture, edgy but not overly dark, driving energy",
        "subjects": ["dramatic stone architecture with bold light", "powerful campus composition with movement", "energetic archway with striking contrast"],
    },
}

QUALITY_TAGS = "highly detailed, cinematic composition, professional indie album cover art, trending on artstation, artistic photography"

NEGATIVE_PROMPT = (
    "text, letters, watermark, logo, blurry, distorted, bad anatomy, extra limbs, "
    "official Duke logo, trademark, people, faces, portraits, bodies, humans"
)


def describe_mood(mood_features: dict) -> str:
    """Extract mood descriptors from audio features (tempo, energy, brightness)."""
    tempo = mood_features.get("tempo_bpm", 100)
    energy = mood_features.get("energy", 0.1)
    brightness = mood_features.get("brightness", 2000)

    descriptors = []
    if tempo > 140:
        descriptors.append("energetic, fast-paced")
    elif tempo < 80:
        descriptors.append("slow, contemplative")
    else:
        descriptors.append("steady, mid-tempo")

    if energy > 0.15:
        descriptors.append("intense")
    elif energy < 0.05:
        descriptors.append("soft, quiet")

    if brightness > 3500:
        descriptors.append("bright, sharp")
    elif brightness < 1500:
        descriptors.append("dark, muted")

    return ", ".join(descriptors)


def lyrics_to_mood(lyrics: str) -> str:
    """
    Extract emotional mood from lyrics using simple keyword rules.
    Returns one of: melancholic, romantic, nostalgic, energetic, dark, dreamy, rebellious
    """
    if not lyrics:
        return ""

    lyrics_lower = lyrics.lower()

    # Define mood keywords (in priority order)
    mood_keywords = {
        "dark": ["dark", "pain", "suffer", "death", "alone", "lost", "devil", "evil", "nightmare"],
        "rebellious": ["rebel", "fight", "break", "run", "free", "wild", "scream", "defy", "burn"],
        "melancholic": ["sad", "lonely", "tears", "cry", "gone", "miss", "goodbye", "heart", "blue"],
        "nostalgic": ["remember", "back then", "used to", "old", "memory", "then", "before", "once"],
        "romantic": ["love", "heart", "forever", "kiss", "sweet", "care", "hold", "embrace", "forever"],
        "dreamy": ["dream", "float", "sky", "cloud", "wonder", "imagine", "ethereal", "stars", "night"],
        "energetic": ["dance", "move", "jump", "feel", "alive", "fly", "high", "rush", "power"],
    }

    # Count keyword matches per mood
    mood_scores = {mood: 0 for mood in mood_keywords}
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            mood_scores[mood] += lyrics_lower.count(keyword)

    # Return the mood with the highest score
    if max(mood_scores.values()) > 0:
        return max(mood_scores, key=mood_scores.get)
    return "introspective"


def is_valid_refinement(refinement_text: str) -> tuple[bool, str | None]:
    """
    Check if refinement is related to album cover design.
    Returns: (is_valid, error_message)

    Valid topics: color, mood, lighting, composition, campus style, visual elements
    Invalid topics: unrelated questions, jokes, personal advice, etc.
    """
    if not refinement_text or len(refinement_text.strip()) < 3:
        return False, "Refinement text is too short."

    # Keywords that indicate valid refinement feedback
    valid_keywords = {
        "color", "colorful", "bright", "dark", "light", "lighting", "mood", "tone",
        "warm", "cool", "saturated", "vibrant", "muted", "cinematic", "minimal",
        "gothic", "campus", "architecture", "composition", "style", "aesthetic",
        "rain", "sky", "clouds", "gold", "blue", "navy", "shadows", "contrast",
        "sharp", "soft", "focus", "blur", "texture", "grain", "film", "vintage",
        "modern", "indie", "editorial", "professional", "energetic", "peaceful",
        "dramatic", "subtle", "bold", "delicate", "intensity", "emphasis",
        "foreground", "background", "perspective", "angle", "framing", "crop",
        "saturate", "desaturate", "desaturated", "less", "more", "add", "remove",
        "black", "white", "grayscale", "monochrome", "bw", "noir",
    }

    refinement_lower = refinement_text.lower()

    # Check if any valid keywords appear
    has_valid_keyword = any(keyword in refinement_lower for keyword in valid_keywords)

    if not has_valid_keyword:
        return False, (
            "Please give feedback related to the album cover, such as: "
            "color, mood, lighting, composition, campus style, visual elements, "
            "or other design aspects (e.g., 'more colorful', 'less dark', 'add rain')."
        )

    return True, None


def map_refinement_to_prompt(refinement_text: str) -> str:
    """
    Map user refinement input to stronger descriptive prompt language.

    Examples:
    - "more colorful" → "richer color palette, vibrant tones, more saturated colors"
    - "less gothic" → "less dramatic, more approachable, warmer aesthetic"
    - "more campus" → "more campus quad, more architecture, university atmosphere"
    """
    refinement_lower = refinement_text.lower().strip()

    # Mapping dictionary: user input → prompt enhancement
    mappings = {
        "black and white": "BLACK AND WHITE monochromatic, grayscale aesthetic, no color, monochrome photography, black and white film, high contrast black and white, noir style, desaturated completely",
        "grayscale": "BLACK AND WHITE monochromatic, grayscale aesthetic, no color, monochrome photography, black and white film, high contrast black and white, noir style, desaturated completely",
        "bw": "BLACK AND WHITE monochromatic, grayscale aesthetic, no color, monochrome photography, black and white film, high contrast black and white, noir style, desaturated completely",
        "more colorful": "HIGHLY SATURATED colors, vibrant rich palette, bold color grading, strong Duke Royal Blue (#00539B) and gold (#D4AF37) tones, vivid chromatic intensity, maximum color saturation",
        "colorful": "HIGHLY SATURATED colors, vibrant rich palette, bold color grading, strong Duke Royal Blue and gold tones, vivid chromatic intensity, maximum color saturation",
        "darker": "VERY DARK overall exposure, LOW-KEY lighting, HEAVY shadows, REDUCED brightness, moody dark color grading, deep tones, nighttime aesthetic, dramatic darkness, underexposed, shadow emphasis",
        "more dark": "VERY DARK overall exposure, LOW-KEY lighting, HEAVY shadows, REDUCED brightness, moody dark color grading, deep tones, nighttime aesthetic, dramatic darkness, underexposed, shadow emphasis",
        "dark": "VERY DARK overall exposure, LOW-KEY lighting, HEAVY shadows, REDUCED brightness, moody dark color grading, deep tones, nighttime aesthetic, dramatic darkness, underexposed, shadow emphasis",
        "less gothic": "less dramatic, more approachable, warmer aesthetic, less emphasis on shadows, more inviting feel",
        "less dark": "brighter lighting, less shadow emphasis, warmer tones, more visible detail, higher key lighting",
        "more campus": "more campus quad visible, more architecture, more student spaces, more university atmosphere, prominent buildings",
        "campus": "stronger campus feel, more architectural detail, university environment emphasis, campus elements",
        "more minimal": "cleaner composition, less clutter, focused subject, minimal background, simple elegant design",
        "minimal": "minimal elements, clean aesthetic, focused composition, less complexity",
        "more cinematic": "cinematic lighting, professional color grading, dramatic but balanced, high production value, cinema aesthetic",
        "cinematic": "cinematic composition, professional lighting, refined color grading, polished look, movie-like",
        "more warm": "warm color temperature, golden tones, amber light, cozy atmosphere, soft warmth",
        "warm": "warm lighting, golden tones, inviting warmth, amber highlights",
        "add rain": "rain effect, wet surfaces, moisture in air, rain-soaked aesthetic, reflective surfaces, rainy weather",
        "rain": "rain-soaked environment, wet architecture, rain drops, moisture, reflections",
        "more golden": "golden-hour light, warm gold tones, amber accents, sunset glow, rich warm palette",
        "golden": "golden tones, golden-hour lighting, amber warmth, sun-lit aesthetic",
        "more blue": "stronger Duke Blue tones, more blue color grading, deeper blue palette, blue-hour lighting, cool blue dominant",
        "blue": "blue tones, Duke Royal Blue prominent, cool blue lighting, blue-hour aesthetic",
        "less dramatic": "less contrast, softer shadows, more balanced lighting, less intense, more approachable",
        "less contrast": "reduced contrast, softer transitions, more subtle light, less dramatic shadows",
        "more dramatic": "higher contrast, dramatic shadows, intense lighting, bold composition, strong impact",
        "dramatic": "dramatic shadows, high contrast, intense lighting, bold visual impact",
        "more detail": "visible architectural detail, more texture, fine details visible, crisp focus",
        "detail": "detailed texture, visible architecture, sharp focus on elements, textural emphasis",
        "softer": "soft focus, gentle lighting, smooth gradients, delicate aesthetic, subtle tones",
        "soft": "soft lighting, gentle shadows, smooth transitions, delicate details",
        "sharper": "sharp focus, crisp details, high definition, clear textures, defined edges",
        "sharp": "sharp focus, crisp details, high definition, clear composition",
        "more editorial": "editorial photography style, professional magazine aesthetic, polished refined look, high production value",
        "editorial": "editorial style, magazine-quality, professional photography, polished aesthetic",
        "more indie": "indie aesthetic, artistic vision, unique perspective, creative composition, non-commercial feel",
        "indie": "indie aesthetic, artistic approach, creative vision, unique style, alternative aesthetic",
        "more peaceful": "calm aesthetic, peaceful mood, serene lighting, tranquil atmosphere, gentle tones",
        "peaceful": "peaceful mood, calm lighting, serene aesthetic, tranquil environment",
        "energetic": "energetic mood, vibrant tones, dynamic composition, lively atmosphere, bold colors",
        # NEW: Compositional instructions
        "add sun": "bright sun visible, sunlight prominent, solar element, sun glare, bright sunlit scene",
        "sun right": "sun positioned on right side, right-side lighting, sun from right, bright right-side illumination",
        "sun left": "sun positioned on left side, left-side lighting, sun from left, bright left-side illumination",
        "add moon": "moon visible, moonlit scene, lunar element, moonlight, night sky with moon",
        "moon": "moonlit aesthetic, moon prominent, lunar lighting, night with moon",
        "sky": "prominent sky, visible sky, sky as main element, sky-focused composition",
        "clouds": "cloudy atmosphere, cloud formations, overcast sky, dramatic clouds",
        "sunset": "sunset lighting, golden sunset, sunset hour, warm sunset tones",
        "sunrise": "sunrise lighting, golden sunrise, dawn light, morning light",
        "night": "nighttime scene, night lighting, dark night sky, nocturnal atmosphere",
        "dusk": "dusk lighting, twilight hour, dusk atmosphere, early evening light",
        "fog": "foggy atmosphere, mist, fog effect, atmospheric fog, hazy conditions",
        "snow": "snow visible, snowy scene, snow-covered, winter aesthetic",
        "autumn": "autumn colors, fall aesthetic, warm autumn tones, fall foliage",
        "summer": "summer light, bright summer day, summer aesthetic, warm daytime",
        "winter": "winter aesthetic, cold tones, winter lighting, frosty atmosphere",
    }

    # Try exact matches first, then partial matches
    for key, enhancement in mappings.items():
        if refinement_lower == key or refinement_lower.startswith(key):
            return enhancement

    # If no mapping found, just use the refinement as-is
    return refinement_text


def refine_prompt(base_prompt: str, refinement_instruction: str) -> tuple[str, bool]:
    """
    Refine a prompt while preserving original structure.

    Returns:
    - (refined_prompt, is_valid)

    The refined prompt keeps the original Duke/genre/mood base and appends
    the refinement as an enhancement.

    Examples:
    - Input: "more colorful"
    - Output: "{original prompt}, richer color palette, vibrant tones, saturated colors, bold color grading..."
    """
    if not refinement_instruction:
        return base_prompt, True

    # Validate refinement
    is_valid, error_msg = is_valid_refinement(refinement_instruction)
    if not is_valid:
        return base_prompt, False  # Return original prompt and invalid flag

    # Map refinement to stronger language
    mapped_refinement = map_refinement_to_prompt(refinement_instruction)

    # Append to base prompt
    refined = f"{base_prompt}, {mapped_refinement.strip()}"

    return refined, True


@dataclass
class Prompt:
    positive: str
    negative: str

    def __str__(self) -> str:
        return self.positive


def get_expanded_genre_modifiers(genre: str, mood_features: dict | None = None) -> str:
    """
    Map base GTZAN genres to expanded style descriptors.

    Since we can't retrain, we expand prompts conditionally based on:
    1. The base genre
    2. Audio mood features (tempo, energy, brightness)

    This allows us to generate diverse covers without requiring new datasets.
    """
    expanded_modifiers = ""

    # Map base genres to extended style descriptors
    genre_expansions = {
        "hiphop": ["R&B aesthetic", "hip-hop energy", "urban vibe"],
        "disco": ["electronic texture", "dance beat aesthetic", "synth-inspired"],
        "classical": ["ambient atmosphere", "orchestral elements", "refined elegance"],
        "rock": ["indie rock sensibility", "alternative edge", "raw creative energy"],
        "electronic": ["lo-fi aesthetics", "digital texture", "modern electronic vibe"],
        "techno": ["electronic energy", "futuristic tech aesthetic"],
        "pop": ["indie pop sensibility", "contemporary aesthetic"],
        "jazz": ["ambient jazz influence", "sophisticated electronic edge"],
        "blues": ["soulful aesthetic", "emotional depth"],
        "country": ["folk indie sensibility", "organic warmth"],
        "reggae": ["lo-fi reggae aesthetic", "organic island vibe"],
        "metal": ["intense electronic edge", "powerful production aesthetic"],
    }

    # Get base modifiers for this genre
    if genre in genre_expansions:
        expanded_modifiers = genre_expansions[genre][0]

        # Add conditional modifiers based on mood features
        if mood_features:
            tempo = mood_features.get("tempo_bpm", 100)
            energy = mood_features.get("energy", 0.1)

            # Slow tempo (< 80) → add ambient/lo-fi descriptors
            if tempo < 80:
                expanded_modifiers += ", lo-fi aesthetic, relaxed pace, ambient atmosphere"
            # Fast tempo (> 140) → add energetic/electronic descriptors
            elif tempo > 140:
                expanded_modifiers += ", energetic electronic texture, vibrant pace"

            # High energy → add intensity
            if energy > 0.15:
                expanded_modifiers += ", high energy production, dynamic intensity"
            # Low energy → add calmness
            elif energy < 0.05:
                expanded_modifiers += ", calm ambient texture, gentle aesthetic"

    return expanded_modifiers


def build_prompt(
    genre: str,
    mood_features: dict | None = None,
    lyrics: str | None = None,
    subject_choice: int = 0,
) -> Prompt:
    """
    Build a Duke-inspired Stable Diffusion prompt from genre, audio mood features, and optional lyrics.

    Args:
        genre: Music genre (must be in GENRE_STYLES)
        mood_features: Dict with tempo_bpm, energy, brightness from audio analysis
        lyrics: Optional lyrics text to extract emotional mood from
        subject_choice: Which subject variant to use for this genre

    Returns:
        Prompt object with positive and negative prompts
    """
    if genre not in GENRE_STYLES:
        raise ValueError(f"Unknown genre '{genre}'. Known: {list(GENRE_STYLES)}")

    style_info = GENRE_STYLES[genre]
    subject = style_info["subjects"][subject_choice % len(style_info["subjects"])]
    style = style_info["style"]

    # Build the Duke-inspired base
    parts = [DUKE_AESTHETIC_BASE, subject, style]

    # Add expanded genre modifiers (R&B, electronic, lo-fi, etc.)
    expanded = get_expanded_genre_modifiers(genre, mood_features)
    if expanded:
        parts.append(expanded)

    # Add audio-derived mood
    if mood_features:
        mood_desc = describe_mood(mood_features)
        if mood_desc:
            parts.append(mood_desc)

    # Add lyrics-derived mood if provided
    if lyrics:
        lyrics_mood = lyrics_to_mood(lyrics)
        if lyrics_mood:
            parts.append(f"{lyrics_mood} emotional tone")

    parts.append(QUALITY_TAGS)

    return Prompt(positive=", ".join(parts), negative=NEGATIVE_PROMPT)


if __name__ == "__main__":
    test_mood = {"tempo_bpm": 130, "energy": 0.12, "brightness": 2500}
    test_lyrics = "I remember when we were young, now everything feels different and lost"

    for genre in GENRE_STYLES:
        p = build_prompt(genre, mood_features=test_mood, lyrics=test_lyrics)
        print(f"\n--- {genre.upper()} ---")
        print(f"Positive: {p.positive}")

    # Test lyrics mood extraction
    print("\n\n=== LYRICS MOOD EXTRACTION ===")
    test_lyrics_samples = {
        "dark": "alone in the dark, pain and suffering, death surrounds me",
        "energetic": "dance, jump, feel alive, rushing through the night",
        "dreamy": "floating in clouds, dreaming under stars, wonderland",
        "nostalgic": "I remember back then, used to be so free",
    }
    for mood_type, sample in test_lyrics_samples.items():
        detected_mood = lyrics_to_mood(sample)
        print(f"  {mood_type}: '{sample[:40]}...' → {detected_mood}")
