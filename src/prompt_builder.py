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
        "style": "melancholic album cover set on Duke campus at dusk, featuring Duke Chapel in soft blue twilight, wet stone reflecting ambient light, subtle golden window glow, mist in the air, deep navy and amber color palette, cinematic lighting, shallow depth of field, emotional and introspective mood, slight film grain, soft shadows",
        "subjects": ["Duke Chapel at dusk", "wet stone courtyard at twilight", "chapel window with golden glow"],
    },
    "classical": {
        "style": "elegant classical album cover with perfectly symmetrical composition inside Duke Chapel, towering gothic arches and columns, soft diffused daylight, marble and stone textures, muted blue and ivory tones, balanced framing, minimalistic and refined, high detail, no clutter, timeless and sophisticated aesthetic",
        "subjects": ["Duke Chapel interior with symmetrical arches", "gothic columns and vaulted ceiling", "chapel architecture in soft daylight"],
    },
    "country": {
        "style": "warm country album cover set in Duke Gardens at golden hour, soft sunlight filtering through trees, fallen leaves, wooden bench or pathway, earthy tones (brown, gold, muted green), relaxed and nostalgic atmosphere, slight haze, natural textures, gentle depth, Americana southern charm",
        "subjects": ["Duke Gardens at golden hour", "tree-lined pathways at sunset", "natural campus landscape with autumn leaves"],
    },
    "disco": {
        "style": "vibrant disco album cover on Duke campus at night, glowing neon blue and gold lights reflecting on pavement, dynamic lighting, motion blur, energetic composition, saturated colors, lens flares, celebratory mood, high contrast, stylized lighting, glossy surfaces, party atmosphere",
        "subjects": ["Duke campus with neon lights at night", "glowing stone architecture with light reflections", "dynamic nighttime campus scene with energy"],
    },
    "hiphop": {
        "style": "bold hip-hop album cover on Duke campus, low-angle perspective, strong shadows, dramatic lighting, campus architecture with street-art overlays, deep contrast, cool blue tones with gold highlights, gritty texture, urban energy, confident composition, high sharpness, cinematic intensity",
        "subjects": ["Duke campus from low-angle perspective", "architectural detail with dramatic shadows", "urban campus composition with strong contrast"],
    },
    "jazz": {
        "style": "smooth jazz album cover with Duke campus at night, soft spotlighting, calm atmosphere, deep blue tones with subtle gold accents, reflections on stone or pavement, minimal composition, elegant negative space, slightly blurred edges, relaxed and sophisticated mood",
        "subjects": ["Duke campus at night with soft lighting", "architectural reflection in stone", "minimalist composition with negative space"],
    },
    "metal": {
        "style": "intense metal album cover featuring Duke Chapel under a stormy sky, dramatic clouds, lightning or harsh contrast lighting, dark shadows, heavy textures, desaturated tones with sharp highlights, aggressive composition, cinematic darkness, high contrast, powerful and ominous mood",
        "subjects": ["Duke Chapel under stormy sky", "dark dramatic architecture with harsh lighting", "chapel silhouette against dramatic clouds"],
    },
    "pop": {
        "style": "bright pop album cover on Duke campus during a sunny day, vibrant blue sky, saturated Duke blue and gold palette, clean modern composition, cheerful mood, soft shadows, high clarity, minimal grain, energetic and uplifting, crisp lighting, playful aesthetic",
        "subjects": ["Duke campus on bright sunny day", "vibrant blue sky with saturated colors", "cheerful campus scene with clear lighting"],
    },
    "reggae": {
        "style": "relaxed reggae album cover in Duke Gardens at sunset, lush greenery, warm golden light, soft shadows, hints of green and amber tones, peaceful composition, breezy and uplifting mood, natural textures, slight haze, calm and rhythmic atmosphere",
        "subjects": ["Duke Gardens at sunset with lush greenery", "warm golden light through natural elements", "peaceful campus landscape at dusk"],
    },
    "rock": {
        "style": "energetic rock album cover on Duke campus, gritty texture, high contrast black and white with blue accents, film grain, motion blur, dynamic framing, edgy composition, strong shadows, raw and powerful mood, slightly chaotic energy",
        "subjects": ["Duke campus with dynamic framing", "high contrast architectural detail", "energetic campus composition with motion"],
    },
}

QUALITY_TAGS = "album cover, square format, centered composition, highly detailed, cinematic lighting, 4k, professional photography, trending on artstation, artistic aesthetic, no text"

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
        "sky", "clouds", "fog", "gold", "blue", "navy", "shadows", "contrast",
        "sharp", "soft", "focus", "blur", "texture", "grain", "film", "vintage",
        "modern", "indie", "editorial", "professional", "energetic", "peaceful",
        "dramatic", "subtle", "bold", "delicate", "intensity", "emphasis",
        "foreground", "background", "perspective", "angle", "framing", "crop",
        "saturate", "desaturate", "desaturated", "less", "more", "add", "remove",
        "black", "white", "grayscale", "monochrome", "bw", "noir",
        "sunset", "sunrise", "night", "dusk", "dawn", "daytime", "midday",
        "sunny", "dreamy", "gloomy", "moody", "calm", "peaceful", "intense", "energetic", "bold",
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
    Map user refinement input to AGGRESSIVE descriptive prompt language.

    Key principle: Refinements must OVERRIDE previous prompt instructions.
    Use repetition, caps, and strong descriptors to force Stable Diffusion to comply.

    Examples:
    - "darker" → "VERY DARK, DARK, dark, LOW-KEY, SHADOWS, nighttime..."
    - "add rain" → "RAIN, RAIN FALLING, wet, rainy, rain-soaked, precipitation..."
    - "black and white" → "MONOCHROMATIC, GRAYSCALE, BLACK, WHITE, no color..."
    """
    refinement_lower = refinement_text.lower().strip()

    # AGGRESSIVE Mapping dictionary: user input → STRONG prompt modification
    # Key: Use repetition, caps, and extremes to force the model to listen
    mappings = {
        # Black & White / Grayscale
        "black and white": "MONOCHROMATIC, GRAYSCALE, BLACK AND WHITE, no color, desaturated, gray, greyscale, noir, black and white photography, high contrast monochrome, completely desaturated",
        "grayscale": "MONOCHROMATIC, GRAYSCALE, BLACK AND WHITE, no color, desaturated, gray, greyscale, noir, black and white photography, high contrast monochrome",
        "bw": "MONOCHROMATIC, GRAYSCALE, BLACK AND WHITE, no color, desaturated",
        "monochrome": "MONOCHROMATIC, GRAYSCALE, BLACK AND WHITE, no color, desaturated, gray scale",

        # Darkness & Gloom - Aggressive dark color palette
        "darker": "VERY DARK, DARK, DARK, BLACK dominant color, DARK GREY dominant, LOW-KEY lighting, HEAVY SHADOWS, deep shadows, NIGHTTIME, underexposed, moody, gloomy, shadowy, dim, darkness, BLACK and grey tones, no bright colors",
        "make it darker": "EXTREMELY DARK, BLACK dominant, DARK GREY dominant, VERY DARK, DARK, DARK, BLACK and grey color palette, LOW-KEY lighting, HEAVY SHADOWS, NIGHTTIME aesthetic, underexposed, minimal light, dark mood, no brightness",
        "dark": "VERY DARK, DARK, LOW-KEY lighting, HEAVY SHADOWS, NIGHTTIME aesthetic, BLACK tones, dark grey dominant",
        "more dark": "VERY DARK, DARK, BLACK dominant, DARK GREY, LOW-KEY lighting, HEAVY SHADOWS, NIGHTTIME, underexposed, dark color palette",
        "gloomier": "GLOOMY, MOODY, VERY DARK, DARK, shadowy, LOW-KEY, overcast, cloudy, depressing, sad lighting, dark clouds, dark mood, BLACK and DARK GREY tones, minimal light",
        "gloomy": "GLOOMY, MOODY, DARK, shadowy, LOW-KEY, overcast, cloudy, depressing lighting, dark tones",
        "dim": "DIM lighting, low brightness, DARK, DARK GREY, BLACK shadows, underlit, LOW-KEY, dark atmosphere",
        "less bright": "LESS BRIGHT, VERY DARK, darker, dim, LOW-KEY, reduced brightness, HEAVY SHADOWS, dark tones",

        # Weather - NOTE: Rain removed (Stable Diffusion 1.5 unreliable with precipitation details)
        "add fog": "FOG, FOGGY, mist, misty, fog effect, atmospheric, hazy, fog-covered, foggy atmosphere",
        "fog": "FOG, FOGGY, mist, misty, fog effect, hazy, fog-covered",
        "misty": "MISTY, MIST, fog, foggy, hazy, atmospheric",
        "add clouds": "CLOUDY, CLOUDS, cloud formations, overcast, cloudy sky, dramatic clouds, cloud cover",
        "clouds": "CLOUDY, CLOUDS, cloud formations, overcast, dramatic clouds",
        "overcast": "OVERCAST, CLOUDY, CLOUDS, gray sky, gray clouds, no sun, cloud cover",
        "stormy": "STORMY, STORM, dark storm clouds, dramatic storm, lightning sky, threatening weather",
        "snow": "SNOW, SNOWY, snow-covered, snowfall, winter snow, snowy landscape, white snow",

        # Lighting & Brightness
        "brighter": "BRIGHT, BRIGHT, BRIGHT, HIGH-KEY lighting, illuminated, well-lit, bright light, sunny, luminous, daylight",
        "brighten": "BRIGHT, BRIGHT, HIGH-KEY lighting, illuminated, bright, well-lit, sunny",
        "bright": "BRIGHT, BRIGHT, BRIGHT, HIGH-KEY lighting, luminous, well-lit, sunny, daylight",
        "lighter": "LIGHTER, BRIGHT, lighter tones, illuminated, high-key lighting, brighter",
        "add light": "BRIGHT LIGHT, illumination, bright lighting, well-lit, luminous, sunlit",
        "sunny": "SUNNY, SUNSHINE, bright sun, daylight, golden hour, sunlit, bright",

        # Colors & Saturation
        "more colorful": "SATURATED, VIBRANT, COLORFUL, BOLD colors, vivid, chromatic, rich colors, saturated palette, bright colors",
        "colorful": "SATURATED, VIBRANT, COLORFUL, BOLD colors, vivid, chromatic, rich colors",
        "vibrant": "VIBRANT, SATURATED, bold colors, vivid, chromatic intensity, bright colors",
        "less colorful": "DESATURATED, muted, pale, low saturation, washed out, monochromatic feel",
        "muted": "MUTED colors, desaturated, pale, subdued, soft tones, low saturation",
        "saturated": "SATURATED, VIBRANT colors, bold colors, vivid, intense colors",
        "desaturated": "DESATURATED, muted, pale, subdued, low saturation, washed out",

        # Specific Colors
        "add blue": "BLUE, BLUE tones, BLUE, Duke Blue, blue dominant, blue color grading, blue atmosphere",
        "more blue": "MORE BLUE, BLUE tones, stronger blue, blue color grading, blue-hour lighting",
        "blue": "BLUE, BLUE tones, blue dominant, cool blue, blue lighting",
        "add gold": "GOLD, GOLDEN, GOLD tones, golden light, amber, warm gold, golden hues",
        "more golden": "GOLDEN, GOLDEN-HOUR, GOLDEN light, warm gold, amber, sunset glow, golden tones",
        "golden": "GOLDEN, GOLDEN-HOUR light, golden tones, warm gold, amber, sun-lit",
        "add purple": "PURPLE, PURPLE tones, violet, purple lighting, purple atmosphere, regal purple",
        "purple": "PURPLE, PURPLE tones, violet, purple atmosphere",
        "add red": "RED, RED tones, crimson, scarlet, red lighting, warm red",
        "red": "RED, RED tones, crimson, red atmosphere",

        # Time of Day
        "sunset": "SUNSET, SUNSET LIGHTING, GOLDEN, golden sunset, warm sunset, orange and gold, WARM TONES, bright golden hour, vibrant dusk light, sunset hour, SATURATED GOLDEN",
        "sunrise": "SUNRISE, SUNRISE LIGHTING, GOLDEN, golden sunrise, dawn light, morning sun, golden hour, warm morning, BRIGHT GOLDEN LIGHT",
        "night": "NIGHT, NIGHTTIME, night sky, nocturnal, stars, moonlight, dark evening, cool night tones",
        "dusk": "DUSK, TWILIGHT, WARM, early evening, dusk lighting, twilight hour, golden hour transition, GOLDEN LIGHT",
        "dawn": "DAWN, EARLY MORNING, sunrise time, first light, GOLDEN morning light, golden dawn, WARM DAWN",
        "daytime": "DAYTIME, DAY, bright daylight, sunny day, full daylight, midday, COLORFUL BRIGHT",
        "midday": "MIDDAY, NOON, bright sun overhead, bright daylight, bright, sunny, VIBRANT",

        # Mood & Atmosphere
        "gloomy": "GLOOMY, MOODY, DARK mood, sad, melancholic, depressing, dark atmosphere",
        "moody": "MOODY, DARK mood, atmospheric, brooding, intense mood, dark feeling",
        "peaceful": "PEACEFUL, CALM, serene, tranquil, quiet, relaxing, peaceful atmosphere",
        "calm": "CALM, PEACEFUL, serene, relaxed, tranquil, gentle mood",
        "dramatic": "DRAMATIC, INTENSE, bold, high contrast, powerful, striking, dramatic lighting",
        "intense": "INTENSE, DRAMATIC, bold, powerful, striking, energetic, high impact",
        "energetic": "ENERGETIC, VIBRANT, DYNAMIC, lively, active, bold, powerful, energetic mood",
        "bold": "BOLD, STRONG, striking, powerful, intense, confident, dramatic",
        "dreamy": "DREAMY, ethereal, soft focus, hazy, floating, fantastical, dreamy atmosphere, magical, surreal",
        "more dreamy": "DREAMY, ethereal, soft, hazy, floating, dreamy atmosphere, magical, surreal lighting",
        "dreamy mood": "DREAMY, ethereal, soft, hazy, dreamy atmosphere, magical aesthetic",

        # Architecture & Style
        "gothic": "GOTHIC, gothic architecture, gothic style, dark gothic, gothic mood, historical gothic",
        "more gothic": "GOTHIC, gothic architecture, gothic elements, dark gothic mood, gothic aesthetic",
        "cinematic": "CINEMATIC, cinematic lighting, cinematic mood, dramatic cinematic, cinema",
        "more cinematic": "CINEMATIC, cinematic lighting, dramatic cinematic style, cinema quality, cinematic mood",

        # Weather & Nature
        "add sun": "SUN, SUNLIGHT, bright sun visible, sun glare, solar, sunlit scene, sun element",
        "sunny": "SUNNY, SUNSHINE, bright sun, sunlit, golden light, daylight",
        "add moon": "MOON, MOONLIGHT, lunar, moonlit scene, night moon, moon visible, crescent moon",
        "moonlit": "MOONLIT, MOON, moonlight, lunar lighting, night moon, crescent",
        "add wind": "WINDY, WIND, windswept, wind effects, breezy, wind-blown",
        "windy": "WINDY, WIND, windswept, breezy, wind effects",
        "add snow": "SNOW, SNOWY, snowfall, winter snow, snow-covered, snowy landscape",

        # Detail & Clarity
        "sharper": "SHARP, CRISP, HIGH DEFINITION, clear, detailed, sharp focus, defined details, clear textures",
        "sharp": "SHARP, CRISP, clear, detailed, sharp focus, defined, high definition",
        "softer": "SOFT, SOFT FOCUS, blurry, gentle, smooth, delicate, soft aesthetic",
        "soft": "SOFT, SOFT FOCUS, smooth, gentle, delicate, soft aesthetic",
        "more detail": "DETAILED, TEXTURE, fine details, visible texture, intricate, detailed architecture",
        "detailed": "DETAILED, TEXTURE, fine details, visible texture, crisp, intricate",
        "less detail": "SOFT FOCUS, BLURRY, less detail, soft, out of focus, minimalist",
        "minimal": "MINIMAL, MINIMALIST, simple, clean, uncluttered, sparse, minimal elements",
        "simplify": "SIMPLE, MINIMAL, clean, uncluttered, reduced elements, minimal design",

        # Contrast & Composition
        "more contrast": "HIGH CONTRAST, CONTRAST, strong contrast, bold shadows, bright highlights, dramatic",
        "high contrast": "HIGH CONTRAST, CONTRAST, strong contrast, bold shadows, bright lights, dramatic",
        "less contrast": "LOW CONTRAST, soft, subtle, gentle transitions, muted, soft shadows",
        "low contrast": "LOW CONTRAST, soft transitions, subtle, gentle lighting, muted tones",
        "add texture": "TEXTURE, TEXTURED, textural, rough, detailed texture, visible texture, grain",
        "texture": "TEXTURE, TEXTURED, textural, detailed texture, rough, visible grain",
        "film grain": "FILM GRAIN, GRAIN, grainy, analog film, vintage film look, film aesthetic",
        "grainy": "GRAINY, GRAIN, film grain, vintage, analog aesthetic",
    }

    # Try exact matches first
    for key, enhancement in mappings.items():
        if refinement_lower == key:
            return enhancement

    # Then try partial matches (for "make it darker" → "darker")
    for key, enhancement in mappings.items():
        if key in refinement_lower:
            return enhancement

    # If no mapping found, return strong version of the refinement text itself
    return f"{refinement_text.upper()}, {refinement_text}, {refinement_text.lower()}"


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

    is_valid, error_msg = is_valid_refinement(refinement_instruction)
    if not is_valid:
        return base_prompt, False

    mapped_refinement = map_refinement_to_prompt(refinement_instruction)

    refinement_lower = refinement_instruction.lower()
    color_light_words = ["sunset", "sunrise", "golden", "bright", "colorful", "vibrant", "warm", "sunny"]
    if any(word in refinement_lower for word in color_light_words):
        refined = f"{mapped_refinement.strip()}, {base_prompt}"
    else:
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
