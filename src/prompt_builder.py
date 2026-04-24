"""Build a text prompt for Stable Diffusion from CNN predictions + audio features.

This is the glue between the audio-classification and image-generation stages.
The quality of the final cover art depends heavily on the prompt quality.

Design principles:
    - Genre sets the overall visual language.
    - Mood features (tempo, energy, brightness) add expressive variation.
    - The prompt is structured as (subject) + (style) + (mood) + (quality tags).
    - A negative prompt pushes Stable Diffusion away from common failure modes.

# AI-generated via Claude (scaffold). Author owns the prompt design / tuning.
"""
from __future__ import annotations

from dataclasses import dataclass


GENRE_STYLES: dict[str, dict] = {
    "blues": {
        "style": "vintage blues album cover, 1960s analog photograph of a smoky jazz club, warm sepia tones",
        "subjects": ["a solitary guitarist silhouette", "an empty barstool at a dim club", "worn guitar strings close-up"],
    },
    "classical": {
        "style": "elegant classical album cover, baroque oil painting, ornate gold detailing, chiaroscuro lighting",
        "subjects": ["a grand piano in a candlelit hall", "a violin on velvet", "a marble statue in soft light"],
    },
    "country": {
        "style": "country album cover, rural Americana photograph, golden-hour warm sunlight, film grain",
        "subjects": ["a weathered cowboy boot on a wooden porch", "open prairie at sunset", "a dusty pickup truck"],
    },
    "disco": {
        "style": "disco album cover, 1970s glitter glam, mirror ball reflections, saturated neon pink and purple",
        "subjects": ["a mirror ball casting light rays", "a dance floor with colored tiles", "platform shoes mid-step"],
    },
    "hiphop": {
        "style": "hip hop album cover, bold graffiti typography, urban street photography, high-contrast black and gold",
        "subjects": ["a boombox on a concrete wall", "a subway car with spray-paint tags", "gold chain against a dark hoodie"],
    },
    "jazz": {
        "style": "jazz album cover, Blue Note Records aesthetic, cool blue tones, geometric typography, cigarette smoke",
        "subjects": ["a saxophone silhouette", "a trumpet on a stool", "a jazz quartet in dim stage light"],
    },
    "metal": {
        "style": "metal album cover, dark fantasy art, dramatic stormy sky, heavy shadows, gothic typography",
        "subjects": ["a skull wreathed in chains", "a lone tower on a mountain at night", "a cracked obsidian monument"],
    },
    "pop": {
        "style": "pop album cover, bright pastel gradient, clean studio lighting, glossy modern aesthetic",
        "subjects": ["a minimalist portrait under colored gels", "a pastel heart balloon", "bold typography on a candy-pink wall"],
    },
    "reggae": {
        "style": "reggae album cover, tropical sunset, green-gold-red color palette, warm organic textures",
        "subjects": ["palm leaves against a gold sunset", "a wooden guitar by the ocean", "dreadlocks silhouetted against the sky"],
    },
    "rock": {
        "style": "classic rock album cover, gritty film grain, dramatic stage lighting, leather and denim textures",
        "subjects": ["an electric guitar mid-swing", "a packed concert crowd under spotlights", "a beat-up amplifier"],
    },
}

QUALITY_TAGS = "highly detailed, cinematic composition, professional album cover art, trending on artstation"

NEGATIVE_PROMPT = (
    "text, letters, words, watermark, signature, low quality, blurry, pixelated, "
    "distorted, deformed, extra limbs, bad anatomy, flat, boring"
)


def describe_mood(mood_features: dict) -> str:
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


@dataclass
class Prompt:
    positive: str
    negative: str

    def __str__(self) -> str:
        return self.positive


def build_prompt(
    genre: str,
    mood_features: dict | None = None,
    subject_choice: int = 0,
) -> Prompt:
    if genre not in GENRE_STYLES:
        raise ValueError(f"Unknown genre '{genre}'. Known: {list(GENRE_STYLES)}")

    style_info = GENRE_STYLES[genre]
    subject = style_info["subjects"][subject_choice % len(style_info["subjects"])]
    style = style_info["style"]

    parts = [subject, style]
    if mood_features:
        mood_desc = describe_mood(mood_features)
        if mood_desc:
            parts.append(mood_desc)
    parts.append(QUALITY_TAGS)

    return Prompt(positive=", ".join(parts), negative=NEGATIVE_PROMPT)


if __name__ == "__main__":
    for genre in GENRE_STYLES:
        mood = {"tempo_bpm": 130, "energy": 0.12, "brightness": 2500}
        p = build_prompt(genre, mood)
        print(f"\n--- {genre.upper()} ---")
        print(f"Positive: {p.positive}")
