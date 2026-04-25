"""Chapel Covers: Duke-inspired album cover art generator.

A one-page Streamlit app that takes an audio file and optional lyrics,
predicts the genre with a custom CNN, and generates Duke-inspired indie album covers
using Stable Diffusion with chapel, gothic architecture, and campus aesthetics.

Run:
    streamlit run app/streamlit_app.py

Branding:
    - Slogan: "From the Chapel to your headphones."
    - Colors: Duke Navy Blue (#012169), Duke Royal Blue (#00539B), White

# AI-generated scaffold via Claude. Author owns UX layout and feature decisions.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.pipeline import CoverArtPipeline
from src.prompt_builder import Prompt, build_prompt, refine_prompt, NEGATIVE_PROMPT

# ===== PAGE CONFIG & STYLING =====
st.set_page_config(
    page_title="Chapel Covers",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Duke branding
st.markdown(
    """
    <style>
    :root {
        --duke-navy: #012169;
        --duke-royal: #00539B;
        --duke-white: #FFFFFF;
        --gold: #D4AF37;
    }

    /* Main heading */
    .header-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: var(--gold);
        margin-bottom: 0.2em;
        letter-spacing: 0.05em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .header-slogan {
        text-align: center;
        font-size: 1.2em;
        color: var(--duke-white);
        font-style: italic;
        margin-bottom: 1.5em;
    }

    .section-divider {
        border-bottom: 1px solid var(--duke-royal);
        margin: 1.5em 0;
    }

    /* Section titles - WHITE TEXT */
    h2 {
        color: var(--duke-white) !important;
    }

    h3 {
        color: var(--duke-white) !important;
    }

    .stSubheader {
        color: var(--duke-white) !important;
    }

    /* Info and success boxes */
    .info-box {
        background-color: rgba(0, 82, 155, 0.1);
        border-left: 4px solid var(--duke-royal);
        padding: 1em;
        border-radius: 0.3em;
        margin: 1em 0;
    }

    .success-box {
        background-color: rgba(1, 33, 105, 0.1);
        border-left: 4px solid var(--gold);
        padding: 1em;
        border-radius: 0.3em;
        margin: 1em 0;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--duke-navy) !important;
        color: white !important;
        border-radius: 0.5em;
        font-weight: bold;
        padding: 0.6em 1.5em !important;
        border: 2px solid var(--duke-royal) !important;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: var(--duke-royal) !important;
        border-color: var(--gold) !important;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.3);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        color: var(--duke-white) !important;
    }

    /* Text styling */
    .stMarkdown {
        color: var(--duke-white) !important;
    }

    /* Metric styling */
    .stMetric {
        background-color: rgba(0, 82, 155, 0.05);
        padding: 1em;
        border-radius: 0.5em;
        border-left: 3px solid var(--duke-royal);
    }

    .stMetricLabel {
        color: var(--duke-white) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== HEADER =====
st.markdown(
    '<div class="header-title">🏛️ Chapel Covers</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="header-slogan">From the Chapel to your headphones.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align: center; color: #FFFFFF; font-size: 1.05em; margin-bottom: 1.5em;">'
    'Transform your music into Duke-inspired indie album cover art. '
    'Upload a song, optionally add lyrics, and AI will generate a moody chapel-themed cover.'
    '</p>',
    unsafe_allow_html=True,
)

# ===== SESSION STATE INITIALIZATION =====
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "genre" not in st.session_state:
    st.session_state.genre = None
if "genre_probs" not in st.session_state:
    st.session_state.genre_probs = None
if "mood_features" not in st.session_state:
    st.session_state.mood_features = None
if "lyrics" not in st.session_state:
    st.session_state.lyrics = ""
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "refinement_history" not in st.session_state:
    st.session_state.refinement_history = []

# ===== CONFIGURATION =====
DEFAULT_CKPT = "models/cnn_default_best.pt"


@st.cache_resource(
    show_spinner="Loading models (first run downloads Stable Diffusion ~4 GB)..."
)
def load_pipeline(ckpt_path: str, diffusion_model_id: str) -> CoverArtPipeline:
    return CoverArtPipeline(
        cnn_checkpoint=ckpt_path,
        diffusion_model_id=diffusion_model_id,
    )


# ===== STEP 1: AUDIO UPLOAD =====
st.markdown('<h3 style="color: #FFFFFF;">📁 Step 1: Upload Your Song</h3>', unsafe_allow_html=True)
st.markdown('<p style="color: #CCCCCC;">Choose an audio file (MP3, WAV, FLAC, OGG, M4A)</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # Save uploaded file to temp location
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.session_state.audio_path = tmp_path

    # Show player
    st.audio(uploaded_file)

    # Load pipeline
    if not Path(DEFAULT_CKPT).exists():
        st.error(
            f"CNN checkpoint not found at `{DEFAULT_CKPT}`. "
            "Train first with `python -m src.train`."
        )
        st.stop()

    st.session_state.pipeline = load_pipeline(DEFAULT_CKPT, "runwayml/stable-diffusion-v1-5")

    # Analyze audio
    if st.session_state.genre is None:
        with st.spinner("🎼 Analyzing your song..."):
            genre, genre_probs, mood = st.session_state.pipeline.classify_audio(
                st.session_state.audio_path
            )
            st.session_state.genre = genre
            st.session_state.genre_probs = genre_probs
            st.session_state.mood_features = mood

    # Display genre prediction
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #FFFFFF; font-size: 1.1em;"><strong>Predicted Genre:</strong> {st.session_state.genre.title()}</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tempo", f"{st.session_state.mood_features['tempo_bpm']:.0f} BPM")
    with col2:
        st.metric("Energy", f"{st.session_state.mood_features['energy']:.2f}")
    with col3:
        st.metric("Brightness", f"{st.session_state.mood_features['brightness']:.0f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ===== STEP 2: OPTIONAL LYRICS =====
    st.markdown('<h3 style="color: #FFFFFF;">✍️ Step 2: Optional — Add Lyrics (for emotional influence)</h3>', unsafe_allow_html=True)

    use_lyrics = st.checkbox(
        "Include lyrics to influence the mood of the cover art",
        value=False,
    )

    if use_lyrics:
        st.session_state.lyrics = st.text_area(
            "Paste lyrics here",
            value=st.session_state.lyrics,
            height=120,
            label_visibility="collapsed",
            placeholder="Paste song lyrics (or a few key lines) to add emotional tone...",
        )
    else:
        st.session_state.lyrics = ""

    # ===== STEP 3: GENERATE INITIAL COVER =====
    st.markdown('<h3 style="color: #FFFFFF;">🎨 Step 3: Generate Cover Art</h3>', unsafe_allow_html=True)

    if st.button("Generate Album Cover", type="primary", use_container_width=True):
        with st.spinner("✨ Creating your chapel-inspired cover (30-90 seconds)..."):
            prompt = build_prompt(
                st.session_state.genre,
                mood_features=st.session_state.mood_features,
                lyrics=st.session_state.lyrics if st.session_state.lyrics else None,
            )
            st.session_state.current_prompt = prompt

            image = st.session_state.pipeline.generate_image(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=None,
            )
            st.session_state.generated_image = image
            st.session_state.refinement_history = []

    # ===== DISPLAY GENERATED IMAGE =====
    if st.session_state.generated_image is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_img, col_info = st.columns([2, 1])

        with col_img:
            st.image(
                st.session_state.generated_image,
                use_column_width=True,
                caption="Your Chapel Cover Art",
            )

        with col_info:
            st.markdown('<h4 style="color: #FFFFFF;">Details</h4>', unsafe_allow_html=True)
            st.write(f"**Genre:** {st.session_state.genre.title()}")
            if st.session_state.lyrics:
                st.write("**Lyrics:** ✓ Included")
            with st.expander("View Prompt"):
                st.code(st.session_state.current_prompt.positive, language=None)

            # Download button
            tmp_out = Path(tempfile.mkdtemp()) / "cover.png"
            st.session_state.generated_image.save(tmp_out)
            with open(tmp_out, "rb") as f:
                st.download_button(
                    "⬇️ Download Cover",
                    data=f.read(),
                    file_name=f"{Path(uploaded_file.name).stem}_chapel_cover.png",
                    mime="image/png",
                    use_container_width=True,
                )

        # ===== STEP 4: REFINE VIA CHATBOT =====
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #FFFFFF;">🎭 Step 4: Refine (Optional)</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: #CCCCCC;">Don\'t like it? Describe what you want to change, and we\'ll regenerate.</p>',
            unsafe_allow_html=True,
        )

        refinement_input = st.text_input(
            "How would you like to adjust the cover?",
            placeholder="e.g., make it darker, add more rain, more gothic, less colorful...",
            label_visibility="collapsed",
        )

        if st.button("Refine & Regenerate", use_container_width=True):
            if refinement_input.strip():
                with st.spinner("🔄 Refining your cover..."):
                    # Append refinement to the current prompt
                    refined_prompt_text = refine_prompt(
                        st.session_state.current_prompt.positive,
                        refinement_input,
                    )

                    # Generate new image with refined prompt
                    new_image = st.session_state.pipeline.generate_from_prompt_text(
                        prompt_text=refined_prompt_text,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        seed=None,
                    )

                    # Update session state
                    st.session_state.generated_image = new_image
                    st.session_state.refinement_history.append(refinement_input)

                    # Rerun to show new image
                    st.rerun()
            else:
                st.warning("Please describe what you'd like to change.")

        # Show refinement history
        if st.session_state.refinement_history:
            with st.expander(
                f"📝 Refinement History ({len(st.session_state.refinement_history)})"
            ):
                for i, refine_text in enumerate(st.session_state.refinement_history):
                    st.write(f"{i + 1}. {refine_text}")

else:
    # No file uploaded yet
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write(
        "**Getting started:**\n\n"
        "1. Upload a song (20-30 seconds works great)\n"
        "2. (Optional) Add lyrics for emotional influence\n"
        "3. Click 'Generate Album Cover'\n"
        "4. Refine until you love it"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "🏛️ Chapel Covers · CS 372 final project · Custom CNN on GTZAN · Stable Diffusion generation · "
    "Duke-inspired aesthetics, no official logos"
)
