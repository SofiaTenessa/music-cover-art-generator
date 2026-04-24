"""Streamlit web app for the Music Genre → Cover Art Generator.

Run:
    streamlit run app/streamlit_app.py

# AI-generated via Claude (scaffold). Author owns UX layout decisions.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.pipeline import CoverArtPipeline


st.set_page_config(
    page_title="Music → Cover Art",
    page_icon="🎵",
    layout="wide",
)

st.title("Music Genre → Album Cover Art Generator")
st.caption("Upload a song clip. A CNN predicts its genre, and Stable Diffusion generates cover art to match.")


DEFAULT_CKPT = "models/cnn_default_best.pt"


@st.cache_resource(show_spinner="Loading models (first run downloads Stable Diffusion ~4 GB)...")
def load_pipeline(ckpt_path: str, diffusion_model_id: str) -> CoverArtPipeline:
    return CoverArtPipeline(
        cnn_checkpoint=ckpt_path,
        diffusion_model_id=diffusion_model_id,
    )


with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("CNN checkpoint", value=DEFAULT_CKPT)
    diffusion_id = st.selectbox(
        "Diffusion model",
        [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/sdxl-turbo",
        ],
        help="SDXL-Turbo is ~5x faster but slightly lower quality.",
    )
    steps = st.slider("Inference steps", 4, 50, 30)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, step=0.5)
    seed_input = st.text_input("Seed (blank = random)", value="")


uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    accept_multiple_files=False,
)

if uploaded is not None:
    st.audio(uploaded)

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    if not Path(ckpt_path).exists():
        st.error(f"CNN checkpoint not found at `{ckpt_path}`. Train first with `python -m src.train`.")
        st.stop()

    pipeline = load_pipeline(ckpt_path, diffusion_id)

    with st.spinner("Analyzing audio..."):
        genre, probs, mood = pipeline.classify_audio(tmp_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted genre")
        st.markdown(f"### 🎼 **{genre.title()}**")
        top5 = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
        st.bar_chart({g: [p] for g, p in top5})
    with col2:
        st.subheader("Audio features")
        st.metric("Tempo (BPM)", f"{mood['tempo_bpm']:.0f}")
        st.metric("Energy", f"{mood['energy']:.3f}")
        st.metric("Brightness", f"{mood['brightness']:.0f}")
        st.metric("Estimated key", mood["key_estimate"])

    from src.prompt_builder import build_prompt
    prompt = build_prompt(genre, mood_features=mood)
    with st.expander("View generated prompt"):
        st.code(prompt.positive, language=None)
        st.caption("Negative prompt: " + prompt.negative)

    if st.button("Generate cover art", type="primary"):
        with st.spinner("Generating... (30-90s on Mac, faster on GPU)"):
            seed = int(seed_input) if seed_input.strip() else None
            image = pipeline.generate_image(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
            )
        st.subheader("Generated cover art")
        st.image(image, use_column_width=True)

        tmp_out = Path(tempfile.mkdtemp()) / "cover.png"
        image.save(tmp_out)
        with open(tmp_out, "rb") as f:
            st.download_button(
                "Download cover art",
                data=f.read(),
                file_name=f"{Path(uploaded.name).stem}_cover.png",
                mime="image/png",
            )
else:
    st.info("Upload an audio file to get started. For best results, use a 20-30 second clip.")

st.markdown("---")
st.caption("CS 372 final project · custom CNN trained on GTZAN · Stable Diffusion for generation.")
