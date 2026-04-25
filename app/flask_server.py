"""
Flask API server for Chapel Covers.

Exposes the ML pipeline as REST endpoints for the custom HTML frontend.
Handles audio upload, genre classification, image generation, and refinement.

Run:
    python app/flask_server.py

Then open http://localhost:5000 in your browser.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from io import BytesIO
import base64

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from PIL import Image

from src.pipeline import CoverArtPipeline
from src.prompt_builder import build_prompt, refine_prompt, NEGATIVE_PROMPT

# ===== FLASK SETUP =====
app = Flask(__name__)
APP_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "chapel_covers_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

DEFAULT_CKPT = "models/cnn_default_best.pt"
DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ===== GLOBAL PIPELINE =====
pipeline = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_pipeline():
    """Initialize the ML pipeline (cached in memory)."""
    global pipeline
    if pipeline is None:
        ckpt_path = Path(DEFAULT_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found at {DEFAULT_CKPT}. "
                "Train first with: python -m src.train"
            )
        pipeline = CoverArtPipeline(
            cnn_checkpoint=str(ckpt_path),
            diffusion_model_id=DIFFUSION_MODEL_ID,
        )
    return pipeline


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for JSON response."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ===== ROUTES =====

@app.route("/", methods=["GET"])
def index():
    """Serve the custom HTML frontend."""
    html_file = APP_DIR / "frontend.html"

    if not html_file.exists():
        return f"<h1>Error 404</h1><p>frontend.html not found at {html_file}</p>", 404

    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return Response(html_content, mimetype="text/html")
    except Exception as e:
        return f"<h1>Error</h1><p>Could not load frontend: {str(e)}</p>", 500


@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """
    Analyze uploaded audio file: extract genre, mood features, and build initial prompt.

    POST /analyze
    - File: audio file
    - Optional: lyrics (form field)

    Returns:
    {
        "success": true,
        "genre": "rock",
        "genre_probs": {...},
        "mood_features": {...},
        "prompt": {...}
    }
    """
    try:
        # Check for file
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid file type. Allowed: MP3, WAV, FLAC, OGG, M4A",
                    }
                ),
                400,
            )

        # Save temp file
        filename = secure_filename(file.filename)
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Initialize pipeline
        pipeline = init_pipeline()

        # Analyze audio
        genre, genre_probs, mood_features = pipeline.classify_audio(tmp_path)

        # Get lyrics if provided
        lyrics = request.form.get("lyrics", "").strip()

        # Build initial prompt
        prompt = build_prompt(
            genre, mood_features=mood_features, lyrics=lyrics if lyrics else None
        )

        return jsonify(
            {
                "success": True,
                "genre": genre,
                "genre_probs": genre_probs,
                "mood_features": mood_features,
                "prompt": {
                    "positive": prompt.positive,
                    "negative": prompt.negative,
                },
                "audio_path": tmp_path,
                "lyrics": lyrics,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate_image():
    """
    Generate album cover art from a prompt.

    POST /generate
    - prompt: text prompt
    - negative_prompt: negative prompt
    - steps: number of inference steps (default 30)
    - guidance: guidance scale (default 7.5)

    Returns:
    {
        "success": true,
        "image": "data:image/png;base64,..."
    }
    """
    try:
        data = request.get_json()
        prompt_text = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        steps = int(data.get("steps", 30))
        guidance = float(data.get("guidance", 7.5))

        if not prompt_text:
            return jsonify({"success": False, "error": "Prompt is required"}), 400

        pipeline = init_pipeline()

        image = pipeline.generate_from_prompt_text(
            prompt_text=prompt_text,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=None,
        )

        # Convert to base64
        image_b64 = image_to_base64(image)

        return jsonify(
            {"success": True, "image": f"data:image/png;base64,{image_b64}"}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/refine", methods=["POST"])
def refine_image():
    """
    Refine an existing prompt with a refinement instruction and regenerate.

    POST /refine
    - current_prompt: the current positive prompt
    - refinement: the refinement instruction (e.g., "make it darker")
    - negative_prompt: the negative prompt to keep

    Returns:
    {
        "success": true,
        "image": "data:image/png;base64,...",
        "refined_prompt": "..."
    }

    If refinement is invalid (off-topic), returns:
    {
        "success": false,
        "error": "Please give feedback related to the album cover..."
    }
    """
    try:
        data = request.get_json()
        current_prompt = data.get("current_prompt", "")
        refinement = data.get("refinement", "")
        negative_prompt = data.get("negative_prompt", "")

        if not current_prompt or not refinement:
            return (
                jsonify(
                    {"success": False, "error": "Current prompt and refinement required"}
                ),
                400,
            )

        # Refine the prompt (now returns tuple: (prompt, is_valid))
        refined_prompt_text, is_valid = refine_prompt(current_prompt, refinement)

        # Check if refinement was valid
        if not is_valid:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Please give feedback related to the album cover, such as: "
                        "color, mood, lighting, composition, campus style, visual elements, "
                        "or other design aspects (e.g., 'more colorful', 'less dark', 'add rain')."
                    ),
                }
            ), 400

        # Generate new image with refined prompt
        pipeline = init_pipeline()
        image = pipeline.generate_from_prompt_text(
            prompt_text=refined_prompt_text,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=None,
        )

        image_b64 = image_to_base64(image)

        return jsonify(
            {
                "success": True,
                "image": f"data:image/png;base64,{image_b64}",
                "refined_prompt": refined_prompt_text,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        pipeline = init_pipeline()
        return jsonify({"status": "ok", "pipeline_loaded": pipeline is not None})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("🏛️ Chapel Covers API Server")
    print("Starting on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
