"""
Flask API server with LoRA support.

Enhanced version of flask_server.py that loads and uses trained LoRA weights
for Duke-specific visual style.

Usage:
    # After training LoRA weights
    python app/flask_server_lora.py --lora-path lora_weights/chapel_covers_lora

Then visit http://localhost:5000
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from io import BytesIO
import base64
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from PIL import Image

from src.lora_integration import CoverArtPipelineWithLoRA
from src.prompt_builder import build_prompt, refine_prompt, NEGATIVE_PROMPT

# ===== FLASK SETUP =====
app = Flask(__name__)
APP_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "chapel_covers_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

DEFAULT_CKPT = "models/cnn_default_best.pt"
DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ===== GLOBAL STATE =====
pipeline = None
lora_path = None
use_lora = False


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

        pipeline = CoverArtPipelineWithLoRA(
            cnn_checkpoint=str(ckpt_path),
            diffusion_model_id=DIFFUSION_MODEL_ID,
        )

        # Load LoRA weights if provided
        if lora_path and Path(lora_path).exists():
            try:
                pipeline.load_lora_weights(lora_path, lora_scale=0.7)
                print(f"✓ LoRA weights loaded from {lora_path}")
            except Exception as e:
                print(f"⚠️  Failed to load LoRA: {e}")

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


@app.route("/status", methods=["GET"])
def status():
    """Get pipeline status and LoRA info."""
    try:
        init_pipeline()
        return jsonify({
            "success": True,
            "pipeline_loaded": pipeline is not None,
            "lora_loaded": use_lora and lora_path is not None,
            "lora_path": str(lora_path) if lora_path else None,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_audio():
    """Analyze uploaded audio file."""
    try:
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        file = request.files["audio"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file"}), 400

        filename = secure_filename(file.filename)
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        pipeline = init_pipeline()
        genre, genre_probs, mood_features = pipeline.classify_audio(tmp_path)

        lyrics = request.form.get("lyrics", "").strip()
        prompt = build_prompt(
            genre, mood_features=mood_features, lyrics=lyrics if lyrics else None
        )

        return jsonify({
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
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate_image():
    """Generate album cover art with optional LoRA."""
    try:
        data = request.get_json()
        prompt_text = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        steps = int(data.get("steps", 30))
        guidance = float(data.get("guidance", 7.5))
        use_lora_flag = data.get("use_lora", use_lora)
        lora_scale = float(data.get("lora_scale", 0.7))

        if not prompt_text:
            return jsonify({"success": False, "error": "Prompt is required"}), 400

        pipeline = init_pipeline()

        image = pipeline.generate_from_prompt_text(
            prompt_text=prompt_text,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=None,
            use_lora=use_lora_flag,
            lora_scale=lora_scale,
        )

        image_b64 = image_to_base64(image)

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_b64}",
            "used_lora": use_lora_flag and use_lora,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/refine", methods=["POST"])
def refine_image():
    """Refine an existing prompt."""
    try:
        data = request.get_json()
        current_prompt = data.get("current_prompt", "")
        refinement = data.get("refinement", "")
        negative_prompt = data.get("negative_prompt", "")
        use_lora_flag = data.get("use_lora", use_lora)

        if not current_prompt or not refinement:
            return jsonify(
                {"success": False, "error": "Current prompt and refinement required"}
            ), 400

        refined_prompt_text, is_valid = refine_prompt(current_prompt, refinement)

        if not is_valid:
            return jsonify({
                "success": False,
                "error": (
                    "Please give feedback related to the album cover, such as: "
                    "color, mood, lighting, composition, style, visual elements, "
                    "or other design aspects (e.g., 'more colorful', 'less dark', 'add rain')."
                ),
            }), 400

        pipeline = init_pipeline()
        image = pipeline.generate_from_prompt_text(
            prompt_text=refined_prompt_text,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=None,
            use_lora=use_lora_flag,
        )

        image_b64 = image_to_base64(image)

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_b64}",
            "refined_prompt": refined_prompt_text,
            "used_lora": use_lora_flag and use_lora,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        pipeline = init_pipeline()
        return jsonify({
            "status": "ok",
            "pipeline_loaded": pipeline is not None,
            "lora_loaded": use_lora and lora_path is not None,
        })
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
    parser = argparse.ArgumentParser(description="Chapel Covers Flask server with LoRA")
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to trained LoRA weights",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port",
    )

    args = parser.parse_args()

    lora_path = args.lora_path
    use_lora = lora_path is not None and Path(lora_path).exists()

    print("🏛️ Chapel Covers API Server (with LoRA support)")
    print(f"Starting on http://{args.host}:{args.port}")
    if use_lora:
        print(f"✓ LoRA enabled: {lora_path}")
    print("Press Ctrl+C to stop\n")

    app.run(debug=False, host=args.host, port=args.port, threaded=True)
