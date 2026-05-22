"""
OpenAI-compatible video generation server for LTX-2.3-GGUF on Kaggle (T4x2).

Usage on Kaggle:
1. Add this script as a Kaggle Notebook
2. Run the cell to start the server
3. Use the public URL (via ngrok/cloudflared) to send requests from your local machine
4. The generated video is streamed back and saved locally on your machine, NOT on Kaggle

Install dependencies first:
    pip install -r requirements.txt
"""

import os
import sys
import json
import uuid
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify, Response, stream_with_context
from flask.typing import ResponseReturnValue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Lightricks/LTX-2.3"  # Base config repo for LTX-2.3
# Unsloth GGUF Q4_K_M checkpoint
GGUF_CKPT = "https://huggingface.co/unsloth/LTX-2.3-GGUF/blob/main/ltx-2.3-22b-dev-Q4_K_M.gguf"

DEVICE = "cuda:0"
DTYPE = "bfloat16"
SEED = 42

# Kaggle / server settings
PORT = 5000
# Set this via Kaggle secrets or environment variable
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", None)

# ---------------------------------------------------------------------------
# Imports & model loading
# ---------------------------------------------------------------------------
import torch
from diffusers import (
    GGUFQuantizationConfig,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.export_utils import encode_video
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Global pipeline (loaded once)
_pipe = None
_pipe_lock = threading.Lock()


def load_pipeline() -> LTX2Pipeline:
    """Load the LTX-2.3 GGUF pipeline with memory-efficient settings for T4x2."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print("[LOAD] Step 1/5: Preparing dtype and quantization config...")
    dtype = getattr(torch, DTYPE)

    transformer_kwargs = {}
    _, ext = os.path.splitext(GGUF_CKPT)
    if ext == ".gguf":
        transformer_kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
        print("[LOAD]   Quantization config: GGUF with compute_dtype=bfloat16")

    print("[LOAD] Step 2/5: Downloading LTX-2.3 Q4_K_M GGUF transformer (14.3 GB)...")
    print("[LOAD]   This may take a few minutes depending on connection speed.")

    print("[LOAD] Step 3/5: Loading transformer from_single_file with version=2.3...")
    transformer = LTX2VideoTransformer3DModel.from_single_file(
        GGUF_CKPT,
        config=MODEL_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        single_file_version="2.3",
        **transformer_kwargs,
    )
    print("[LOAD]   Transformer loaded successfully.")

    print("[LOAD] Step 4/5: Building LTX2Pipeline from_pretrained...")
    pipe = LTX2Pipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=dtype,
    )
    print("[LOAD]   Pipeline built successfully.")

    print("[LOAD] Step 5/5: Applying memory optimizations for T4x2...")
    pipe.enable_model_cpu_offload(device=DEVICE)
    print("[LOAD]   CPU offload enabled.")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
        print("[LOAD]   VAE slicing enabled.")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        print("[LOAD]   VAE tiling enabled.")

    _pipe = pipe
    print("[LOAD] Pipeline ready! All steps complete.")
    return _pipe


# ---------------------------------------------------------------------------
# OpenAI-compatible helpers
# ---------------------------------------------------------------------------

def make_openai_chunk(data: dict) -> str:
    """Serialize a dict as an SSE chunk."""
    return f"data: {json.dumps(data)}\n\n"


def generate_video_stream(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    frame_rate: float,
    seed: int,
):
    """
    Run inference and yield SSE events.
    The final event contains the MP4 bytes as a base64 string so the client
    can reconstruct and save the file locally.
    """
    import base64

    pipe = load_pipeline()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Progress callback
    progress = {"current": 0, "total": num_inference_steps}

    def callback_on_step_end(pipe_obj, step_index, timestep, callback_kwargs):
        progress["current"] = step_index + 1
        return callback_kwargs

    yield make_openai_chunk({
        "id": f"gen-{uuid.uuid4().hex[:8]}",
        "object": "video.generation.chunk",
        "status": "started",
        "progress": {"current": 0, "total": num_inference_steps},
    })

    # Run inference
    start_time = time.time()
    video, audio = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
        return_dict=False,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    inference_time = time.time() - start_time

    # Post-process to uint8 video tensor
    video_np = (video * 255).round().astype("uint8")
    video_tensor = torch.from_numpy(video_np)

    # Encode to MP4 in memory
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    encode_video(
        video_tensor[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=tmp_path,
    )

    with open(tmp_path, "rb") as f:
        mp4_bytes = f.read()
    os.remove(tmp_path)

    b64_video = base64.b64encode(mp4_bytes).decode("utf-8")

    yield make_openai_chunk({
        "id": f"gen-{uuid.uuid4().hex[:8]}",
        "object": "video.generation.chunk",
        "status": "done",
        "progress": {"current": num_inference_steps, "total": num_inference_steps},
        "inference_time_sec": round(inference_time, 2),
    })

    # Final payload with the actual file
    yield make_openai_chunk({
        "id": f"gen-{uuid.uuid4().hex[:8]}",
        "object": "video.generation",
        "status": "completed",
        "format": "video/mp4",
        "data": b64_video,
    })

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/v1/video/generations", methods=["POST"])
def video_generations() -> ResponseReturnValue:
    """
    OpenAI-compatible endpoint for video generation.

    Request JSON body (example):
    {
        "model": "ltx-2.3-22b-dev-q4_k_m",
        "prompt": "A serene mountain lake at sunrise with mist rising",
        "negative_prompt": "worst quality, inconsistent motion, blurry",
        "width": 768,
        "height": 512,
        "num_frames": 121,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "frame_rate": 24.0,
        "seed": 42
    }

    Returns SSE stream. The client should read the final chunk which contains
    base64-encoded MP4 data and save it locally.
    """
    body = request.get_json(force=True, silent=True) or {}

    prompt = body.get("prompt", "")
    if not prompt:
        return jsonify({"error": {"message": "Missing 'prompt'", "type": "invalid_request"}}), 400

    negative_prompt = body.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    width = int(body.get("width", 768))
    height = int(body.get("height", 512))
    num_frames = int(body.get("num_frames", 121))
    num_inference_steps = int(body.get("num_inference_steps", 40))
    guidance_scale = float(body.get("guidance_scale", 4.0))
    frame_rate = float(body.get("frame_rate", 24.0))
    seed = int(body.get("seed", SEED))

    # Validate divisibility constraints for LTX-2.3
    if width % 32 != 0 or height % 32 != 0:
        return jsonify({
            "error": {
                "message": "Width and height must be divisible by 32",
                "type": "invalid_request"
            }
        }), 400
    if (num_frames - 1) % 8 != 0:
        return jsonify({
            "error": {
                "message": "num_frames - 1 must be divisible by 8 (i.e., num_frames = 8k + 1)",
                "type": "invalid_request"
            }
        }), 400

    return Response(
        stream_with_context(generate_video_stream(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            frame_rate=frame_rate,
            seed=seed,
        )),
        mimetype="text/event-stream",
    )


@app.route("/v1/models", methods=["GET"])
def list_models() -> ResponseReturnValue:
    """List available models (OpenAI-compatible)."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "ltx-2.3-22b-dev-q4_k_m",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "unsloth",
            }
        ],
    })


@app.route("/health", methods=["GET"])
def health() -> ResponseReturnValue:
    return jsonify({"status": "ok", "pipeline_loaded": _pipe is not None})


# ---------------------------------------------------------------------------
# Ngrok tunnel (optional, for Kaggle public URL)
# ---------------------------------------------------------------------------

def start_ngrok_tunnel(port: int, auth_token: Optional[str] = None):
    """Start an ngrok tunnel so Kaggle notebook is reachable from local machine."""
    try:
        from pyngrok import ngrok
    except ImportError:
        print("[WARN] pyngrok not installed. Skipping ngrok tunnel.")
        return None

    if auth_token:
        ngrok.set_auth_token(auth_token)

    public_url = ngrok.connect(port, "http")
    print(f"[NGROK] Public URL: {public_url}")
    return public_url


# ---------------------------------------------------------------------------
# Client-side helper script (to be run on your LOCAL machine)
# ---------------------------------------------------------------------------
LOCAL_CLIENT_SCRIPT = '''
"""
Local client to send requests to the Kaggle LTX video server and save the
resulting MP4 file on your local disk (NOT on Kaggle).

Usage:
    python local_client.py --url https://<your-ngrok-url>.ngrok-free.app/v1/video/generations \\
                           --prompt "A cat playing piano" \\
                           --output my_video.mp4
"""
import argparse
import base64
import json
import sys
import requests


def stream_generate(url: str, payload: dict, output_path: str):
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()

    final_data = None
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data_str = decoded[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            obj = data.get("object", "")
            status = data.get("status", "")

            if obj == "video.generation.chunk" and status in ("started", "done"):
                progress = data.get("progress", {})
                print(f"  Progress: {progress.get('current', 0)}/{progress.get('total', 0)}", end="\\r")
            elif obj == "video.generation" and status == "completed":
                final_data = data
                print("\\n[CLIENT] Generation complete. Saving file...")

    if final_data is None:
        print("[ERROR] No video data received.", file=sys.stderr)
        sys.exit(1)

    b64_video = final_data.get("data")
    if not b64_video:
        print("[ERROR] Empty video payload.", file=sys.stderr)
        sys.exit(1)

    mp4_bytes = base64.b64decode(b64_video)
    with open(output_path, "wb") as f:
        f.write(mp4_bytes)
    print(f"[CLIENT] Saved to: {output_path} ({len(mp4_bytes)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Local client for Kaggle LTX video server")
    parser.add_argument("--url", required=True, help="Server URL (e.g., ngrok public URL)")
    parser.add_argument("--prompt", required=True, help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--output", default="output.mp4", help="Local output MP4 path")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = {
        "model": "ltx-2.3-22b-dev-q4_k_m",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "frame_rate": args.frame_rate,
        "seed": args.seed,
    }

    stream_generate(args.url, payload, args.output)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Pre-load the pipeline so first request is faster
    print("[INIT] Pre-loading pipeline...")
    load_pipeline()

    # Optionally start ngrok tunnel
    public_url = start_ngrok_tunnel(PORT, NGROK_AUTH_TOKEN)

    # Write local client script to disk so user can download it
    client_path = Path("local_client.py")
    client_path.write_text(LOCAL_CLIENT_SCRIPT, encoding="utf-8")
    print(f"[INIT] Wrote local client script: {client_path.absolute()}")
    print("[INIT] Download this file to your local machine and run it to generate videos.")

    print(f"[INIT] Starting Flask server on 0.0.0.0:{PORT}")
    if public_url:
        print(f"[INIT] Public endpoint: {public_url}/v1/video/generations")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
