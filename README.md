# LTX-2.3 GGUF OpenAI-Compatible Video Generation Server (Kaggle T4x2)

This project sets up an OpenAI-compatible HTTP server on **Kaggle** using a **T4x2** GPU to run the **unsloth/LTX-2.3-GGUF** model (Q4_K_M quantization). You send a prompt from your **local machine**, the video is generated on Kaggle, and the resulting MP4 file is streamed back and saved **locally** (not on Kaggle).

---

## Files

| File | Description |
|------|-------------|
| `ltx_video_server.py` | Flask server that loads the LTX-2.3 GGUF model and exposes `/v1/video/generations` |
| `local_client.py` | Run this on your **local machine** to send requests and save the video file |
| `kaggle_install.sh` | Dependency installation script for Kaggle notebooks |
| `requirements.txt` | Python dependencies (av, diffusers, transformers, etc.) |
| `README.md` | This file |

---

## How It Works

1. **Kaggle Notebook** runs `ltx_video_server.py`.
2. The server downloads the **Q4_K_M** GGUF transformer from `unsloth/LTX-2.3-GGUF`.
3. An **ngrok** tunnel exposes the local Flask server to the internet.
4. Your **local machine** runs `local_client.py`, pointing at the ngrok URL.
5. The client sends a JSON payload (prompt, resolution, frames, etc.).
6. The server streams progress updates via **SSE** (Server-Sent Events).
7. When generation completes, the server base64-encodes the MP4 and sends it in the final SSE chunk.
8. The **client decodes and saves the MP4 locally**.

---

## Step-by-Step Setup

### 1. Kaggle Notebook Setup

Create a new Kaggle Notebook with **GPU T4x2** enabled.

#### Install dependencies (run in a cell):

```bash
!bash kaggle_install.sh
```

Or manually:

```bash
!pip install -q -r requirements.txt
```

#### Set your ngrok auth token (optional but recommended for stable URLs):

Get a free token from [ngrok.com](https://ngrok.com) and add it to Kaggle Secrets as `NGROK_AUTH_TOKEN`, or set it inline:

```python
import os
os.environ["NGROK_AUTH_TOKEN"] = "your_token_here"
```

#### Upload and run the server:

Upload `ltx_video_server.py` to Kaggle (Add Data → Upload) and run:

```python
!python ltx_video_server.py
```

You will see output like:

```
[INIT] Pre-loading pipeline...
[LOAD] Downloading / loading LTX-2.3 Q4_K_M GGUF transformer...
[LOAD] Building LTX2Pipeline from pretrained config...
[LOAD] Pipeline ready.
[NGROK] Public URL: https://xxxx.ngrok-free.app
[INIT] Starting Flask server on 0.0.0.0:5000
```

**Copy the ngrok URL** — this is what your local client will use.

---

### 2. Local Machine Setup

Download `local_client.py` to your local machine (or copy-paste it).

#### Requirements:

```bash
pip install -r requirements.txt
```

#### Generate a video:

```bash
python local_client.py \
    --url https://xxxx.ngrok-free.app/v1/video/generations \
    --prompt "A serene mountain lake at sunrise with mist rising" \
    --output my_video.mp4 \
    --width 768 \
    --height 512 \
    --num-frames 121 \
    --steps 40 \
    --guidance-scale 4.0 \
    --frame-rate 24.0 \
    --seed 42
```

The video will be saved as `my_video.mp4` on your **local disk**.

---

## API Specification

### `POST /v1/video/generations`

OpenAI-compatible video generation endpoint. Returns an **SSE stream**.

#### Request Body

```json
{
  "model": "ltx-2.3-22b-dev-q4_k_m",
  "prompt": "A serene mountain lake at sunrise with mist rising",
  "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "width": 768,
  "height": 512,
  "num_frames": 121,
  "num_inference_steps": 40,
  "guidance_scale": 4.0,
  "frame_rate": 24.0,
  "seed": 42
}
```

#### Constraints

- `width` and `height` must be **divisible by 32**.
- `num_frames` must satisfy `(num_frames - 1) % 8 == 0` (i.e., `num_frames = 8k + 1`).

#### Response (SSE Stream)

```
data: {"id": "gen-abc123", "object": "video.generation.chunk", "status": "started", ...}

data: {"id": "gen-abc123", "object": "video.generation.chunk", "status": "done", ...}

data: {"id": "gen-abc123", "object": "video.generation", "status": "completed", "format": "video/mp4", "data": "<base64-encoded-mp4>"}

data: [DONE]
```

### `GET /v1/models`

Lists available models (OpenAI-compatible).

### `GET /health`

Health check. Returns `{"status": "ok", "pipeline_loaded": true/false}`.

---

## Memory Optimization for T4 (16 GB VRAM)

The server applies the following optimizations automatically:

- `pipe.enable_model_cpu_offload()` — offloads inactive models to CPU.
- `pipe.vae.enable_slicing()` — reduces VAE memory spikes.
- `pipe.vae.enable_tiling()` — tiles large latents.
- **Q4_K_M quantization** keeps the transformer at ~14.3 GB on disk, with much lower VRAM usage at runtime.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--num-frames`, `--width`, `--height`, or `--steps`. |
| `ngrok tunnel not starting` | Ensure `NGROK_AUTH_TOKEN` is set. Free ngrok accounts have connection limits. |
| `ModuleNotFoundError` | Re-run `kaggle_install.sh` to install dependencies. |
| Video looks corrupted | Ensure `width % 32 == 0`, `height % 32 == 0`, and `(num_frames - 1) % 8 == 0`. |

---

## Notes

- The **Q4_K_M** model is the recommended balance of quality and speed for T4.
- You can switch to the **distilled** variant for faster generation (fewer steps) by changing `GGUF_CKPT` in `ltx_video_server.py`.
- The server pre-loads the pipeline on startup so the first request doesn't incur a cold-start delay.
- All heavy computation happens on Kaggle; your local machine only runs the lightweight client.

---

## License

This project uses the LTX-2.3 model under the [LTX-2 Community License Agreement](https://huggingface.co/unsloth/LTX-2.3-GGUF).
