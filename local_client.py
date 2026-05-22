"""
Local client to send requests to the Kaggle LTX video server and save the
resulting MP4 file on your local disk (NOT on Kaggle).

Usage:
    python local_client.py --url https://<your-ngrok-url>.ngrok-free.app/v1/video/generations \
                           --prompt "A cat playing piano" \
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
                print(f"  Progress: {progress.get('current', 0)}/{progress.get('total', 0)}", end="\r")
            elif obj == "video.generation" and status == "completed":
                final_data = data
                print("\n[CLIENT] Generation complete. Saving file...")

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
