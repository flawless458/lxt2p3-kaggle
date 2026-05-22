#!/bin/bash
# Run this in a Kaggle notebook cell to install dependencies before starting the server

set -e

echo "[INSTALL] Upgrading pip..."
pip install -q --upgrade pip

echo "[INSTALL] Installing diffusers, transformers, accelerate, huggingface_hub..."
pip install -q diffusers transformers accelerate huggingface_hub

echo "[INSTALL] Installing Flask and pyngrok..."
pip install -q flask pyngrok

echo "[INSTALL] Verifying torch CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

echo "[INSTALL] Done. You can now run: python ltx_video_server.py"
