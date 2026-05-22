#!/bin/bash
# Run this in a Kaggle notebook cell to install dependencies before starting the server

set -e

echo "[INSTALL] Upgrading pip..."
pip install -q --upgrade pip

echo "[INSTALL] Installing dependencies from requirements.txt..."
pip install -q -r requirements.txt

echo "[INSTALL] Verifying torch CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

echo "[INSTALL] Done. You can now run: python ltx_video_server.py"
