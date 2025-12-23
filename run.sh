#!/usr/bin/env sh
set -eu

# Simple runner for the Batch OCR container on Linux/macOS
# - Maps current directory to /workspace in the container
# - Exposes Gradio on port 7888 (change if needed)

IMAGE="batch-ocr:latest"

# Build tip:
#   docker build -t ${IMAGE} -f Dockerfile .

docker run --gpus all --rm -it \
  --shm-size=8gb --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 7888:7888 \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache":/root/.cache \
  ${IMAGE} bash -lc "python /workspace/app.py"

