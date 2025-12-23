
# FROM nvcr.io/nvidia/paddlepaddle:24.12-py3
# FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5
FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.9-cudnn9.9

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python tooling
# ---------------------------------------------------------
# RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# PaddleOCR + PDF + UI stack (SAFE)
# ---------------------------------------------------------
RUN pip install \
    paddleocr \
    gradio \
    PyMuPDF \
    pypdfium2 \
    opencv-python-headless \
    pillow \
    addict \
    frontend

# /root/.paddlex/official_models/PP-OCRv5_server_det