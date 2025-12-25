#!/usr/bin/env bash
set -euo pipefail

# macOS setup + run + test helper for batch-ocr/app.py
# - Creates a local venv under batch-ocr/.venv
# - Installs required CPU-only dependencies
# - Runs a smoke test that OCRs a generated 1-page PDF
# - Leaves the venv active only for the subshell; does not alter user shell

HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE_DIR}/.." && pwd)"
VENV_DIR="${HERE_DIR}/.venv"

PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

echo "[1/4] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[2/4] Upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "[3/4] Installing dependencies (CPU)"
# Notes:
# - Pin paddleocr to 2.7.x for the legacy .ocr() result format used by app.py
# - Pin numpy<2 for OpenCV compatibility with paddleocr 2.7.x
# - Install huggingface_hub<1.0 to satisfy Gradio's HfFolder import
pip install \
  paddlepaddle \
  'paddleocr==2.7.3' \
  'numpy<2' \
  pymupdf \
  pillow \
  'huggingface_hub<1.0' \
  gradio

echo "[4/4] Running smoke test against app.py (CPU)"
export HERE_DIR
export APP_PATH="${HERE_DIR}/app.py"
python - <<'PY'
import os, sys, tempfile
from pathlib import Path

APP_PATH = Path(os.environ.get('APP_PATH', '')).resolve()

if not APP_PATH.exists():
    raise SystemExit(f"app.py not found at {APP_PATH}")

# Dynamically load app.py as a module since folder name has a hyphen
import importlib.util
spec = importlib.util.spec_from_file_location("batch_ocr_app", str(APP_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

# Create a simple 1-page PDF from an image with clear text
from PIL import Image, ImageDraw, ImageFont

tmpdir = Path(tempfile.gettempdir())
pdf_path = tmpdir / "batch_ocr_smoketest.pdf"

W, H = 1600, 600
img = Image.new('RGB', (W, H), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Try to load a system font; fall back to default
font = None
for cand in [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Helvetica.ttf",
]:
    if Path(cand).exists():
        try:
            font = ImageFont.truetype(cand, 72)
            break
        except Exception:
            pass
if font is None:
    font = ImageFont.load_default()

text = "PaddleOCR smoke test: Hello 123 ABC"
draw.text((80, H//3), text, fill=(0,0,0), font=font)

# Save as PDF (raster PDF)
img.save(str(pdf_path), "PDF", resolution=150.0)
print(f"Created test PDF at: {pdf_path}")

# Initialize OCR (force CPU on macOS)
init_msg = mod.initialize_ocr(use_gpu=False)
print(init_msg)

# Run OCR on the generated PDF
res = mod.ocr_pdf_file(str(pdf_path))
if 'error' in res:
    print('OCR error:', res['error'])
    print(res.get('traceback',''))
    raise SystemExit(2)

full_text = res.get('full_text','')
print('--- OCR Extract Start ---')
print(full_text[:500])
print('--- OCR Extract End ---')

# Basic assertion: make sure some of the known words appear
ok = any(k in full_text for k in ["PaddleOCR", "Hello", "ABC", "123"]) and len(full_text.strip()) > 0
if not ok:
    raise SystemExit("OCR smoke test did not find expected text")

print("Smoke test passed: extracted expected text.")
PY

echo "All good. To launch the UI (CPU):"
echo "  source ${VENV_DIR}/bin/activate && python ${HERE_DIR}/app.py"
echo "Then open: http://127.0.0.1:7888"
